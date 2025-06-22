import sys
with open(sys.argv[0]) as f:
    code = f.read()

from utils import *
import pandas as pd

EVALUATE_TRADEOFF_EVERY = 100

run_id, rank, world_size, device, master_process = setup_distributed_training()
print0, run_id_full, logfile = setup_logging(run_id, master_process)
log_system_info(print0, code)

args = Hyperparameters()
model, optimizers, opt2params = create_model_and_optimizers(args, rank, world_size)
model = torch.compile(model, dynamic=False)
bias_variance_records_pure = []
bias_variance_records_momentum = []
warmup_kernels(model, optimizers, args)

torch.cuda.reset_peak_memory_stats()
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
dist.barrier()
t0 = time.perf_counter()

for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step, args.num_iterations))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{args.num_iterations} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        dist.barrier()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id_full}", exist_ok=True)
            torch.save(log, f"logs/{run_id_full}/state_step{step:06d}.pt")
        break

    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step, args.num_iterations)).backward()
    
    # Bias/variance analysis
    if step % EVALUATE_TRADEOFF_EVERY == 0 and step > 0:
        print0(f"Running bias/variance analysis at step {step}", console=True)
        
        # Get all block parameters with gradients for analysis
        all_params = [(name, param) for name, param in model.named_parameters() if param.ndim >= 2]
        block_params = [(name, param) for name, param in all_params if 'blocks' in name]
        params_with_grad = [(name, param) for name, param in block_params if param.grad is not None]
        
        # Find the Muon optimizer
        muon_optimizer = None
        for opt in optimizers:
            if hasattr(opt, '__class__') and opt.__class__.__name__ == 'Muon':
                muon_optimizer = opt
                break
        
        # Extract parameters for bias-variance analysis
        analysis_params_pure = []
        analysis_params_momentum = []
        
        for name, param in params_with_grad:
            # Clean the parameter name
            clean_name = name.replace('module.', '').replace('_orig_mod.', '')
            
            # Handle MLP parameters
            if any(param_type in clean_name for param_type in ['mlp.fc_w', 'mlp.proj_w']):
                analysis_params_pure.append((clean_name, param.grad))
                
                # Get momentum buffer if this param is optimized by Muon
                if muon_optimizer is not None and param in muon_optimizer.state and "momentum_buffer" in muon_optimizer.state[param]:
                    momentum_buffer = muon_optimizer.state[param]["momentum_buffer"]
                    momentum = muon_optimizer.param_groups[0]["momentum"]
                    momentum_grad = momentum * momentum_buffer + (1 - momentum) * param.grad.float()
                    analysis_params_momentum.append((clean_name, momentum_grad.bfloat16()))
            
            # Handle attention parameters
            elif any(param_type in clean_name for param_type in ['attn.qkvo_w', 'attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'attn.c_attn', 'attn.c_proj']):
                # Check if it's a merged QKVO parameter (3D tensor)
                if 'attn.qkvo_w' in clean_name and param.grad.ndim == 3 and param.grad.shape[0] == 4:
                    base_name = clean_name.replace('qkvo_w', '')
                    # qkvo_w has shape (4, hdim, dim) - split into Q, K, V, O components
                    
                    # Check if this 3D parameter has momentum buffer
                    has_momentum = (muon_optimizer is not None and 
                                  param in muon_optimizer.state and 
                                  "momentum_buffer" in muon_optimizer.state[param])
                    
                    if has_momentum:
                        momentum_buffer = muon_optimizer.state[param]["momentum_buffer"]
                        momentum = muon_optimizer.param_groups[0]["momentum"]
                        momentum_grad = momentum * momentum_buffer + (1 - momentum) * param.grad.float()
                    
                    for i, component in enumerate(['q', 'k', 'v', 'o']):
                        component_name = f"{base_name}{component}"
                        component_grad = param.grad[i]
                        analysis_params_pure.append((component_name, component_grad))
                        
                        # Add momentum component if available
                        if has_momentum:
                            component_momentum_grad = momentum_grad[i].bfloat16()
                            analysis_params_momentum.append((component_name, component_momentum_grad))
                elif param.grad.ndim == 2:
                    # Handle individual attention parameters
                    analysis_params_pure.append((clean_name, param.grad))
                    
                    # Get momentum buffer if this param is optimized by Muon
                    if muon_optimizer is not None and param in muon_optimizer.state and "momentum_buffer" in muon_optimizer.state[param]:
                        momentum_buffer = muon_optimizer.state[param]["momentum_buffer"]
                        momentum = muon_optimizer.param_groups[0]["momentum"]
                        momentum_grad = momentum * momentum_buffer + (1 - momentum) * param.grad.float()
                        analysis_params_momentum.append((clean_name, momentum_grad.bfloat16()))
        
        print0(f"Analyzing {len(analysis_params_pure)} pure gradient parameters and {len(analysis_params_momentum)} momentum parameters", console=True)
        
        # Ensure all ranks have the same parameter count (critical for NCCL)
        param_counts = torch.tensor([len(analysis_params_pure), len(analysis_params_momentum)], device=device)
        dist.all_reduce(param_counts, op=dist.ReduceOp.MAX)
        if param_counts[0] != len(analysis_params_pure) or param_counts[1] != len(analysis_params_momentum):
            print0(f"FATAL: Parameter count mismatch across ranks. Max counts: {param_counts.tolist()}, This rank: {[len(analysis_params_pure), len(analysis_params_momentum)]}", console=True)
            return
        
        # Analyze pure gradients
        for param_name, param_grad in analysis_params_pure:
            gradients_all_gpus = [torch.zeros_like(param_grad) for _ in range(world_size)]
            dist.all_gather(gradients_all_gpus, param_grad)
            
            for subset_size in range(1, min(8, world_size)):
                try:
                    residual_norms = jackknife_gradient_residuals_on_subsets(
                        gradients_all_gpus, subset_size, norm="fro"
                    )
                    if master_process:
                        bias_variance_records_pure.append({
                            'minibatch_idx': step,
                            'parameter_name': param_name,
                            'subset_size': subset_size,
                            'residual_norms': residual_norms
                        })
                except Exception as e:
                    print0(f"Error in pure gradient jackknife analysis for {param_name}, subset_size {subset_size}: {e}")
        
        # Analyze momentum gradients
        for param_name, momentum_grad in analysis_params_momentum:
            gradients_all_gpus = [torch.zeros_like(momentum_grad) for _ in range(world_size)]
            dist.all_gather(gradients_all_gpus, momentum_grad)
            
            for subset_size in range(1, min(8, world_size)):
                try:
                    residual_norms = jackknife_gradient_residuals_on_subsets(
                        gradients_all_gpus, subset_size, norm="fro"
                    )
                    if master_process:
                        bias_variance_records_momentum.append({
                            'minibatch_idx': step,
                            'parameter_name': param_name,
                            'subset_size': subset_size,
                            'residual_norms': residual_norms
                        })
                except Exception as e:
                    print0(f"Error in momentum gradient jackknife analysis for {param_name}, subset_size {subset_size}: {e}")
        
        print0(f"Completed bias/variance analysis at step {step}")
    
    opt2futures = {
        opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
        for opt, params in opt2params.items()
    }
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step, args.num_iterations, args.cooldown_frac)
    if len(optimizers) > 1:
        for group in optimizers[1].param_groups:
            frac = min(step / 300, 1)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt in optimizers:
        torch.futures.collect_all(opt2futures[opt]).wait()
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

# Save bias/variance analysis results
if master_process:
    # Save pure gradient analysis
    if bias_variance_records_pure:
        print0("Saving pure gradient bias/variance analysis results...")
        df_records_pure = []
        for record in bias_variance_records_pure:
            for i, norm_val in enumerate(record['residual_norms']):
                df_records_pure.append({
                    'minibatch_idx': record['minibatch_idx'],
                    'parameter_name': record['parameter_name'],
                    'subset_size': record['subset_size'],
                    'subset_idx': i,
                    'residual_norm': norm_val
                })
        
        df_pure = pd.DataFrame(df_records_pure)
        csv_filename_pure = f"logs/{run_id_full}_bias_variance_analysis_pure.csv"
        df_pure.to_csv(csv_filename_pure, index=False)
        print0(f"Pure gradient bias/variance analysis saved to {csv_filename_pure}")
    else:
        print0("No pure gradient bias/variance records to save")
    
    # Save momentum gradient analysis
    if bias_variance_records_momentum:
        print0("Saving momentum gradient bias/variance analysis results...")
        df_records_momentum = []
        for record in bias_variance_records_momentum:
            for i, norm_val in enumerate(record['residual_norms']):
                df_records_momentum.append({
                    'minibatch_idx': record['minibatch_idx'],
                    'parameter_name': record['parameter_name'],
                    'subset_size': record['subset_size'],
                    'subset_idx': i,
                    'residual_norm': norm_val
                })
        
        df_momentum = pd.DataFrame(df_records_momentum)
        csv_filename_momentum = f"logs/{run_id_full}_bias_variance_analysis_momentum.csv"
        df_momentum.to_csv(csv_filename_momentum, index=False)
        print0(f"Momentum gradient bias/variance analysis saved to {csv_filename_momentum}")
    else:
        print0("No momentum gradient bias/variance records to save")

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()