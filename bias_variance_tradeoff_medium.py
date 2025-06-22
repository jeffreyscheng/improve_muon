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
    
    # Get Muon optimizer (needed for both analysis and assertions)
    muon_optimizer = None
    for opt in optimizers:
        if hasattr(opt, '__class__') and opt.__class__.__name__ == 'Muon':
            muon_optimizer = opt
            break
    
    # Bias/variance analysis
    if step % EVALUATE_TRADEOFF_EVERY == 0 and step > 0:
        print0(f"Running bias/variance analysis at step {step}", console=True)
        
        # Get all block parameters optimized by Muon
        muon_params = set(muon_optimizer.param_groups[0]['params']) if muon_optimizer else set()
        block_params = [(name, param) for name, param in model.named_parameters() 
                       if 'blocks' in name and param in muon_params and param.grad is not None]
        
        analysis_params_pure = []
        analysis_params_momentum = []
        
        # Get all block parameters optimized by Muon, but only analyze owned ones
        muon_params = muon_optimizer.param_groups[0]['params']
        owned_params = []
        for base_i in range(len(muon_params))[::world_size]:
            if base_i + rank < len(muon_params):
                owned_params.append(muon_params[base_i + rank])
        
        owned_param_set = set(owned_params)
        
        # Get owned block parameters for analysis
        block_params = [(name, param) for name, param in model.named_parameters() 
                       if 'blocks' in name and param in owned_param_set and param.grad is not None]
        
        # Collect all analysis parameters first, then do collective operations
        all_analysis_params_pure = []
        all_analysis_params_momentum = []
        
        for name, param in block_params:
            clean_name = name.replace('module.', '').replace('_orig_mod.', '')
            
            # Get momentum buffer - guaranteed to exist for owned parameters
            state = muon_optimizer.state[param]
            momentum_buffer = state["momentum_buffer"]
            momentum = muon_optimizer.param_groups[0]["momentum"]
            momentum_grad = momentum * momentum_buffer + (1 - momentum) * param.grad.float()
            
            # Handle MLP parameters (2D)
            if any(param_type in clean_name for param_type in ['mlp.fc_w', 'mlp.proj_w']):
                analysis_params_pure.append((clean_name, param.grad))
                analysis_params_momentum.append((clean_name, momentum_grad.bfloat16()))
            
            # Handle attention QKVO parameters (3D) - split into components
            elif 'attn.qkvo_w' in clean_name and param.grad.ndim == 3:
                base_name = clean_name.replace('qkvo_w', '')
                for i, component in enumerate(['q', 'k', 'v', 'o']):
                    component_name = f"{base_name}{component}"
                    analysis_params_pure.append((component_name, param.grad[i]))
                    analysis_params_momentum.append((component_name, momentum_grad[i].bfloat16()))
        
        print0(f"Analyzing {len(analysis_params_pure)} pure and {len(analysis_params_momentum)} momentum parameters", console=True)
        
        # Synchronize all ranks before starting collective operations
        dist.barrier()
        
        
        # Collect analysis data from all ranks using the same pattern as Muon optimizer
        futures = []
        
        # First collect all pure gradients using distributed pattern
        for base_i in range(len(analysis_params_pure))[::world_size]:
            if base_i + rank < len(analysis_params_pure):
                param_name, param_grad = analysis_params_pure[base_i + rank]
                gradients_all_gpus = [torch.zeros_like(param_grad) for _ in range(world_size)]
                future = dist.all_gather(gradients_all_gpus, param_grad, async_op=True)
                futures.append((future.get_future(), param_name, gradients_all_gpus, 'pure'))
        
        # Then collect all momentum gradients
        for base_i in range(len(analysis_params_momentum))[::world_size]:
            if base_i + rank < len(analysis_params_momentum):
                param_name, momentum_grad = analysis_params_momentum[base_i + rank]
                gradients_all_gpus = [torch.zeros_like(momentum_grad) for _ in range(world_size)]
                future = dist.all_gather(gradients_all_gpus, momentum_grad, async_op=True)
                futures.append((future.get_future(), param_name, gradients_all_gpus, 'momentum'))
        
        # Wait for all collectives to complete, then do analysis
        torch.futures.collect_all([f[0] for f in futures]).wait()
        
        # Process results
        for future, param_name, gradients_all_gpus, analysis_type in futures:
            for subset_size in range(1, min(8, world_size)):
                try:
                    residual_norms = jackknife_gradient_residuals_on_subsets(
                        gradients_all_gpus, subset_size, norm="fro"
                    )
                    if master_process:
                        target_records = bias_variance_records_pure if analysis_type == 'pure' else bias_variance_records_momentum
                        target_records.append({
                            'minibatch_idx': step,
                            'parameter_name': param_name,
                            'subset_size': subset_size,
                            'residual_norms': residual_norms
                        })
                except Exception as e:
                    print0(f"Error in {analysis_type} gradient analysis for {param_name}, subset_size {subset_size}: {e}")
    
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
    
    
    # Assert Muon parameters owned by this rank have proper momentum state
    if muon_optimizer is not None and step > 10:
        muon_params = muon_optimizer.param_groups[0]['params']
        for base_i in range(len(muon_params))[::world_size]:
            if base_i + rank < len(muon_params):
                param = muon_params[base_i + rank]
                state = muon_optimizer.state[param]
                assert len(state) > 0, f"Empty state for Muon parameter with shape {param.shape} on rank {rank}"
                assert "momentum_buffer" in state, f"Missing momentum_buffer for parameter with shape {param.shape} on rank {rank}"
                momentum_buffer = state["momentum_buffer"]
                assert momentum_buffer.numel() > 0, f"Empty momentum buffer for parameter with shape {param.shape} on rank {rank}"
                assert (momentum_buffer != 0).any(), f"All-zero momentum buffer for parameter with shape {param.shape} on rank {rank}"
    
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

# Save bias/variance analysis results
if master_process:
    if bias_variance_records_pure:
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
        print0(f"Pure gradient analysis saved to {csv_filename_pure}")
    
    if bias_variance_records_momentum:
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
        print0(f"Momentum gradient analysis saved to {csv_filename_momentum}")

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()