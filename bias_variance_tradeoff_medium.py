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
        
        # Hypothesis 5: Check all optimizers
        print0(f"Rank {rank}: Total optimizers: {len(optimizers)}", console=True)
        for i, opt in enumerate(optimizers):
            print0(f"Rank {rank}: Optimizer {i}: {opt.__class__.__name__}", console=True)
        
        # Hypothesis 4: Check all parameter groups  
        print0(f"Rank {rank}: Muon has {len(muon_optimizer.param_groups)} param groups", console=True)
        total_muon_params = []
        for i, group in enumerate(muon_optimizer.param_groups):
            print0(f"Rank {rank}: Group {i} has {len(group['params'])} parameters", console=True)
            total_muon_params.extend(group['params'])
        
        # Hypothesis 1: Check distributed ownership calculation
        print0(f"Rank {rank}: world_size={world_size}, rank={rank}", console=True)
        expected_owned_indices = [i for i in range(len(total_muon_params))[::world_size] if i + rank < len(total_muon_params)]
        print0(f"Rank {rank}: Should own parameter indices: {expected_owned_indices}", console=True)
        
        # Hypothesis 6: Check which parameters actually have state
        params_with_state = len(muon_optimizer.state)
        print0(f"Rank {rank}: Parameters with state: {params_with_state}/{len(total_muon_params)}", console=True)
        
        for name, param in block_params[:3]:  # Only check first 3 to avoid spam
            clean_name = name.replace('module.', '').replace('_orig_mod.', '')
            
            # Hypothesis 3: Check parameter identity
            param_id = id(param)
            muon_param_ids = [id(p) for p in total_muon_params]
            id_match = param_id in muon_param_ids
            param_idx = muon_param_ids.index(param_id) if id_match else -1
            
            # Hypothesis 1: Check if this rank should own this parameter
            should_own = param_idx in expected_owned_indices if param_idx >= 0 else False
            
            # Check state
            has_state = param in muon_optimizer.state
            state_keys = list(muon_optimizer.state[param].keys()) if has_state else []
            
            print0(f"Rank {rank}: {clean_name} - idx={param_idx}, id_match={id_match}, should_own={should_own}, has_state={has_state}, keys={state_keys}", console=True)
            
            # Get momentum buffer - every Muon parameter must have one
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
        
        print0(f"Analyzing {len(analysis_params_pure)} parameters", console=True)
        
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
                    print0(f"Error in pure gradient analysis for {param_name}, subset_size {subset_size}: {e}")
        
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
                    print0(f"Error in momentum gradient analysis for {param_name}, subset_size {subset_size}: {e}")
    
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
    
    # Debug momentum buffer issues when they first appear
    if muon_optimizer is not None and step == 11:
        print0(f"=== DEBUGGING AT STEP {step} ===", console=True)
        
        # Check all optimizers
        print0(f"Rank {rank}: Total optimizers: {len(optimizers)}", console=True)
        for i, opt in enumerate(optimizers):
            print0(f"Rank {rank}: Optimizer {i}: {opt.__class__.__name__}", console=True)
        
        # Check all parameter groups  
        print0(f"Rank {rank}: Muon has {len(muon_optimizer.param_groups)} param groups", console=True)
        total_muon_params = []
        for i, group in enumerate(muon_optimizer.param_groups):
            print0(f"Rank {rank}: Group {i} has {len(group['params'])} parameters", console=True)
            total_muon_params.extend(group['params'])
        
        # Check distributed ownership calculation
        print0(f"Rank {rank}: world_size={world_size}, rank={rank}", console=True)
        expected_owned_indices = [i for i in range(len(total_muon_params))[::world_size] if i + rank < len(total_muon_params)]
        print0(f"Rank {rank}: Should own parameter indices: {expected_owned_indices}", console=True)
        
        # Check which parameters actually have state
        params_with_state = len(muon_optimizer.state)
        print0(f"Rank {rank}: Parameters with state: {params_with_state}/{len(total_muon_params)}", console=True)
        
        # Check first few parameters that this rank should own
        for idx in expected_owned_indices[:5]:  # Check first 5 owned parameters
            param = total_muon_params[idx]
            has_state = param in muon_optimizer.state
            state_keys = list(muon_optimizer.state[param].keys()) if has_state else []
            print0(f"Rank {rank}: Param {idx} (shape {param.shape}) - has_state={has_state}, keys={state_keys}", console=True)
    
    # Assert Muon parameters owned by this rank have proper momentum state
    if muon_optimizer is not None and step > 10:
        muon_params = muon_optimizer.param_groups[0]['params']
        for base_i in range(len(muon_params))[::world_size]:
            if base_i + rank < len(muon_params):
                param_idx = base_i + rank
                param = muon_params[param_idx]
                print0(f"Rank {rank}: Checking param {param_idx} (shape {param.shape})", console=True)
                
                if param not in muon_optimizer.state:
                    print0(f"Rank {rank}: ERROR - param {param_idx} not in optimizer state!", console=True)
                    continue
                    
                state = muon_optimizer.state[param]
                assert len(state) > 0, f"Empty state for Muon parameter {param_idx} with shape {param.shape} on rank {rank}"
                
                if "momentum_buffer" not in state:
                    print0(f"Rank {rank}: ERROR - param {param_idx} missing momentum_buffer, has keys: {list(state.keys())}", console=True)
                    continue
                    
                momentum_buffer = state["momentum_buffer"]
                assert momentum_buffer.numel() > 0, f"Empty momentum buffer for parameter {param_idx} with shape {param.shape} on rank {rank}"
                assert (momentum_buffer != 0).any(), f"All-zero momentum buffer for parameter {param_idx} with shape {param.shape} on rank {rank}"
                print0(f"Rank {rank}: param {param_idx} OK - momentum buffer shape {momentum_buffer.shape}, non-zero: {(momentum_buffer != 0).any()}", console=True)
    
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