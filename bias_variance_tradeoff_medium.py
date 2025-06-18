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
bias_variance_records = []
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
        named_matrix_params = [
            (name.replace('module.', '').replace('_orig_mod.', '').replace('weight', '').strip('.'), param)
            for name, param in model.named_parameters()
            if param.ndim == 2 and param.grad is not None and 'blocks' in name
        ]
        
        for param_name, param in named_matrix_params:
            gradients_all_gpus = [torch.zeros_like(param.grad) for _ in range(world_size)]
            dist.all_gather(gradients_all_gpus, param.grad)
            
            for subset_size in range(1, min(8, world_size)):
                try:
                    residual_norms = jackknife_gradient_residuals_on_subsets(
                        gradients_all_gpus, subset_size, norm="fro"
                    )
                    if master_process:
                        bias_variance_records.append({
                            'minibatch_idx': step,
                            'parameter_name': param_name,
                            'subset_size': subset_size,
                            'residual_norms': residual_norms
                        })
                except Exception as e:
                    print0(f"Error in jackknife analysis for {param_name}, subset_size {subset_size}: {e}")
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
if master_process and bias_variance_records:
    print0("Saving bias/variance analysis results...")
    df_records = []
    for record in bias_variance_records:
        for i, norm_val in enumerate(record['residual_norms']):
            df_records.append({
                'minibatch_idx': record['minibatch_idx'],
                'parameter_name': record['parameter_name'],
                'subset_size': record['subset_size'],
                'subset_idx': i,
                'residual_norm': norm_val
            })
    
    if df_records:
        df = pd.DataFrame(df_records)
        csv_filename = f"logs/{run_id_full}_bias_variance_analysis.csv"
        df.to_csv(csv_filename, index=False)
        print0(f"Bias/variance analysis saved to {csv_filename}")
    else:
        print0("No bias/variance records to save")

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()