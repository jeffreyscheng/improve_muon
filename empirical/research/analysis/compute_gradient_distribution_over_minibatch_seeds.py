#!/usr/bin/env python3
"""
Compute gradient distributions over multiple minibatch SGD seeds for noise model analysis.
This script performs the expensive computation once per training run, saving results to CSV files.

Usage:
    torchrun --standalone --nproc_per_node=8 compute_gradient_distribution_over_minibatch_seeds.py <run_id> [--testing] [--force]
    
Example:
    torchrun --standalone --nproc_per_node=8 compute_gradient_distribution_over_minibatch_seeds.py 000_2db40055-63ed-48dd-bb98-8cccdb78a501
"""

import os
import sys
import glob
import re
import json
import time
from pathlib import Path

# Memory optimization like training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch._inductor.config.coordinate_descent_tuning = False
import torch.distributed as dist
import numpy as np
import pandas as pd

# Import all dependencies at top level
from empirical.research.training.training_core import setup_distributed_training, get_window_size_blocks, Hyperparameters, warmup_kernels
from empirical.research.analysis.offline_logging import deserialize_model_checkpoint, compute_singular_values, compute_stable_rank
from empirical.research.analysis.map import get_weight_matrix_iterator, get_research_log_path


def precompile_svd_kernels(device: torch.device, rank: int):
    """Precompile SVD kernels for all expected tensor shapes to avoid compilation overhead during analysis."""
    if rank == 0:
        print("Precompiling SVD kernels for expected tensor shapes...")
    
    # Expected shapes for GPT(vocab_size, 16, 8, 1024, context_len):
    # - Attention Q/K/V/O: (1024, 1024) 
    # - MLP Input: (4096, 1024)
    # - MLP Output: (1024, 4096)
    shapes = [(1024, 1024), (4096, 1024), (1024, 4096)]
    
    start_time = time.time()
    for i, shape in enumerate(shapes):
        if rank == 0:
            print(f"  Compiling SVD kernel {i+1}/{len(shapes)}: {shape}")
        
        # Create dummy tensor and trigger compilation
        dummy = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=False)
        with torch.no_grad():
            _, _, _ = torch.linalg.svd(dummy, full_matrices=False)
        
        # Clean up
        del dummy
        torch.cuda.empty_cache()
    
    # Ensure all ranks complete compilation before proceeding
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        compile_time = time.time() - start_time
        print(f"SVD kernel precompilation completed in {compile_time:.1f}s")
        print("All ranks synchronized - ready for analysis")


def extract_step_from_checkpoint_path(checkpoint_path: Path) -> int:
    """Extract step number from checkpoint filename."""
    step_match = re.search(r'step_(\d+)', str(checkpoint_path))
    return int(step_match.group(1)) if step_match else 0


def setup_model_from_checkpoint(checkpoint_file: str, device: torch.device):
    """Load and setup model from checkpoint matching training's model creation."""
    from empirical.research.training.architecture import GPT
    
    rank = dist.get_rank()
    if rank == 0:
        print(f"Rank {rank}: Loading checkpoint {checkpoint_file}")
    
    # Create fresh model exactly like training does
    args = Hyperparameters()
    model = GPT(args.vocab_size, 16, 8, 1024, max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    
    # Broadcast parameters like training does
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    
    # Load checkpoint weights
    checkpoint_data = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint_data['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # Compile model like training does
    model = torch.compile(model, dynamic=False)
    
    # Warmup kernels like training does to coordinate compilation across ranks
    args = Hyperparameters()
    warmup_kernels(model, [], args)  # No optimizers needed for analysis
    
    if rank == 0:
        print(f"Rank {rank}: Model loaded and compiled")
    
    return model


def create_real_data_loader(args, rank: int, world_size: int):
    """Create real training data loader using FineWeb dataset."""
    from empirical.research.training.training_core import distributed_data_generator
    return distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)


def extract_model_analysis_distributed(model, use_gradients: bool = True, accumulated_gradients: dict = None):
    """Extract analysis from model parameters using distributed GPU SVD."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Get all parameters (model is replicated on all ranks like training)
    layer_properties = get_weight_matrix_iterator(model, only_hidden=True)
    all_params = list(layer_properties.items())
    
    # Distribute SVD computation work across ranks, not the parameters themselves
    all_param_splits = []
    for (param_type, layer_num), param in all_params:
        if use_gradients:
            param_key = (param_type, layer_num)
            if accumulated_gradients is not None:
                # Use accumulated gradients if provided
                tensor = accumulated_gradients.get(param_key, None)
                if tensor is None:
                    continue
            else:
                # Use raw gradients from param.grad
                tensor = param.grad
                if tensor is None:
                    continue
        else:
            tensor = param
        
        # No need to split - already handled by get_weight_matrix_iterator
        all_param_splits.append(((param_type, layer_num), tensor))
    
    # Distribute SVD work across ranks
    splits_per_rank = len(all_param_splits) // world_size
    start_idx = rank * splits_per_rank
    end_idx = start_idx + splits_per_rank if rank < world_size - 1 else len(all_param_splits)
    my_splits = all_param_splits[start_idx:end_idx]
    
    if rank == 0:
        print(f"    Distributed {len(all_param_splits)} SVD computations across {world_size} ranks ({len(my_splits)} per rank)")
    
    # Process my assigned SVD computations
    local_results = {}
    for (param_type, layer_num), tensor in my_splits:
        # Use GPU SVD instead of CPU numpy
        tensor_f32 = tensor.detach().to(torch.float32)
        try:
            _, s, _ = torch.linalg.svd(tensor_f32, full_matrices=False)
            sv = s.cpu().numpy()
            stable_rank = compute_stable_rank(sv)
            local_results[(param_type, layer_num)] = {'sv': sv, 'stable_rank': stable_rank}
        except Exception:
            pass
    
    # Gather all results on rank 0
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)
    
    if rank == 0:
        # Merge results from all ranks
        merged_results = {}
        for rank_results in all_results:
            merged_results.update(rank_results)
        return merged_results
    
    return {}


def compute_c_with_mean_truth_distributed(my_gradient_dict, rank, world_size):
    """
    Distributed computation of C estimates with mean truth using accumulated gradients.
    Each rank stores one minibatch accumulated gradient, we compute mean across ranks, 
    then each rank computes C for its accumulated gradient vs the mean.
    
    Args:
        my_gradient_dict: Dict of {param_name: accumulated_gradient_tensor} for this rank's minibatch
        rank: Current rank
        world_size: Total number of ranks
    
    Returns:
        c_estimates_dict: Dict of {param_name: c_values_array} 
    """
    c_estimates = {}
    
    for param_key, my_gradient in my_gradient_dict.items():
        try:
            # All-reduce to compute mean gradient across ranks
            mean_gradient = my_gradient.clone()
            dist.all_reduce(mean_gradient, op=dist.ReduceOp.AVG)
            
            # Compute SVD of mean gradient (truth) 
            U0, _, V0h = torch.linalg.svd(mean_gradient, full_matrices=False)
            V0 = V0h.T
            
            # Compute SVD of my gradient
            U, _, Vh = torch.linalg.svd(my_gradient, full_matrices=False) 
            V = Vh.T
            
            # Compute C estimate: |<u, u0>| * |<v, v0>|
            cu = torch.abs(torch.einsum('mr,mr->r', U, U0))  # (r,)
            cv = torch.abs(torch.einsum('nr,nr->r', V, V0))  # (r,)
            c_estimate = cu * cv  # (r,)
            
            c_estimates[param_key] = c_estimate.cpu().numpy()
            
        except Exception as e:
            print(f"Error computing C for {param_key}: {e}")
            r = min(my_gradient.shape)
            c_estimates[param_key] = np.zeros(r)
    
    return c_estimates

def aggregate_singular_values_and_stable_ranks(all_gradient_results, weight_results, c_estimates_dict=None):
    """Aggregate singular values and stable rank statistics across minibatches."""
    param_data = {}
    
    all_param_keys = set()
    for mb_grads in all_gradient_results:
        all_param_keys.update(mb_grads.keys())
    
    for param_key in all_param_keys:
        # param_key is already a tuple (param_type, layer_num)
        param_type, layer_num = param_key
        
        sv_arrays = [mb_grads[param_key]['sv'] for mb_grads in all_gradient_results if param_key in mb_grads]
        stable_ranks = [mb_grads[param_key]['stable_rank'] for mb_grads in all_gradient_results if param_key in mb_grads]
        
        min_len = min(len(sv) for sv in sv_arrays)
        aligned_svs = [sv[:min_len] for sv in sv_arrays]
        aligned_svs_array = np.array(aligned_svs)
        
        key = (param_type, layer_num)
        param_data[key] = {
            'gradient_singular_values': aligned_svs_array,  # 8x1024 ndarray of all gradient SVs
            'weight_stable_rank': weight_results[param_key]['stable_rank'],
            'gradient_stable_rank_mean': np.mean(stable_ranks),
            'gradient_stable_rank_std': np.std(stable_ranks)
        }
        
        # Add C estimates if provided
        if c_estimates_dict and param_key in c_estimates_dict:
            # Only c_with_mean_truth now
            param_data[key]['c_with_mean_truth'] = c_estimates_dict[param_key]
    
    return param_data


def compute_analysis_for_step(step: int, checkpoint_file: str, num_minibatches: int, rank: int, world_size: int, device: torch.device):
    """Compute singular values and stable rank analysis using accumulated gradients (with momentum)."""
    
    model = setup_model_from_checkpoint(checkpoint_file, device)
    args = Hyperparameters()
    
    if rank == 0:
        print(f"Rank {rank}: Starting weight analysis")
        weight_start = time.time()
    weight_results = extract_model_analysis_distributed(model, use_gradients=False)
    torch.cuda.empty_cache()
    if rank == 0:
        print(f"Rank {rank}: Weight analysis took {time.time() - weight_start:.1f}s")
    
    # Initialize momentum buffers for accumulated gradient computation
    momentum_buffers = {}
    layer_properties = get_weight_matrix_iterator(model, only_hidden=True)
    for (param_type, layer_num), param in layer_properties.items():
        momentum_buffers[(param_type, layer_num)] = torch.zeros_like(param, dtype=torch.float32)
    
    # Get momentum value for this step (matches training_core.py logic)
    frac = step / args.num_iterations
    momentum_value = (1 - frac) * 0.85 + frac * 0.95
    
    if rank == 0:
        print(f"Using momentum value: {momentum_value:.3f} for step {step}")
    
    # Create real data loader like training
    data_loader = create_real_data_loader(args, rank, world_size)
    all_gradient_results = []
    my_gradient_dict = {}  # Store accumulated gradient for my assigned minibatch
    
    for mb_idx in range(num_minibatches):
        if rank == 0:
            print(f"  Minibatch {mb_idx+1}/{num_minibatches}")
        
        inputs, targets = next(data_loader)
        
        model.zero_grad()
        model.train()
        
        window_size_blocks = get_window_size_blocks(step, args.num_iterations).to(device)
        if rank == 0:
            print(f"    Starting forward pass")
        loss = model(inputs, targets, window_size_blocks)
        if rank == 0:
            print(f"    Starting backward pass")
        loss.backward()
        
        if rank == 0:
            print(f"    Starting gradient sync")
        # Synchronize gradients across ranks
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"    Gradient sync complete")
        
        # Update momentum buffers and compute accumulated gradients (like Muon does)
        accumulated_gradients = {}
        layer_properties = get_weight_matrix_iterator(model, only_hidden=True)
        for (param_type, layer_num), param in layer_properties.items():
            if param.grad is None:
                continue
            
            # Update momentum buffer: momentum_buffer = momentum * momentum_buffer + (1 - momentum) * grad
            grad_f32 = param.grad.float()
            param_key = (param_type, layer_num)
            momentum_buffers[param_key] = momentum_value * momentum_buffers[param_key] + (1 - momentum_value) * grad_f32
            
            # Compute accumulated gradient: accumulated_grad = momentum * momentum_buffer + (1 - momentum) * grad
            # This matches exactly what Muon does in zeropower.py:199
            accumulated_grad = momentum_value * momentum_buffers[param_key] + (1 - momentum_value) * grad_f32
            accumulated_gradients[param_key] = accumulated_grad
        
        # Store accumulated gradient for C computation if this is my assigned minibatch
        if mb_idx == rank and mb_idx < world_size:
            if rank == 0:
                print(f"    Storing accumulated gradients for C estimation on rank {rank}")
            for param_key, accumulated_grad in accumulated_gradients.items():
                # No need to split - already handled by get_weight_matrix_iterator
                # Keep accumulated gradient on GPU in float32
                my_gradient_dict[param_key] = accumulated_grad.detach().to(torch.float32)
        
        if rank == 0:
            print(f"    Rank {rank}: Extracting accumulated gradients for mb {mb_idx}")
            mb_start = time.time()
        mb_results = extract_model_analysis_distributed(model, use_gradients=True, accumulated_gradients=accumulated_gradients)
        if rank == 0:
            all_gradient_results.append(mb_results)
            print(f"    Rank {rank}: Accumulated gradient extraction took {time.time() - mb_start:.1f}s")
        
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
    if rank == 0:
        print(f"Rank {rank}: Computing distributed C estimates")
        c_start = time.time()
    
    # Compute C estimates using distributed approach
    c_estimates_dict = compute_c_with_mean_truth_distributed(my_gradient_dict, rank, world_size)
    
    if rank == 0:
        print(f"Rank {rank}: C estimation took {time.time() - c_start:.1f}s")
        print(f"Rank {rank}: Aggregating results for {len(all_gradient_results)} minibatches")
        agg_start = time.time()
        result = aggregate_singular_values_and_stable_ranks(all_gradient_results, weight_results, c_estimates_dict)
        print(f"Rank {rank}: Aggregation took {time.time() - agg_start:.1f}s")
        return step, result
    return step, {}


def save_step_to_csvs(step: int, step_data: dict, sv_output_dir: Path, sr_output_dir: Path):
    """Save step data to the new CSV format."""
    sv_cache_data = []
    sr_cache_data = []
    
    for (param_type, layer_num), layer_data in step_data.items():
        sv_row = {
            'param_type': param_type,
            'layer_num': layer_num,
            'means': json.dumps(layer_data['means'].tolist()),
            'stds': json.dumps(layer_data['stds'].tolist()),
        }
        
        # Add C estimate if available
        if 'c_with_mean_truth' in layer_data:
            sv_row['c_with_mean_truth'] = json.dumps(layer_data['c_with_mean_truth'].tolist())
            
        sv_cache_data.append(sv_row)
        
        sr_cache_data.append({
            'param_type': param_type,
            'layer_num': layer_num,
            'weight_stable_rank': layer_data['weight_stable_rank'],
            'gradient_stable_rank_mean': layer_data['gradient_stable_rank_mean'],
            'gradient_stable_rank_std': layer_data['gradient_stable_rank_std']
        })
    
    sv_output_dir.mkdir(parents=True, exist_ok=True)
    sr_output_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(sv_cache_data).to_csv(sv_output_dir / f"step_{step:06d}.csv", index=False)
    pd.DataFrame(sr_cache_data).to_csv(sr_output_dir / f"step_{step:06d}.csv", index=False)


def find_all_checkpoints(run_id: str) -> list[tuple[int, str]]:
    """Find all checkpoint files and return sorted list of (step, filepath) tuples."""
    checkpoints_dir = Path("research_logs/checkpoints")
    checkpoint_pattern = str(checkpoints_dir / run_id / "model_step_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    unique_steps = {}
    for file in checkpoint_files:
        step = extract_step_from_checkpoint_path(Path(file))
        unique_steps[step] = file
    
    return [(step, unique_steps[step]) for step in sorted(unique_steps.keys())]


def setup_output_directories(run_name: str) -> tuple[Path, Path]:
    """Setup output directories and return paths."""
    sv_output_dir = get_research_log_path("singular_values_distribution", run_name, "")
    sr_output_dir = get_research_log_path("stable_rank_distribution", run_name, "")
    
    return sv_output_dir, sr_output_dir


def main():
    _, rank, world_size, device, master_process = setup_distributed_training()
    
    # Initialize global print function for training_core patch
    from empirical.research.training.training_core import _global_print0
    import empirical.research.training.training_core as training_core
    training_core._global_print0 = lambda s, console=False: print(s) if rank == 0 else None
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python compute_gradient_distribution_over_minibatch_seeds.py <run_id> [--testing] [--force]")
        print("Example: python compute_gradient_distribution_over_minibatch_seeds.py 000_2db40055-63ed-48dd-bb98-8cccdb78a501")
        sys.exit(1)
    
    run_id = sys.argv[1]
    run_name = run_id  # Use run_id as run_name for output directories
    testing = "--testing" in sys.argv
    force_recompute = "--force" in sys.argv
    num_minibatches = 8
    
    # Find all checkpoints
    selected_checkpoints = find_all_checkpoints(run_id)
    
    # Precompile SVD kernels to avoid compilation overhead during analysis
    precompile_svd_kernels(device, rank)
    
    # NOTE: Checkpoint 0 often has NaN/None gradients that break computation
    # Starting from checkpoint 1 avoids this issue.
    if testing:
        # Use 24 checkpoints from step 100 or earlier (skipping checkpoint 0)
        available_checkpoints = [(step, path) for step, path in selected_checkpoints 
                                if step > 0 and step <= 100]  # Skip checkpoint 0, only use step <= 100
        if len(available_checkpoints) >= 24:
            indices = np.linspace(0, len(available_checkpoints) - 1, 24, dtype=int)
            selected_checkpoints = [available_checkpoints[i] for i in indices]
        else:
            selected_checkpoints = available_checkpoints
    
    # Setup directories
    sv_output_dir, sr_output_dir = setup_output_directories(run_name)
    
    # Check which steps already exist
    existing_sv_files = set()
    if sv_output_dir.exists():
        for csv_file in sv_output_dir.glob("step_*.csv"):
            step_match = re.search(r'step_(\d+)\.csv', csv_file.name)
            if step_match:
                existing_sv_files.add(int(step_match.group(1)))
    
    # Process checkpoints
    for i, (step, checkpoint_file) in enumerate(selected_checkpoints):
        if not force_recompute and step in existing_sv_files:
            if master_process:
                print(f"Step {step} already exists, skipping ({i+1}/{len(selected_checkpoints)})")
            continue
        
        if master_process:
            print(f"Processing step {step} ({i+1}/{len(selected_checkpoints)})")
            start_time = time.time()
        
        _, param_data = compute_analysis_for_step(step, checkpoint_file, num_minibatches, rank, world_size, device)
        
        if master_process:
            print(f"Step {step} completed in {time.time() - start_time:.1f}s")
            save_step_to_csvs(step, param_data, sv_output_dir, sr_output_dir)
            print(f"Step {step} saved to CSV files")
    
    if master_process:
        print(f"Gradient distribution computation complete. Results saved to:")
        print(f"  - {sv_output_dir}")
        print(f"  - {sr_output_dir}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()