#!/usr/bin/env python3
"""
Compute gradient distributions over multiple minibatch SGD seeds for noise model analysis.
This script performs the expensive computation once per training run, saving results to CSV files.

Usage:
    torchrun --standalone --nproc_per_node=8 compute_gradient_distribution_over_minibatch_seeds.py checkpoint_run_name
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
from empirical.research.training.training_core import setup_distributed_training, get_window_size_blocks, Hyperparameters
from empirical.research.analysis.offline_logging import deserialize_model_checkpoint, split_qkv_weight, compute_singular_values, compute_stable_rank, categorize_parameter
from empirical.research.analysis.map import get_weight_matrix_iterator, get_research_log_path


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
    all_params = list(get_weight_matrix_iterator(model, only_hidden=True))
    
    # Distribute SVD computation work across ranks, not the parameters themselves
    all_param_splits = []
    for param_name, param in all_params:
        if use_gradients:
            if accumulated_gradients is not None:
                # Use accumulated gradients if provided
                tensor = accumulated_gradients.get(param_name, None)
                if tensor is None:
                    continue
            else:
                # Use raw gradients from param.grad
                tensor = param.grad
                if tensor is None:
                    continue
        else:
            tensor = param
        
        split_tensors = split_qkv_weight(param_name, tensor)
        for split_name, split_tensor in split_tensors.items():
            all_param_splits.append((split_name, split_tensor))
    
    # Distribute SVD work across ranks
    splits_per_rank = len(all_param_splits) // world_size
    start_idx = rank * splits_per_rank
    end_idx = start_idx + splits_per_rank if rank < world_size - 1 else len(all_param_splits)
    my_splits = all_param_splits[start_idx:end_idx]
    
    if rank == 0:
        print(f"    Distributed {len(all_param_splits)} SVD computations across {world_size} ranks ({len(my_splits)} per rank)")
    
    # Process my assigned SVD computations
    local_results = {}
    for split_name, split_tensor in my_splits:
        # Use GPU SVD instead of CPU numpy
        tensor_f32 = split_tensor.detach().to(torch.float32)
        try:
            _, s, _ = torch.linalg.svd(tensor_f32, full_matrices=False)
            sv = s.cpu().numpy()
            stable_rank = compute_stable_rank(sv)
            local_results[split_name] = {'sv': sv, 'stable_rank': stable_rank}
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
    
    for param_name, my_gradient in my_gradient_dict.items():
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
            
            c_estimates[param_name] = c_estimate.cpu().numpy()
            
        except Exception as e:
            print(f"Error computing C for {param_name}: {e}")
            r = min(my_gradient.shape)
            c_estimates[param_name] = np.zeros(r)
    
    return c_estimates



def compute_c_estimates_for_param(gradient_matrices, param_name):
    """Compute the actual C estimates for a parameter using the provided functions."""
    try:
        # Stack gradients: (B, m, n) 
        G = np.stack(gradient_matrices, axis=0)
        
        # Use the exact functions provided
        c_pairwise = estimate_c_pairwise_signsync(G)
        c_mean_truth_per_batch = estimate_c_with_mean_truth(G)  # Shape: (B, r)
        c_jackknife_per_batch = estimate_c_jackknife(G)        # Shape: (B, r)
        
        # Average across batches for the per-batch estimators
        c_mean_truth = np.mean(c_mean_truth_per_batch, axis=0)
        c_jackknife = np.mean(c_jackknife_per_batch, axis=0)
        
        return {
            'c_pairwise_signsync': c_pairwise,
            'c_with_mean_truth': c_mean_truth, 
            'c_jackknife': c_jackknife
        }
    except Exception as e:
        print(f"Error computing C estimates for {param_name}: {e}")
        # Return zeros of appropriate length
        min_dim = min(G.shape[1], G.shape[2]) if len(gradient_matrices) > 0 else 10
        return {
            'c_pairwise_signsync': np.zeros(min_dim),
            'c_with_mean_truth': np.zeros(min_dim),
            'c_jackknife': np.zeros(min_dim)
        }


def aggregate_singular_values_and_stable_ranks(all_gradient_results, weight_results, c_estimates_dict=None):
    """Aggregate singular values and stable rank statistics across minibatches."""
    param_data = {}
    
    all_param_names = set()
    for mb_grads in all_gradient_results:
        all_param_names.update(mb_grads.keys())
    
    for param_name in all_param_names:
        param_type, layer_num = categorize_parameter(param_name)
        
        sv_arrays = [mb_grads[param_name]['sv'] for mb_grads in all_gradient_results if param_name in mb_grads]
        stable_ranks = [mb_grads[param_name]['stable_rank'] for mb_grads in all_gradient_results if param_name in mb_grads]
        
        min_len = min(len(sv) for sv in sv_arrays)
        aligned_svs = [sv[:min_len] for sv in sv_arrays]
        aligned_svs_array = np.array(aligned_svs)
        
        key = (param_type, layer_num)
        param_data[key] = {
            'means': np.mean(aligned_svs_array, axis=0),
            'stds': np.std(aligned_svs_array, axis=0),
            'weight_stable_rank': weight_results[param_name]['stable_rank'],
            'gradient_stable_rank_mean': np.mean(stable_ranks),
            'gradient_stable_rank_std': np.std(stable_ranks)
        }
        
        # Add C estimates if provided
        if c_estimates_dict and param_name in c_estimates_dict:
            # Only c_with_mean_truth now
            param_data[key]['c_with_mean_truth'] = c_estimates_dict[param_name]
    
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
    for param_name, param in get_weight_matrix_iterator(model, only_hidden=True):
        momentum_buffers[param_name] = torch.zeros_like(param, dtype=torch.float32)
    
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
        all_params = list(get_weight_matrix_iterator(model, only_hidden=True))
        for param_name, param in all_params:
            if param.grad is None:
                continue
            
            # Update momentum buffer: momentum_buffer = momentum * momentum_buffer + (1 - momentum) * grad
            grad_f32 = param.grad.float()
            momentum_buffers[param_name] = momentum_value * momentum_buffers[param_name] + (1 - momentum_value) * grad_f32
            
            # Compute accumulated gradient: accumulated_grad = momentum * momentum_buffer + (1 - momentum) * grad
            # This matches exactly what Muon does in zeropower.py:199
            accumulated_grad = momentum_value * momentum_buffers[param_name] + (1 - momentum_value) * grad_f32
            accumulated_gradients[param_name] = accumulated_grad
        
        # Store accumulated gradient for C computation if this is my assigned minibatch
        if mb_idx == rank and mb_idx < world_size:
            if rank == 0:
                print(f"    Storing accumulated gradients for C estimation on rank {rank}")
            for param_name, accumulated_grad in accumulated_gradients.items():
                split_tensors = split_qkv_weight(param_name, accumulated_grad)
                for split_name, split_tensor in split_tensors.items():
                    # Keep accumulated gradient on GPU in float32
                    my_gradient_dict[split_name] = split_tensor.detach().to(torch.float32)
        
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


def find_all_checkpoints() -> list[tuple[int, str]]:
    """Find all checkpoint files and return sorted list of (step, filepath) tuples."""
    checkpoints_dir = Path("research_logs/checkpoints")
    # Filter for the specific successful sharp cutoff run
    run_id = "000_e37b459d-3a35-48d2-b102-a33ac52584c6"
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
    run_name = sys.argv[1] if len(sys.argv) >= 2 else "gradient_noise_run"
    testing = len(sys.argv) >= 3 and sys.argv[2] == "--testing"
    force_recompute = "--force" in sys.argv
    num_minibatches = 8
    
    # Find all checkpoints
    selected_checkpoints = find_all_checkpoints()
    
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