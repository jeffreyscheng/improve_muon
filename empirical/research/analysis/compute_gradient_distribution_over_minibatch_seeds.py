#!/usr/bin/env python3
"""
Functional gradient distribution analysis over multiple minibatch seeds.
This script uses a clean functional programming approach with GPTLayerProperty transformations
to compute gradient distributions, SVDs, basis similarities, and spectral projection coefficients.

Key Features:
- Functional programming approach with composable transformations
- Distributed computation with parameter sharding
- Basis cosine similarity and spectral projection coefficient analysis
- Stable rank computation for both gradients and weights
- Clean separation of mathematical operations and data flow

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
from empirical.research.training.training_core import (
    setup_distributed_training, get_window_size_blocks, Hyperparameters, 
    warmup_kernels, distributed_data_generator
)
from empirical.research.analysis.offline_logging import compute_stable_rank
from empirical.research.analysis.map import (
    get_weight_matrices, get_research_log_path, get_accumulated_gradient_matrices,
    combine_layer_properties, gather_layer_properties_to_rank_zero, apply_stable_rank
)


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
    
    # Lightweight warmup for analysis - use small sequence to avoid OOM
    small_seq_len = 128  # Much smaller than training's 8192
    vocab_size = args.vocab_size
    inputs = targets = torch.randint(0, vocab_size, size=(small_seq_len,), device=device)
    sliding_window_blocks = 1  # Minimal sliding window
    
    # Single warmup pass
    model(inputs.to(torch.int32), targets, sliding_window_blocks).backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()  # Clear memory after warmup
    
    if rank == 0:
        print(f"Rank {rank}: Model loaded and compiled")
    
    return model


"""
SVD and Mathematical Functions for Gradient Analysis
"""

def apply_batched_svd(layer_properties) -> tuple:
    """Apply SVD to each batched tensor (batch×n×m) in the GPTLayerProperty."""
    def compute_batched_svd(key, tensor):
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor for batched SVD, got {tensor.ndim}D for {key}")
        
        batch_size = tensor.shape[0]
        U_list, S_list, Vh_list = [], [], []
        
        for i in range(batch_size):
            U_i, S_i, Vh_i = torch.linalg.svd(tensor[i].float(), full_matrices=False)
            U_list.append(U_i)
            S_list.append(S_i)
            Vh_list.append(Vh_i)
        
        U = torch.stack(U_list, dim=0)  # batch×n×n
        S = torch.stack(S_list, dim=0)  # batch×min(n,m)
        Vh = torch.stack(Vh_list, dim=0)  # batch×m×m
        
        return U, S, Vh
    
    U_prop, S_prop, Vh_prop = {}, {}, {}
    
    for key, tensor in layer_properties.items():
        U, S, Vh = compute_batched_svd(key, tensor)
        U_prop[key] = U
        S_prop[key] = S
        Vh_prop[key] = Vh
    
    return U_prop, S_prop, Vh_prop


def apply_svd(layer_properties) -> tuple:
    """Apply SVD to each matrix (n×m) in the GPTLayerProperty."""
    def compute_svd(key, tensor):
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor for SVD, got {tensor.ndim}D for {key}")
        
        U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
        return U, S, Vh
    
    U_prop, S_prop, Vh_prop = {}, {}, {}
    
    for key, tensor in layer_properties.items():
        U, S, Vh = compute_svd(key, tensor)
        U_prop[key] = U
        S_prop[key] = S
        Vh_prop[key] = Vh
    
    return U_prop, S_prop, Vh_prop


def compute_basis_cosine_similarity(batched_basis: torch.Tensor, reference_basis: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between each batched basis and reference basis."""
    # Ensure same device and dtype
    batched_basis = batched_basis.to(reference_basis.device).to(reference_basis.dtype)
    
    # Compute inner products: batch×k×k @ k×k -> batch×k×k
    # We want the diagonal elements, which are the inner products of corresponding columns
    inner_products = torch.bmm(batched_basis.transpose(-2, -1), reference_basis.unsqueeze(0).expand(batched_basis.shape[0], -1, -1))
    
    # Extract diagonal elements (inner products of corresponding basis vectors)
    cosines = torch.diagonal(inner_products, dim1=-2, dim2=-1)  # batch×k
    
    # Return absolute values (cosine similarity)
    return torch.abs(cosines)


def compute_spectral_projection_coefficients(left_cosines: torch.Tensor, right_cosines: torch.Tensor) -> torch.Tensor:
    """Compute spectral projection coefficients as element-wise products."""
    # Take minimum rank to get valid spectral coefficients
    min_rank = min(left_cosines.shape[-1], right_cosines.shape[-1])
    
    # Truncate to minimum rank and compute element-wise product
    left_truncated = left_cosines[..., :min_rank].abs()
    right_truncated = right_cosines[..., :min_rank].abs()
    
    return left_truncated * right_truncated


def compute_basis_similarities_and_spectral_coeffs(per_gradient_U, per_gradient_Vh, average_gradient_U, average_gradient_Vh) -> tuple:
    """Compute basis similarities and spectral projection coefficients."""
    # Compute left cosine similarities
    left_cosines = combine_layer_properties(
        compute_basis_cosine_similarity,
        per_gradient_U,
        average_gradient_U
    )
    
    # Compute right cosine similarities (need to transpose Vh to get column vectors)
    right_cosines = combine_layer_properties(
        lambda batched_vh, ref_vh: compute_basis_cosine_similarity(
            batched_vh.transpose(-2, -1),  # Convert to column-major
            ref_vh.transpose(-2, -1)       # Convert to column-major
        ),
        per_gradient_Vh,
        average_gradient_Vh
    )
    
    # Compute spectral projection coefficients
    spectral_coeffs = combine_layer_properties(
        compute_spectral_projection_coefficients,
        left_cosines,
        right_cosines
    )
    
    return left_cosines, right_cosines, spectral_coeffs


def compute_gradient_singular_value_std(per_minibatch_singular_values, average_gradient_singular_values):
    """Compute standard deviation of singular values across minibatches."""
    def compute_std(batched_sv, avg_sv):
        # Compute standard deviation across batch dimension
        return batched_sv.std(dim=0, unbiased=True)
    
    return combine_layer_properties(compute_std, per_minibatch_singular_values, average_gradient_singular_values)


def compute_sharded_gradients(model: torch.nn.Module, data_loader, num_minibatches: int, momentum_buffers: dict, momentum_value: float, step: int):
    """Compute per-minibatch gradients with momentum accumulation."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = next(model.parameters()).device
    
    args = Hyperparameters()
    
    # Get all parameters that we'll need for sharding
    all_layer_properties = get_weight_matrices(model, only_hidden=True)
    all_param_keys = list(all_layer_properties.keys())
    
    # Shard parameters across ranks
    params_per_rank = len(all_param_keys) // world_size
    start_idx = rank * params_per_rank
    end_idx = start_idx + params_per_rank if rank < world_size - 1 else len(all_param_keys)
    my_param_keys = set(all_param_keys[start_idx:end_idx])
    
    if rank == 0:
        print(f"    Rank {rank}: Processing {len(my_param_keys)} parameters out of {len(all_param_keys)} total")
    
    # Collect gradients for my assigned parameters across all minibatches
    per_minibatch_gradients = {}
    
    for mb_idx in range(num_minibatches):
        if rank == 0:
            print(f"    Processing minibatch {mb_idx + 1}/{num_minibatches}")
        
        # Forward and backward pass
        model.zero_grad()
        model.train()
        
        inputs, targets = next(data_loader)
        window_size_blocks = get_window_size_blocks(step, args.num_iterations).to(device)
        
        loss = model(inputs, targets, window_size_blocks)
        loss.backward()
        
        # Synchronize gradients across ranks (like in training)
        for param in model.parameters():
            if param.grad is not None:
                if dist.is_initialized():
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
        # Update momentum buffers and get accumulated gradients
        accumulated_gradients = get_accumulated_gradient_matrices(model, momentum_buffers, momentum_value)
        
        # Store gradients for my assigned parameters
        for param_key, grad_tensor in accumulated_gradients.items():
            if param_key in my_param_keys:
                if param_key not in per_minibatch_gradients:
                    # Initialize tensor to store all minibatches: num_minibatches×n×m
                    per_minibatch_gradients[param_key] = torch.zeros(
                        (num_minibatches,) + grad_tensor.shape,
                        dtype=grad_tensor.dtype,
                        device=grad_tensor.device
                    )
                
                # Store this minibatch's gradient
                per_minibatch_gradients[param_key][mb_idx] = grad_tensor.detach()
        
        # Clear gradients to save memory
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
    if rank == 0:
        print(f"    Rank {rank}: Collected gradients for {len(per_minibatch_gradients)} parameters")
    
    return per_minibatch_gradients


def convert_to_record_format(
    average_gradient_singular_values,
    per_minibatch_gradient_singular_values, 
    gradient_singular_value_standard_deviations,
    spectral_projection_coefficients,
    per_minibatch_gradient_stable_rank,
    weight_matrix_stable_rank
) -> dict:
    """Convert functional computation results back to the record format expected by CSV output."""
    records = {}
    
    # Get all keys (should be the same across all inputs on rank 0)
    all_keys = set(average_gradient_singular_values.keys())
    
    for param_key in all_keys:
        param_type, layer_num = param_key
        
        records[param_key] = {
            'per_minibatch_gradient_singular_values': per_minibatch_gradient_singular_values[param_key].cpu().numpy(),
            'gradient_singular_value_standard_deviations': gradient_singular_value_standard_deviations[param_key].cpu().numpy(),
            'weight_stable_rank': weight_matrix_stable_rank[param_key].cpu().item(),
            'per_minibatch_gradient_stable_rank': per_minibatch_gradient_stable_rank[param_key].cpu().numpy(),
            'spectral_projection_coefficients_from_8x_mean_gradient': spectral_projection_coefficients[param_key].cpu().numpy()
        }
    
    return records


def compute_analysis_for_step(step: int, checkpoint_file: str, num_minibatches: int, rank: int, world_size: int, device: torch.device):
    """
    Functional implementation of gradient analysis using GPTLayerProperty transformations.
    This replaces the procedural approach with clean functional transformations.
    """
    
    model = setup_model_from_checkpoint(checkpoint_file, device)
    args = Hyperparameters()
    
    if rank == 0:
        print(f"Rank {rank}: Starting functional gradient analysis")
        start_time = time.time()
    
    # 1. Get weight matrices (full on all ranks, used for stable rank computation)
    checkpoint_weight_matrices = get_weight_matrices(model, only_hidden=True)
    
    # 2. Initialize momentum buffers
    momentum_buffers = {}
    for param_key, param in checkpoint_weight_matrices.items():
        momentum_buffers[param_key] = torch.zeros_like(param, dtype=torch.float32)
    
    # 3. Get momentum value for this step
    frac = step / args.num_iterations
    momentum_value = (1 - frac) * 0.85 + frac * 0.95
    
    if rank == 0:
        print(f"Using momentum value: {momentum_value:.3f} for step {step}")
    
    # 4. Create data loader
    data_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
    
    # 5. Core computation: compute sharded per-minibatch gradients
    if rank == 0:
        print(f"Rank {rank}: Computing sharded gradients")
        grad_start = time.time()
    
    per_minibatch_gradient = compute_sharded_gradients(
        model, data_loader, num_minibatches, momentum_buffers, momentum_value, step
    )
    
    if rank == 0:
        print(f"Rank {rank}: Gradient computation took {time.time() - grad_start:.1f}s")
    
    # 6. Statistical transforms: compute average gradient
    if rank == 0:
        print(f"Rank {rank}: Computing average gradients")
    
    def apply_mean_across_batch_dim(layer_properties, dim=0):
        """Apply mean reduction across specified dimension."""
        return {key: tensor.mean(dim=dim) for key, tensor in layer_properties.items()}
    
    average_gradient = apply_mean_across_batch_dim(per_minibatch_gradient, dim=0)
    
    # 7. SVD transforms
    if rank == 0:
        print(f"Rank {rank}: Computing SVDs")
        svd_start = time.time()
    
    # Batched SVD for per-minibatch gradients
    per_grad_U, per_grad_S, per_grad_Vh = apply_batched_svd(per_minibatch_gradient)
    
    # Regular SVD for average gradients
    avg_grad_U, avg_grad_S, avg_grad_Vh = apply_svd(average_gradient)
    
    if rank == 0:
        print(f"Rank {rank}: SVD computation took {time.time() - svd_start:.1f}s")
    
    # 8. Basis similarity and spectral projection coefficients
    if rank == 0:
        print(f"Rank {rank}: Computing basis similarities")
    
    left_cosines, right_cosines, spectral_coeffs = compute_basis_similarities_and_spectral_coeffs(
        per_grad_U, per_grad_Vh, avg_grad_U, avg_grad_Vh
    )
    
    # 9. Gradient singular value standard deviations
    gradient_sv_std = compute_gradient_singular_value_std(per_grad_S, avg_grad_S)
    
    # 10. Stable ranks
    if rank == 0:
        print(f"Rank {rank}: Computing stable ranks")
    
    per_minibatch_stable_ranks = apply_stable_rank(per_minibatch_gradient)  # 8-element tensors
    weight_stable_ranks = apply_stable_rank(checkpoint_weight_matrices)      # scalars
    
    # 11. Gather to rank 0
    if rank == 0:
        print(f"Rank {rank}: Gathering results to rank 0")
        gather_start = time.time()
    
    gathered_results = gather_layer_properties_to_rank_zero(
        avg_grad_S, per_grad_S, gradient_sv_std, spectral_coeffs, 
        per_minibatch_stable_ranks, weight_stable_ranks
    )
    
    if rank == 0:
        print(f"Rank {rank}: Gathering took {time.time() - gather_start:.1f}s")
        
        # Unpack gathered results
        (avg_grad_S_full, per_grad_S_full, gradient_sv_std_full,
         spectral_coeffs_full, per_mb_ranks_full, weight_ranks_full) = gathered_results
        
        # 12. Convert to record format for CSV compatibility
        print(f"Rank {rank}: Converting to record format")
        records = convert_to_record_format(
            avg_grad_S_full, per_grad_S_full, gradient_sv_std_full,
            spectral_coeffs_full, per_mb_ranks_full, weight_ranks_full
        )
        
        total_time = time.time() - start_time
        print(f"Rank {rank}: Functional analysis completed in {total_time:.1f}s")
        
        return step, records
    
    return step, {}


def unpivot_and_save_step_to_csvs(step: int, step_data: dict, sv_output_dir: Path, sr_output_dir: Path):
    """Save step data to the new CSV format."""
    sv_cache_data = []
    
    for (param_type, layer_num), layer_data in step_data.items():
        sv_row = {
            'param_type': param_type,
            'layer_num': layer_num,
            'weight_stable_rank': layer_data['weight_stable_rank'],
            'per_minibatch_gradient_singular_values': json.dumps(layer_data['per_minibatch_gradient_singular_values'].tolist()),
            'gradient_singular_value_standard_deviations': json.dumps(layer_data['gradient_singular_value_standard_deviations'].tolist()),
            'per_minibatch_gradient_stable_rank': json.dumps(layer_data['per_minibatch_gradient_stable_rank'].tolist()),
            'spectral_projection_coefficients_from_8x_mean_gradient': json.dumps(layer_data['spectral_projection_coefficients_from_8x_mean_gradient'].tolist())
        }
            
        sv_cache_data.append(sv_row)
    
    sv_output_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(sv_cache_data).to_csv(sv_output_dir / f"step_{step:06d}.csv", index=False)


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
            unpivot_and_save_step_to_csvs(step, param_data, sv_output_dir, sr_output_dir)
            print(f"Step {step} saved to CSV files")
    
    if master_process:
        print(f"Gradient distribution computation complete. Results saved to:")
        print(f"  - {sv_output_dir}")
        print(f"  - {sr_output_dir}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()