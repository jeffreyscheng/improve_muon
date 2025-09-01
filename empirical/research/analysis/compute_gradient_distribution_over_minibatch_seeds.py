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
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Import all dependencies at top level
from empirical.research.training.training_core import (
    setup_distributed_training, get_window_size_blocks, Hyperparameters, 
    warmup_kernels, distributed_data_generator, safe_torch_compile
)
from empirical.research.analysis.offline_logging import compute_stable_rank
from empirical.research.analysis.map import (
    get_weight_matrices, get_research_log_path, get_accumulated_gradient_matrices,
    combine_layer_properties, gather_layer_properties_to_rank_zero, apply_stable_rank,
    GPTLayerProperty
)
from empirical.research.analysis.predict_spectral_projection_torch import (
    predict_spectral_projection_batched,
    matrix_shape_beta,
    estimate_noise_level_numpy,
    get_denoised_squared_singular_value_numpy,
    estimate_spectral_projection_coefficients_numpy,
)

# ---- Small utilities for viz/gather ----
from typing import Dict, Tuple, Any, List
import math


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
    
    # Compile model using distributed-safe compilation
    model = safe_torch_compile(model, dynamic=False)
    
    # Warm up kernels using the same helper as training.
    # training_core.warmup_kernels expects an `optimizers` dict; for analysis
    # we can hand it a simple throwaway optimizer to compile the optimizer path.
    optimizers = {"sgd": torch.optim.SGD(model.parameters(), lr=1.0)}
    warmup_kernels(model, args, optimizers=optimizers)
    
    if rank == 0:
        print(f"Rank {rank}: Model loaded and compiled")
    
    return model


"""
Visualization Helper Functions for Noise Estimation Analysis  
"""

PARAM_TYPES = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'MLP Input', 'MLP Output']

def mp_pdf_singular(s: np.ndarray, beta: float, sigma: float):
    """Marchenko-Pastur density for singular values (scale sigma)."""
    s = np.asarray(s, float)
    lam_m = (1 - np.sqrt(beta))**2
    lam_p = (1 + np.sqrt(beta))**2
    u = s / sigma
    lam = u*u
    inside = (lam > lam_m) & (lam < lam_p) & (u > 0)
    out = np.zeros_like(s)
    if np.any(inside):
        num = np.sqrt((lam_p - lam[inside]) * (lam[inside] - lam_m))
        out[inside] = (num / (np.pi * beta * u[inside])) * (1.0 / sigma)
    return out


def create_gif_from_frames(frame_paths, gif_path, fps=12):
    """Create looping GIF from frame files."""
    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    print(f"Looping GIF saved: {gif_path}")


def create_subplot_grid(param_types, figsize, get_data_fn, plot_fn, title, output_path):
    """Generic subplot grid creator."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    
    viridis = plt.cm.viridis
    for i, param_type in enumerate(param_types):
        ax = axes[i // 3, i % 3]
        layer_data_list = get_data_fn(param_type)
        max_layers = max([layer_num for layer_num, _ in layer_data_list], default=1) + 1
        plot_fn(ax, param_type, layer_data_list, viridis, max_layers)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


"""
SVD and Mathematical Functions for Gradient Analysis
"""

def apply_batched_svd(layer_properties) -> tuple:
    """Apply SVD to each batched tensor (batch×n×m) in the GPTLayerProperty."""
    # 3F: vectorized SVD (supports batched B×n×m) instead of Python loop
    def compute_batched_svd(key, tensor):
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor for batched SVD, got {tensor.ndim}D for {key}")
        with torch.no_grad():  # 3E: no autograd needed
            U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
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
        # 3E: analysis math doesn't require grad
        with torch.no_grad():
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
    
    # Collect gradients and momentum buffers for my assigned parameters across all minibatches
    per_minibatch_gradients = {}
    per_minibatch_momentum_buffers = {}
    
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
        
        # (2) Do NOT all-reduce per-parameter grads in analysis.
        #     Each rank only needs grads for its shard of params; avoid broadcast+network cost.
        #     (We keep data parallel input sharding in the loader.)
        #     Intentionally no dist.all_reduce here.
        
        # Store current momentum buffers before they get updated
        for param_key in my_param_keys:
            if param_key in momentum_buffers:
                if param_key not in per_minibatch_momentum_buffers:
                    # Initialize tensor to store all minibatches: num_minibatches×n×m
                    per_minibatch_momentum_buffers[param_key] = torch.zeros(
                        (num_minibatches,) + momentum_buffers[param_key].shape,
                        dtype=momentum_buffers[param_key].dtype,
                        device=momentum_buffers[param_key].device
                    )
                
                # Store current momentum buffer before update
                per_minibatch_momentum_buffers[param_key][mb_idx] = momentum_buffers[param_key].detach()
        
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
        
        # Clear gradients to save memory (no per-minibatch empty_cache: 3E)
        model.zero_grad(set_to_none=True)
    
    if rank == 0:
        print(f"    Rank {rank}: Collected gradients and momentum buffers for {len(per_minibatch_gradients)} parameters")
    
    return per_minibatch_gradients, per_minibatch_momentum_buffers


def convert_to_record_format(
    average_gradient_singular_values,
    per_minibatch_gradient_singular_values, 
    gradient_singular_value_standard_deviations,
    spectral_projection_coefficients,
    per_minibatch_gradient_stable_rank,
    weight_matrix_stable_rank,
    predicted_spectral_coefficients
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
            'spectral_projection_coefficients_from_8x_mean_gradient': spectral_projection_coefficients[param_key].cpu().numpy(),
            'predicted_spectral_projection_coefficients': predicted_spectral_coefficients[param_key].cpu().numpy()
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
    checkpoint_weight_matrices: GPTLayerProperty = get_weight_matrices(model, only_hidden=True)
    
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
    
    per_minibatch_gradient: GPTLayerProperty
    per_minibatch_momentum_buffers: GPTLayerProperty
    per_minibatch_gradient, per_minibatch_momentum_buffers = compute_sharded_gradients(
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
    
    average_gradient: GPTLayerProperty = apply_mean_across_batch_dim(per_minibatch_gradient, dim=0)
    
    # 7. SVD transforms
    if rank == 0:
        print(f"Rank {rank}: Computing SVDs")
        svd_start = time.time()
    
    # Batched SVD for per-minibatch gradients
    with torch.no_grad():  # 3E
        per_grad_U: GPTLayerProperty
        per_grad_S: GPTLayerProperty
        per_grad_Vh: GPTLayerProperty
        per_grad_U, per_grad_S, per_grad_Vh = apply_batched_svd(per_minibatch_gradient)
    
    # Regular SVD for average gradients
    with torch.no_grad():  # 3E
        avg_grad_U: GPTLayerProperty
        avg_grad_S: GPTLayerProperty
        avg_grad_Vh: GPTLayerProperty
        avg_grad_U, avg_grad_S, avg_grad_Vh = apply_svd(average_gradient)
    
    if rank == 0:
        print(f"Rank {rank}: SVD computation took {time.time() - svd_start:.1f}s")
    
    # 8. Basis similarity and spectral projection coefficients
    if rank == 0:
        print(f"Rank {rank}: Computing basis similarities")
    
    left_cosines: GPTLayerProperty
    right_cosines: GPTLayerProperty
    spectral_coeffs: GPTLayerProperty
    left_cosines, right_cosines, spectral_coeffs = compute_basis_similarities_and_spectral_coeffs(
        per_grad_U, per_grad_Vh, avg_grad_U, avg_grad_Vh
    )
    
    # 9. Gradient singular value standard deviations
    gradient_sv_std: GPTLayerProperty = compute_gradient_singular_value_std(per_grad_S, avg_grad_S)
    
    # 10. Compute spectral projection coefficients using torch (batched, sharded on this rank)
    if rank == 0:
        print(f"Rank {rank}: Computing spectral projection coefficients")
    
    def apply_predict_spc_torch(grad_tensor, momentum_tensor):
        # grad_tensor, momentum_tensor: [B, n, m] for this param on THIS rank
        # returns [B, min(n,m)] — computation sharded across ranks by param key
        # Ensure each invocation is a new cudagraph "step" to avoid output reuse hazards
        try:
            import torch
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        # Also clone outside compiled calls (predict_spectral_projection_batched already clones,
        # but keeping this here makes the contract explicit at the boundary)
        return predict_spectral_projection_batched(grad_tensor, momentum_tensor).clone()

    predicted_spectral_coeffs: GPTLayerProperty = combine_layer_properties(
        apply_predict_spc_torch, per_minibatch_gradient, per_minibatch_momentum_buffers
    )

    # 10b. (3A) Build compact per-layer viz stats on *this* rank only.
    def _sample_numpy_1d(a: np.ndarray, k: int) -> np.ndarray:
        if a.size <= k:
            return a
        idx = np.random.default_rng(0).choice(a.size, size=k, replace=False)
        out = a[idx]
        # 3B: ensure contiguous, no negative strides surprises later
        return np.ascontiguousarray(out)

    BETA_CACHE: Dict[Tuple[int,int], float] = {}
    def _beta_for(shape: Tuple[int,int]) -> float:
        if shape not in BETA_CACHE:
            BETA_CACHE[shape] = matrix_shape_beta(shape)
        return BETA_CACHE[shape]

    viz_stats_shard: Dict[Tuple[str,int], Dict[str, Any]] = {}
    with torch.no_grad():
        for key in per_minibatch_gradient.keys():
            param_type, layer_num = key
            grads = per_minibatch_gradient[key]          # [B,n,m]
            moms  = per_minibatch_momentum_buffers[key]  # [B,n,m]

            innov = grads - moms                         # [B,n,m]
            B, n, m = innov.shape
            # Get singular values for all minibatches (vectorized)
            try:
                s_all = torch.linalg.svdvals(innov.float())  # [B, K]
            except AttributeError:
                # Fallback if svdvals not available
                _, s_all, _ = torch.linalg.svd(innov.float(), full_matrices=False)
            K = s_all.shape[-1]

            # Flatten + downsample on device, then move a small sample to CPU
            s_flat = s_all.reshape(-1)
            # take up to 4096 samples for a nice histogram
            ksample = min(4096, s_flat.numel())
            if ksample < s_flat.numel():
                perm = torch.randperm(s_flat.numel(), device=s_flat.device)
                s_sample = s_flat.index_select(0, perm[:ksample])
            else:
                s_sample = s_flat
            s_np = s_sample.detach().cpu().contiguous().numpy()
            # Sorted descending (3B: .copy() avoids negative stride)
            s_np_sorted_desc = np.sort(s_np)[::-1].copy()

            beta = _beta_for((n, m))
            sigma_hat = float(estimate_noise_level_numpy(s_np_sorted_desc, beta))

            # For "SPC vs Singular": take first two minibatches
            mb_take = min(2, B)
            y_list: List[np.ndarray] = []
            spc_list: List[np.ndarray] = []
            for mb in range(mb_take):
                s_mb = s_all[mb].detach().cpu().contiguous().numpy()
                y = s_mb / max(sigma_hat, 1e-30)
                # spikes only
                edge = 1.0 + math.sqrt(beta)
                spike_mask = y > edge
                if spike_mask.any():
                    t_hat = get_denoised_squared_singular_value_numpy(y, beta)
                    spc   = estimate_spectral_projection_coefficients_numpy(t_hat, beta)
                    y_list.append(np.ascontiguousarray(y[spike_mask]))
                    spc_list.append(np.ascontiguousarray(spc[spike_mask]))
            if len(y_list) == 0:
                y_spikes = np.empty((0,), dtype=np.float64)
                spc_pred = np.empty((0,), dtype=np.float64)
            else:
                y_spikes = np.ascontiguousarray(np.concatenate(y_list))
                spc_pred = np.ascontiguousarray(np.concatenate(spc_list))
                # cap to 512 points for plotting
                y_spikes = _sample_numpy_1d(y_spikes, 512)
                spc_pred = _sample_numpy_1d(spc_pred, 512)

            # For "Predicted vs Actual": sample pairs
            pred = predicted_spectral_coeffs[key].detach().cpu().contiguous().numpy().ravel()
            act  = spectral_coeffs[key].detach().cpu().contiguous().numpy().ravel()
            kpair = min(2048, min(pred.size, act.size))
            if pred.size and act.size:
                idx = np.random.default_rng(0).choice(min(pred.size, act.size), size=kpair, replace=False)
                pred_s = np.ascontiguousarray(pred[idx])
                act_s  = np.ascontiguousarray(act[idx])
            else:
                pred_s = np.empty((0,), dtype=np.float64)
                act_s  = np.empty((0,), dtype=np.float64)

            viz_stats_shard[key] = {
                "shape": (n, m),
                "beta": beta,
                "sigma_hat": sigma_hat,
                "innovation_sample": s_np,            # for histogram
                "y_spikes": y_spikes,                 # x for SPC vs y
                "spc_pred": spc_pred,                 # y for SPC vs y
                "pred_vs_actual": (act_s, pred_s),    # scatter pairs
            }

    # Helper: gather Python objects to rank 0
    def _gather_viz_stats_to_rank0(local_obj):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return local_obj if rank == 0 else None
        if rank == 0:
            all_objs: List[Dict] = [None for _ in range(world_size)]
            dist.gather_object(local_obj, object_gather_list=all_objs, dst=0)
            merged = {}
            for d in all_objs:
                if not d:
                    continue
                merged.update(d)
            return merged
        else:
            dist.gather_object(local_obj, dst=0)
            return None
    
    # 11. Stable ranks
    if rank == 0:
        print(f"Rank {rank}: Computing stable ranks")
    
    per_minibatch_stable_ranks: GPTLayerProperty = apply_stable_rank(per_minibatch_gradient)  # 8-element tensors
    weight_stable_ranks: GPTLayerProperty = apply_stable_rank(checkpoint_weight_matrices)      # scalars
    
    # 12. Gather to rank 0
    if rank == 0:
        print(f"Rank {rank}: Gathering results to rank 0")
        gather_start = time.time()
    
    gathered_results = gather_layer_properties_to_rank_zero(
        avg_grad_S, per_grad_S, gradient_sv_std, spectral_coeffs, 
        per_minibatch_stable_ranks, weight_stable_ranks, predicted_spectral_coeffs
    )
    
    if rank == 0:
        print(f"Rank {rank}: Gathering took {time.time() - gather_start:.1f}s")
        
        # Unpack gathered results
        (avg_grad_S_full, per_grad_S_full, gradient_sv_std_full,
         spectral_coeffs_full, per_mb_ranks_full, weight_ranks_full, predicted_spectral_coeffs_full) = gathered_results

        # Gather compact viz stats objects
        viz_stats_full = _gather_viz_stats_to_rank0(viz_stats_shard)
        
        # 13. Convert to record format for CSV compatibility
        print(f"Rank {rank}: Converting to record format")
        records = convert_to_record_format(
            avg_grad_S_full, per_grad_S_full, gradient_sv_std_full,
            spectral_coeffs_full, per_mb_ranks_full, weight_ranks_full, predicted_spectral_coeffs_full
        )
        
        total_time = time.time() - start_time
        print(f"Rank {rank}: Functional analysis completed in {total_time:.1f}s")
        
        # Return additional visualization data for rank 0
        viz_data = {"viz_stats": viz_stats_full} if rank == 0 else {}
        
        return step, records, viz_data
    
    return step, {}, {}


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
            'spectral_projection_coefficients_from_8x_mean_gradient': json.dumps(layer_data['spectral_projection_coefficients_from_8x_mean_gradient'].tolist()),
            'predicted_spectral_projection_coefficients': json.dumps(layer_data['predicted_spectral_projection_coefficients'].tolist())
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


def create_noise_estimation_visualizations(
    step: int, 
    viz_stats: Dict[Tuple[str,int], Dict[str, Any]],
    gif_frames: dict,
    rank: int
):
    """Create visualization frames for noise estimation analysis."""
    if rank != 0:  # Only create visualizations on rank 0
        return
    
    # Create frames directory
    frames_dir = Path(f"research_logs/visualizations/noise_estimation_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    def get_layer_data_for_param_type(param_type):
        """Get compact viz stats for this parameter type."""
        items = []
        for (p_type, layer_num), d in viz_stats.items():
            if p_type == param_type and layer_num >= 0:
                items.append((layer_num, d))
        return sorted(items, key=lambda x: x[0])
    
    # 1. Bulk vs Spike Estimation (Spectrum with MP density)
    def plot_bulk_vs_spike(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_title(f'{param_type}')
        ax.set_xlabel('Singular value')
        ax.set_ylabel('Density')
        
        # Sum up per-layer innovation samples (already downsampled) and render a histogram
        samples = []
        betas = []
        sigmas = []
        for _, data in layer_data_list:
            s = np.asarray(data["innovation_sample"], dtype=float)
            if s.size:
                samples.append(s)
                betas.append(float(data["beta"]))
                sigmas.append(float(data["sigma_hat"]))
        if samples:
            all_innov = np.ascontiguousarray(np.concatenate(samples))
            ax.hist(all_innov, bins=60, density=True, alpha=0.35, label="Empirical spectrum")
            # Use median β, σ̂ across layers for the overlay
            if betas and sigmas:
                beta = float(np.median(np.asarray(betas)))
                sigma_hat = float(np.median(np.asarray(sigmas)))
                xs = np.linspace(0, all_innov.max()*1.05, 1000)
                mp_density = mp_pdf_singular(xs, beta, sigma_hat)
                ax.plot(xs, mp_density, 'r--', lw=2, label=f"MP (σ̂={sigma_hat:.3f})")
                edge = sigma_hat * (1 + np.sqrt(beta))
                ax.axvline(edge, color='k', lw=1.5, ls='--', label=f"Edge τ̂={edge:.3f}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 2. SPC vs Singular Values  
    def plot_spc_vs_singular_values(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_title(f'{param_type}')
        ax.set_xlabel('Whitened singular value y = s/σ̂')
        ax.set_ylabel('SPC')
        ax.set_xscale('log')
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        
        for layer_num, data in layer_data_list:
            color = viridis(layer_num / (max_layers - 1))
            y = np.asarray(data["y_spikes"])
            spc_pred = np.asarray(data["spc_pred"])
            if y.size and spc_pred.size:
                ax.scatter(y, spc_pred, alpha=0.6, s=20, c=[color], label=f'L{layer_num}')
        
        # Add theoretical SPC curve
        ygrid = np.logspace(0, 2, 200)  # y from 1 to 100
        # Use a typical shape's β for reference (3D: fixed once)
        beta_sample = matrix_shape_beta((1024, 1024))
        tgrid = get_denoised_squared_singular_value_numpy(ygrid, beta_sample)
        spcgrid = estimate_spectral_projection_coefficients_numpy(tgrid, beta_sample)
        ax.plot(ygrid, spcgrid, 'k--', lw=2, alpha=0.7, label='Theoretical SPC')
        
        ax.legend(loc='lower right', fontsize=8)
    
    # 3. Predicted vs Actual SPC
    def plot_predicted_vs_actual_spc(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_title(f'{param_type}')
        ax.set_xlabel('Actual SPC')
        ax.set_ylabel('Predicted SPC')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect prediction')
        
        for layer_num, data in layer_data_list:
            color = viridis(layer_num / (max_layers - 1))
            act, pred = data["pred_vs_actual"]
            if act.size and pred.size:
                ax.scatter(np.asarray(act), np.asarray(pred), alpha=0.3, s=10, c=[color],
                           label=f'L{layer_num}' if layer_num <= 2 else None)
        
        ax.legend(loc='upper left', fontsize=8)
    
    # Create all three frame types
    try:
        # Frame 1: Bulk vs Spike
        frame_path_1 = frames_dir / f"bulk_spike_{step:06d}.png"
        create_subplot_grid(PARAM_TYPES, (20, 10), get_layer_data_for_param_type, 
                           plot_bulk_vs_spike, f'Bulk vs Spike Estimation - Step {step}', frame_path_1)
        gif_frames['bulk_spike'].append(str(frame_path_1))
        
        # Frame 2: SPC vs Singular Values
        frame_path_2 = frames_dir / f"spc_singular_{step:06d}.png"
        create_subplot_grid(PARAM_TYPES, (20, 10), get_layer_data_for_param_type,
                           plot_spc_vs_singular_values, f'SPC vs Singular Values - Step {step}', frame_path_2)
        gif_frames['spc_singular'].append(str(frame_path_2))
        
        # Frame 3: Predicted vs Actual SPC
        frame_path_3 = frames_dir / f"pred_actual_{step:06d}.png"
        create_subplot_grid(PARAM_TYPES, (20, 10), get_layer_data_for_param_type,
                           plot_predicted_vs_actual_spc, f'Predicted vs Actual SPC - Step {step}', frame_path_3)
        gif_frames['pred_actual'].append(str(frame_path_3))
        
        print(f"Rank {rank}: Created visualization frames for step {step}")
    except Exception as e:
        print(f"Rank {rank}: Warning - Failed to create visualization frames for step {step}: {e}")


def finalize_noise_estimation_gifs(gif_frames: dict, rank: int):
    """Create the final GIFs from all collected frames."""
    if rank != 0 or not any(gif_frames.values()):
        return
    
    vis_dir = Path("research_logs/visualizations/noise_estimation")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create GIF 1: Bulk vs Spike Estimation
        if gif_frames['bulk_spike']:
            create_gif_from_frames(gif_frames['bulk_spike'], 
                                 vis_dir / "bulk_vs_spike_estimation.gif", fps=8)
        
        # Create GIF 2: SPC vs Singular Values
        if gif_frames['spc_singular']:
            create_gif_from_frames(gif_frames['spc_singular'],
                                 vis_dir / "spc_vs_singular_values.gif", fps=8)
        
        # Create GIF 3: Predicted vs Actual SPC  
        if gif_frames['pred_actual']:
            create_gif_from_frames(gif_frames['pred_actual'],
                                 vis_dir / "predicted_vs_actual_spc.gif", fps=8)
        
        # Clean up frame files
        for frame_list in gif_frames.values():
            for frame_path in frame_list:
                Path(frame_path).unlink(missing_ok=True)
        
        frames_dir = Path("research_logs/visualizations/noise_estimation_frames")
        if frames_dir.exists():
            frames_dir.rmdir()
        
        print(f"Rank {rank}: Created noise estimation GIFs in {vis_dir}")
        
    except Exception as e:
        print(f"Rank {rank}: Warning - Failed to create GIFs: {e}")


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
    
    # Initialize GIF frame collections for visualization
    gif_frames = {
        'bulk_spike': [],
        'spc_singular': [],
        'pred_actual': []
    }
    
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
        
        step_result, param_data, viz_data = compute_analysis_for_step(step, checkpoint_file, num_minibatches, rank, world_size, device)
        
        if master_process:
            analysis_time = time.time() - start_time
            print(f"Step {step} functional analysis completed in {analysis_time:.1f}s")
            unpivot_and_save_step_to_csvs(step, param_data, sv_output_dir, sr_output_dir)
            print(f"Step {step} saved to CSV files")
        
        # Create visualizations (only on rank 0)
        if rank == 0 and viz_data and viz_data.get("viz_stats"):
            if master_process:
                print(f"Step {step} creating visualizations...")
                viz_start_time = time.time()
            try:
                create_noise_estimation_visualizations(
                    step,
                    viz_data["viz_stats"],
                    gif_frames,
                    rank
                )
                if master_process:
                    viz_time = time.time() - viz_start_time
                    total_time = time.time() - start_time
                    print(f"Step {step} visualizations completed in {viz_time:.1f}s")
                    print(f"Step {step} fully completed in {total_time:.1f}s")
            except Exception as e:
                if master_process:
                    total_time = time.time() - start_time
                    print(f"Step {step} visualization failed: {e}")
                    print(f"Step {step} completed (without visualization) in {total_time:.1f}s")
        
        # Print completion message for non-rank-0 processes or when no visualization data
        if not (rank == 0 and viz_data) and master_process:
            total_time = time.time() - start_time
            print(f"Step {step} completed in {total_time:.1f}s")
    
    # Create final GIFs from collected frames
    finalize_noise_estimation_gifs(gif_frames, rank)
    
    if master_process:
        print(f"Gradient distribution computation complete. Results saved to:")
        print(f"  - {sv_output_dir}")
        print(f"  - {sr_output_dir}")
        print(f"  - research_logs/visualizations/noise_estimation/ (GIFs)")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()