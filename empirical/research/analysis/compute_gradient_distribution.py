#!/usr/bin/env python3
"""
Refactored gradient distribution analysis with clean property pipeline architecture.

This script computes gradient analysis using a declarative property pipeline approach.
The core insight: gradient analysis is just a dependency graph of transformations
applied across model layers. By separating the "what" (property definitions) from
the "how" (execution), we achieve dramatically improved readability and maintainability.

Usage:
    torchrun --standalone --nproc_per_node=8 compute_gradient_distribution.py <run_id> [--testing] [--force]
"""

import os
import sys
import time
import glob
import re
import json
from pathlib import Path
import csv
import os
from collections import defaultdict
from typing import Dict, Tuple, Any

# Memory optimization like training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch._inductor.config.coordinate_descent_tuning = False
import torch.distributed as dist
import numpy as np
import pandas as pd

from empirical.research.training.training_core import (
    setup_distributed_training, Hyperparameters, get_window_size_blocks,
    distributed_data_generator, safe_torch_compile
)
from empirical.research.analysis.model_utilities import (
    get_weight_matrices, get_accumulated_gradient_matrices,
    combine_layer_properties, gather_layer_properties_to_rank_zero, GPTLayerProperty
)
from empirical.research.analysis.property_pipeline import PropertySpec, PropertyPipeline
from empirical.research.analysis.core_math import (
    stable_rank_from_tensor, safe_svd,
    compute_basis_cosine_similarity, compute_spectral_projection_coefficients_from_cosines,
    compute_innovation_spectrum, matrix_shape_beta
)
from empirical.research.analysis.core_visualization import (
    create_visualization_frames, finalize_gifs
)
from empirical.research.analysis.wishart import (
    fit_sigma_with_wishart,
    aspect_ratio_beta as wishart_aspect_ratio_beta,
    squared_true_signal_from_quadratic_formula,
    predict_spectral_projection_coefficient_from_squared_true_signal,
    set_sv_tables_from_npz,
    set_current_shape,
    precompute_quantile_table_for_shape,
)
from empirical.research.training.architecture import GPT
import empirical.research.training.training_core as training_core
from empirical.research.training.training_core import _global_print0


ANALYSIS_SPECS = [
    # Stable rank computations
    PropertySpec("weights_stable_rank", ["checkpoint_weights"], 
                stable_rank_from_tensor),
    PropertySpec("gradients_stable_rank", ["per_minibatch_gradient"], 
                lambda grads: stable_rank_from_tensor(grads.view(-1, grads.shape[-1]))),
    
    # Core gradient analysis
    PropertySpec("mean_gradient", ["per_minibatch_gradient"], 
                lambda grads: grads.mean(dim=0)),
    PropertySpec("minibatch_gradient_svd", ["per_minibatch_gradient"], 
                safe_svd),
    PropertySpec("mean_gradient_svd", ["mean_gradient"], 
                safe_svd),
    
    # Singular value analysis
    PropertySpec("minibatch_singular_values", ["minibatch_gradient_svd"], 
                lambda svd_tuple: svd_tuple[1]),
    PropertySpec("mean_singular_values", ["mean_gradient_svd"], 
                lambda svd_tuple: svd_tuple[1]),
    PropertySpec("singular_value_std", 
                ["minibatch_singular_values", "mean_singular_values"],
                lambda mb_sv, mean_sv: torch.std(mb_sv, dim=0)),
    
    # Spectral projection analysis
    PropertySpec("basis_cosine_similarities", 
                ["minibatch_gradient_svd", "mean_gradient_svd"],
                lambda mb_svd, mean_svd: (
                    compute_basis_cosine_similarity(mb_svd[0], mean_svd[0]),  # U matrices
                    compute_basis_cosine_similarity(mb_svd[2], mean_svd[2])   # Vh matrices
                )),
    PropertySpec("spectral_projection_coefficients", ["basis_cosine_similarities"],
                lambda cosines: compute_spectral_projection_coefficients_from_cosines(*cosines)),
    
    PropertySpec("innovation_spectrum", ["per_minibatch_gradient", "mean_gradient"],
                compute_innovation_spectrum),

    # Strict Wishart-based estimates
    PropertySpec("noise_sigma", ["minibatch_singular_values", "checkpoint_weights"],
                lambda sv, w: (set_current_shape(w.shape), float(fit_sigma_with_wishart(sv)))[1]),
    PropertySpec("aspect_ratio_beta", ["checkpoint_weights"],
                lambda w: float(wishart_aspect_ratio_beta(w))),
    PropertySpec("squared_true_signal_t", ["minibatch_singular_values", "noise_sigma", "aspect_ratio_beta"],
                squared_true_signal_from_quadratic_formula),
    PropertySpec("predicted_spectral_projection_coefficient", ["squared_true_signal_t", "aspect_ratio_beta"],
                predict_spectral_projection_coefficient_from_squared_true_signal),
]


def precompile_svd_kernels(device: torch.device, rank: int):
    """Pre-compile SVD kernels to avoid compilation during analysis."""
    if rank == 0:
        print("Pre-compiling SVD kernels...")
    
    with torch.no_grad():
        # Small matrices for compilation
        test_single = torch.randn(128, 64, device=device, dtype=torch.float32)
        test_batch = torch.randn(4, 128, 64, device=device, dtype=torch.float32)
        
        # Single SVD
        torch.linalg.svd(test_single)
        # Batched SVD  
        torch.linalg.svd(test_batch)
    
    if rank == 0:
        print("SVD kernel compilation complete.")


def setup_model_from_checkpoint(checkpoint_file: str, device: torch.device, compile_model: bool = True):
    """Load model from checkpoint with proper device placement."""
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
    
    # Optionally compile model (Inductor); can be disabled by caller
    if compile_model:
        model = training_core.safe_torch_compile(model, dynamic=False)
    
    if rank == 0:
        print(f"Rank {rank}: Model loaded and compiled")
    
    return model, checkpoint_data.get('step', 0)


def compute_analysis_for_step(
    step: int,
    checkpoint_file: str, 
    num_minibatches: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> GPTLayerProperty:
    """Core analysis function - clean and focused."""
    
    # 1. Setup (5 LOC)
    model, _ = setup_model_from_checkpoint(checkpoint_file, device, compile_model=True)
    
    # Lightweight warmup for analysis - use small sequence to avoid OOM
    args = Hyperparameters()
    small_seq_len = 128  # Much smaller than training's 8192
    vocab_size = args.vocab_size
    inputs = targets = torch.randint(0, vocab_size, size=(small_seq_len,), device=device)
    
    # Single warmup pass with compile fallback on backend failures (e.g., Triton/PTX)
    try:
        window_size_blocks = get_window_size_blocks(0, args.num_iterations).to(device)
        model(inputs.to(torch.int32), targets, window_size_blocks).backward()
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        if rank == 0:
            print(f"Warmup failed under compiled model, falling back to eager: {e}")
        # Recreate uncompiled model and retry warmup once
        model, _ = setup_model_from_checkpoint(checkpoint_file, device, compile_model=False)
        window_size_blocks = get_window_size_blocks(0, args.num_iterations).to(device)
        model(inputs.to(torch.int32), targets, window_size_blocks).backward()
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    precompile_svd_kernels(device, rank)
    
    # Synchronize after setup
    if dist.is_initialized():
        dist.barrier()
    
    # 2. Parameter sharding - each rank gets subset of parameters
    args = Hyperparameters()
    
    # Get all parameters for sharding
    all_weights = get_weight_matrices(model, only_hidden=True)
    all_param_keys = list(all_weights.keys())
    
    # Shard parameters across ranks
    params_per_rank = len(all_param_keys) // world_size
    start_idx = rank * params_per_rank
    end_idx = start_idx + params_per_rank if rank < world_size - 1 else len(all_param_keys)
    my_param_keys = set(all_param_keys[start_idx:end_idx])
    
    if rank == 0:
        print(f"    Rank {rank}: Processing a shard of {len(my_param_keys)} parameters out of {len(all_param_keys)} total")
    
    # Get sharded gradients (only for my assigned parameters)
    gradients = get_accumulated_gradient_matrices(
        model, args, step, num_minibatches=num_minibatches, assigned_params=my_param_keys
    )
    
    # Filter weights to only assigned parameters
    my_weights = {key: tensor for key, tensor in all_weights.items() if key in my_param_keys}
    
    # Create initial properties only for assigned parameters
    initial_props = combine_layer_properties(
        lambda w, g: {"checkpoint_weights": w, "per_minibatch_gradient": g},
        my_weights, gradients
    )
    
    # 3. Execute analysis pipeline (5 LOC)
    pipeline = PropertyPipeline(ANALYSIS_SPECS)
    
    def progress_callback(completed: int, total: int):
        if rank == 0 and completed % 5 == 0:
            print(f"  Analyzed {completed}/{total} layers")
    
    local_results = pipeline.execute_for_all_layers(initial_props, progress_callback)

    # Stream results to per-rank CSV to avoid large in-memory payloads
    stream_write_analysis_results(local_results, step, rank)
    local_records = {}

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print(f"Step {step}: Analysis complete (streamed to CSV)")
        return {}, {}
    else:
        return {}, {}


def convert_pipeline_results_to_record_format(layer_props: GPTLayerProperty) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Convert pipeline results to compact, serializable record format."""
    records = {}

    def safe_tensor_to_numpy(value, default_value=0.0):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        elif isinstance(value, (int, float)):
            return np.array(value)
        else:
            return np.array(default_value)

    for (param_type, layer_num), props in layer_props.items():
        record = {
            'weight_stable_rank': float(props.get('weights_stable_rank', 0.0)),
            'per_minibatch_gradient_stable_rank': safe_tensor_to_numpy(props.get('gradients_stable_rank', [])),
            'per_minibatch_gradient_singular_values': safe_tensor_to_numpy(props.get('minibatch_singular_values', [])),
            'gradient_singular_value_standard_deviations': safe_tensor_to_numpy(props.get('singular_value_std', [])),
            'spectral_projection_coefficients_from_8x_mean_gradient': safe_tensor_to_numpy(props.get('spectral_projection_coefficients', [])),
        }
        records[(param_type, layer_num)] = record

    return records


def _build_viz_stats_from_pipeline(viz_payload: GPTLayerProperty) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Map pipeline outputs to the per-layer viz dict used by plotting."""
    viz_stats: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for k, props in viz_payload.items():
        if 'innovation_spectrum' not in props:
            continue
        innov_s = props['innovation_spectrum']
        beta = matrix_shape_beta(props['checkpoint_weights'].shape)
        B = int(innov_s.shape[0]) if isinstance(innov_s, torch.Tensor) and innov_s.ndim >= 2 else 1
        sigma_eff = float(props['noise_sigma']) * (max(B - 1, 1) / max(B, 1)) ** 0.5
        viz_stats[k] = {
            'innovation_sample': innov_s[:, :min(100, innov_s.shape[1])].cpu().numpy(),
            'beta': float(beta),
            'sigma_hat': sigma_eff,
            'per_minibatch_singular_values': props['minibatch_singular_values'].cpu().numpy(),
            'spectral_projection_coefficients': props['spectral_projection_coefficients'].cpu().numpy(),
            'shape': tuple(int(x) for x in props['checkpoint_weights'].shape),
        }
    return viz_stats


def save_analysis_results(results: Dict[Tuple[str, int], Dict[str, Any]], step: int):
    """Save analysis results to CSV files."""
    # Create output directory
    base_dir = Path("research_logs/singular_values_distribution")
    sv_dir = base_dir
    sv_dir.mkdir(parents=True, exist_ok=True)

    # Convert to flat records for CSV
    step_records = []
    for (param_type, layer_num), record_data in results.items():
        if layer_num < 0:
            continue
        step_records.append({
            'param_type': param_type,
            'layer_num': layer_num,
            'weight_stable_rank': record_data.get('weight_stable_rank', 0.0),
            'per_minibatch_gradient_singular_values': json.dumps(record_data.get('per_minibatch_gradient_singular_values', []).tolist()),
            'gradient_singular_value_standard_deviations': json.dumps(record_data.get('gradient_singular_value_standard_deviations', []).tolist()),
            'per_minibatch_gradient_stable_rank': json.dumps(record_data.get('per_minibatch_gradient_stable_rank', []).tolist()),
            'spectral_projection_coefficients_from_8x_mean_gradient': json.dumps(record_data.get('spectral_projection_coefficients_from_8x_mean_gradient', []).tolist()),
        })

    df = pd.DataFrame(step_records)
    if not df.empty:
        csv_path = sv_dir / f"step_{step:06d}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved analysis results to {csv_path}")


def stream_write_analysis_results(layer_props: GPTLayerProperty, step: int, rank: int):
    base_dir = Path("research_logs/singular_values_distribution")
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / f"step_{step:06d}_rank{rank}.csv"

    fieldnames = [
        'param_type', 'layer_num', 'weight_stable_rank',
        'per_minibatch_gradient_singular_values',
        'gradient_singular_value_standard_deviations',
        'per_minibatch_gradient_stable_rank',
        'spectral_projection_coefficients_from_8x_mean_gradient',
    ]

    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for (param_type, layer_num), props in layer_props.items():
            def to_np16(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().to(torch.float16).cpu().numpy()
                return np.asarray(x)

            row = {
                'param_type': param_type,
                'layer_num': layer_num,
                'weight_stable_rank': float(props.get('weights_stable_rank', 0.0)),
                'per_minibatch_gradient_singular_values': json.dumps(to_np16(props.get('minibatch_singular_values', [])).tolist()),
                'gradient_singular_value_standard_deviations': json.dumps(to_np16(props.get('singular_value_std', [])).tolist()),
                'per_minibatch_gradient_stable_rank': json.dumps(to_np16(props.get('gradients_stable_rank', [])).tolist()),
                'spectral_projection_coefficients_from_8x_mean_gradient': json.dumps(to_np16(props.get('spectral_projection_coefficients', [])).tolist()),
            }
            writer.writerow(row)


def find_all_checkpoints(run_id: str) -> list[tuple[int, str]]:
    """Find all checkpoint files for the given run."""
    checkpoints_dir = Path("research_logs/checkpoints")
    checkpoint_pattern = str(checkpoints_dir / run_id / "model_step_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    unique_steps = {}
    for file in checkpoint_files:
        step_match = re.search(r'step_(\d+)', str(file))
        if step_match:
            step = int(step_match.group(1))
            unique_steps[step] = file
    
    return [(step, unique_steps[step]) for step in sorted(unique_steps.keys())]

def main():
    """Main entry point with clean argument parsing."""
    if len(sys.argv) < 2:
        print("Usage: python compute_gradient_distribution.py <run_id> [--testing] [--force]")
        return 1
    
    # Lightweight local mock mode (CPU-only, no distributed, synthetic data)
    if "--local-mock" in sys.argv:
        torch.manual_seed(0)
        device = torch.device("cpu")
        rank, world_size = 0, 1

        # Create tiny synthetic layer properties for 2 layers
        def _make_local_mock_initial_props(num_layers: int = 2, batch: int = 8, h: int = 32, w: int = 32):
            param_types = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'MLP Input', 'MLP Output']
            initial = {}
            for layer in range(num_layers):
                for ptype in param_types:
                    cw = torch.randn(h, w, device=device, dtype=torch.float32)
                    mb = torch.randn(batch, h, w, device=device, dtype=torch.float32)
                    initial[(ptype, layer)] = {"checkpoint_weights": cw, "per_minibatch_gradient": mb}
            return initial

        # Ensure the finite-size Wishart table contains the mock shape; precompute if missing
        def _ensure_sv_table_shape(npz_path: str, shape: tuple[int, int]):
            p, n = int(shape[0]), int(shape[1])
            path = Path(npz_path)
            base_keys = set()
            blob = None
            if path.exists():
                blob = np.load(npz_path, allow_pickle=True)
                base_keys = {k[:-2] for k in blob.files if k.endswith(("_u", "_q"))}
            has_shape = (f"{p}x{n}" in base_keys) or (f"{n}x{p}" in base_keys)
            if not has_shape:
                tbl = precompute_quantile_table_for_shape((p, n), draws=80)
                out = {}
                if blob is not None:
                    for k in blob.files:
                        out[k] = blob[k]
                out[f"{p}x{n}_u"] = tbl.u_grid
                out[f"{p}x{n}_q"] = tbl.q_singular_sigma1
                np.savez_compressed(npz_path, **out)

        mock_shape = (32, 32)
        _ensure_sv_table_shape("sv_quantiles_sigma1.npz", mock_shape)
        set_sv_tables_from_npz("sv_quantiles_sigma1.npz")

        NUM_CHECKPOINTS = 24
        pipeline = PropertyPipeline(ANALYSIS_SPECS)
        gif_frames = defaultdict(list)
        vis_output_dir = Path("research_logs/visualizations/noise_estimation")

        for step_idx in range(NUM_CHECKPOINTS):
            # Build synthetic grads that yield innovations like test_wishart_fitting:
            # G_b = S + sigma * E_b, then pipeline will form innovations as G_b - mean(G)
            def _make_signal(h: int, w: int, r: int = 8):
                r = min(r, h, w)
                U, _ = torch.linalg.qr(torch.randn(h, r, device=device, dtype=torch.float32))
                V, _ = torch.linalg.qr(torch.randn(w, r, device=device, dtype=torch.float32))
                a = (2.0/3.0) ** 0.5
                s = torch.pow(torch.full((r,), a, device=device, dtype=torch.float32), torch.arange(0, r, device=device, dtype=torch.float32))
                s = s * s.max()  # scale
                return (U * s) @ V.T

            def _make_local_props_with_innov(num_layers: int = 2, batch: int = 8, h: int = 32, w: int = 32):
                param_types = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'MLP Input', 'MLP Output']
                props = {}
                for layer in range(num_layers):
                    # shared signal per layer (cancels in innovations after mean)
                    S = _make_signal(h, w, r=min(8, h, w))
                    # choose noise sigma per layer (log-uniform)
                    sigma = float(10.0 ** torch.empty(1).uniform_(-3.5, -1.5).item())
                    for ptype in param_types:
                        E = torch.randn(batch, h, w, device=device, dtype=torch.float32)
                        G = S.unsqueeze(0) + sigma * E
                        props[(ptype, layer)] = {
                            "checkpoint_weights": torch.randn(h, w, device=device, dtype=torch.float32),
                            "per_minibatch_gradient": G,
                        }
                return props

            initial_props = _make_local_props_with_innov(batch=8)
            local_results = pipeline.execute_for_all_layers(initial_props)

            # Save CSV per checkpoint
            records = convert_pipeline_results_to_record_format(local_results)
            save_analysis_results(records, step=step_idx)

            # Build viz stats for this checkpoint
            viz_payload = local_results
            viz_stats = _build_viz_stats_from_pipeline(viz_payload)
            create_visualization_frames(step_idx, viz_stats, gif_frames, vis_output_dir, rank)

        finalize_gifs(gif_frames, vis_output_dir, rank=rank)
        return 0

    run_id = sys.argv[1]
    testing_mode = "--testing" in sys.argv
    force_recompute = "--force" in sys.argv
    
    # Distributed setup
    _, rank, world_size, device, master_process = setup_distributed_training()
    
    # Initialize global print function for training_core patch
    from empirical.research.training.training_core import _global_print0
    import empirical.research.training.training_core as training_core
    training_core._global_print0 = lambda s, console=False: print(s) if rank == 0 else None
    
    if rank == 0:
        print(f"Starting gradient distribution analysis for run: {run_id}")
        print(f"Testing mode: {testing_mode}, Force recompute: {force_recompute}")
    
    # Find checkpoints
    checkpoints = find_all_checkpoints(run_id)
    if not checkpoints:
        if rank == 0:
            print(f"No checkpoints found for run {run_id}")
        return 1
    
    if testing_mode:
        checkpoints = checkpoints[:2]  # Only process first 2 steps
    
    if rank == 0:
        print(f"Found {len(checkpoints)} checkpoints to process")

    # Load finite-size Wishart tables (strict)
    set_sv_tables_from_npz("sv_quantiles_sigma1.npz")

    # Process each checkpoint
    gif_frames = defaultdict(list)
    total_expected_frames = 0
    num_minibatches = 8  # Match training configuration
    
    for step, checkpoint_file in checkpoints:
        if rank == 0:
            print(f"\nProcessing step {step}...")
            start_time = time.time()
        
        # Run analysis
        records, viz_payload = compute_analysis_for_step(
            step, checkpoint_file, num_minibatches, rank, world_size, device
        )

        if rank == 0:
            # Ensure all ranks have flushed their CSVs before reading for visualization
            if dist.is_initialized():
                dist.barrier()
            # Build viz stats from the pipeline results (or read from CSVs inside viz)
            viz_stats = _build_viz_stats_from_pipeline(viz_payload)
            vis_output_dir = Path("research_logs/visualizations/noise_estimation")
            create_visualization_frames(step, viz_stats, gif_frames, vis_output_dir, rank)
            # Accumulate expected frames based on minibatch dimension
            any_entry = next(iter(viz_stats.values())) if viz_stats else None
            total_expected_frames += int(any_entry['innovation_sample'].shape[0]) if any_entry is not None else 1
    
    # Create final GIFs
    if rank == 0:
        print("\nCreating visualization GIFs...")
        vis_output_dir = Path("research_logs/visualizations/noise_estimation")
        finalize_gifs(gif_frames, vis_output_dir, rank=rank, expected_len=total_expected_frames)
        print("Analysis complete!")

    # Ensure all ranks reach a common point before teardown
    if dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()

    return 0



if __name__ == "__main__":
    exit_code = main()
    # Global teardown (no try/except)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0 if exit_code is None else exit_code)
