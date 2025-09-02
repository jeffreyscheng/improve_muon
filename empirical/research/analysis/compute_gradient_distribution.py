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
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Any

# Memory optimization like training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch._inductor.config.coordinate_descent_tuning = False
import torch.distributed as dist
import numpy as np

from empirical.research.training.training_core import (
    setup_distributed_training, Hyperparameters, warmup_kernels,
    distributed_data_generator
)
from empirical.research.analysis.model_utilities import (
    get_weight_matrices, get_research_log_path, get_accumulated_gradient_matrices,
    combine_layer_properties, gather_layer_properties_to_rank_zero, GPTLayerProperty
)
from empirical.research.analysis.property_pipeline import PropertySpec, PropertyPipeline
from empirical.research.analysis.core_math import (
    stable_rank_from_tensor, batched_svd, single_svd, 
    compute_basis_cosine_similarity, compute_spectral_projection_coefficients_from_cosines,
    compute_spectral_projection_coefficients_direct, compute_innovation_statistics
)
from empirical.research.analysis.core_visualization import (
    create_visualization_frames, finalize_gifs
)


# The complete analysis pipeline specification - direct function references
ANALYSIS_PIPELINE = [
    # Stable rank computations
    PropertySpec("weights_stable_rank", ["checkpoint_weights"], 
                stable_rank_from_tensor),
    PropertySpec("gradients_stable_rank", ["per_minibatch_gradient"], 
                lambda grads: stable_rank_from_tensor(grads.view(-1, grads.shape[-1]))),
    
    # Core gradient analysis
    PropertySpec("mean_gradient", ["per_minibatch_gradient"], 
                lambda grads: grads.mean(dim=0)),
    PropertySpec("minibatch_gradient_svd", ["per_minibatch_gradient"], 
                batched_svd),
    PropertySpec("mean_gradient_svd", ["mean_gradient"], 
                single_svd),
    
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
    
    # Innovation statistics for visualization
    PropertySpec("innovation_statistics", ["per_minibatch_gradient", "mean_gradient"],
                compute_innovation_statistics),
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


def extract_step_from_checkpoint_path(checkpoint_path: Path) -> int:
    """Extract step number from checkpoint filename."""
    return int(checkpoint_path.stem.split('_')[1])


def setup_model_from_checkpoint(checkpoint_file: str, device: torch.device):
    """Load model from checkpoint with proper device placement."""
    from empirical.research.training.training_core import GPT
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Create model with same config as training
    model_config = checkpoint['model_args']
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['step']


def compute_analysis_for_step(
    step: int,
    checkpoint_file: str, 
    num_minibatches: int,
    rank: int,
    world_size: int,
    device: torch.device
) -> GPTLayerProperty:
    """Core analysis function - clean and focused."""
    
    # 1. Setup (5 LOC)
    model, _ = setup_model_from_checkpoint(checkpoint_file, device)
    setup_distributed_training(rank)
    warmup_kernels(model, [torch.optim.SGD(model.parameters(), lr=1.0)], 
                  Hyperparameters())
    precompile_svd_kernels(device, rank)
    
    # 2. Get initial layer properties (10 LOC) 
    weights = get_weight_matrices(model)
    
    # Create dummy data loader for gradient computation
    args = Hyperparameters()
    data_loader = distributed_data_generator(args, rank, world_size)
    
    gradients = get_accumulated_gradient_matrices(
        model, args, step, num_minibatches=num_minibatches
    )
    
    initial_props = combine_layer_properties(
        lambda w, g: {"checkpoint_weights": w, "per_minibatch_gradient": g},
        weights, gradients
    )
    
    # 3. Execute analysis pipeline (5 LOC)
    pipeline = PropertyPipeline(ANALYSIS_PIPELINE)
    
    def progress_callback(completed: int, total: int):
        if rank == 0 and completed % 5 == 0:
            print(f"  Analyzed {completed}/{total} layers")
    
    local_results = pipeline.execute_for_all_layers(initial_props, progress_callback)
    
    # 4. Distributed coordination (10 LOC)
    if rank == 0:
        all_results = gather_layer_properties_to_rank_zero(local_results)
        print(f"Step {step}: Analysis complete for {len(all_results)} layers")
        return all_results
    else:
        gather_layer_properties_to_rank_zero(local_results)
        return {}


def create_compact_visualization_stats(layer_props: GPTLayerProperty) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Extract compact statistics for visualization."""
    viz_stats = {}
    
    for (param_type, layer_num), props in layer_props.items():
        if "innovation_statistics" not in props:
            continue
            
        innov_stats = props["innovation_statistics"]
        
        viz_stats[(param_type, layer_num)] = {
            "innovation_sample": innov_stats["innovation_samples"].cpu().numpy(),
            "beta": float(innov_stats["beta"]),
            "sigma_hat": float(torch.median(innov_stats["innovation_singular_values"]).item()),
            "per_minibatch_singular_values": props["minibatch_singular_values"].cpu().numpy(),
            "spectral_projection_coefficients": props["spectral_projection_coefficients"].cpu().numpy(),
        }
    
    return viz_stats


def save_analysis_results(results: GPTLayerProperty, step: int):
    """Save analysis results to CSV files."""
    # Create output directories
    base_dir = Path("research_logs/fitted_noise_model")
    sv_dir = base_dir / "singular_values"
    sr_dir = base_dir / "stable_ranks"
    sv_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to record format and save
    step_records = []
    
    for (param_type, layer_num), props in results.items():
        if layer_num < 0:  # Skip invalid layers
            continue
            
        # Extract key metrics for CSV export
        record = {
            "step": step,
            "param_type": param_type,
            "layer": layer_num,
            "weights_stable_rank": float(props.get("weights_stable_rank", 0)),
            "gradients_stable_rank": float(props.get("gradients_stable_rank", 0)),
        }
        
        # Add singular value statistics
        if "minibatch_singular_values" in props:
            sv = props["minibatch_singular_values"].cpu().numpy()
            record.update({
                "mean_singular_value": float(sv.mean()),
                "std_singular_value": float(sv.std()),
                "max_singular_value": float(sv.max()),
            })
        
        step_records.append(record)
    
    # Save to CSV (simplified version - expand as needed)
    import pandas as pd
    df = pd.DataFrame(step_records)
    
    if not df.empty:
        csv_path = sr_dir / f"step_{step:06d}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved analysis results to {csv_path}")


def find_all_checkpoints(run_id: str) -> list[tuple[int, str]]:
    """Find all checkpoint files for the given run."""
    log_dir = get_research_log_path(run_id)
    checkpoint_pattern = log_dir / "step_*.pt"
    
    checkpoints = []
    for path in sorted(log_dir.glob("step_*.pt")):
        try:
            step = extract_step_from_checkpoint_path(path)
            checkpoints.append((step, str(path)))
        except (ValueError, IndexError):
            continue
    
    return checkpoints


def main():
    """Main entry point with clean argument parsing."""
    if len(sys.argv) < 2:
        print("Usage: python compute_gradient_distribution.py <run_id> [--testing] [--force]")
        sys.exit(1)
    
    run_id = sys.argv[1]
    testing_mode = "--testing" in sys.argv
    force_recompute = "--force" in sys.argv
    
    # Distributed setup
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    
    if rank == 0:
        print(f"Starting gradient distribution analysis for run: {run_id}")
        print(f"Testing mode: {testing_mode}, Force recompute: {force_recompute}")
    
    # Find checkpoints
    checkpoints = find_all_checkpoints(run_id)
    if not checkpoints:
        print(f"No checkpoints found for run {run_id}")
        sys.exit(1)
    
    if testing_mode:
        checkpoints = checkpoints[:2]  # Only process first 2 steps
    
    if rank == 0:
        print(f"Found {len(checkpoints)} checkpoints to process")
    
    # Process each checkpoint
    gif_frames = defaultdict(list)
    num_minibatches = 8  # Match training configuration
    
    for step, checkpoint_file in checkpoints:
        if rank == 0:
            print(f"\nProcessing step {step}...")
            start_time = time.time()
        
        # Run analysis
        results = compute_analysis_for_step(
            step, checkpoint_file, num_minibatches, rank, world_size, device
        )
        
        # Save results and create visualizations (rank 0 only)
        if rank == 0:
            save_analysis_results(results, step)
            
            viz_stats = create_compact_visualization_stats(results)
            vis_output_dir = Path("research_logs/visualizations/noise_estimation")
            create_visualization_frames(step, viz_stats, gif_frames, vis_output_dir, rank)
            
            elapsed = time.time() - start_time
            print(f"Step {step} completed in {elapsed:.1f}s")
    
    # Create final GIFs
    if rank == 0:
        print("\nCreating visualization GIFs...")
        vis_output_dir = Path("research_logs/visualizations/noise_estimation")
        finalize_gifs(gif_frames, vis_output_dir, rank=rank)
        print("Analysis complete!")


if __name__ == "__main__":
    main()