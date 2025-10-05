#!/usr/bin/env python3
"""
Refactored gradient distribution analysis with clean property pipeline architecture.

This script computes gradient analysis using a declarative property pipeline approach.
The core insight: gradient analysis is just a dependency graph of transformations
applied across model layers. By separating the "what" (property definitions) from
the "how" (execution), we achieve dramatically improved readability and maintainability.

Usage:
    torchrun --standalone --nproc_per_node=8 compute_gradient_distribution.py <run_id> [--testing] [--force]

Local testing:
    python -m empirical.research.analysis.compute_gradient_distribution --local-mock
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
    create_subplot_grid, finalize_gifs, PARAM_TYPES
)
from empirical.research.analysis.wishart import (
    aspect_ratio_beta as wishart_aspect_ratio_beta,
    squared_true_signal_from_quadratic_formula,
    predict_spectral_projection_coefficient_from_squared_true_signal,
)
from empirical.research.analysis.logging_utilities import deserialize_model_checkpoint, categorize_parameter


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
    
    # Noise sigma is provided externally from serialized checkpoints; no Wishart fitting
    PropertySpec("aspect_ratio_beta", ["checkpoint_weights"],
                lambda w: float(wishart_aspect_ratio_beta(w))),
    PropertySpec("squared_true_signal_t", ["minibatch_singular_values", "noise_sigma", "aspect_ratio_beta"],
                squared_true_signal_from_quadratic_formula),
    PropertySpec("predicted_spectral_projection_coefficient", ["squared_true_signal_t", "aspect_ratio_beta"],
                predict_spectral_projection_coefficient_from_squared_true_signal),
]

# Mock analysis specs (can diverge from real pipeline if needed)
MOCK_SPECS = ANALYSIS_SPECS


def _make_mock_step_props(num_layers: int, batch: int, h: int, w: int, device: torch.device):
    def _signal(r: int):
        r = min(r, h, w)
        U, _ = torch.linalg.qr(torch.randn(h, r, device=device, dtype=torch.float32))
        V, _ = torch.linalg.qr(torch.randn(w, r, device=device, dtype=torch.float32))
        a = (2.0/3.0) ** 0.5
        s = torch.pow(torch.full((r,), a, device=device, dtype=torch.float32), torch.arange(0, r, device=device, dtype=torch.float32))
        s = s * s.max()
        return (U * s) @ V.T
    param_types = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'MLP Input', 'MLP Output']
    props = {}
    for layer in range(num_layers):
        S = _signal(r=min(8, h, w))
        sigma = float(10.0 ** torch.empty(1, device=device).uniform_(-3.5, -1.5).item())
        for ptype in param_types:
            E = torch.randn(batch, h, w, device=device, dtype=torch.float32)
            G = S.unsqueeze(0) + sigma * E
            props[(ptype, layer)] = {"checkpoint_weights": torch.randn(h, w, device=device), "per_minibatch_gradient": G}
    return props


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


def build_compiled_model(device: torch.device):
    """Build and compile the model once per process."""
    from empirical.research.training.architecture import GPT
    import empirical.research.training.training_core as training_core
    args = Hyperparameters()
    model = GPT(args.vocab_size, 16, 8, 1024, max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    model = training_core.safe_torch_compile(model, dynamic=False)
    return model


def load_weights_into_model(checkpoint_file: str, model: torch.nn.Module, device: torch.device):
    """Load checkpoint weights into an existing compiled model and broadcast from rank 0."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"Rank {rank}: Loading checkpoint {checkpoint_file}")
    # Use the unified deserializer; schema expects 'model'
    checkpoint_data = deserialize_model_checkpoint(Path(checkpoint_file))
    state_dict = checkpoint_data['model']

    # If compiling wrapped the module, load into the original module
    target = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Normalize checkpoints that may contain '_orig_mod.' prefix in keys
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}

    target.load_state_dict(state_dict)

    # Broadcast parameters so all ranks are in sync with rank 0
    if dist.is_initialized():
        for param in target.parameters():
            dist.broadcast(param.detach(), 0)
    return int(checkpoint_data['step'])


def compute_analysis_for_step(
    step: int,
    checkpoint_file: str, 
    num_minibatches: int,
    rank: int,
    world_size: int,
    device: torch.device,
    model: torch.nn.Module,
    *,
    initial_props: GPTLayerProperty | None = None,
    specs: list[PropertySpec] | None = None,
) -> GPTLayerProperty:
    """Core analysis function - clean and focused."""
    
    # 1. Build initial properties (either provided for mock, or computed from model)
    args = Hyperparameters()
    
    if initial_props is None:
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
        # Get sharded gradients
        gradients = get_accumulated_gradient_matrices(
            model, args, step, num_minibatches=num_minibatches, assigned_params=my_param_keys
        )
        my_weights = {key: tensor for key, tensor in all_weights.items() if key in my_param_keys}
        initial_props = combine_layer_properties(
            lambda w, g: {"checkpoint_weights": w, "per_minibatch_gradient": g},
            my_weights, gradients
        )

        # Inject noise_sigma from serialized checkpoint
        try:
            ckpt = deserialize_model_checkpoint(Path(checkpoint_file))
            muon_sigma_map: Dict[str, float] = ckpt['muon_sigma']
        except Exception as e:
            muon_sigma_map = {}

        # Build per-layer sigma mappings from parameter names
        sigma_attention: Dict[int, float] = {}
        sigma_mlp_in: Dict[int, float] = {}
        sigma_mlp_out: Dict[int, float] = {}
        for name, _param in model.named_parameters():
            try:
                ptype, layer = categorize_parameter(name)
            except Exception:
                continue
            if layer < 0:
                continue
            if name in muon_sigma_map:
                s = float(muon_sigma_map[name])
                if ptype == 'attention':
                    sigma_attention[layer] = s
                elif ptype == 'mlp_input':
                    sigma_mlp_in[layer] = s
                elif ptype == 'mlp_output':
                    sigma_mlp_out[layer] = s

        # Attach noise_sigma to initial_props per (param_type, layer)
        for (ptype, layer), props in initial_props.items():
            if ptype.startswith('Attention '):
                sigma = sigma_attention.get(layer, 0.0)
            elif ptype == 'MLP Input':
                sigma = sigma_mlp_in.get(layer, 0.0)
            elif ptype == 'MLP Output':
                sigma = sigma_mlp_out.get(layer, 0.0)
            else:
                sigma = 0.0
            props['noise_sigma'] = float(sigma)
    
    # 3. Execute analysis pipeline (5 LOC)
    pipeline = PropertyPipeline(specs or ANALYSIS_SPECS)
    
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
        return {}, local_results
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
    """Map pipeline outputs to the per-layer viz dict used by plotting (SPC only)."""
    viz_stats: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for k, props in viz_payload.items():
        beta = matrix_shape_beta(props['checkpoint_weights'].shape)
        sigma_hat = float(props.get('noise_sigma', 0.0))
        entry = {
            'beta': float(beta),
            'sigma_hat': sigma_hat,
            'per_minibatch_singular_values': props['minibatch_singular_values'].detach().cpu().numpy(),
            'spectral_projection_coefficients': props['spectral_projection_coefficients'].detach().cpu().numpy(),
            'shape': tuple(int(x) for x in props['checkpoint_weights'].shape),
        }
        viz_stats[k] = entry
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
        'shape',
        'noise_sigma',
    ]

    # If file exists with an older/different schema, replace it to keep CSV consistent
    write_header = True
    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            if existing_header == fieldnames:
                write_header = False
            else:
                csv_path.unlink()  # remove stale file with incompatible header
        except Exception:
            csv_path.unlink(missing_ok=True)
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
                'shape': json.dumps(list(props.get('checkpoint_weights', torch.empty(0)).shape[-2:])),
                'noise_sigma': float(props.get('noise_sigma', 0.0)) if isinstance(props.get('noise_sigma', None), (int, float)) else float(props.get('noise_sigma', torch.tensor(0.0)).item() if isinstance(props.get('noise_sigma', None), torch.Tensor) else 0.0),
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
    
    # Shared visualization accumulators
    gif_frames = defaultdict(list)
    viz_timeseries: Dict[int, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    vis_output_dir = Path("research_logs/visualizations/noise_estimation")
    expected_len = 0
    is_mock = "--local-mock" in sys.argv
    if is_mock:
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

        NUM_CHECKPOINTS = 24

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
            _, viz_payload = compute_analysis_for_step(
                step_idx, "mock", num_minibatches=0, rank=rank, world_size=world_size,
                device=device, model=None, initial_props=initial_props, specs=MOCK_SPECS
            )
            viz_timeseries[step_idx] = _build_viz_stats_from_pipeline(viz_payload)
        expected_len = NUM_CHECKPOINTS

    if not is_mock:
        run_id = sys.argv[1]
        testing_mode = "--testing" in sys.argv
        force_recompute = "--force" in sys.argv
    
    # Distributed setup
    if not is_mock:
        _, rank, world_size, device, master_process = setup_distributed_training()
    
    # Initialize global print function for training_core patch
    from empirical.research.training.training_core import _global_print0
    import empirical.research.training.training_core as training_core
    training_core._global_print0 = lambda s, console=False: print(s) if rank == 0 else None
    
    if not is_mock and rank == 0:
        print(f"Starting gradient distribution analysis for run: {run_id}")
        print(f"Testing mode: {testing_mode}, Force recompute: {force_recompute}")
    
    # Find checkpoints
    if not is_mock:
        checkpoints = find_all_checkpoints(run_id)
        if not checkpoints:
            if rank == 0:
                print(f"No checkpoints found for run {run_id}")
            return 1
    
    if not is_mock and testing_mode:
        checkpoints = checkpoints[:2]  # Only process first 2 steps
    
    if not is_mock and rank == 0:
        print(f"Found {len(checkpoints)} checkpoints to process")

    # Clean stale per-step CSVs before processing (rank 0), then sync
    if not is_mock and rank == 0:
        sv_dir = Path("research_logs/singular_values_distribution")
        if sv_dir.exists():
            for csv_file in sv_dir.glob("step_*.csv"):
                csv_file.unlink()
    if not is_mock and dist.is_initialized():
        dist.barrier()

    # Finite-size Wishart tables are loaded on-demand from CSVs

    # Build and warm up model once per process
    if not is_mock:
        model = build_compiled_model(device)
    # Lightweight warmup for analysis - use small sequence to avoid OOM
    if not is_mock:
        args_local = Hyperparameters()
        small_seq_len = 128
        vocab_size = args_local.vocab_size
        inputs = targets = torch.randint(0, vocab_size, size=(small_seq_len,), device=device)
        window_size_blocks = get_window_size_blocks(0, args_local.num_iterations).to(device)
        model(inputs.to(torch.int32), targets, window_size_blocks).backward()
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        precompile_svd_kernels(device, rank)
        if dist.is_initialized():
            dist.barrier()

    # Process each checkpoint
    if not is_mock:
        num_minibatches = 8  # Match training configuration
        for step, checkpoint_file in checkpoints:
            if rank == 0:
                print(f"\nProcessing step {step}...")
                start_time = time.time()
            _ = load_weights_into_model(checkpoint_file, model, device)
            records, viz_payload = compute_analysis_for_step(
                step, checkpoint_file, num_minibatches, rank, world_size, device, model
            )
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                viz_timeseries[step] = _build_viz_stats_from_pipeline(viz_payload)
    
    # Create final GIFs
    # Common visualization (single call for mock and real)
    if (is_mock or (not is_mock and rank == 0)):
        print("\nCreating visualization GIFs...")
        create_visualization_frames(0, viz_timeseries, gif_frames, vis_output_dir, rank if not is_mock else 0, frame_types=['spc_singular'])
        # Only finalize SPC GIF
        gif_frames = {'spc_singular': gif_frames.get('spc_singular', [])}
        finalize_gifs(gif_frames, vis_output_dir, rank=(rank if not is_mock else 0),)
        print("Analysis complete!")

    # Ensure all ranks reach a common point before teardown
    if dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()

    return 0



def _plot_pred_vs_actual(ax, prop: Dict[Tuple[str,int], Any], param_type: str, viridis, max_layers: int):
    ax.set_title(param_type)
    ax.set_xlabel('Predicted SPC (log scale)')
    ax.set_ylabel('Actual SPC (log scale)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-8, 1.0); ax.set_ylim(1e-8, 1.0)
    denom = max(1, max_layers - 1)
    for (pt, layer), arr in sorted(prop.items(), key=lambda x: x[0][1]):
        if pt != param_type: continue
        a = np.asarray(arr)
        if a.ndim != 2 or (a.shape[0] != 2 and a.shape[1] != 2):
            continue
        if a.shape[0] == 2:
            pred, actual = a[0], a[1]
        else:
            pred, actual = a[:, 0], a[:, 1]
        pred = np.clip(pred.flatten(), 1e-8, 1.0)
        actual = np.clip(actual.flatten(), 1e-8, 1.0)
        color = viridis(layer / denom)
        ax.scatter(pred, actual, s=6, alpha=0.2, c=[color])
    # y=x reference
    xs = np.geomspace(1e-8, 1.0, 200)
    ax.plot(xs, xs, ls='--', lw=1.0, color='black')
    return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m empirical.research.analysis.compute_gradient_distribution <run_id>")
        return 1
    run_id = sys.argv[1]
    _, rank, world_size, device, _ = setup_distributed_training()
    model = build_compiled_model(device)
    checkpoints = find_all_checkpoints(run_id)
    if rank == 0:
        frames = []
        out_dir = Path("research_logs/visualizations/spc_pred_vs_actual"); out_dir.mkdir(parents=True, exist_ok=True)
        for step, ckpt in checkpoints:
            load_weights_into_model(ckpt, model, device)
            _, payload = compute_analysis_for_step(step, ckpt, num_minibatches=8, rank=rank, world_size=world_size, device=device, model=model)
            # Build property map: (ptype,layer)->2xN [pred; actual]
            prop_all: Dict[Tuple[str,int], Any] = {}
            for key, props in payload.items():
                pred = props['predicted_spectral_projection_coefficient'].detach().cpu().numpy().flatten()
                actual = props['spectral_projection_coefficients'].detach().cpu().numpy().flatten()
                n = min(len(pred), len(actual))
                if n:
                    prop_all[key] = np.vstack([pred[:n], actual[:n]])
            frame_path = out_dir / f"pred_vs_actual_spc_{step:06d}.png"
            create_subplot_grid(PARAM_TYPES, (20,10), prop_all, _plot_pred_vs_actual, f"Predicted vs Actual SPC - Step {step}", frame_path, wants_colorbar=True)
            frames.append(str(frame_path))
        finalize_gifs({'spc_pred_vs_actual': frames}, out_dir, gif_configs={'spc_pred_vs_actual': 'pred_vs_actual_spc.gif'}, rank=0)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
