#!/usr/bin/env python3
"""
Refactored gradient distribution analysis with clean property pipeline architecture.

This script computes gradient analysis using a declarative property pipeline approach.
The core insight: gradient analysis is just a dependency graph of transformations
applied across model layers. By separating the "what" (property definitions) from
the "how" (execution), we achieve dramatically improved readability and maintainability.

Usage:
    torchrun --standalone --nproc_per_node=8 -m empirical.research.analysis.compute_gradient_distribution <run_id> [--testing]
"""

import os
import sys
import glob
import re
from pathlib import Path
import csv
import json
from typing import Dict, Tuple, Any

# Memory optimization like training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch._inductor.config.coordinate_descent_tuning = False
import torch.distributed as dist
import numpy as np

from empirical.research.training.training_core import (
    setup_distributed_training, Hyperparameters,
)
from empirical.research.analysis.model_utilities import (
    get_weight_matrices, get_accumulated_gradient_matrices,
    combine_layer_properties, gather_layer_properties_to_rank_zero, GPTLayerProperty
)
from empirical.research.analysis.property_pipeline import PropertySpec, PropertyPipeline
from empirical.research.analysis.core_math import (
    stable_rank_from_tensor, safe_svd,
    compute_basis_cosine_similarity, compute_spectral_projection_coefficients_from_cosines,
)
from empirical.research.analysis.core_visualization import (
    make_gif_from_layer_property_time_series,
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

## removed mock and SVD precompile helpers for simplicity


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

    # Aggressive debug: on rank 0, print singular value stats per layer
    if rank == 0:
        try:
            for (ptype, layer), props in list(local_results.items())[:32]:  # cap to first 32 layers for readability
                sv = props.get('minibatch_singular_values', None)
                if isinstance(sv, torch.Tensor):
                    sv_cpu = sv.detach().float().cpu().reshape(-1)
                    if sv_cpu.numel() > 0:
                        pos = sv_cpu[sv_cpu > 0]
                        n_pos = int(pos.numel())
                        n = int(sv_cpu.numel())
                        sv_min = float(torch.min(sv_cpu).item())
                        sv_max = float(torch.max(sv_cpu).item())
                        frac_pos = (n_pos / n) if n > 0 else 0.0
                        print(f"[rank0] SV stats {ptype} L{layer}: n={n} pos={n_pos} ({frac_pos:.2%}) min={sv_min:.3e} max={sv_max:.3e}")
        except Exception:
            pass

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print(f"Step {step}: Analysis complete (streamed to CSV)")
        return {}, local_results
    else:
        return {}, {}


## removed legacy record/viz builders in favor of direct GPTLayerProperty usage



def stream_write_analysis_results(layer_props: GPTLayerProperty, step: int, rank: int):
    base_dir = Path("research_logs/singular_values_distribution")
    if dist.is_initialized():
        (rank == 0) and base_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
    else:
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

## removed legacy main; simplified main is defined below



def create_pred_vs_actual_spc_log_log_subplot(ax, prop: GPTLayerProperty, param_type: str, viridis, max_layers: int):
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

def create_spc_vs_sv_semilog_subplot(ax, prop: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(param_type)
    ax.set_xlabel('Singular value (log scale)')
    ax.set_ylabel('Spectral projection coefficient')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    denom = max(1, max_layers - 1)
    for (pt, layer), arr in sorted(prop.items(), key=lambda x: x[0][1]):
        if pt != param_type:
            continue
        a = np.asarray(arr)
        if a.ndim != 2 or (a.shape[0] != 2 and a.shape[1] != 2):
            continue
        sv, spc = (a[0], a[1]) if a.shape[0] == 2 else (a[:, 0], a[:, 1])
        sv = sv.flatten(); spc = np.clip(spc.flatten(), 0.0, 1.0)
        color = viridis(layer / denom)
        ax.scatter(sv, spc, s=6, alpha=0.2, c=[color])
    return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m empirical.research.analysis.compute_gradient_distribution <run_id>")
        return 1
    run_id = sys.argv[1]
    _, rank, world_size, device, _ = setup_distributed_training()
    model = build_compiled_model(device)
    checkpoints = find_all_checkpoints(run_id)
    # Skip step 0 checkpoints (many weights are zero-initialized there)
    checkpoints = [(s, p) for (s, p) in checkpoints if s > 0]
    if rank == 0:
        print(f"Filtered checkpoints (skip step 0): {len(checkpoints)} found")
    # Optional testing flag: only run on first 2 checkpoints
    testing_mode = "--testing" in sys.argv
    if testing_mode:
        checkpoints = checkpoints[:2]
        if rank == 0:
            print(f"Testing mode enabled: processing {len(checkpoints)} checkpoints")
    # Collect time series for two GIFs (rank 0 only)
    pred_actual_gptlp_ts: Dict[int, GPTLayerProperty] = {}
    spc_singular_gptlp_ts: Dict[int, GPTLayerProperty] = {}
    for step, ckpt in checkpoints:
        load_weights_into_model(ckpt, model, device)
        _, payload = compute_analysis_for_step(step, ckpt, num_minibatches=8, rank=rank, world_size=world_size, device=device, model=model)
        if rank == 0:
            # Pred vs actual SPC: GPTLayerProperty mapping (ptype, layer) -> 2xN [pred; actual]
            pred_actual_gptlp: GPTLayerProperty = {}
            spc_singular_gptlp: GPTLayerProperty = {}
            for key, props in payload.items():
                pred = props['predicted_spectral_projection_coefficient'].detach().cpu().numpy().flatten()
                actual = props['spectral_projection_coefficients'].detach().cpu().numpy().flatten()
                n = min(len(pred), len(actual))
                if n:
                    pred_actual_gptlp[key] = np.vstack([pred[:n], actual[:n]])
                # SPC vs Singular Values: (ptype, layer) -> 2xN [sv; spc]
                sv = props['minibatch_singular_values'].detach().cpu().numpy().flatten()
                spc = props['spectral_projection_coefficients'].detach().cpu().numpy().flatten()
                m = min(len(sv), len(spc))
                if m:
                    spc_singular_gptlp[key] = np.vstack([sv[:m], spc[:m]])
            pred_actual_gptlp_ts[step] = pred_actual_gptlp
            spc_singular_gptlp_ts[step] = spc_singular_gptlp
        if dist.is_initialized():
            dist.barrier()
    if rank == 0:
        # Build GIFs using the simplified interface
        make_gif_from_layer_property_time_series(pred_actual_gptlp_ts, create_pred_vs_actual_spc_log_log_subplot, f"{run_id} pred_vs_actual_spc")
        make_gif_from_layer_property_time_series(spc_singular_gptlp_ts, create_spc_vs_sv_semilog_subplot, f"{run_id} spc_vs_singular_values")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
