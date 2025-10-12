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
    matrix_shape_beta,
    stable_rank_from_tensor, safe_svd,
    compute_spc_and_alignment, gather_aligned_singulars,
    estimate_gradient_noise_sigma2,
    fit_empirical_phase_constant_tau2,
)
from empirical.research.analysis.core_visualization import (
    make_gif_from_layer_property_time_series,
    compute_panel_xs,
    predict_spc_curve_np,
    newton_schulz_quintic_function,
)
from empirical.research.analysis.logging_utilities import deserialize_model_checkpoint
import logging


# Small configuration knobs
NUM_MINIBATCHES = 8
LOG_EVERY = 5


def gradients_stable_rank(grads: torch.Tensor) -> float:
    return stable_rank_from_tensor(grads.view(-1, grads.shape[-1]))


def singular_value_std(mb_sv: torch.Tensor, mean_sv: torch.Tensor) -> torch.Tensor:
    return torch.std(mb_sv, dim=0)


ANALYSIS_SPECS = [
    # Stable rank computations
    PropertySpec("weights_stable_rank", ["checkpoint_weights"], 
                stable_rank_from_tensor),
    PropertySpec("gradients_stable_rank", ["per_minibatch_gradient"], gradients_stable_rank),
    
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
    PropertySpec("singular_value_std", ["minibatch_singular_values", "mean_singular_values"], singular_value_std),
    
    # Spectral projection analysis (with Procrustes alignment)
    PropertySpec("spc_and_alignment",
                ["minibatch_gradient_svd", "mean_gradient_svd"],
                compute_spc_and_alignment),
    PropertySpec("spectral_projection_coefficients", ["spc_and_alignment"], lambda t: t[0]),
    PropertySpec("aligned_minibatch_singular_values", ["spc_and_alignment", "minibatch_singular_values"], lambda sa, sv: gather_aligned_singulars(sv, sa[1])),

    # Noise sigma is provided externally from serialized checkpoints; no Wishart fitting
    PropertySpec("aspect_ratio_beta", ["checkpoint_weights"],
                lambda w: float(matrix_shape_beta(w.shape[-2:] if hasattr(w, 'shape') else w))),
    PropertySpec("worker_count", ["per_minibatch_gradient"], lambda g: int(g.shape[0])),
    PropertySpec("m_big", ["checkpoint_weights"], lambda w: int(max(w.shape[-2], w.shape[-1]))),
    # PropertySpec("squared_true_signal_t", ["minibatch_singular_values", "noise_sigma", "aspect_ratio_beta"],
    #             squared_true_signal_from_quadratic_formula),
    # PropertySpec("predicted_spectral_projection_coefficient", ["squared_true_signal_t", "aspect_ratio_beta"],
    #             predict_spectral_projection_coefficient_from_squared_true_signal),

    PropertySpec("gradient_noise_sigma2", ["per_minibatch_gradient", "mean_gradient"], estimate_gradient_noise_sigma2),
    PropertySpec("empirical_phase_constant_tau2", ["aligned_minibatch_singular_values", "spectral_projection_coefficients"], fit_empirical_phase_constant_tau2),
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


def shard_param_keys(all_keys: list[Tuple[str, int]], rank: int, world_size: int) -> set:
    """Shard parameter keys across ranks uniformly."""
    n = len(all_keys)
    if world_size <= 1:
        return set(all_keys)
    per = n // world_size
    start = rank * per
    end = start + per if rank < world_size - 1 else n
    return set(all_keys[start:end])


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
    run_id: str | None = None,
) -> GPTLayerProperty:
    """Core analysis function - clean and focused."""
    
    # 1. Build initial properties (either provided for mock, or computed from model)
    args = Hyperparameters()
    
    if initial_props is None:
        # Get all parameters for sharding
        all_weights = get_weight_matrices(model, only_hidden=True)
        all_param_keys = list(all_weights.keys())
        # Shard parameters across ranks
        my_param_keys = shard_param_keys(all_param_keys, rank, world_size)
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
        # Removed noise_sigma injection from checkpoint; sigma^2 is computed per layer in-pipeline
    
    # 3. Execute analysis pipeline (5 LOC)
    pipeline = PropertyPipeline(specs or ANALYSIS_SPECS)
    
    def progress_callback(completed: int, total: int):
        if rank == 0 and completed % LOG_EVERY == 0:
            print(f"  Analyzed {completed}/{total} layers")
    
    local_results = pipeline.execute_for_all_layers(initial_props, progress_callback)

    # Stream results to per-rank CSV to avoid large in-memory payloads
    stream_write_analysis_results(local_results, step, rank, run_id or "unknown_run")

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"Step {step}: Analysis complete (streamed to CSV)")
    return local_results


## removed legacy record/viz builders in favor of direct GPTLayerProperty usage



def to_np16(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float16).cpu().numpy()
    return np.asarray(x)


def open_layer_stats_writer(csv_path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    """Ensure CSV exists with matching header; return (file_handle, writer)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            if existing_header == fieldnames:
                write_header = False
            else:
                csv_path.unlink()
        except Exception:
            csv_path.unlink(missing_ok=True)
    f = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    return f, writer


def stream_write_analysis_results(layer_props: GPTLayerProperty, step: int, rank: int, run_id: str):
    base_dir = Path(f"research_logs/per_layer_statistics/{run_id}")
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
        'gradient_noise_sigma2',
        'empirical_phase_constant_tau2',
        'empirical_noise_to_phase_slope_kappa',
    ]

    f, writer = open_layer_stats_writer(csv_path, fieldnames)
    try:
        for (param_type, layer_num), props in layer_props.items():
            # Pre-compute scalar extras
            grad_sigma2_val = float(props.get('gradient_noise_sigma2', 0.0))
            tau2_val = float(props.get('empirical_phase_constant_tau2', 0.0))
            empirical_kappa_val = (tau2_val / grad_sigma2_val) if grad_sigma2_val > 0 else 0.0

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
                'gradient_noise_sigma2': grad_sigma2_val,
                'empirical_phase_constant_tau2': tau2_val,
                'empirical_noise_to_phase_slope_kappa': empirical_kappa_val,
            }
            writer.writerow(row)
    finally:
        f.close()


def find_all_checkpoints(run_id: str) -> list[tuple[int, str]]:
    """Find all checkpoint files for the given run."""
    ckpt_dir = Path("research_logs/checkpoints") / run_id
    unique: Dict[int, str] = {}
    for p in ckpt_dir.glob("model_step_*.pt"):
        m = re.search(r'step_(\d+)', p.stem)
        if m:
            unique[int(m.group(1))] = str(p)
    return [(s, unique[s]) for s in sorted(unique)]

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

def create_spc_vs_sv_semilog_subplot(ax, panel: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(param_type)
    ax.set_xlabel('Singular value s (log scale)')
    ax.set_ylabel('Spectral projection coefficient')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    # Actual SPC vs s per layer (scatter)
    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sv = d.get('sv'); spc = d.get('spc')
        if sv is None or spc is None:
            continue
        sv = np.asarray(sv, dtype=float).flatten()
        spc = np.clip(np.asarray(spc, dtype=float).flatten(), 0.0, 1.0)
        if sv.size == 0 or spc.size == 0:
            continue
        m = min(sv.size, spc.size)
        sv = sv[:m]; spc = spc[:m]
        color = viridis(layer / denom)
        order = np.argsort(sv)
        ax.scatter(sv[order], spc[order], s=6, alpha=0.25, c=[color])
    # Common x-grid across panel
    xs = compute_panel_xs(panel)
    xs_max = float(np.max(xs)) if xs.size else 1.0
    # Overlay NS quintic in black (normalized xs)
    if xs.size:
        xnorm = np.clip(xs / max(xs_max, 1e-12), 0.0, 1.0)
        y_ns = np.clip(newton_schulz_quintic_function(xnorm), 0.0, 1.0)
        ax.plot(xs, y_ns, color='black', lw=1.2, alpha=0.9)
    # Overlay predicted E[SPC] for each layer using tau2
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        tau2 = d.get('tau2', None)
        if tau2 is None:
            continue
        color = viridis(layer / denom)
        y_pred = predict_spc_curve_np(xs, float(tau2)) if xs.size else np.array([])
        if y_pred.size:
            ax.plot(xs, y_pred, color=color, lw=1.0, alpha=0.9)
    return []


def create_noise_to_phase_slope_subplot(ax, panel: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(param_type)
    ax.set_xlabel('Gradient noise variance σ^2')
    ax.set_ylabel('Phase constant τ^2')
    ax.grid(True, alpha=0.3)
    # Collect points
    xs, ys, layers = [], [], []
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sigma2 = d.get('sigma2'); tau2 = d.get('tau2')
        if sigma2 is None or tau2 is None:
            continue
        xs.append(float(sigma2)); ys.append(float(tau2)); layers.append(layer)
    if not xs:
        return []
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    denom = max(1, max_layers - 1)
    for x, y, layer in zip(xs, ys, layers):
        ax.scatter([x], [y], c=[viridis(layer / denom)], s=22, alpha=0.9)
    valid = xs > 0
    kappas = ys[valid] / xs[valid]
    if kappas.size:
        mean_kappa = float(np.mean(kappas))
        std_kappa = float(np.std(kappas, ddof=1)) if kappas.size > 1 else 0.0
        ci_kappa = 1.96 * (std_kappa / np.sqrt(max(1, kappas.size)))
        xline = np.linspace(0.0, float(np.max(xs)) * 1.05, 200)
        y_center = mean_kappa * xline
        y_lo = max(0.0, mean_kappa - ci_kappa) * xline
        y_hi = (mean_kappa + ci_kappa) * xline
        ax.plot(xline, y_center, color='black', lw=1.5)
        ax.fill_between(xline, y_lo, y_hi, color='black', alpha=0.15)
    return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m empirical.research.analysis.compute_gradient_distribution <run_id>")
        return 1
    # Lightweight logging setup
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
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
    noise_to_phase_ts: Dict[int, GPTLayerProperty] = {}
    for step, ckpt in checkpoints:
        load_weights_into_model(ckpt, model, device)
        local_payload = compute_analysis_for_step(step, ckpt, num_minibatches=NUM_MINIBATCHES, rank=rank, world_size=world_size, device=device, model=model, run_id=run_id)
        # Optional verbose per-layer diagnostic logging (enable with LOG_LAYER_METRICS=1)
        if os.getenv("LOG_LAYER_METRICS") == "1":
            for (ptype, layer), props in local_payload.items():
                prefix = f"[step={step} rank={rank}] layer=({ptype}, {layer})"
                for key in (
                    'kappa_diag', 'kappa_sketch', 'john_sphericity_U', 'kappa_LW',
                    'anisotropy_tau2'
                ):
                    val = props.get(key, None)
                    if val is not None:
                        logging.info(f"{prefix} metric={key} value={float(val):.6g}")
        # Gather layer properties from all ranks to rank 0 so we plot all layers
        aggregated_payload = gather_layer_properties_to_rank_zero(local_payload)
        if rank == 0 and aggregated_payload is not None:
            pred_actual_gptlp_ts[step] = build_pred_actual_gptlp(aggregated_payload)
            spc_singular_gptlp_ts[step] = build_spc_singular_gptlp(aggregated_payload)
            noise_to_phase_ts[step] = build_noise_to_phase_gptlp(aggregated_payload)
        if dist.is_initialized():
            dist.barrier()
    if rank == 0:
        out_dir = Path(f"research_logs/visualizations/{run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_gifs_for_run(out_dir, pred_actual_gptlp_ts, spc_singular_gptlp_ts, noise_to_phase_ts)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


def build_pred_actual_gptlp(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        sv = props['aligned_minibatch_singular_values'].detach().cpu().numpy().flatten()
        actual = props['spectral_projection_coefficients'].detach().cpu().numpy().flatten()
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        n = min(len(sv), len(actual))
        if n:
            pred = predict_spc_curve_np(sv[:n], tau2)
            out[key] = np.vstack([np.clip(pred, 1e-8, 1.0), np.clip(actual[:n], 1e-8, 1.0)])
    return out


def build_spc_singular_gptlp(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        sv = props['aligned_minibatch_singular_values'].detach().cpu().numpy().flatten()
        spc = props['spectral_projection_coefficients'].detach().cpu().numpy().flatten()
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        n = min(len(sv), len(spc))
        if n:
            out[key] = {
                'sv': sv[:n],
                'spc': spc[:n],
                'tau2': tau2,
                'shape': tuple(int(x) for x in props['checkpoint_weights'].shape[-2:]),
            }
    return out


def build_noise_to_phase_gptlp(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        sigma2 = float(props.get('gradient_noise_sigma2', np.nan))
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        out[key] = {'sigma2': sigma2, 'tau2': tau2}
    return out


def generate_gifs_for_run(out_dir: Path,
                          pred_ts: Dict[int, GPTLayerProperty],
                          spc_ts: Dict[int, GPTLayerProperty],
                          noise_ts: Dict[int, GPTLayerProperty]):
    make_gif_from_layer_property_time_series(pred_ts, create_pred_vs_actual_spc_log_log_subplot, title="pred_vs_actual_spc", output_dir=out_dir)
    make_gif_from_layer_property_time_series(spc_ts, create_spc_vs_sv_semilog_subplot, title="spc_vs_singular_values", output_dir=out_dir)
    make_gif_from_layer_property_time_series(noise_ts, create_noise_to_phase_slope_subplot, title="noise_to_phase_slope", output_dir=out_dir)


if __name__ == "__main__":
    sys.exit(main())
