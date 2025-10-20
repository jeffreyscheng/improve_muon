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
import logging
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
    compute_reverb_echo_and_alignment, gather_aligned_singulars,
    estimate_gradient_noise_sigma2,
    fit_empirical_phase_constant_tau2,
)
from empirical.research.analysis.core_visualization import (
    make_gif_from_layer_property_time_series,
    compute_panel_xs,
    predict_spectral_echo_curve_np,
    newton_schulz_quintic_function,
    PARAM_TYPES,
)
from empirical.research.analysis.logging_utilities import deserialize_model_checkpoint, log_from_rank
from empirical.research.analysis.constants import (
    FIELD_NAMES,
    NUM_MINIBATCHES,
    LOG_EVERY
)

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
    
    PropertySpec("spectral_echo_and_alignment",
                ["minibatch_gradient_svd"],
                compute_reverb_echo_and_alignment),
    PropertySpec("spectral_echo", ["spectral_echo_and_alignment"], lambda t: t[0]),
    PropertySpec("aligned_minibatch_singular_values", ["spectral_echo_and_alignment", "minibatch_singular_values"], lambda sa, sv: gather_aligned_singulars(sv, sa[1])),
    # Use mean singular values per mode (constant across minibatches) as the x-axis signal s

    PropertySpec("aspect_ratio_beta", ["checkpoint_weights"],
                lambda w: float(matrix_shape_beta(w.shape[-2:] if hasattr(w, 'shape') else w))),
    PropertySpec("worker_count", ["per_minibatch_gradient"], lambda g: int(g.shape[0])),
    PropertySpec("m_big", ["checkpoint_weights"], lambda w: int(max(w.shape[-2], w.shape[-1]))),
    PropertySpec("gradient_noise_sigma2", ["per_minibatch_gradient", "mean_gradient"], estimate_gradient_noise_sigma2),
    PropertySpec("empirical_phase_constant_tau2", ["aligned_minibatch_singular_values", "spectral_echo"], fit_empirical_phase_constant_tau2),
]


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
    log_from_rank(f"Loading checkpoint {checkpoint_file}", rank)
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
    num_minibatches: int,
    rank: int,
    world_size: int,
    model: torch.nn.Module,
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
        # Shard parameters across ranks (owners)
        my_param_keys = shard_param_keys(all_param_keys, rank, world_size)
        log_from_rank(f"Processing a shard of {len(my_param_keys)} parameters out of {len(all_param_keys)} total (owners)", rank)

        # Build owner map deterministically across ranks
        owner_map = {}
        for r in range(world_size):
            keys_r = shard_param_keys(all_param_keys, r, world_size)
            for k in keys_r:
                owner_map[k] = r

        # Accumulate and gather microgradients per-key to owners only; owners will have 8xA replicates
        gradients = get_accumulated_gradient_matrices(
            model, args, step, num_minibatches=num_minibatches, assigned_params=my_param_keys, owner_map=owner_map
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
        if completed % LOG_EVERY == 0:
            log_from_rank(f"Analyzed {completed}/{total} layers", rank)
    
    local_results = pipeline.execute_for_all_layers(initial_props, progress_callback)

    # Stream results to per-rank CSV to avoid large in-memory payloads
    stream_write_analysis_results(local_results, step, rank, run_id or "unknown_run")

    if dist.is_initialized():
        dist.barrier()
    log_from_rank(f"Step {step}: Analysis complete (streamed to CSV)", rank)
    return local_results


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

    f, writer = open_layer_stats_writer(csv_path, fieldnames=list(layer_props.keys()))
    try:
        for (param_type, layer_num), props in layer_props.items():
            # Pre-compute scalar extras
            grad_sigma2_val = float(props['gradient_noise_sigma2'])
            tau2_val = float(props['empirical_phase_constant_tau2'])

            row = {
                'param_type': param_type,
                'layer_num': layer_num,
                'weight_stable_rank': float(props['weights_stable_rank']),
                'per_minibatch_gradient_singular_values': json.dumps(to_np16(props['minibatch_singular_values']).tolist()),
                'gradient_singular_value_standard_deviations': json.dumps(to_np16(props['singular_value_std']).tolist()),
                'per_minibatch_gradient_stable_rank': json.dumps(to_np16(props['gradients_stable_rank']).tolist()),
                'spectral_echo_from_reverb': json.dumps(to_np16(props['spectral_echo']).tolist()),
                'shape': json.dumps(list(props['checkpoint_weights'].shape[-2:])),
                'gradient_noise_sigma2': grad_sigma2_val,
                'empirical_phase_constant_tau2': tau2_val,
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



def create_pred_vs_actual_spectral_echo_subplot(ax, prop: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(param_type)
    ax.set_xlabel('Predicted spectral echo')
    ax.set_ylabel('Actual spectral echo')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
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
    xs = np.linspace(0.0, 1.0, 200)
    ax.plot(xs, xs, ls='--', lw=1.0, color='black')
    return []

def create_spectral_echo_vs_sv_semilog_subplot(ax, panel: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(param_type)
    ax.set_xlabel('Singular value s (log scale)')
    ax.set_ylabel('Spectral projection coefficient')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    # Actual spectral echo vs s per layer (scatter)
    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sv = d.get('sv'); echo = d.get('spectral_echo')
        if sv is None or echo is None:
            continue
        sv = np.asarray(sv, dtype=float).flatten()
        echo = np.clip(np.asarray(echo, dtype=float).flatten(), 0.0, 1.0)
        if sv.size == 0 or echo.size == 0:
            continue
        m = min(sv.size, echo.size)
        sv = sv[:m]; echo = echo[:m]
        color = viridis(layer / denom)
        order = np.argsort(sv)
        ax.scatter(sv[order], echo[order], s=6, alpha=0.25, c=[color])
    # Common x-grid across panel
    xs = compute_panel_xs(panel)
    xs_max = float(np.max(xs)) if xs.size else 1.0
    # Overlay NS quintic in black (normalized xs)
    if xs.size:
        xnorm = np.clip(xs / max(xs_max, 1e-12), 0.0, 1.0)
        y_ns = np.clip(newton_schulz_quintic_function(xnorm), 0.0, 1.0)
        ax.plot(xs, y_ns, color='black', lw=1.2, alpha=0.9)
    # Overlay predicted E[spectral_echo] for each layer using tau2
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        tau2 = d.get('tau2', None)
        if tau2 is None:
            continue
        color = viridis(layer / denom)
        y_pred = predict_spectral_echo_curve_np(xs, float(tau2)) if xs.size else np.array([])
        if y_pred.size:
            ax.plot(xs, y_pred, color=color, lw=1.0, alpha=0.9)
    return []


def create_spectral_echo_vs_sv_semilog_normalized_subplot(ax, panel: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(f"{param_type} (normalized s)")
    ax.set_xlabel('Normalized singular value s/max(s) (log scale)')
    ax.set_ylabel('Spectral echo')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sv = d.get('sv'); echo = d.get('spectral_echo')
        if sv is None or echo is None:
            continue
        sv = np.asarray(sv, dtype=float).flatten()
        echo = np.clip(np.asarray(echo, dtype=float).flatten(), 0.0, 1.0)
        if sv.size == 0 or echo.size == 0:
            continue
        m = min(sv.size, echo.size)
        sv = sv[:m]; echo = echo[:m]
        smax = np.max(sv)
        if smax <= 0:
            continue
        svn = sv / smax
        order = np.argsort(svn)
        color = viridis(layer / denom)
        ax.scatter(svn[order], echo[order], s=6, alpha=0.25, c=[color])
        tau2 = d.get('tau2', None)
        if tau2 is None:
            continue
        y_pred = predict_spectral_echo_curve_np(sv[order], float(tau2))
        ax.plot(svn[order], y_pred, color=color, lw=1.0, alpha=0.9)
    return []


def create_kappa_calibration_subplot(ax, panel: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    # Try to extract the fitted kappa for this param_type from the panel entries
    kappa_val = None
    for (pt, _layer), d in panel.items():
        if pt == param_type and isinstance(d, dict) and 'kappa' in d:
            try:
                kappa_val = float(d['kappa'])
            except Exception:
                kappa_val = None
            break
    title = f"{param_type}" if kappa_val is None else f"{param_type} (κ={kappa_val:.3g})"
    ax.set_title(title)
    ax.set_xlabel('σ^2 · κ (log)')
    ax.set_ylabel('Fitted τ^2 (log)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, which='both', alpha=0.3)
    # Collect per-layer points
    xs, ys, layers = [], [], []
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        if 'x_cal' not in d or 'y' not in d:
            continue
        x_cal = float(d['x_cal']); tau2 = float(d['y'])
        if x_cal <= 0 or not np.isfinite(tau2) or tau2 <= 0:
            continue
        xs.append(x_cal); ys.append(tau2); layers.append(layer)
    if not xs:
        return []
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    denom = max(1, max_layers - 1)
    for x, y, layer in zip(xs, ys, layers):
        ax.scatter([x], [y], c=[viridis(layer / denom)], s=26, alpha=0.9)
    # y=x reference
    x_max = float(max(np.max(xs), np.max(ys)))
    xline = np.linspace(0.0, x_max * 1.05, 200)
    ax.plot(xline, xline, ls='--', lw=1.2, color='black')
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
    log_from_rank(f"Filtered checkpoints (skip step 0): {len(checkpoints)} found", rank)
    # Optional testing flag: only run on first 2 checkpoints
    testing_mode = "--testing" in sys.argv
    if testing_mode:
        checkpoints = checkpoints[:2]
        log_from_rank(f"Testing mode enabled: processing {len(checkpoints)} checkpoints", rank)
    # Collect time series (rank 0 only)
    pred_actual_gptlp_ts: Dict[int, GPTLayerProperty] = {}
    echo_singular_direct_ts: Dict[int, GPTLayerProperty] = {}
    echo_singular_from_kappa_ts: Dict[int, GPTLayerProperty] = {}
    kappa_calibration_ts: Dict[int, GPTLayerProperty] = {}
    noise_panel_ts: Dict[int, GPTLayerProperty] = {}
    for step, ckpt in checkpoints:
        load_weights_into_model(ckpt, model, device)
        local_payload = compute_analysis_for_step(step, num_minibatches=NUM_MINIBATCHES, rank=rank, world_size=world_size, model=model, run_id=run_id)
        # Gather layer properties from all ranks to rank 0 so we plot all layers
        aggregated_payload = gather_layer_properties_to_rank_zero(local_payload)
        if rank == 0 and aggregated_payload is not None:
            pred_actual_gptlp_ts[step] = build_pred_actual_gptlp(aggregated_payload)
            noise_panel_ts[step] = build_noise_to_phase_gptlp(aggregated_payload)
            echo_singular_direct_ts[step] = build_spectral_echo_vs_sv_panel(aggregated_payload)
            # Drop GPU-heavy payload immediately after extracting CPU arrays
            aggregated_payload = None
            local_payload = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        if dist.is_initialized():
            dist.barrier()
    if rank == 0:
        # Fit global kappa across all checkpoints (per param type)
        # Aggregate all noise panels
        combined_noise: GPTLayerProperty = {}
        for panel in noise_panel_ts.values():
            combined_noise.update(panel)
        kappa_map = fit_per_type_kappa_loglog(combined_noise)

        # Build per-step kappa-derived panels using the same kappa_map
        for step, panel in noise_panel_ts.items():
            kappa_calibration_ts[step] = build_kappa_calibration_panel(panel, kappa_map)
            echo_singular_from_kappa_ts[step] = build_spectral_echo_vs_sv_panel_from_kappa(echo_singular_direct_ts[step], panel, kappa_map)

        out_dir = Path(f"research_logs/visualizations/{run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_gifs_for_run(out_dir, pred_actual_gptlp_ts, echo_singular_direct_ts, echo_singular_from_kappa_ts, kappa_calibration_ts)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


def build_pred_actual_gptlp(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        sv = props['aligned_minibatch_singular_values'].detach().cpu().numpy().flatten()
        actual = props['spectral_echo'].detach().cpu().numpy().flatten()
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        n = min(len(sv), len(actual))
        if n:
            pred = predict_spectral_echo_curve_np(sv[:n], tau2)
            out[key] = np.vstack([np.clip(pred, 1e-8, 1.0), np.clip(actual[:n], 1e-8, 1.0)])
    return out


def build_spectral_echo_vs_sv_panel(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        sv = props['aligned_minibatch_singular_values'].detach().cpu().numpy().flatten()
        echo = props['spectral_echo'].detach().cpu().numpy().flatten()
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        n = min(len(sv), len(echo))
        if n:
            out[key] = {
                'sv': sv[:n],
                'spectral_echo': echo[:n],
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


def fit_per_type_kappa_loglog(noise_panel: GPTLayerProperty) -> Dict[str, float]:
    """Fit per-parameter-type kappa via log-log OLS with slope fixed to 1.

    For each param_type, solve log(tau2) = log(kappa) + log(sigma2) and estimate
    log(kappa) = mean(log(tau2) - log(sigma2)) across the 16 layers.
    """
    kappa_map: Dict[str, float] = {}
    for param_type in PARAM_TYPES:
        diffs = []
        for (pt, _layer), d in noise_panel.items():
            if pt != param_type or not isinstance(d, dict):
                continue
            sigma2 = float(d.get('sigma2', float('nan')))
            tau2 = float(d.get('tau2', float('nan')))
            if np.isfinite(sigma2) and np.isfinite(tau2) and sigma2 > 0 and tau2 > 0:
                diffs.append(np.log(tau2) - np.log(sigma2))
        kappa_map[param_type] = float(np.exp(np.mean(diffs))) if diffs else float('nan')
    return kappa_map


def build_spectral_echo_vs_sv_panel_from_kappa(direct_panel: GPTLayerProperty,
                                               noise_panel: GPTLayerProperty,
                                               kappa_map: Dict[str, float]) -> GPTLayerProperty:
    """Use existing CPU direct panel (sv, spectral_echo, shape) and noise panel (sigma2) to set tau2 = kappa*sigma2."""
    out: GPTLayerProperty = {}
    for key, d in direct_panel.items():
        param_type, _ = key
        sv = np.asarray(d.get('sv', []))
        echo = np.asarray(d.get('spectral_echo', []))
        shape = tuple(d.get('shape', ()))
        np_noise = noise_panel.get(key, {}) if isinstance(noise_panel.get(key, None), dict) else {}
        sigma2 = float(np_noise.get('sigma2', np.nan))
        kappa = float(kappa_map.get(param_type, np.nan))
        tau2 = sigma2 * kappa if np.isfinite(sigma2) and np.isfinite(kappa) else np.nan
        n = min(len(sv), len(echo))
        if n:
            out[key] = {
                'sv': sv[:n],
                'spectral_echo': echo[:n],
                'tau2': tau2,
                'shape': shape,
            }
    return out


def build_kappa_calibration_panel(noise_panel: GPTLayerProperty, kappa_map: Dict[str, float]) -> GPTLayerProperty:
    """Transform (sigma2, tau2) into (x_cal = sigma2*kappa, y = tau2) per layer; include kappa for titles."""
    out: GPTLayerProperty = {}
    for (pt, layer), d in noise_panel.items():
        sigma2 = float(d.get('sigma2', np.nan))
        tau2 = float(d.get('tau2', np.nan))
        kappa = float(kappa_map.get(pt, np.nan))
        if np.isfinite(sigma2) and np.isfinite(tau2) and np.isfinite(kappa) and sigma2 > 0 and tau2 > 0 and kappa > 0:
            out[(pt, layer)] = {'x_cal': sigma2 * kappa, 'y': tau2, 'kappa': kappa}
    return out


def generate_gifs_for_run(out_dir: Path,
                          pred_ts: Dict[int, GPTLayerProperty],
                          echo_ts_direct: Dict[int, GPTLayerProperty],
                          echo_ts_kappa: Dict[int, GPTLayerProperty],
                          kappa_ts: Dict[int, GPTLayerProperty]):
    make_gif_from_layer_property_time_series(pred_ts, create_pred_vs_actual_spectral_echo_subplot, title="pred_vs_actual_spectral_echo", output_dir=out_dir)
    make_gif_from_layer_property_time_series(echo_ts_direct, create_spectral_echo_vs_sv_semilog_subplot, title="spectral_echo_vs_singular_values_direct", output_dir=out_dir)
    make_gif_from_layer_property_time_series(echo_ts_kappa, create_spectral_echo_vs_sv_semilog_subplot, title="spectral_echo_vs_singular_values_from_kappa", output_dir=out_dir)
    make_gif_from_layer_property_time_series(kappa_ts, create_kappa_calibration_subplot, title="kappa_calibration", output_dir=out_dir)
    # Normalized-x spectral-echo plot (using direct tau2 overlays)
    make_gif_from_layer_property_time_series(echo_ts_direct, create_spectral_echo_vs_sv_semilog_normalized_subplot, title="spectral_echo_vs_singular_values_normalized_direct", output_dir=out_dir)


if __name__ == "__main__":
    sys.exit(main())
