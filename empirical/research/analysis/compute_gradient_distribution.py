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
    compute_spc_by_permutation_alignment,
)
from empirical.research.analysis.core_visualization import (
    make_gif_from_layer_property_time_series,
    compute_panel_xs,
    predict_spc_curve_np,
)
from empirical.research.analysis.wishart import (
    aspect_ratio_beta as wishart_aspect_ratio_beta,
    squared_true_signal_from_quadratic_formula,
    predict_spectral_projection_coefficient_from_squared_true_signal,
)
from empirical.research.analysis.logging_utilities import deserialize_model_checkpoint, categorize_parameter
from empirical.research.analysis.anisotropy import (
    center_across_workers,
    diagonal_spread,
    sketch_condition,
    john_sphericity,
    lanczos_condition_shrunk,
    matrix_normal_flipflop,
    anisotropy_dispersion_tau2,
    predicted_spc_soft_plugin,
)
import logging


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
    
    # Spectral projection analysis (with Procrustes alignment)
    PropertySpec("spectral_projection_coefficients",
                ["minibatch_gradient_svd", "mean_gradient_svd"],
                compute_spc_by_permutation_alignment),

    # Noise sigma is provided externally from serialized checkpoints; no Wishart fitting
    PropertySpec("aspect_ratio_beta", ["checkpoint_weights"],
                lambda w: float(wishart_aspect_ratio_beta(w))),
    PropertySpec("worker_count", ["per_minibatch_gradient"], lambda g: int(g.shape[0])),
    PropertySpec("m_big", ["checkpoint_weights"], lambda w: int(max(w.shape[-2], w.shape[-1]))),
    PropertySpec("nu_dof", ["per_minibatch_gradient", "checkpoint_weights"],
                lambda g, w: int(max(g.shape[0]-1, 0) * int(w.shape[-2]) * int(w.shape[-1]))),
    PropertySpec("squared_true_signal_t", ["minibatch_singular_values", "noise_sigma", "aspect_ratio_beta"],
                squared_true_signal_from_quadratic_formula),
    PropertySpec("predicted_spectral_projection_coefficient", ["squared_true_signal_t", "aspect_ratio_beta"],
                predict_spectral_projection_coefficient_from_squared_true_signal),

    # Anisotropy metrics (Noise Anisotropy Test)
    PropertySpec("worker_centered_residuals", ["per_minibatch_gradient"], center_across_workers),
    PropertySpec("kappa_diag", ["worker_centered_residuals"], diagonal_spread),
    PropertySpec("kappa_sketch", ["worker_centered_residuals"], lambda E: sketch_condition(E, s=128, repeats=3, seed=1234)),
    PropertySpec("john_sphericity_U", ["worker_centered_residuals"], john_sphericity),
    PropertySpec("kappa_LW", ["worker_centered_residuals"], lanczos_condition_shrunk),
    PropertySpec("matrix_normal_kappas", ["worker_centered_residuals"], matrix_normal_flipflop),
    PropertySpec("anisotropy_tau2", ["worker_centered_residuals"], anisotropy_dispersion_tau2),

    # Softened SPC using (sigma_eff, tau2, beta) and mean singular values
    PropertySpec("predicted_spc_soft_plugin", ["mean_singular_values", "noise_sigma", "anisotropy_tau2", "aspect_ratio_beta", "worker_count", "m_big", "nu_dof"],
                predicted_spc_soft_plugin),

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
    stream_write_analysis_results(local_results, step, rank, run_id or "unknown_run")
    local_records = {}

    # (no debug logs)

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"Step {step}: Analysis complete (streamed to CSV)")
    return {}, local_results


## removed legacy record/viz builders in favor of direct GPTLayerProperty usage



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

def create_spc_vs_sv_semilog_subplot(ax, panel: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(param_type)
    ax.set_xlabel('Singular value (log scale)')
    ax.set_ylabel('Spectral projection coefficient')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    # X-grid across all layers in this panel
    xs = compute_panel_xs(panel)
    # Newton–Schulz quintic reference
    from empirical.research.analysis.core_visualization import newton_schulz_quintic_function
    y_ns = np.clip(newton_schulz_quintic_function(xs), 0.0, 1.0)
    ax.plot(xs, y_ns, color='black', lw=1.5)
    # Predicted SPC curves per layer (softened using tau^2)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type:
            continue
        if not isinstance(d, dict):
            continue
        shape = tuple(int(x) for x in d.get('shape', ()))
        if len(shape) != 2:
            continue
        p, n = shape
        beta = min(p, n) / max(p, n)
        sigma = float(d.get('sigma', 0.0))
        tau2 = float(d.get('tau2', 0.0))
        m_big = int(max(p, n))
        # Effective averaging count approximated by number of per-minibatch grads used
        W_eff = int(d.get('W_eff', 1))
        # Softened SPC curve via anisotropy averaging
        xs_t = torch.from_numpy(xs.astype(np.float32))
        # Approximate ν = (W-1) * n * m from layer shape and W
        nu = int(max(W_eff - 1, 0) * p * n)
        y_pred_t = predicted_spc_soft_plugin(xs_t, sigma, tau2, beta, worker_count=W_eff, m_big=m_big, nu_dof=nu)
        y_pred = y_pred_t.detach().cpu().numpy()
        color = viridis(layer / max(1, max_layers - 1))
        ax.plot(xs, y_pred, color=color, lw=1.0)
        # Optional: faint scatter of actual SV/SPC
        sv = d.get('sv'); spc = d.get('spc')
        if sv is not None and spc is not None:
            sv = np.asarray(sv).flatten(); spc = np.clip(np.asarray(spc).flatten(), 0.0, 1.0)
            if sv.size and spc.size:
                ax.scatter(sv, spc, s=6, alpha=0.05, c=[color])
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
    for step, ckpt in checkpoints:
        load_weights_into_model(ckpt, model, device)
        _, local_payload = compute_analysis_for_step(step, ckpt, num_minibatches=8, rank=rank, world_size=world_size, device=device, model=model, run_id=run_id)
        # Log anisotropy metrics for each local layer on each rank
        for (ptype, layer), props in local_payload.items():
            # Only log if properties were computed
            kd = props.get('kappa_diag', None)
            ks = props.get('kappa_sketch', None)
            ju = props.get('john_sphericity_U', None)
            klw = props.get('kappa_LW', None)
            mnk = props.get('matrix_normal_kappas', None)
            tau2 = props.get('anisotropy_tau2', None)
            spc_soft = props.get('predicted_spc_soft_plugin', None)
            prefix = f"[step={step} rank={rank}] layer=({ptype}, {layer})"
            if kd is not None:
                logging.info(f"{prefix} metric=kappa_diag value={float(kd):.6g}")
            if ks is not None:
                logging.info(f"{prefix} metric=kappa_sketch value={float(ks):.6g}")
            if ju is not None:
                logging.info(f"{prefix} metric=john_sphericity_U value={float(ju):.6g}")
            if klw is not None:
                logging.info(f"{prefix} metric=kappa_LW value={float(klw):.6g}")
            if isinstance(mnk, (tuple, list)) and len(mnk) == 3:
                ku, kv, kmn = mnk
                logging.info(f"{prefix} metric=matrix_normal_kappa_U value={float(ku):.6g}")
                logging.info(f"{prefix} metric=matrix_normal_kappa_V value={float(kv):.6g}")
                logging.info(f"{prefix} metric=matrix_normal_kappa_MN value={float(kmn):.6g}")
            if tau2 is not None:
                logging.info(f"{prefix} metric=anisotropy_tau2 value={float(tau2):.6g}")
            if spc_soft is not None and hasattr(spc_soft, 'detach'):
                ss = spc_soft.detach().cpu().float().numpy().reshape(-1)
                if ss.size:
                    logging.info(f"{prefix} metric=predicted_spc_soft stats={{min:{ss.min():.4f}, med:{np.median(ss):.4f}, max:{ss.max():.4f}}}")
        # Gather layer properties from all ranks to rank 0 so we plot all layers
        aggregated_payload = gather_layer_properties_to_rank_zero(local_payload)
        if rank == 0 and aggregated_payload is not None:
            # Pred vs actual SPC: GPTLayerProperty mapping (ptype, layer) -> 2xN [pred; actual]
            pred_actual_gptlp: GPTLayerProperty = {}
            spc_singular_gptlp: GPTLayerProperty = {}
            for key, props in aggregated_payload.items():
                # Pred vs actual SPC: keep as 2xN
                pred = props['predicted_spectral_projection_coefficient'].detach().cpu().numpy().flatten()
                actual = props['spectral_projection_coefficients'].detach().cpu().numpy().flatten()
                n = min(len(pred), len(actual))
                if n:
                    pred_actual_gptlp[key] = np.vstack([pred[:n], actual[:n]])
                # SPC vs SV: store dict with sv/spc/sigma/shape
                sv = props['minibatch_singular_values'].detach().cpu().numpy().flatten()
                spc = props['spectral_projection_coefficients'].detach().cpu().numpy().flatten()
                m = min(len(sv), len(spc))
                if m:
                    spc_singular_gptlp[key] = {
                        'sv': sv[:m],
                        'spc': spc[:m],
                        'sigma': float(props.get('noise_sigma', 0.0)),
                        'tau2': float(props.get('anisotropy_tau2', 0.0)),
                        'W_eff': int(props.get('worker_count', int(props['minibatch_singular_values'].shape[0]) if hasattr(props.get('minibatch_singular_values', None), 'shape') else 1)),
                        'shape': tuple(int(x) for x in props['checkpoint_weights'].shape[-2:]),
                    }
            pred_actual_gptlp_ts[step] = pred_actual_gptlp
            spc_singular_gptlp_ts[step] = spc_singular_gptlp
        if dist.is_initialized():
            dist.barrier()
    if rank == 0:
        # Build GIFs using the simplified interface (nested under run folder)
        out_dir = Path(f"research_logs/visualizations/{run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        make_gif_from_layer_property_time_series(pred_actual_gptlp_ts, create_pred_vs_actual_spc_log_log_subplot, title="pred_vs_actual_spc", output_dir=out_dir)
        make_gif_from_layer_property_time_series(spc_singular_gptlp_ts, create_spc_vs_sv_semilog_subplot, title="spc_vs_singular_values", output_dir=out_dir)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
