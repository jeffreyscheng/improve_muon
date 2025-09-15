"""
Core visualization utilities for gradient analysis.

This module provides all the plotting and gif generation functionality needed
across different analysis scripts. It eliminates duplication by providing
a single, consistent interface for all visualization needs.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, List, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio.v2 as imageio
import pandas as pd
import json
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator

# no direct math imports required here
from .wishart import (
    load_sv_quantile_tables_npz,
    select_table,
    predict_counts_from_tabulated,
    predict_spectral_projection_coefficient_from_squared_true_signal,
    F_noise_sigma,
)


# Standard parameter types for consistent visualization
PARAM_TYPES = [
    'Attention Q', 'Attention K', 'Attention V', 
    'Attention O', 'MLP Input', 'MLP Output'
]


_SV_TABLES = load_sv_quantile_tables_npz("sv_quantiles_sigma1.npz")



def create_gif_from_frames(frame_paths: List[str], gif_path: Path, fps: int = 12):
    """Create looping GIF from frame files."""
    if not frame_paths:
        return
    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(gif_path, images, fps=fps, loop=0)


def create_subplot_grid(
    param_types: List[str], 
    figsize: Tuple[int, int],
    get_data_fn: Callable[[str], List[Tuple[int, Dict]]],
    plot_fn: Callable,
    title: str,
    output_path: Path,
    layout: str = "constrained",
    wants_colorbar: bool = False,
):
    """
    Generic subplot grid creator for 6-panel visualizations.
    
    Args:
        param_types: List of parameter type names
        figsize: Figure size (width, height)
        get_data_fn: Function that returns [(layer_num, data)] for a param type
        plot_fn: Function to plot single panel
        title: Overall figure title
        output_path: Where to save the figure
        layout: Layout engine ('constrained', 'tight', or 'none')
    """
    if layout == "constrained":
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    fig.suptitle(title, fontsize=14)
    axes = axes.flatten()
    viridis = plt.cm.viridis

    # Precompute global max layer index for consistent colorbar mapping
    layer_lists = {pt: get_data_fn(pt) for pt in param_types}
    global_max_layers = max((max([ln for ln, _ in lst], default=-1) for lst in layer_lists.values()), default=-1) + 1

    for i, param_type in enumerate(param_types):
        ax = axes[i]
        layer_data_list = layer_lists[param_type]
        plot_fn(ax, param_type, layer_data_list, viridis, global_max_layers)
    
    if layout == "tight":
        plt.tight_layout()

    # Optional single colorbar for layer mapping
    if wants_colorbar and global_max_layers > 0:
        sm = cm.ScalarMappable(cmap=viridis, norm=mcolors.Normalize(vmin=0, vmax=max(global_max_layers - 1, 1)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.02, pad=0.02)
        cbar.set_label('Layer', rotation=270, labelpad=12)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def setup_log_scale_axis(ax, xlabel: str = "singular value s (log scale)", ylabel: str = "density"):
    """Configure axis for log-scale singular value plots."""
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelsize=8)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(NullLocator())  # Avoid ZeroDivisionError
    ax.xaxis.set_major_formatter(LogFormatterMathtext())


def plot_bulk_vs_spike(ax, param_type: str, layer_data_list: List, viridis, max_layers: int):
    """Plot bulk vs spike estimation with MP density overlay."""
    ax.set_title(f'{param_type}')
    setup_log_scale_axis(ax)
    ax.set_yscale('log')

    # Collect singular values from layers (opinionated: requires innovation_sample, beta, sigma_hat)
    samples, betas, sigmas = [], [], []
    for _, data in layer_data_list:
        assert 'innovation_sample' in data, "innovation_sample missing in viz_stats"
        assert 'beta' in data, "beta missing in viz_stats"
        assert 'sigma_hat' in data, "sigma_hat missing in viz_stats"
        s = np.asarray(data['innovation_sample'], dtype=float)
        if s.size:
            # Only positive for log scale
            s = s[s > 0]
            if s.size:
                samples.append(np.ascontiguousarray(s))
                betas.append(float(data['beta']))
                sigmas.append(float(data['sigma_hat']))
    
    if not samples:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return

    all_innov = np.ascontiguousarray(np.concatenate(samples))
    if all_innov.size == 0:
        return
        
    beta = float(np.median(np.asarray(betas))) if betas else 1.0
    sigma_hat = float(np.median(np.asarray(sigmas))) if sigmas else 1e-8
    edge = sigma_hat * (1.0 + np.sqrt(max(beta, 0.0)))

    # Log-safe binning
    x_min = max(all_innov.min(), edge * 1e-2, 1e-8)
    x_max = float(all_innov.max() * 1.05)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return
        
    bins = np.geomspace(x_min, x_max, 60)
    counts_emp, _ = np.histogram(all_innov, bins=bins)
    dens_emp = counts_emp.astype(np.float64) / (len(all_innov) * np.diff(bins))
    ax.hist(all_innov, bins=bins, density=True, alpha=0.35, label="Empirical spectrum")

    # Finite-size Wishart overlay (expected counts as density curve) — requires shape in viz_stats
    if not layer_data_list:
        return
    first_data = layer_data_list[0][1]
    assert 'shape' in first_data, f"shape missing in viz_stats for {param_type}"
    p, n = map(int, first_data['shape'])
    table = select_table(_SV_TABLES, p, n)
    counts, _ = np.histogram(all_innov, bins=bins)
    mu = predict_counts_from_tabulated(bins, table, sigma_hat, total=len(all_innov))
    centers = 0.5 * (bins[1:] + bins[:-1])
    density_pred = mu.astype(np.float64) / (len(all_innov) * np.diff(bins))
    # Clip to log-safe minimum so the line is visible on log-y
    density_pred = np.clip(density_pred, 1e-8, None)
    ax.plot(centers, density_pred, ls='--', lw=2, color='tab:orange', label=f"Wishart FS (σ̂={sigma_hat:.3g})")
    # Edge line
    ax.axvline(edge, color='k', lw=1.0, ls=':', label='τ̂')

    # Metrics: KS distance and mass above edge
    s_sorted = np.sort(all_innov)
    F_emp = np.arange(1, len(s_sorted) + 1, dtype=np.float64) / float(len(s_sorted))
    F_pred = F_noise_sigma(s_sorted, table, sigma_hat)
    ks = float(np.max(np.abs(F_emp - F_pred))) if s_sorted.size else 0.0
    mass_above = float(np.mean(all_innov > edge)) if all_innov.size else 0.0
    txt = (f"β={beta:.3f}\nσ̂={sigma_hat:.3e}\nτ̂={edge:.3e}\n"
           f"KS={ks:.3f}\nP(s>τ̂)={mass_above:.1%}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, lw=0.0))
    ax.legend(loc='upper right', fontsize=8, frameon=False)


def plot_spc_vs_singular_values(ax, param_type: str, layer_data_list: List, viridis, max_layers: int):
    """Plot spectral projection coefficients vs singular values."""
    ax.set_title(f'{param_type}')
    ax.set_xlabel('Singular value (log scale)')
    ax.set_ylabel('Spectral projection coefficient')
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    denom = max(1, max_layers - 1)
    x_min, x_max = np.inf, 0.0
    for layer_num, data in layer_data_list:
        if 'per_minibatch_singular_values' not in data or 'spectral_projection_coefficients' not in data:
            continue
            
        color = viridis(layer_num / denom)
        sv = np.asarray(data['per_minibatch_singular_values'])
        spc = np.asarray(data['spectral_projection_coefficients'])
        
        if sv.size > 0 and spc.size > 0:
            # Flatten if needed and plot
            sv_flat = sv.flatten()
            spc_flat = spc.flatten()
            
            # Subsample for visualization if too many points
            if len(sv_flat) > 1000:
                indices = np.random.choice(len(sv_flat), 1000, replace=False)
                sv_flat = sv_flat[indices]
                spc_flat = spc_flat[indices]
            
            ax.scatter(sv_flat, spc_flat, alpha=0.05, s=12, c=[color])
            # Track x-range for curves
            pos = sv_flat[sv_flat > 0]
            if pos.size:
                x_min = min(x_min, float(pos.min()))
                x_max = max(x_max, float(pos.max()))

    # Overlay curves: Newton–Schulz quintic (black) and predicted SPC per layer (colored)
    if np.isfinite(x_min) and x_max > x_min:
        xs = np.geomspace(max(x_min, 1e-8), x_max, 256)
        # Newton–Schulz quintic in black (required)
        y_ns = newton_schulz_quintic_function(xs)
        ax.plot(xs, y_ns, color='black', lw=1.5, label='Newton–Schulz quintic')
        # Predicted SPC per layer using sigma_hat and beta (calls math util with piecewise rule)
        for layer_num, data in layer_data_list:
            if 'sigma_hat' not in data or 'beta' not in data:
                continue
            sigma = float(data['sigma_hat'])
            beta = float(data['beta'])
            if sigma <= 0:
                continue
            y2 = (xs / max(sigma, 1e-30))**2
            Bcoef = -(y2 - (1.0 + beta))
            disc = np.clip(Bcoef*Bcoef - 4.0*beta, 0.0, None)
            t = 0.5 * (-Bcoef + np.sqrt(disc))
            pred = predict_spectral_projection_coefficient_from_squared_true_signal(t, beta)
            color = viridis(layer_num / denom)
            ax.plot(xs, pred, color=color, lw=1.0)


## removed residual panel per user request


## removed unused plot_stable_rank_evolution


def create_visualization_frames(
    step: int,
    viz_stats: Dict[Tuple[str, int], Dict[str, Any]] | Dict[int, Dict[Tuple[str, int], Dict[str, Any]]],
    gif_frames: Dict[str, List[str]],
    output_dir: Path,
    rank: int = 0
):
    """
    Create visualization frames.

    Supports two input shapes:
      - per-step dict: {(param_type, layer): data} → renders frames for the given step arg
      - timeseries dict: {step_num: {(param_type, layer): data}} → renders frames for each step_num
    """
    if rank != 0:
        return

    # Create frames directory
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Compute consistent SPC x-axis ranges across all steps (from saved CSVs)
    sv_dir = Path("research_logs/singular_values_distribution")
    all_data = load_noise_model_data(sv_dir)
    spc_axis_ranges = compute_visualization_axis_ranges(all_data)

    # If per-step stats not provided, load them from CSVs
    provided = viz_stats if isinstance(viz_stats, dict) else {}
    top_keys = list(provided.keys())
    if not top_keys or isinstance(top_keys[0], int):
        per_step_stats = all_data.get(step, {})
    else:
        per_step_stats = provided

    # Compute bulk x-axis ranges from innovation samples (preferably across steps)
    def compute_bulk_axis_ranges(viz: Dict) -> Dict[str, Tuple[float, float]]:
        ranges: Dict[str, Tuple[float, float]] = {}
        # Detect timeseries: top-level keys are ints
        if viz and isinstance(next(iter(viz.keys())), int):
            steps = sorted(viz.keys())
            per_step = [viz[s] for s in steps]
        else:
            per_step = [viz]
        for param_type in PARAM_TYPES:
            vals = []
            for step_dict in per_step:
                for (p_type, _), d in step_dict.items():
                    if p_type == param_type and 'innovation_sample' in d:
                        s = np.asarray(d['innovation_sample'], dtype=float)
                        s = s[np.isfinite(s) & (s > 0)]
                        if s.size:
                            vals.append(s)
            if not vals:
                continue
            all_s = np.concatenate(vals)
            logs = np.log10(all_s)
            lo_p, hi_p = np.percentile(logs, [1, 99])
            lo, hi = float(10 ** (lo_p - 0.1)), float(10 ** (hi_p + 0.1))
            lo = max(lo, 1e-8)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ranges[param_type] = (lo, hi)
        return ranges

    bulk_axis_ranges = compute_bulk_axis_ranges(viz_stats)
    # Also compute a simple global y max per param type for bulk (from current step data, log-safe)
    bulk_ymax: Dict[str, float] = {}
    for param_type in PARAM_TYPES:
        if param_type not in bulk_axis_ranges:
            continue
        lo, hi = bulk_axis_ranges[param_type]
        bins = np.geomspace(lo, hi, 60)
        xs = []
        for (p_type, _), d in (per_step_stats.items() if isinstance(per_step_stats, dict) else []):
            if p_type == param_type and 'innovation_sample' in d:
                s = np.asarray(d['innovation_sample'], dtype=float)
                s = s[np.isfinite(s) & (s > 0)]
                if s.size:
                    xs.append(s)
        if xs:
            all_s = np.concatenate(xs)
            counts, _ = np.histogram(all_s, bins=bins)
            dens = counts.astype(np.float64) / (len(all_s) * np.diff(bins))
            bulk_ymax[param_type] = float(dens.max() * 1.1 if dens.size else 1.0)

    def _wrap_bulk(base_plot_fn):
        def _wrapped(ax, param_type, layer_data_list, viridis, max_layers):
            base_plot_fn(ax, param_type, layer_data_list, viridis, max_layers)
            if param_type in bulk_axis_ranges:
                lo, hi = bulk_axis_ranges[param_type]
                lo = max(float(lo), 1e-8)
                if hi > lo:
                    ax.set_xlim(lo, float(hi))
            if param_type in bulk_ymax:
                ax.set_ylim(1e-8, bulk_ymax[param_type])
        return _wrapped

    def _wrap_spc(base_plot_fn):
        def _wrapped(ax, param_type, layer_data_list, viridis, max_layers):
            base_plot_fn(ax, param_type, layer_data_list, viridis, max_layers)
            rng = spc_axis_ranges.get(param_type)
            if rng and 'x' in rng:
                lo, hi = rng['x']
                lo = max(float(lo), 1e-8)
                if hi > lo:
                    ax.set_xlim(lo, float(hi))
        return _wrapped

    def render_one_step(step_num: int, per_step_stats: Dict[Tuple[str, int], Dict[str, Any]]):
        def get_layer_data_for_param_type(param_type: str):
            items = []
            for (p_type, layer_num), d in per_step_stats.items():
                if p_type == param_type and layer_num >= 0:
                    items.append((layer_num, d))
            return sorted(items, key=lambda x: x[0])

        frame_configs = {
            'bulk_spike': {
                'plot_fn': _wrap_bulk(plot_bulk_vs_spike),
                'title': f'Bulk vs Spike Estimation - Step {step_num}',
                'filename': f"bulk_spike_{step_num:06d}.png"
            },
            'spc_singular': {
                'plot_fn': _wrap_spc(plot_spc_vs_singular_values),
                'title': f'SPC vs Singular Values - Step {step_num}',
                'filename': f"spc_singular_{step_num:06d}.png"
            }
        }

        for frame_type, config in frame_configs.items():
            frame_path = frames_dir / config['filename']
            create_subplot_grid(
                PARAM_TYPES, (20, 10),
                get_layer_data_for_param_type,
                config['plot_fn'],
                config['title'],
                frame_path,
                wants_colorbar=(frame_type == 'spc_singular')
            )
            gif_frames[frame_type].append(str(frame_path))
        print(f"Rank {rank}: Created visualization frames for step {step_num}")

    # Dispatch: detect timeseries vs single-step input
    top_keys = list(viz_stats.keys())
    if top_keys and isinstance(top_keys[0], int):
        # timeseries: iterate all steps
        for s in sorted(top_keys):
            render_one_step(s, viz_stats[s])
    else:
        render_one_step(step, viz_stats)  # single-step


def finalize_gifs(
    gif_frames: Dict[str, List[str]], 
    output_dir: Path,
    gif_configs: Optional[Dict[str, str]] = None,
    rank: int = 0,
    fps: int = 12,
):
    """
    Create final GIFs from collected frames.
    
    Args:
        gif_frames: Frame paths by gif type
        output_dir: Output directory for GIFs
        gif_configs: Optional mapping of frame_type -> gif_filename
        rank: Process rank
        fps: Frames per second
    """
    if rank != 0 or not any(gif_frames.values()):
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default GIF names
    if gif_configs is None:
        gif_configs = {
            'bulk_spike': 'bulk_vs_spike_estimation.gif',
            'spc_singular': 'spc_vs_singular_values.gif'
        }
    
    for frame_type, frames in gif_frames.items():
        if not frames:
            continue
        gif_name = gif_configs[frame_type]
        gif_path = output_dir / gif_name
        create_gif_from_frames(frames, gif_path, fps=fps)
        for frame_path in frames:
            Path(frame_path).unlink()
        print(f"Wrote GIF: {gif_path}")
    frames_dir = output_dir / "frames"
    if frames_dir.exists() and not any(frames_dir.iterdir()):
        frames_dir.rmdir()


# Additional visualization functions consolidated from visualize_noise_models.py

from empirical.research.training.zeropower import NEWTON_SCHULZ_QUINTIC_COEFFICIENTS

def newton_schulz_quintic_function(x):
    """Newton-Schulz quintic function for overlay."""
    out = x
    for a, b, c in NEWTON_SCHULZ_QUINTIC_COEFFICIENTS:
        out = a * out + b * (out **3) + c * (out ** 5)
    return out


def load_noise_model_data(sv_dir: Path) -> dict:
    """Load singular values distribution data from CSV files."""
    results = {}
    
    def _parse_arrayish(val):
        if isinstance(val, str) and val:
            return np.array(json.loads(val))
        if isinstance(val, (list, tuple, np.ndarray)):
            return np.array(val)
        # Opinionated: must be provided in expected schema
        raise ValueError("Expected JSON string or array for array-like CSV field")

    for csv_file in sv_dir.glob("step_*.csv"):
        step_num = int(csv_file.stem.split('_')[1])
        df = pd.read_csv(csv_file)
        step_data = {}
        
        for _, row in df.iterrows():
            param_type = row['param_type']
            layer_num = int(row['layer_num'])
            key = (param_type, layer_num)
            
            gradient_svs = _parse_arrayish(row.get('per_minibatch_gradient_singular_values'))
            
            layer_data = {'per_minibatch_singular_values': gradient_svs.flatten()}

            # Optional derived fields if present in CSV
            if 'shape' in row and pd.notna(row['shape']):
                shp = json.loads(row['shape']) if isinstance(row['shape'], str) else row['shape']
                layer_data['shape'] = tuple(int(x) for x in shp)
            if 'noise_sigma' in row and pd.notna(row['noise_sigma']):
                layer_data['noise_sigma'] = float(row['noise_sigma'])
            
            # Load spectral projection coefficients
            val_spc = row.get('spectral_projection_coefficients_from_8x_mean_gradient') if isinstance(row, dict) else row['spectral_projection_coefficients_from_8x_mean_gradient']
            spc_coeffs = _parse_arrayish(val_spc)
            if spc_coeffs.size:
                layer_data['spectral_projection_coefficients_from_8x_mean_gradient'] = spc_coeffs.flatten()
            
            step_data[key] = layer_data
        
        results[step_num] = step_data
    
    return results


def compute_visualization_axis_ranges(data):
    """Compute consistent x-axis ranges across all data for each param type (log-aware)."""
    ranges = {}
    for param_type in PARAM_TYPES:
        x_vals = []
        for step_data in data.values():
            for (p_type, layer_num), layer_data in step_data.items():
                if p_type == param_type and layer_num >= 0:
                    x_vals.extend(layer_data.get('per_minibatch_singular_values', []))
        x_vals = np.array(x_vals, dtype=float)
        x_vals = x_vals[np.isfinite(x_vals) & (x_vals > 0)]
        if x_vals.size > 0:
            logs = np.log10(x_vals)
            lo_p, hi_p = np.percentile(logs, [1, 99])
            lo, hi = float(10 ** (lo_p - 0.1)), float(10 ** (hi_p + 0.1))
            lo = max(lo, 1e-8)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 1e-3, 1e1
            x_range = (lo, hi)
        else:
            x_range = (1e-3, 1e1)
        ranges[param_type] = {'x': x_range}
    return ranges
