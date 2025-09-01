#!/usr/bin/env python3
"""
Clean visualization of spectral projection coefficients vs gradient singular values.
Focuses on data visualization with Newton-Schulz quintic overlay.

Usage:
    python visualize_noise_models.py run_name [fps] [--testing]
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from empirical.research.analysis.map import get_research_log_path
from empirical.research.analysis.predict_spectral_projection_torch import (
    estimate_noise_level_numpy,
    matrix_shape_beta,
)


PARAM_TYPES = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'MLP Input', 'MLP Output']


def newton_schulz_quintic_function(x):
    """Newton-Schulz quintic function for overlay."""
    from empirical.research.training.zeropower import NEWTON_SCHULZ_QUINTIC_COEFFICIENTS
    out = x
    for a, b, c in NEWTON_SCHULZ_QUINTIC_COEFFICIENTS:
        out = a * out + b * (out **3) + c * (out ** 5)
    return out


def load_data(sv_dir: Path) -> dict:
    """Load singular values distribution data from CSV files."""
    results = {}
    
    for csv_file in sv_dir.glob("step_*.csv"):
        step_num = int(csv_file.stem.split('_')[1])
        df = pd.read_csv(csv_file)
        step_data = {}
        
        for _, row in df.iterrows():
            param_type = row['param_type']
            layer_num = int(row['layer_num'])
            key = (param_type, layer_num)
            
            gradient_svs = np.array(json.loads(row['per_minibatch_gradient_singular_values']))
            gradient_sv_stds = np.array(json.loads(row['gradient_singular_value_standard_deviations']))
            
            layer_data = {
                'per_minibatch_singular_values': gradient_svs.flatten(),
                'stds': np.tile(gradient_sv_stds, len(gradient_svs)),
                'weight_stable_rank': row['weight_stable_rank'],
            }
            
            # Load gradient stable rank
            if 'per_minibatch_gradient_stable_rank' in row:
                per_mb_ranks = np.array(json.loads(row['per_minibatch_gradient_stable_rank']))
                layer_data.update({
                    'gradient_stable_rank_mean': per_mb_ranks.mean(),
                    'gradient_stable_rank_std': per_mb_ranks.std(),
                    'per_minibatch_gradient_stable_rank': per_mb_ranks
                })
            
            # Load spectral projection coefficients
            if 'spectral_projection_coefficients_from_8x_mean_gradient' in row and pd.notna(row['spectral_projection_coefficients_from_8x_mean_gradient']):
                spc_coeffs = np.array(json.loads(row['spectral_projection_coefficients_from_8x_mean_gradient']))
                layer_data['spectral_projection_coefficients_from_8x_mean_gradient'] = spc_coeffs.flatten()
            
            step_data[key] = layer_data
        
        results[step_num] = step_data
    
    return results


def create_gif_from_frames(frame_paths, gif_path, fps=12):
    """Create looping GIF from frame files."""
    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    print(f"Looping GIF saved: {gif_path}")


def _mp_pdf_singular_np(s: np.ndarray, beta: float, sigma: float) -> np.ndarray:
    """
    Marchenko–Pastur density for singular values (not eigenvalues).
    s, beta, sigma are numpy scalars/arrays; returns pdf(s) with same shape as s.
    """
    s = np.asarray(s, dtype=np.float64)
    sigma = float(sigma)
    sqrtb = np.sqrt(beta)
    lam_m = (1.0 - sqrtb) ** 2
    lam_p = (1.0 + sqrtb) ** 2
    u = np.clip(s / max(sigma, 1e-30), 1e-30, None)
    lam = u * u
    inside = (lam > lam_m) & (lam < lam_p)
    out = np.zeros_like(s, dtype=np.float64)
    num = np.sqrt(np.clip((lam_p - lam) * (lam - lam_m), 0.0, None))
    out[inside] = num[inside] / (np.pi * beta * sigma * u[inside])
    return out

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


def compute_axis_ranges(data):
    """Compute consistent axis ranges across all data."""
    ranges = {}
    for param_type in PARAM_TYPES:
        x_vals, y_vals = [], []
        for step_data in data.values():
            for (p_type, layer_num), layer_data in step_data.items():
                if p_type == param_type and layer_num >= 0:
                    x_vals.extend(layer_data['per_minibatch_singular_values'])
                    y_vals.extend(layer_data['stds'])
        
        x_vals, y_vals = np.array(x_vals), np.array(y_vals)
        x_vals, y_vals = x_vals[x_vals > 0], y_vals[y_vals > 0]  # Remove zeros/negatives
        
        ranges[param_type] = {
            'x_min': max(1e-6, np.nanmin(x_vals)) if len(x_vals) > 0 else 1e-6,
            'x_max': np.nanmax(x_vals) if len(x_vals) > 0 else 1.0,
            'y_min': max(1e-6, np.nanmin(y_vals)) if len(y_vals) > 0 else 1e-6,
            'y_max': np.nanmax(y_vals) if len(y_vals) > 0 else 1.0,
        }
    return ranges


def create_frame_with_scatter(step, step_data, axis_ranges, output_dir):
    """Create scatter plot frame for std vs mean."""
    def get_frame_data(param_type):
        return [(layer_num, data) for (p_type, layer_num), data in step_data.items() 
                if p_type == param_type and layer_num >= 0]
    
    def plot_std_mean_scatter(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_xlabel('Per-minibatch gradient singular value')
        ax.set_ylabel('Standard deviation')
        ax.set_title(f'{param_type}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        rng = axis_ranges[param_type]
        ax.set_xlim(rng['x_min'], rng['x_max'])
        ax.set_ylim(rng['y_min'], rng['y_max'])
        
        for layer_num, data in layer_data_list:
            color = viridis(layer_num / (max_layers - 1))
            ax.scatter(data['per_minibatch_singular_values'], data['stds'], 
                      alpha=0.08, s=15, c=[color])
    
    frame_path = output_dir / f"frame_scatter_{step:06d}.png"
    create_subplot_grid(PARAM_TYPES, (20, 10), get_frame_data, plot_std_mean_scatter, 
                       f'Std–Mean Scatter - Step {step}', frame_path)
    return str(frame_path)


def create_stable_rank_plots(data, output_dir):
    """Create stable rank evolution plots."""
    def get_gradient_data(param_type):
        data_by_layer = defaultdict(list)
        for step, step_data in data.items():
            for (p_type, layer_num), layer_data in step_data.items():
                if p_type == param_type and layer_num >= 0:
                    if 'gradient_stable_rank_mean' in layer_data:
                        data_by_layer[layer_num].append({
                            'step': step,
                            'mean': layer_data['gradient_stable_rank_mean'],
                            'std': layer_data['gradient_stable_rank_std']
                        })
        return list(data_by_layer.items())
    
    def get_weight_data(param_type):
        data_by_layer = defaultdict(list)
        for step, step_data in data.items():
            for (p_type, layer_num), layer_data in step_data.items():
                if p_type == param_type and layer_num >= 0:
                    if 'weight_stable_rank' in layer_data:
                        data_by_layer[layer_num].append({
                            'step': step,
                            'value': layer_data['weight_stable_rank']
                        })
        return list(data_by_layer.items())
    
    def plot_gradient_stable_rank(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Stable Rank')
        ax.set_title(f'{param_type} - Gradient Stable Rank')
        ax.grid(True, alpha=0.3)
        
        for layer_num, data in layer_data_list:
            step_vals = [d['step'] for d in data]
            means = [d['mean'] for d in data]
            stds = [d['std'] for d in data]
            color = viridis(layer_num / (max_layers - 1))
            ax.plot(step_vals, means, color=color, alpha=0.8, linewidth=1.5, label=f'Layer {layer_num}')
            upper = np.array(means) + 1.96 * np.array(stds)
            lower = np.array(means) - 1.96 * np.array(stds)
            ax.fill_between(step_vals, lower, upper, color=color, alpha=0.2)
    
    def plot_weight_stable_rank(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Weight Stable Rank')
        ax.set_title(f'{param_type} - Weight Stable Rank')
        ax.grid(True, alpha=0.3)
        
        for layer_num, data in layer_data_list:
            step_vals = [d['step'] for d in data]
            values = [d['value'] for d in data]
            color = viridis(layer_num / (max_layers - 1))
            ax.plot(step_vals, values, color=color, alpha=0.8, linewidth=1.5, label=f'Layer {layer_num}')
    
    create_subplot_grid(PARAM_TYPES, (18, 10), get_gradient_data, plot_gradient_stable_rank, 
                       'Gradient Stable Rank Evolution', output_dir / "gradient_stable_rank.png")
    create_subplot_grid(PARAM_TYPES, (18, 10), get_weight_data, plot_weight_stable_rank, 
                       'Weight Stable Rank Evolution', output_dir / "weight_stable_rank.png")


def create_spc_plots(data, output_dir):
    """Create spectral projection coefficient plots with Newton-Schulz overlay."""
    # Check if SPC data is available
    has_spc = any('spectral_projection_coefficients_from_8x_mean_gradient' in layer_data
                  for step_data in data.values() for layer_data in step_data.values())
    if not has_spc:
        print("No spectral projection coefficient data found; skipping SPC plots.")
        return
    
    # Compute axis ranges
    x_min, x_max = 1e-6, 1.0
    for step_data in data.values():
        for (param_type, _), d in step_data.items():
            if param_type not in PARAM_TYPES or 'spectral_projection_coefficients_from_8x_mean_gradient' not in d:
                continue
            svals = d['per_minibatch_singular_values']
            if len(svals) > 0:
                x_min = min(x_min, max(1e-6, np.nanmin(svals)))
                x_max = max(x_max, np.nanmax(svals))
    
    frame_paths = []
    for step in sorted(data.keys()):
        def get_data(param_type):
            return [(layer_num, data) for (p_type, layer_num), data in data[step].items()
                    if p_type == param_type and layer_num >= 0 and 
                    'spectral_projection_coefficients_from_8x_mean_gradient' in data]
        
        def plot_frame(ax, param_type, layer_list, viridis, max_layers):
            ax.set_xlabel('Per-minibatch gradient singular value')
            ax.set_ylabel('Spectral projection coefficient')
            ax.set_title(f'{param_type}')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log'); ax.set_yscale('linear')
            ax.set_xlim(x_min, x_max); ax.set_ylim(0.0, 1.0)
            
            for layer_num, d in layer_list:
                color = viridis(layer_num / (max_layers - 1))
                svals = d['per_minibatch_singular_values']
                spc = d['spectral_projection_coefficients_from_8x_mean_gradient']
                m = min(len(svals), len(spc))
                if m > 0:
                    ax.scatter(svals[:m], spc[:m], alpha=0.12, s=12, c=[color])
            
            # Add Newton-Schulz quintics function curve in black
            x_ns = np.logspace(np.log10(x_min), np.log10(x_max), 200)
            y_ns = newton_schulz_quintic_function(x_ns)
            ax.plot(x_ns, y_ns, 'k--', linewidth=2, alpha=0.8, label='Newton-Schulz Quintics')
            ax.legend(loc='upper left', fontsize=6)
        
        frame_path = output_dir / f"frame_spc_{step:06d}.png"
        create_subplot_grid(PARAM_TYPES, (18, 10), get_data, plot_frame, 
                           f'SPC vs s - Step {step}', frame_path)
        frame_paths.append(str(frame_path))
    
    print(f"Creating SPC GIF with {len(frame_paths)} frames...")
    create_gif_from_frames(frame_paths, output_dir / "spc_vs_s.gif", 12)
    for fp in frame_paths:
        Path(fp).unlink()


def create_bulk_spike_plots(data, output_dir, bins_per_axis: int = 64):
    """
    Bulk vs Spike plots with per-layer *lines* (no bars):
    - empirical density: solid viridis line per layer
    - fitted MP bulk:    dashed viridis line per layer (same color)
    x-axis is log scale to match the noise model visualizations.
    """
    # Shapes used for beta (matches training architecture)
    # If your shapes change, update this map.
    SHAPES = {
        'Attention Q': (1024, 1024),
        'Attention K': (1024, 1024),
        'Attention V': (1024, 1024),
        'Attention O': (1024, 1024),
        'MLP Input'  : (4096, 1024),
        'MLP Output' : (1024, 4096),
    }

    def _geom_bins_for_param(step_data_for_type):
        """Compute shared geometric bins for one param type at a given step."""
        smins, smaxs = [], []
        for _, d in step_data_for_type:
            sv = np.asarray(d.get('per_minibatch_singular_values', []), dtype=np.float64)
            sv = sv[sv > 0]
            if sv.size:
                smins.append(sv.min())
                smaxs.append(sv.max())
        if not smaxs:
            # Fallback to a tiny positive range to avoid logspace errors
            return np.geomspace(1e-12, 1.0, bins_per_axis + 1)
        lo = max(1e-12, float(np.min(smins)))
        hi = float(np.max(smaxs)) * 1.05
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 1e-12, 1.0
        return np.geomspace(lo, hi, bins_per_axis + 1)

    def _get_step_param_type(step_dict, param_type):
        return [(layer_num, d) for (p_type, layer_num), d in step_dict.items()
                if p_type == param_type and layer_num >= 0 and
                'per_minibatch_singular_values' in d]

    def _plot_param_type(ax, param_type, layer_list, viridis, max_layers, bins):
        ax.set_title(f'{param_type}')
        ax.set_xlabel('singular value s  (log scale)')
        ax.set_ylabel('density')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Common bin centers and widths
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths  = np.diff(bins)

        # β from known shape
        shape = SHAPES.get(param_type, (1024, 1024))
        beta = matrix_shape_beta(shape)

        # Color scale keyed to layer id (avoid divide-by-zero)
        denom = max(1, max_layers - 1)

        for layer_num, d in layer_list:
            color = viridis(layer_num / denom)
            sv = np.asarray(d['per_minibatch_singular_values'], dtype=np.float64)
            sv = sv[sv > 0]
            if sv.size == 0:
                continue
            # Empirical density on geometric bins
            counts, _ = np.histogram(sv, bins=bins)
            density = counts.astype(np.float64) / (sv.size * widths)
            # MP fit: estimate σ from the *descending* singulars
            s_desc = np.sort(sv)[::-1].copy()
            sigma_hat = estimate_noise_level_numpy(s_desc, beta)
            mp = _mp_pdf_singular_np(centers, beta, sigma_hat)
            # Draw both lines
            ax.plot(centers, density, color=color, lw=1.6, alpha=0.9)
            ax.plot(centers, mp,      color=color, lw=1.6, ls='--', alpha=0.9)

    # Build frames, one per step, 6 subplots (param types)
    frame_paths = []
    for step in sorted(data.keys()):
        def _get_data(param_type):
            return _get_step_param_type(data[step], param_type)

        # Precompute bins per param type for this step (shared within panel)
        bins_map = {
            pt: _geom_bins_for_param(_get_data(pt)) for pt in PARAM_TYPES
        }

        def _plot(ax, param_type, layer_list, viridis, max_layers):
            _plot_param_type(ax, param_type, layer_list, viridis, max_layers, bins_map[param_type])

        frame_path = output_dir / f"bulk_spike_lines_{step:06d}.png"
        create_subplot_grid(PARAM_TYPES, (20, 10), _get_data, _plot,
                            f'Bulk vs Spike (lines) - Step {step}', frame_path)
        frame_paths.append(str(frame_path))

    if frame_paths:
        print(f"Creating Bulk/Spike lines GIF with {len(frame_paths)} frames...")
        create_gif_from_frames(frame_paths, output_dir / "bulk_spike_lines.gif", 12)
        for fp in frame_paths:
            Path(fp).unlink()
    else:
        print("No singular values found for bulk/spike plots; skipping.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_noise_models.py run_name [fps] [--testing]")
        sys.exit(1)
    
    run_name = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2] != "--testing" else 12
    testing = "--testing" in sys.argv
    
    # Setup directories
    sv_input_dir = get_research_log_path("singular_values_distribution", run_name, "")
    vis_output_dir = get_research_log_path("visualizations", run_name, "")
    
    print(f"Loading data from: {sv_input_dir}")
    data = load_data(sv_input_dir)
    
    if not data:
        print(f"ERROR: No data found in {sv_input_dir}")
        sys.exit(1)
    
    # Apply testing filter: use 24 checkpoints from step 100 or earlier
    if testing:
        available_steps = [step for step in data.keys() if step > 0 and step <= 100]
        if len(available_steps) >= 24:
            indices = np.linspace(0, len(available_steps) - 1, 24, dtype=int)
            selected_steps = [available_steps[i] for i in indices]
        else:
            selected_steps = available_steps
        data = {step: data[step] for step in selected_steps if step in data}
        print(f"Testing mode: filtered to {len(data)} steps (step <= 100)")
    else:
        print(f"Loaded data for {len(data)} steps")
    
    # Create visualizations
    print(f"Creating visualizations in {vis_output_dir}")
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute axis ranges for consistent plotting
    axis_ranges = compute_axis_ranges(data)
    
    # Create GIFs with scatter plots
    frame_paths_fit = []
    for step in sorted(data.keys()):
        step_vis_dir = vis_output_dir / f"step_{step:06d}"
        step_vis_dir.mkdir(exist_ok=True)
        frame_paths_fit.append(create_frame_with_scatter(step, data[step], axis_ranges, step_vis_dir))
    
    print(f"Creating scatter GIF with {len(frame_paths_fit)} frames at {fps} fps...")
    create_gif_from_frames(frame_paths_fit, vis_output_dir / "std_mean_scatter.gif", fps)
    
    # Clean up intermediate frame files
    for frame_path in frame_paths_fit:
        Path(frame_path).unlink()
    for step in sorted(data.keys()):
        step_dir = vis_output_dir / f"step_{step:06d}"
        if step_dir.exists() and not any(step_dir.iterdir()):
            step_dir.rmdir()
    
    # Create stable rank plots
    print("Creating stable rank evolution plots...")
    create_stable_rank_plots(data, vis_output_dir)
    
    # Create SPC plots
    print("Creating SPC plots...")
    create_spc_plots(data, vis_output_dir)
    
    print(f"Visualization complete! Results saved to {vis_output_dir}")


if __name__ == "__main__":
    main()