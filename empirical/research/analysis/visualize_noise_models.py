#!/usr/bin/env python3
"""
Visualize noise models by fitting parameters to gradient distribution data and creating plots/GIFs.
This script reads CSV files from compute_gradient_distribution_over_minibatch_seeds.py and allows
fast iteration on noise model fitting and visualization.

Usage:
    python visualize_noise_models.py run_name [fps]
"""

import sys
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from math import erf
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from empirical.research.analysis.map import get_research_log_path

from empirical.research.training.zeropower import NEWTON_SCHULZ_QUINTIC_COEFFICIENTS

PARAM_TYPES = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'MLP Input', 'MLP Output']
# Note: we no longer use the SVHT constant; we estimate the bulk edge directly from the spectrum.


def newton_schulz_quintic_function(x):
    out = x
    for a, b, c in NEWTON_SCHULZ_QUINTIC_COEFFICIENTS:
        out = a * out + b * (out **3) + c * (out ** 5)
    return out


def _std_normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF using vectorized math.erf; stable for small arrays."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def _robust_log_edge_params(edges: np.ndarray) -> tuple[float, float]:
    """
    Robust fit of lognormal edge dispersion:
      mu = median(log(edges)), sigma = 1.4826 * MAD(log(edges))
    Returns (mu, sigma) with a small floor on sigma.
    """
    e = np.asarray(edges, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    if e.size == 0:
        return 0.0, 0.25  # fallback: neutral scale, mild dispersion
    x = np.log(e)
    mu = float(np.median(x))
    mad = float(np.median(np.abs(x - mu)))
    sigma = max(1.4826 * mad, 1e-6)
    return mu, sigma


def c_pred_mixture_lognormal(s: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Smooth prediction c(s) = E_tau[[1-(tau/s)^2]_+] for ln(tau)~N(mu, sigma^2).
    Closed form:
      Phi((ln s - mu)/sigma) - exp(2mu+2sigma^2)/s^2 * Phi((ln s - mu - 2sigma^2)/sigma)
    Returns values in [0,1].
    """
    s = np.asarray(s, dtype=float)
    sp = np.maximum(s, 1e-30)
    inv_sig = 1.0 / max(sigma, 1e-12)
    Phi1 = _std_normal_cdf((np.log(sp) - mu) * inv_sig)
    Phi2 = _std_normal_cdf((np.log(sp) - mu - 2.0 * sigma * sigma) * inv_sig)
    c = Phi1 - np.exp(2.0 * mu + 2.0 * sigma * sigma) * Phi2 / (sp * sp)
    return np.clip(c, 0.0, 1.0)


def _estimate_bulk_edge_from_spectrum(
    s: np.ndarray,
    *,
    beta: float = 1.0,
    lower_frac: float = 0.8,
    quantile: float = 0.99
) -> float:
    """
    Robust bulk-edge estimator (SVD-only, no model fit):
      1) sort singulars ascending
      2) keep the lower `lower_frac` mass (discard potential spikes)
      3) take the `quantile` as the bulk edge in s-units.
    For beta=1, the natural-unit edge is 2; here we use the edge in observed units.
    """
    if s.size == 0:
        return 0.0
    s_sorted = np.sort(np.asarray(s, dtype=float))
    k = max(1, int(np.floor(lower_frac * s_sorted.size)))
    s_bulk = s_sorted[:k]
    edge = float(np.quantile(s_bulk, quantile))
    if not np.isfinite(edge) or edge <= 0:
        edge = float(np.nanmax(s_sorted)) if s_sorted.size else 0.0
    return max(edge, 1e-12)


def c_pred_square(s: np.ndarray, edge: float | None = None) -> np.ndarray:
    """
    Predict c_i for the square case (beta=1) from observed singular values s.
    Normalization uses a robust bulk edge estimated from the spectrum (no SVHT constants):
      y = s * 2 / edge,  c = 0 if y<=2, else t=((y^2-2)+sqrt((y^2-2)^2-4))/2,  c=1-1/t.
    If `edge` is None, it is estimated from s via _estimate_bulk_edge_from_spectrum. Returns values in [0,1].
    """
    if s.size == 0:
        return s.copy()
    if edge is None or not np.isfinite(edge) or edge <= 0:
        edge = _estimate_bulk_edge_from_spectrum(s)
    if edge <= 0:
        return np.zeros_like(s, dtype=float)
    y = (np.asarray(s, dtype=float) * 2.0) / edge
    c = np.zeros_like(y)
    mask = y > 2.0
    if np.any(mask):
        z = y[mask]**2 - 2.0
        t = (z + np.sqrt(np.maximum(z*z - 4.0, 0.0))) / 2.0
        c[mask] = 1.0 - 1.0 / np.maximum(t, 1e-12)
    return np.clip(c, 0.0, 1.0)


def load_singular_values_data(sv_dir: Path) -> dict:
    """Load singular values distribution data from CSV files."""
    results = {}
    
    for csv_file in sv_dir.glob("step_*.csv"):
        step_match = re.search(r'step_(\d+)\.csv', csv_file.name)
        if not step_match:
            continue
        step = int(step_match.group(1))
        
        df = pd.read_csv(csv_file)
        step_data = {}
        
        for _, row in df.iterrows():
            key = (row['param_type'], int(row['layer_num']))
            layer_data = {
                'means': np.array(json.loads(row['means'])),
                'stds': np.array(json.loads(row['stds']))
            }
            
            # Load C estimate if available
            if 'c_with_mean_truth' in row and pd.notna(row['c_with_mean_truth']):
                layer_data['c_with_mean_truth'] = np.array(json.loads(row['c_with_mean_truth']))
                
            step_data[key] = layer_data
        
        results[step] = step_data
    
    return results


def load_stable_rank_data(sr_dir: Path) -> dict:
    """Load stable rank distribution data from CSV files."""
    results = {}
    
    for csv_file in sr_dir.glob("step_*.csv"):
        step_match = re.search(r'step_(\d+)\.csv', csv_file.name)
        if not step_match:
            continue
        step = int(step_match.group(1))
        
        df = pd.read_csv(csv_file)
        step_data = {}
        
        for _, row in df.iterrows():
            key = (row['param_type'], int(row['layer_num']))
            step_data[key] = {
                'weight_stable_rank': row['weight_stable_rank'],
                'gradient_stable_rank_mean': row['gradient_stable_rank_mean'],
                'gradient_stable_rank_std': row['gradient_stable_rank_std']
            }
        
        results[step] = step_data
    
    return results





def fit_noise_models_to_data(sv_data: dict, sr_data: dict) -> dict:
    """Fit noise model parameters (alpha, beta, tau) to the data."""
    results = {}
    
    for step in sv_data.keys():
        if step not in sr_data:
            continue
        
        step_data = {}
        
        # Merge sv and sr data for this step
        for key in sv_data[step].keys():
            if key in sr_data[step]:
                step_data[key] = {**sv_data[step][key], **sr_data[step][key]}
                # Add c_i prediction from mean singular values (square case, beta=1)
                means = step_data[key].get('means', None)
                if isinstance(means, np.ndarray) and means.size > 0:
                    edge = _estimate_bulk_edge_from_spectrum(means)  # robust bulk edge from that layer's spectrum
                    step_data[key]['c_pred_from_means'] = c_pred_square(means, edge=edge)
                else:
                    step_data[key]['c_pred_from_means'] = np.array([], dtype=float)

        # --- Mixture model (lognormal edge) per param_type at this step ---
        # 1) collect per-layer bulk edges per param_type
        edges_by_pt: dict[str, list[float]] = defaultdict(list)
        for (param_type, layer_num), layer_data in step_data.items():
            means = layer_data.get('means', None)
            if isinstance(means, np.ndarray) and means.size > 0:
                edges_by_pt[param_type].append(_estimate_bulk_edge_from_spectrum(means))
        # 2) fit (mu,sigma) robustly for each param_type
        ms_by_pt: dict[str, tuple[float, float]] = {
            pt: _robust_log_edge_params(np.array(e_list)) for pt, e_list in edges_by_pt.items()
        }
        # 3) produce mixture predictions for each layer of that param_type
        for (param_type, layer_num), layer_data in step_data.items():
            means = layer_data.get('means', None)
            if isinstance(means, np.ndarray) and means.size > 0 and param_type in ms_by_pt:
                mu, sigma = ms_by_pt[param_type]
                layer_data['mixture_mu'] = mu
                layer_data['mixture_sigma'] = sigma
                layer_data['c_pred_mixture_from_means'] = c_pred_mixture_lognormal(means, mu, sigma)
            else:
                layer_data['mixture_mu'] = float('nan')
                layer_data['mixture_sigma'] = float('nan')
                layer_data['c_pred_mixture_from_means'] = np.array([], dtype=float)
        
        
        
        results[step] = step_data
    
    return results


def save_fitted_noise_models(fitted_data: dict, output_dir: Path):
    """Save fitted noise model data to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for step, step_data in fitted_data.items():
        csv_data = []
        
        for (param_type, layer_num), layer_data in step_data.items():
            csv_data.append({
                'param_type': param_type,
                'layer_num': layer_num,
                'means': json.dumps(layer_data['means'].tolist()),
                'stds': json.dumps(layer_data['stds'].tolist()),
                'c_pred_from_means': json.dumps(layer_data.get('c_pred_from_means', np.array([], dtype=float)).tolist()),
                'c_pred_mixture_from_means': json.dumps(layer_data.get('c_pred_mixture_from_means', np.array([], dtype=float)).tolist()),
                'mixture_mu': layer_data.get('mixture_mu', float('nan')),
                'mixture_sigma': layer_data.get('mixture_sigma', float('nan')),
            })
        
        pd.DataFrame(csv_data).to_csv(output_dir / f"step_{step:06d}.csv", index=False)


def compute_axis_ranges(all_results: dict) -> dict:
    """Compute consistent axis ranges across all data."""
    param_types = PARAM_TYPES
    axis_ranges = {}
    for param_type in param_types:
        axis_ranges[param_type] = {'x_min': float('inf'), 'x_max': 0, 'y_min': float('inf'), 'y_max': 0}
    
    all_param_keys = set()
    for step_data in all_results.values():
        for (param_type, layer_num), layer_data in step_data.items():
            all_param_keys.add((param_type, layer_num))
            if param_type in param_types:
                means = layer_data['means']
                stds = layer_data['stds']
                axis_ranges[param_type]['x_min'] = min(axis_ranges[param_type]['x_min'], means.min())
                axis_ranges[param_type]['x_max'] = max(axis_ranges[param_type]['x_max'], means.max())
                axis_ranges[param_type]['y_min'] = min(axis_ranges[param_type]['y_min'], stds.min())
                axis_ranges[param_type]['y_max'] = max(axis_ranges[param_type]['y_max'], stds.max())
    
    # Add margins to axis ranges
    for param_type in param_types:
        x_range = axis_ranges[param_type]['x_max'] - axis_ranges[param_type]['x_min']
        y_range = axis_ranges[param_type]['y_max'] - axis_ranges[param_type]['y_min']
            
        # Ensure positive values for log scale and add margins
        x_min, x_max = axis_ranges[param_type]['x_min'], axis_ranges[param_type]['x_max']
        y_min, y_max = axis_ranges[param_type]['y_min'], axis_ranges[param_type]['y_max']
        
        # Ensure minimum positive values for log scale
        eps = 1e-10
        x_min = max(x_min, eps) if x_min > 0 else eps
        y_min = max(y_min, eps) if y_min > 0 else eps
        x_max = max(x_max, x_min * 10)
        y_max = max(y_max, y_min * 10)
        
        # Apply margins
        if x_min > 0 and x_max > x_min:
            x_margin = min(10.0, max(1.1, (x_max / x_min) ** 0.1))
        else:
            x_margin = 1.1
            
        if y_min > 0 and y_max > y_min:
            y_margin = min(10.0, max(1.1, (y_max / y_min) ** 0.1))
        else:
            y_margin = 1.1
            
        axis_ranges[param_type]['x_min'] = x_min / x_margin
        axis_ranges[param_type]['x_max'] = x_max * x_margin
        axis_ranges[param_type]['y_min'] = y_min / y_margin
        axis_ranges[param_type]['y_max'] = y_max * y_margin
    
    return axis_ranges


def create_gif_from_frames(frame_paths: list[str], gif_path: Path, fps: int):
    """Create GIF from list of frame file paths."""
    with imageio.get_writer(str(gif_path), mode='I', fps=fps, loop=0) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)


def create_subplot_grid(param_types: list, figsize: tuple, data_fn, plot_fn, title: str, output_path: Path):
    """Generic function to create subplot grids for different plot types."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    plt.subplots_adjust(left=0.06, right=0.85, top=0.93, bottom=0.07, wspace=0.25, hspace=0.35)
    
    viridis = plt.cm.viridis
    max_layers = 16
    
    for i, param_type in enumerate(param_types):
        ax = axes[i]
        plot_fn(ax, param_type, data_fn(param_type), viridis, max_layers)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=max_layers-1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Layer Number', rotation=270, labelpad=20)
    cbar.set_ticks(np.arange(0, max_layers, 2))
    
    plt.suptitle(title, fontsize=16)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_frame_with_scatter(step: int, param_data: dict, axis_ranges: dict, output_dir: Path):
    """Std vs Mean scatter plots without fitted curves."""
    param_types = PARAM_TYPES
    def get_frame_data(param_type):
        return [(layer_num, data) for (p_type, layer_num), data in param_data.items() if p_type == param_type and layer_num >= 0]
    def plot_std_mean_scatter(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Mean Singular Value'); ax.set_ylabel('Std Singular Value')
        ax.set_title(f'{param_type}')
        ax.grid(True, alpha=0.3)
        rng = axis_ranges[param_type]; ax.set_xlim(rng['x_min'], rng['x_max']); ax.set_ylim(rng['y_min'], rng['y_max'])
        for layer_num, data in layer_data_list:
            color = viridis(layer_num / (max_layers - 1))
            ax.scatter(data['means'], data['stds'], alpha=0.08, s=15, c=[color])
    frame_path = output_dir / f"frame_scatter_{step:06d}.png"
    create_subplot_grid(param_types, (20, 10), get_frame_data, plot_std_mean_scatter, f'Std–Mean Scatter - Step {step}', frame_path)
    return str(frame_path)


def create_stable_rank_plots(all_results: dict, output_dir: Path):
    """Create static PNG plots for gradient and weight stable rank over training."""
    param_types = PARAM_TYPES
    
    # Collect stable rank data across all steps
    gradient_data = defaultdict(lambda: defaultdict(list))  # param_type -> layer_num -> [values]
    weight_data = defaultdict(lambda: defaultdict(list))    # param_type -> layer_num -> [values]
    steps = sorted(all_results.keys())
    
    for step in steps:
        step_data = all_results[step]
        for (param_type, layer_num), layer_data in step_data.items():
            if param_type in param_types and 'gradient_stable_rank_mean' in layer_data:
                gradient_data[param_type][layer_num].append({
                    'step': step,
                    'mean': layer_data['gradient_stable_rank_mean'],
                    'std': layer_data['gradient_stable_rank_std']
                })
                weight_data[param_type][layer_num].append({
                    'step': step,
                    'value': layer_data['weight_stable_rank']
                })
    
    # Plot functions for different data types
    def get_gradient_data(param_type):
        return [(layer_num, gradient_data[param_type][layer_num]) for layer_num in sorted(gradient_data[param_type].keys())]
    
    def get_weight_data(param_type):
        return [(layer_num, weight_data[param_type][layer_num]) for layer_num in sorted(weight_data[param_type].keys())]
    
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
    
    # Create both plots using generic function
    create_subplot_grid(param_types, (18, 10), get_gradient_data, plot_gradient_stable_rank, 'Gradient Stable Rank Evolution', output_dir / "gradient_stable_rank.png")
    create_subplot_grid(param_types, (18, 10), get_weight_data, plot_weight_stable_rank, 'Weight Stable Rank Evolution', output_dir / "weight_stable_rank.png")


def create_c_estimates_plots(all_results: dict, output_dir: Path):
    """Create GIF showing C estimators vs mean singular values over training steps."""
    param_types = PARAM_TYPES
    
    # Check if C estimates are available
    has_c_estimates = False
    for step_data in all_results.values():
        for layer_data in step_data.values():
            if 'c_with_mean_truth' in layer_data:
                has_c_estimates = True
                break
        if has_c_estimates:
            break
    
    if not has_c_estimates:
        print("No C estimates found in data, skipping C estimates plots")
        return
    
    # Compute shared axis ranges for C estimates across all param types
    global_x_min = float('inf')
    global_x_max = 0
    global_c_min = float('inf') 
    global_c_max = 0
    
    for step_data in all_results.values():
        for (param_type, layer_num), layer_data in step_data.items():
            if param_type in param_types and 'c_with_mean_truth' in layer_data:
                means = layer_data['means']
                c_vals = layer_data['c_with_mean_truth']
                min_len = min(len(means), len(c_vals))
                if min_len > 0:
                    global_x_min = min(global_x_min, means[:min_len].min())
                    global_x_max = max(global_x_max, means[:min_len].max())
                    global_c_min = min(global_c_min, c_vals[:min_len].min())
                    global_c_max = max(global_c_max, c_vals[:min_len].max())
    
    # Set fixed x-axis bounds and add margins to c axis ranges
    global_x_min = 1e-6
    global_x_max = 1.0
    if global_c_min != float('inf'):
        c_margin = (global_c_max / global_c_min) ** 0.1
        global_c_min /= c_margin
        global_c_max *= c_margin
    
    # Create frames for each step
    frame_paths = []
    for step in sorted(all_results.keys()):
        def get_c_data_for_step(param_type):
            return [(layer_num, data) for (p_type, layer_num), data in all_results[step].items() 
                    if p_type == param_type and layer_num >= 0 and 'c_with_mean_truth' in data]
        
        def plot_c_estimates_frame(ax, param_type, layer_data_list, viridis, max_layers):
            ax.set_xlabel('Mean Singular Value')
            ax.set_ylabel('C (Mean Truth)')
            ax.set_title(f'{param_type} - C Estimates')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Set shared axis ranges
            ax.set_xlim(global_x_min, global_x_max)
            ax.set_ylim(global_c_min, global_c_max)
            
            for layer_num, data in layer_data_list:
                color = viridis(layer_num / (max_layers - 1))
                means = data['means']
                c_vals = data['c_with_mean_truth']
                min_len = min(len(means), len(c_vals))
                if min_len > 0:
                    ax.scatter(means[:min_len], c_vals[:min_len], alpha=0.16, s=15, c=[color])
        
        frame_path = output_dir / f"frame_c_estimates_{step:06d}.png"
        create_subplot_grid(param_types, (18, 10), get_c_data_for_step, plot_c_estimates_frame, 
                          f'C Estimates (Mean Truth) - Step {step}', frame_path)
        frame_paths.append(str(frame_path))
    
    # Create GIF with log y-axis
    print(f"Creating C estimates GIF with {len(frame_paths)} frames...")
    create_gif_from_frames(frame_paths, output_dir / "c_estimates_c_with_mean_truth.gif", 12)
    
    # Create frames for linear y-axis version
    frame_paths_linear = []
    for step in sorted(all_results.keys()):
        def get_c_data_for_step(param_type):
            return [(layer_num, data) for (p_type, layer_num), data in all_results[step].items() 
                    if p_type == param_type and layer_num >= 0 and 'c_with_mean_truth' in data]
        
        def plot_c_estimates_frame_linear(ax, param_type, layer_data_list, viridis, max_layers):
            ax.set_xlabel('Mean Singular Value')
            ax.set_ylabel('C (Mean Truth)')
            ax.set_title(f'{param_type} - C Estimates')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('linear')  # Linear y-axis
            
            # Set shared axis ranges
            ax.set_xlim(global_x_min, global_x_max)
            ax.set_ylim(0, 1)  # C estimates are bounded [0, 1]
            
            for layer_num, data in layer_data_list:
                color = viridis(layer_num / (max_layers - 1))
                means = data['means']
                c_vals = data['c_with_mean_truth']
                min_len = min(len(means), len(c_vals))
                if min_len > 0:
                    ax.scatter(means[:min_len], c_vals[:min_len], alpha=0.16, s=15, c=[color])
            
            # --- Predicted c(s) with lognormal edge mixture; semilog x, solid black ---
            # Fit (mu, sigma) from per-layer bulk edges of this param_type at this step
            edges = []
            for _, d in layer_data_list:
                m = d.get('means', None)
                if isinstance(m, np.ndarray) and m.size > 0:
                    edges.append(_estimate_bulk_edge_from_spectrum(m))
            if len(edges) > 0:
                mu, sigma = _robust_log_edge_params(np.array(edges))
                x_pred = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 400)
                c_curve = c_pred_mixture_lognormal(x_pred, mu, sigma)
                ax.plot(x_pred, c_curve, 'k-', linewidth=2, alpha=0.95, label='Predicted c (mixture)')

            # (Optional) keep the bulk-edge single-edge curve for comparison:
            # edge = _estimate_bulk_edge_from_spectrum(np.concatenate([d['means'] for _, d in layer_data_list]))
            # ax.plot(x_pred, c_pred_square(x_pred, edge=edge), 'k-.', lw=1.5, alpha=0.6, label='Predicted c (bulk-edge)')
            
            # Add Newton-Schulz quintics function curve
            x_ns = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 200)
            y_ns = newton_schulz_quintic_function(x_ns)
            ax.plot(x_ns, y_ns, 'k--', linewidth=2, alpha=0.8, label='Newton-Schulz Quintics')
            
            # Add legend for Newton-Schulz curve
            ax.legend(loc='upper left', fontsize=8)
        
        frame_path = output_dir / f"frame_c_estimates_linear_{step:06d}.png"
        create_subplot_grid(param_types, (18, 10), get_c_data_for_step, plot_c_estimates_frame_linear, 
                          f'C Estimates (Mean Truth, Linear Y) - Step {step}', frame_path)
        frame_paths_linear.append(str(frame_path))
    
    # Create GIF with linear y-axis
    print(f"Creating C estimates GIF with linear y-axis...")
    create_gif_from_frames(frame_paths_linear, output_dir / "c_estimates_c_with_mean_truth_linear.gif", 12)
    
    # Create frames for linear-linear version
    frame_paths_linear_linear = []
    for step in sorted(all_results.keys()):
        def get_c_data_for_step(param_type):
            return [(layer_num, data) for (p_type, layer_num), data in all_results[step].items() 
                    if p_type == param_type and layer_num >= 0 and 'c_with_mean_truth' in data]
        
        def plot_c_estimates_frame_linear_linear(ax, param_type, layer_data_list, viridis, max_layers):
            ax.set_xlabel('Mean Singular Value')
            ax.set_ylabel('C (Mean Truth)')
            ax.set_title(f'{param_type} - C Estimates')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('linear')  # Linear x-axis
            ax.set_yscale('linear')  # Linear y-axis
            
            # Set unit square bounds
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            for layer_num, data in layer_data_list:
                color = viridis(layer_num / (max_layers - 1))
                means = data['means']
                c_vals = data['c_with_mean_truth']
                min_len = min(len(means), len(c_vals))
                if min_len > 0:
                    ax.scatter(means[:min_len], c_vals[:min_len], alpha=0.16, s=15, c=[color])
        
        frame_path = output_dir / f"frame_c_estimates_linear_linear_{step:06d}.png"
        create_subplot_grid(param_types, (18, 10), get_c_data_for_step, plot_c_estimates_frame_linear_linear, 
                          f'C Estimates (Linear-Linear) - Step {step}', frame_path)
        frame_paths_linear_linear.append(str(frame_path))
    
    # Create GIF with linear-linear axes
    print(f"Creating C estimates GIF with linear-linear axes...")
    create_gif_from_frames(frame_paths_linear_linear, output_dir / "c_estimates_c_with_mean_truth_linear_linear.gif", 12)
    
    # Clean up frame files
    for frame_path in frame_paths:
        Path(frame_path).unlink()
    for frame_path in frame_paths_linear:
        Path(frame_path).unlink()
    for frame_path in frame_paths_linear_linear:
        Path(frame_path).unlink()


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python visualize_noise_models.py run_name [fps] [--testing]")
        sys.exit(1)
    
    run_name = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2] != "--testing" else 12
    testing = "--testing" in sys.argv
    
    # Setup directories
    sv_input_dir = get_research_log_path("singular_values_distribution", run_name, "")
    sr_input_dir = get_research_log_path("stable_rank_distribution", run_name, "")
    fitted_output_dir = get_research_log_path("fitted_noise_model", run_name, "")
    vis_output_dir = get_research_log_path("visualizations", run_name, "")
    
    print(f"Loading data from:")
    print(f"  - {sv_input_dir}")
    print(f"  - {sr_input_dir}")
    
    # Load data
    sv_data = load_singular_values_data(sv_input_dir)
    sr_data = load_stable_rank_data(sr_input_dir)
    
    if not sv_data:
        print(f"ERROR: No singular values data found in {sv_input_dir}")
        sys.exit(1)
    
    if not sr_data:
        print(f"ERROR: No stable rank data found in {sr_input_dir}")
        sys.exit(1)
    
    # Apply testing filter: use 24 checkpoints from step 100 or earlier
    if testing:
        available_steps = [step for step in sv_data.keys() if step > 0 and step <= 100]
        if len(available_steps) >= 24:
            indices = np.linspace(0, len(available_steps) - 1, 24, dtype=int)
            selected_steps = [available_steps[i] for i in indices]
        else:
            selected_steps = available_steps
        
        # Filter both datasets to only include selected steps
        sv_data = {step: sv_data[step] for step in selected_steps if step in sv_data}
        sr_data = {step: sr_data[step] for step in selected_steps if step in sr_data}
        
        print(f"Testing mode: filtered to {len(sv_data)} steps (step <= 100)")
    else:
        print(f"Loaded data for {len(sv_data)} steps")
    
    # Fit noise models
    print("Fitting noise models...")
    fitted_data = fit_noise_models_to_data(sv_data, sr_data)
    
    # Save fitted models
    print(f"Saving fitted noise models to {fitted_output_dir}")
    save_fitted_noise_models(fitted_data, fitted_output_dir)
    
    # Create visualizations
    print(f"Creating visualizations in {vis_output_dir}")
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute axis ranges for consistent plotting
    axis_ranges = compute_axis_ranges(fitted_data)
    
    # Create GIFs with scatter plots
    frame_paths_fit = []
    for step in sorted(fitted_data.keys()):
        step_vis_dir = vis_output_dir / f"step_{step:06d}"
        step_vis_dir.mkdir(exist_ok=True)
        frame_paths_fit.append(create_frame_with_scatter(step, fitted_data[step], axis_ranges, step_vis_dir))
    
    print(f"Creating GIF with {len(frame_paths_fit)} frames at {fps} fps...")
    create_gif_from_frames(frame_paths_fit, vis_output_dir / "std_mean_scatter.gif", fps)
    
    # Clean up intermediate frame files
    print("Cleaning up intermediate frame files...")
    for frame_path in frame_paths_fit:
        Path(frame_path).unlink()
    # Remove empty step directories
    for step in sorted(fitted_data.keys()):
        step_dir = vis_output_dir / f"step_{step:06d}"
        if step_dir.exists() and not any(step_dir.iterdir()):
            step_dir.rmdir()
    
    # Create stable rank plots
    print("Creating stable rank evolution plots...")
    create_stable_rank_plots(fitted_data, vis_output_dir)
    
    # Create C estimates plots
    print("Creating C estimates plots...")
    create_c_estimates_plots(fitted_data, vis_output_dir)
    
    print(f"Visualization complete! Results saved to {vis_output_dir}")


if __name__ == "__main__":
    main()