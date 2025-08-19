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
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from empirical.research.analysis.map import get_research_log_path
from scipy.optimize import curve_fit

PARAM_TYPES = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'MLP Input', 'MLP Output']


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


def _ls_fit_alpha_tau(x, y, iters: int = 60):
    """
    Log-space Gauss–Newton fit for:
        Y = 0.5 * log( alpha^2 * e^{2X} + tau^2 ),  with X=log(x), Y=log(y).
    Optimizes u=log(alpha), v=log(tau) to minimize sum (Y - r(u,v;X))^2.
    Returns (alpha_hat, tau_hat).
    """
    x = np.asarray(x); y = np.asarray(y)
    m = (x > 0) & (y > 0)
    x = x[m]; y = y[m]
    if x.size < 2:
        return 0.0, 0.0
    # Transform to log space (clip away denorms)
    X = np.log(np.clip(x, 1e-300, None))
    Y = np.log(np.clip(y, 1e-300, None))
    n = X.size

    # Heuristic initialization: alpha from high-x, tau from low-x
    q_hi, q_lo = 0.7, 0.3
    thr_hi = np.quantile(x, q_hi)
    thr_lo = np.quantile(x, q_lo)
    hi = x >= thr_hi
    lo = x <= thr_lo
    alpha0 = np.median(y[hi] / x[hi]) if hi.sum() >= 2 else (y / x).mean()
    tau0   = np.median(y[lo])         if lo.sum() >= 2 else y.min()
    alpha0 = float(max(alpha0, 1e-12))
    tau0   = float(max(tau0,   1e-12))
    u = np.log(alpha0); v = np.log(tau0)

    # Gauss–Newton updates in (u,v). For r = 0.5*logaddexp(A,B) with A=2u+2X, B=2v:
    # dr/du = softmax(A,B),  dr/dv = softmax(B,A).
    for _ in range(iters):
        A = 2.0*u + 2.0*X
        B = 2.0*v + 0.0*X
        r = 0.5 * np.logaddexp(A, B)
        resid = Y - r
        # softmax weights; numerically stable
        wA = 1.0 / (1.0 + np.exp(B - A))  # contribution of the alpha*e^{X} branch
        wB = 1.0 - wA                     # contribution of the tau branch
        # Gradients of sum(resid^2)
        gu = -2.0 * np.sum(resid * wA)
        gv = -2.0 * np.sum(resid * wB)
        # Gauss–Newton diagonal Hessian approx (sum of jac^2)
        Hu = 2.0 * np.sum(wA**2) + 1e-12
        Hv = 2.0 * np.sum(wB**2) + 1e-12
        # Update
        u -= gu / Hu
        v -= gv / Hv
    alpha = float(np.exp(u))
    tau   = float(np.exp(v))
    return alpha, tau


def _local_gaps_from_singulars(s):
    """
    Given a 1D array of descending singular means s, return local gaps g of same length:
      g[0]=s0-s1, g[-1]=s_{n-2}-s_{n-1}, interior = min(forward/backward diffs)
    """
    s = np.asarray(s)
    if s.size <= 1:
        return np.zeros_like(s)
    d = s[:-1] - s[1:]
    g = np.empty_like(s)
    g[0] = d[0]
    g[1:-1] = np.minimum(d[:-1], d[1:])
    g[-1] = d[-1]
    return np.clip(g, 0.0, None)


def _fit_alpha_beta(x, y, gap, iters: int = 60):
    """
    Log-space Gauss–Newton fit (pooled per param_type, per frame):
        log(y^2) ≈ logaddexp( 2*log(alpha) + 2*log(x),
                              2*log(beta)  - 2*log(gap) ).
    Optimizes u=log(alpha), v=log(beta) to minimize sum (Z - r(u,v))^2,
    where Z=2*log(y) and r=logaddexp(2u + 2log(x), 2v - 2log(gap)).
    Returns (alpha_hat, beta_hat).
    """
    x = np.asarray(x); y = np.asarray(y); gap = np.asarray(gap)
    m = (x > 0) & (y > 0) & (gap > 0)
    x = x[m]; y = y[m]; gap = gap[m]
    if x.size < 2:
        return 0.0, 0.0
    # Stable logs
    LX = 2.0 * np.log(np.clip(x,   1e-300, None))   # log(x^2)
    LV = -2.0 * np.log(np.clip(gap, 1e-300, None))  # log(1/gap^2)
    Z  = 2.0 * np.log(np.clip(y,   1e-300, None))   # log(y^2)
    # Heuristic init: α≈median(y/x) on high-x; β≈median(y*gap) on low-x
    q_hi, q_lo = 0.7, 0.3
    thr_hi = np.quantile(x, q_hi);  thr_lo = np.quantile(x, q_lo)
    hi = x >= thr_hi;                lo = x <= thr_lo
    alpha0 = np.median(y[hi] / x[hi]) if hi.sum() >= 2 else (y / x).mean()
    beta0  = np.median(y[lo] * gap[lo]) if lo.sum() >= 2 else (y * gap).mean()
    alpha0 = float(max(alpha0, 1e-12));  beta0 = float(max(beta0, 1e-12))
    u = np.log(alpha0);  v = np.log(beta0)
    # Gauss–Newton in (u,v)
    for _ in range(iters):
        A = 2.0*u + LX                  # α-branch
        B = 2.0*v + LV                  # β/gap branch
        r = np.logaddexp(A, B)          # model prediction: log(y^2)
        resid = Z - r
        # soft assignments per point
        wA = 1.0 / (1.0 + np.exp(B - A))
        wB = 1.0 - wA
        # dr/du, dr/dv
        drdu = 2.0 * wA
        drdv = 2.0 * wB
        # gradients & GN Hessians of sum(resid^2)
        gu = -2.0 * np.sum(resid * drdu)
        gv = -2.0 * np.sum(resid * drdv)
        Hu = 2.0 * np.sum(drdu**2) + 1e-12
        Hv = 2.0 * np.sum(drdv**2) + 1e-12
        u -= gu / Hu
        v -= gv / Hv
    return float(np.exp(u)), float(np.exp(v))


def sigmoid(ln_s, a, b):
    """Sigmoid function: C = 1 / (1 + exp(-a * (ln_s - b)))"""
    return 1.0 / (1.0 + np.exp(-a * (ln_s - b)))


def fit_sigmoid_to_c_estimates(means, c_vals):
    """Fit sigmoid to C estimates vs ln(mean singular values).
    
    Args:
        means: Mean singular values
        c_vals: C estimates
        
    Returns:
        (a, b): Sigmoid parameters, or (None, None) if fitting fails
    """
    try:
        # Filter out invalid values
        valid_mask = (means > 0) & (c_vals > 0) & (c_vals < 1) & np.isfinite(means) & np.isfinite(c_vals)
        if np.sum(valid_mask) < 3:  # Need at least 3 points
            return None, None
            
        means_valid = means[valid_mask]
        c_vals_valid = c_vals[valid_mask]
        
        # Convert to log space
        ln_s = np.log(means_valid)
        
        # Initial guess: b around ln(1e-3) = -6.9, a positive to give sigmoid shape
        # C=0.5 happens around ln(s)=1e-3, so b should be around ln(1e-3) = -6.9
        initial_b = np.log(1e-3)
        initial_a = 1.0  # Positive slope
        
        # Fit sigmoid
        popt, _ = curve_fit(sigmoid, ln_s, c_vals_valid, 
                          p0=[initial_a, initial_b],
                          bounds=([-10, -20], [10, 5]),  # Reasonable bounds
                          maxfev=1000)
        
        return float(popt[0]), float(popt[1])
        
    except Exception as e:
        return None, None


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
        
        # Fit alpha, beta per param_type (pooled across layers)
        for ptype in PARAM_TYPES:
            xs, ys, gs = [], [], []
            keys = [(pt, ln) for (pt, ln) in step_data.keys() if pt == ptype]
            for key in keys:
                m = step_data[key]['means']
                s = step_data[key]['stds']
                g = _local_gaps_from_singulars(m)
                xs.append(m); ys.append(s); gs.append(g)
            if not xs:
                continue
            x_all = np.concatenate(xs); y_all = np.concatenate(ys); g_all = np.concatenate(gs)
            alpha_hat, beta_hat = _fit_alpha_beta(x_all, y_all, g_all)
            for key in keys:
                step_data[key]['alpha'] = alpha_hat
                step_data[key]['beta'] = beta_hat
                # legacy tau derived from typical gap (for backward compat only)
                g_typ = float(np.median(_local_gaps_from_singulars(step_data[key]['means'])))
                step_data[key]['tau'] = (beta_hat / g_typ) if g_typ > 0 else None
        
        # Fit sigmoid parameters per layer for C estimates
        for key in step_data.keys():
            if 'c_with_mean_truth' in step_data[key]:
                means = step_data[key]['means']
                c_vals = step_data[key]['c_with_mean_truth']
                sigmoid_a, sigmoid_b = fit_sigmoid_to_c_estimates(means, c_vals)
                step_data[key]['sigmoid_a'] = sigmoid_a
                step_data[key]['sigmoid_b'] = sigmoid_b
        
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
                'tau': layer_data.get('tau', None),
                'alpha': layer_data.get('alpha', None),
                'beta': layer_data.get('beta', None),
                'sigmoid_a': layer_data.get('sigmoid_a', None),
                'sigmoid_b': layer_data.get('sigmoid_b', None),
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


def create_frame_with_fit(step: int, param_data: dict, axis_ranges: dict, output_dir: Path):
    """Std vs Mean with fitted envelope using alpha & beta: y ≈ sqrt((alpha*x)^2 + (beta/g_typ)^2)."""
    param_types = PARAM_TYPES
    def get_frame_data(param_type):
        return [(layer_num, data) for (p_type, layer_num), data in param_data.items() if p_type == param_type and layer_num >= 0]
    def plot_std_mean_fit(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Mean Singular Value'); ax.set_ylabel('Std Singular Value')
        ax.set_title(f'{param_type} (fit)')
        ax.grid(True, alpha=0.3)
        rng = axis_ranges[param_type]; ax.set_xlim(rng['x_min'], rng['x_max']); ax.set_ylim(rng['y_min'], rng['y_max'])
        alpha_hat = None; beta_hat = None
        g_all = []
        for layer_num, data in layer_data_list:
            color = viridis(layer_num / (max_layers - 1))
            ax.scatter(data['means'], data['stds'], alpha=0.08, s=15, c=[color])
            alpha_hat = data.get('alpha', alpha_hat)
            beta_hat  = data.get('beta',  beta_hat)
            g_all.append(_local_gaps_from_singulars(np.asarray(data['means'])))
        g_all = np.concatenate(g_all) if len(g_all)>0 else np.array([])
        g_typ = float(np.median(g_all[g_all>0])) if g_all.size>0 and np.any(g_all>0) else None
        if alpha_hat is not None and beta_hat is not None and rng['x_min'] > 0 and g_typ:
            xfit = np.logspace(np.log10(rng['x_min']), np.log10(rng['x_max']), 200)
            yfit = np.sqrt((alpha_hat * xfit)**2 + (beta_hat / g_typ)**2)
            ax.plot(xfit, yfit, linewidth=1.5)
        if alpha_hat is not None and beta_hat is not None:
            ax.text(0.02, 0.98, f"α={alpha_hat:.2e}\\nβ={beta_hat:.2e}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9)
    frame_path = output_dir / f"frame_fit_{step:06d}.png"
    create_subplot_grid(param_types, (20, 10), get_frame_data, plot_std_mean_fit, f'Std–Mean with Fit - Step {step}', frame_path)
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
            
            # Collect sigmoid parameters for averaging
            valid_a_vals = []
            valid_b_vals = []
            
            for layer_num, data in layer_data_list:
                color = viridis(layer_num / (max_layers - 1))
                means = data['means']
                c_vals = data['c_with_mean_truth']
                min_len = min(len(means), len(c_vals))
                if min_len > 0:
                    ax.scatter(means[:min_len], c_vals[:min_len], alpha=0.16, s=15, c=[color])
                
                # Plot fitted sigmoid if available
                if 'sigmoid_a' in data and 'sigmoid_b' in data and data['sigmoid_a'] is not None and data['sigmoid_b'] is not None:
                    # Create smooth curve from global x bounds
                    x_curve = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 200)
                    ln_s_curve = np.log(x_curve)
                    y_curve = sigmoid(ln_s_curve, data['sigmoid_a'], data['sigmoid_b'])
                    ax.plot(x_curve, y_curve, color=color, linewidth=1.5, alpha=0.8)
                    
                    valid_a_vals.append(data['sigmoid_a'])
                    valid_b_vals.append(data['sigmoid_b'])
            
            # Add equation text with average parameters
            if valid_a_vals and valid_b_vals:
                avg_a = np.mean(valid_a_vals)
                avg_b = np.mean(valid_b_vals)
                equation_text = f'$C = \\frac{{1}}{{1 + e^{{-{avg_a:.1f}(\\ln s - {avg_b:.1f})}}}}$'
                ax.text(0.95, 0.05, equation_text, transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
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
            
            # Collect sigmoid parameters for averaging
            valid_a_vals = []
            valid_b_vals = []
            
            for layer_num, data in layer_data_list:
                color = viridis(layer_num / (max_layers - 1))
                means = data['means']
                c_vals = data['c_with_mean_truth']
                min_len = min(len(means), len(c_vals))
                if min_len > 0:
                    ax.scatter(means[:min_len], c_vals[:min_len], alpha=0.16, s=15, c=[color])
                
                # Plot fitted sigmoid if available
                if 'sigmoid_a' in data and 'sigmoid_b' in data and data['sigmoid_a'] is not None and data['sigmoid_b'] is not None:
                    # Create smooth curve from global x bounds
                    x_curve = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 200)
                    ln_s_curve = np.log(x_curve)
                    y_curve = sigmoid(ln_s_curve, data['sigmoid_a'], data['sigmoid_b'])
                    ax.plot(x_curve, y_curve, color=color, linewidth=1.5, alpha=0.8)
                    
                    valid_a_vals.append(data['sigmoid_a'])
                    valid_b_vals.append(data['sigmoid_b'])
            
            # Add equation text with average parameters
            if valid_a_vals and valid_b_vals:
                avg_a = np.mean(valid_a_vals)
                avg_b = np.mean(valid_b_vals)
                equation_text = f'$C = \\frac{{1}}{{1 + e^{{-{avg_a:.1f}(\\ln s - {avg_b:.1f})}}}}$'
                ax.text(0.95, 0.05, equation_text, transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
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
            
            # Collect sigmoid parameters for averaging
            valid_a_vals = []
            valid_b_vals = []
            
            for layer_num, data in layer_data_list:
                color = viridis(layer_num / (max_layers - 1))
                means = data['means']
                c_vals = data['c_with_mean_truth']
                min_len = min(len(means), len(c_vals))
                if min_len > 0:
                    ax.scatter(means[:min_len], c_vals[:min_len], alpha=0.16, s=15, c=[color])
                
                # Plot fitted curve if available - in linear space this is just C vs s directly
                if 'sigmoid_a' in data and 'sigmoid_b' in data and data['sigmoid_a'] is not None and data['sigmoid_b'] is not None:
                    # Create curve from 0 to 1
                    s_curve = np.linspace(0.001, 1.0, 200)  # Start slightly above 0 for log
                    ln_s_curve = np.log(s_curve)
                    c_curve = sigmoid(ln_s_curve, data['sigmoid_a'], data['sigmoid_b'])
                    ax.plot(s_curve, c_curve, color=color, linewidth=1.5, alpha=0.8)
                    
                    valid_a_vals.append(data['sigmoid_a'])
                    valid_b_vals.append(data['sigmoid_b'])
            
            # Add equation text with average parameters in simplified power-law form
            if valid_a_vals and valid_b_vals:
                avg_a = np.mean(valid_a_vals)
                avg_b = np.mean(valid_b_vals)
                # Convert to power-law form: C = s^a / (s^a + exp(-a*b))
                k = np.exp(-avg_a * avg_b)
                equation_text = f'$C = \\frac{{s^{{{avg_a:.1f}}}}}{{s^{{{avg_a:.1f}}} + {k:.3f}}}$'
                ax.text(0.95, 0.05, equation_text, transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
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
    
    # Create GIFs with fitted curves
    frame_paths_fit = []
    for step in sorted(fitted_data.keys()):
        step_vis_dir = vis_output_dir / f"step_{step:06d}"
        step_vis_dir.mkdir(exist_ok=True)
        frame_paths_fit.append(create_frame_with_fit(step, fitted_data[step], axis_ranges, step_vis_dir))
    
    print(f"Creating GIF with {len(frame_paths_fit)} frames at {fps} fps...")
    create_gif_from_frames(frame_paths_fit, vis_output_dir / "std_mean_with_fit.gif", fps)
    
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