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
import imageio.v2 as imageio
import pandas as pd
import json
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator

from .core_math import mp_pdf_singular_numpy, matrix_shape_beta


# Standard parameter types for consistent visualization
PARAM_TYPES = [
    'Attention Q', 'Attention K', 'Attention V', 
    'Attention O', 'MLP Input', 'MLP Output'
]


def create_gif_from_frames(frame_paths: List[str], gif_path: Path, fps: int = 12):
    """Create looping GIF from frame files."""
    if not frame_paths:
        print(f"Warning: No frames to create GIF {gif_path}")
        return
        
    images = []
    for fp in frame_paths:
        try:
            images.append(imageio.imread(fp))
        except Exception as e:
            print(f"Warning: Failed to read frame {fp}: {e}")
            continue
    
    if images:
        imageio.mimsave(gif_path, images, fps=fps, loop=0)
        print(f"GIF saved: {gif_path}")
    else:
        print(f"Warning: No valid images found for {gif_path}")


def create_subplot_grid(
    param_types: List[str], 
    figsize: Tuple[int, int],
    get_data_fn: Callable[[str], List[Tuple[int, Dict]]],
    plot_fn: Callable,
    title: str,
    output_path: Path,
    layout: str = "constrained"
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
    
    for i, param_type in enumerate(param_types):
        ax = axes[i]
        layer_data_list = get_data_fn(param_type)
        max_layers = max([layer_num for layer_num, _ in layer_data_list], default=1) + 1
        plot_fn(ax, param_type, layer_data_list, viridis, max_layers)
    
    if layout == "tight":
        plt.tight_layout()
    
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

    # Collect downsampled singular values from layers
    samples, betas, sigmas = [], [], []
    for _, data in layer_data_list:
        s = np.asarray(data.get("innovation_sample", []), dtype=float)
        if s.size:
            # Only positive for log scale
            s = s[s > 0]
            if s.size:
                samples.append(np.ascontiguousarray(s))
                betas.append(float(data.get("beta", 1.0)))
                sigmas.append(float(data.get("sigma_hat", 1e-8)))
    
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
    ax.hist(all_innov, bins=bins, density=True, alpha=0.35, label="Empirical spectrum")

    # MP density overlay
    xs = np.geomspace(x_min, x_max, 800)
    mp = mp_pdf_singular_numpy(xs, beta, sigma_hat)
    ax.plot(xs, mp, ls='--', lw=2, color='tab:orange', label=f"MP (σ̂={sigma_hat:.3f})")
    ax.axvline(max(edge, x_min), color='k', lw=1.0, ls='--', label="Edge τ̂")
    ax.legend(loc='upper right', fontsize=8, frameon=False)


def plot_spc_vs_singular_values(ax, param_type: str, layer_data_list: List, viridis, max_layers: int):
    """Plot spectral projection coefficients vs singular values."""
    ax.set_title(f'{param_type}')
    ax.set_xlabel('Singular value')
    ax.set_ylabel('Spectral projection coefficient')
    ax.set_xlim(1e-6, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    denom = max(1, max_layers - 1)
    
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
            
            ax.scatter(sv_flat, spc_flat, alpha=0.6, s=1, c=[color],
                      label=f'L{layer_num}' if layer_num <= 3 else None)
    
    ax.legend(loc='upper right', fontsize=8)


def plot_predicted_vs_actual_spc(ax, param_type: str, layer_data_list: List, viridis, max_layers: int):
    """Plot predicted vs actual spectral projection coefficients."""
    ax.set_title(f'{param_type}')
    ax.set_xlabel('Actual SPC')
    ax.set_ylabel('Predicted SPC')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect prediction')
    ax.grid(True, alpha=0.3)
    
    denom = max(1, max_layers - 1)
    
    for layer_num, data in layer_data_list:
        if 'spectral_projection_coefficients' not in data:
            continue
            
        color = viridis(layer_num / denom)
        spc = np.asarray(data['spectral_projection_coefficients']).flatten()
        
        if spc.size > 0:
            # For now use identity - in practice you'd have predicted vs actual
            pred = spc  # Replace with actual prediction
            act = spc   # Replace with actual ground truth
            
            # Subsample for visualization
            if len(act) > 500:
                indices = np.random.choice(len(act), 500, replace=False)
                act = act[indices]
                pred = pred[indices]
            
            ax.scatter(act, pred, alpha=0.3, s=10, c=[color],
                      label=f'L{layer_num}' if layer_num <= 2 else None)
    
    ax.legend(loc='upper left', fontsize=8)


def plot_stable_rank_evolution(ax, param_type: str, layer_data_list: List, viridis, max_layers: int, 
                               data_key: str = 'stable_rank'):
    """Plot stable rank evolution over layers."""
    ax.set_title(f'{param_type}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Stable Rank')
    ax.grid(True, alpha=0.3)
    
    layers = []
    ranks = []
    
    for layer_num, data in layer_data_list:
        if data_key in data:
            layers.append(layer_num)
            ranks.append(float(data[data_key]))
    
    if layers:
        ax.plot(layers, ranks, 'o-', alpha=0.7, markersize=4)
        ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)


def plot_bulk_spike_lines(ax, param_type: str, layer_data_list: List, viridis, max_layers: int, 
                         shapes: Dict[str, Tuple[int, int]]):
    """
    Plot bulk vs spike with per-layer lines (no bars).
    Solid lines = empirical density, dashed lines = fitted MP bulk.
    """
    ax.set_title(f'{param_type}')
    setup_log_scale_axis(ax)

    # Get shape for this parameter type
    shape = shapes.get(param_type, (1024, 1024))
    beta = matrix_shape_beta(shape)
    
    denom = max(1, max_layers - 1)
    
    for layer_num, data in layer_data_list:
        if 'per_minibatch_singular_values' not in data:
            continue
            
        color = viridis(layer_num / denom)
        sv = np.asarray(data['per_minibatch_singular_values'], dtype=np.float64)
        sv = sv[sv > 0]  # Only positive values
        
        if sv.size == 0:
            continue
            
        # Create bins for this layer's data
        x_min = max(sv.min(), 1e-12)
        x_max = sv.max() * 1.05
        if x_max <= x_min:
            continue
            
        bins = np.geomspace(x_min, x_max, 64)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)
        
        # Empirical density
        counts, _ = np.histogram(sv, bins=bins)
        density = counts.astype(np.float64) / (sv.size * widths)
        
        # MP fit
        from .core_math import compute_stable_rank
        s_desc = np.sort(sv)[::-1]
        # Simple sigma estimation - in practice use more sophisticated method
        sigma_hat = np.std(sv) if len(sv) > 1 else sv.mean()
        
        mp = mp_pdf_singular_numpy(centers, beta, sigma_hat)
        
        # Plot both lines
        ax.plot(centers, density, color=color, lw=1.6, alpha=0.9, 
                label=f'L{layer_num}' if layer_num <= 3 else None)
        ax.plot(centers, mp, color=color, lw=1.6, ls='--', alpha=0.9)
    
    if max_layers <= 4:
        ax.legend(loc='upper right', fontsize=8)


def create_visualization_frames(
    step: int,
    viz_stats: Dict[Tuple[str, int], Dict[str, Any]], 
    gif_frames: Dict[str, List[str]],
    output_dir: Path,
    rank: int = 0
):
    """
    Create all standard visualization frames for a given step.
    
    Args:
        step: Training step number
        viz_stats: Visualization statistics by (param_type, layer_num)
        gif_frames: Dictionary to accumulate frame paths
        output_dir: Base output directory
        rank: Process rank (only rank 0 creates visualizations)
    """
    if rank != 0:
        return
    
    # Create frames directory
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    def get_layer_data_for_param_type(param_type: str):
        """Get data for this parameter type."""
        items = []
        for (p_type, layer_num), d in viz_stats.items():
            if p_type == param_type and layer_num >= 0:
                items.append((layer_num, d))
        return sorted(items, key=lambda x: x[0])
    
    # Define frame types and their plot functions
    frame_configs = {
        'bulk_spike': {
            'plot_fn': plot_bulk_vs_spike,
            'title': f'Bulk vs Spike Estimation - Step {step}',
            'filename': f"bulk_spike_{step:06d}.png"
        },
        'spc_singular': {
            'plot_fn': plot_spc_vs_singular_values,
            'title': f'SPC vs Singular Values - Step {step}',
            'filename': f"spc_singular_{step:06d}.png"
        },
        'pred_actual': {
            'plot_fn': plot_predicted_vs_actual_spc,
            'title': f'Predicted vs Actual SPC - Step {step}',
            'filename': f"pred_actual_{step:06d}.png"
        }
    }
    
    # Create each frame type
    for frame_type, config in frame_configs.items():
        frame_path = frames_dir / config['filename']
        
        create_subplot_grid(
            PARAM_TYPES, (20, 10), 
            get_layer_data_for_param_type,
            config['plot_fn'],
            config['title'],
            frame_path
        )
        
        gif_frames[frame_type].append(str(frame_path))
    
    print(f"Rank {rank}: Created visualization frames for step {step}")


def finalize_gifs(
    gif_frames: Dict[str, List[str]], 
    output_dir: Path,
    gif_configs: Optional[Dict[str, str]] = None,
    rank: int = 0,
    fps: int = 12
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
            'spc_singular': 'spc_vs_singular_values.gif', 
            'pred_actual': 'predicted_vs_actual_spc.gif'
        }
    
    for frame_type, frames in gif_frames.items():
        if not frames:
            continue
            
        gif_name = gif_configs.get(frame_type, f"{frame_type}.gif")
        gif_path = output_dir / gif_name
        
        create_gif_from_frames(frames, gif_path, fps=fps)
        
        # Clean up individual frames
        for frame_path in frames:
            try:
                Path(frame_path).unlink()
            except FileNotFoundError:
                pass
        
        print(f"Created {gif_path}")
    
    # Clean up empty frames directory
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists() and not any(frames_dir.iterdir()):
            frames_dir.rmdir()
    except OSError:
        pass


# Additional visualization functions consolidated from visualize_noise_models.py

def newton_schulz_quintic_function(x):
    """Newton-Schulz quintic function for overlay."""
    from empirical.research.training.zeropower import NEWTON_SCHULZ_QUINTIC_COEFFICIENTS
    out = x
    for a, b, c in NEWTON_SCHULZ_QUINTIC_COEFFICIENTS:
        out = a * out + b * (out **3) + c * (out ** 5)
    return out


def load_noise_model_data(sv_dir: Path) -> dict:
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


def compute_visualization_axis_ranges(data):
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
        
        if len(x_vals) > 0 and len(y_vals) > 0:
            x_min, x_max = np.percentile(x_vals, [1, 99])
            y_min, y_max = np.percentile(y_vals, [1, 99])
            x_range = (x_min * 0.8, x_max * 1.2)
            y_range = (y_min * 0.8, y_max * 1.2)
        else:
            x_range = (1e-3, 1e1)
            y_range = (1e-3, 1e1)
        
        ranges[param_type] = {'x': x_range, 'y': y_range}
    
    return ranges