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
from empirical.research.training.zeropower import NEWTON_SCHULZ_QUINTIC_COEFFICIENTS
from empirical.research.analysis.wishart import (
    invert_spike_mapping_y2_to_t_np,
    predict_spectral_projection_coefficient_from_squared_true_signal,
)
from empirical.research.analysis.model_utilities import GPTLayerProperty
from empirical.research.analysis.wishart import predict_spectral_projection_coefficient_from_squared_true_signal


# Standard parameter types for consistent visualization
PARAM_TYPES = [
    'Attention Q', 'Attention K', 'Attention V', 
    'Attention O', 'MLP Input', 'MLP Output'
]


def create_subplot_grid(
    layer_property: GPTLayerProperty,
    plot_fn: Callable[[plt.Axes, GPTLayerProperty, str, mcolors.Colormap, int], List[Any]],
    title: str,
    figsize: Tuple[int, int] = (20, 10),
    layout: str = "constrained",
    num_layers: int = 16,
) -> plt.Figure:
    """
    Generic subplot grid creator for 6-panel visualizations.
    
    Args:
        TODO: better descriptions
        layer_property: GPTLayerProperty
        plot_fn: Callable,
        title: str,
        ...
    """
    if layout == "constrained":
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    fig.suptitle(title, fontsize=14)
    axes = axes.flatten()
    viridis = plt.cm.viridis

    # Precompute global max layer index for consistent colorbar mapping
    props_by_type: Dict[str, Dict[Tuple[str, int], Any]] = {pt: {} for pt in PARAM_TYPES}
    for (p_type, layer_num), arr in layer_property.items():
        if p_type in props_by_type:
            props_by_type[p_type][(p_type, layer_num)] = arr

    # Fixed layer count for coloring (GPT‑medium has 16 blocks: 0..15)
    global_max_layers = max(1, int(num_layers))

    for i, param_type in enumerate(PARAM_TYPES):
        ax = axes[i]
        prop = props_by_type[param_type]
        plot_fn(ax, prop, param_type, viridis, global_max_layers)
    
    if layout == "tight":
        plt.tight_layout()

    sm = cm.ScalarMappable(cmap=viridis, norm=mcolors.Normalize(vmin=0, vmax=global_max_layers - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.02, pad=0.02)
    cbar.set_label('Layer', rotation=270, labelpad=12)
    try:
        cbar.set_ticks(list(range(0, global_max_layers)))
        cbar.set_ticklabels([str(i) for i in range(global_max_layers)])
    except Exception:
        pass
    
    return fig

def newton_schulz_quintic_function(x):
    """Newton-Schulz quintic function for overlay."""
    out = x
    for a, b, c in NEWTON_SCHULZ_QUINTIC_COEFFICIENTS:
        out = a * out + b * (out **3) + c * (out ** 5)
    return out


def compute_panel_xs(panel: GPTLayerProperty, eps: float = 1e-8) -> np.ndarray:
    """Build a log-spaced x-grid from positive singulars across panel layers."""
    vals = []
    for (_pt, _layer), d in panel.items():
        sv = d.get('sv') if isinstance(d, dict) else None
        if sv is None:
            continue
        sv = np.asarray(sv, dtype=float)
        sv = sv[np.isfinite(sv) & (sv > 0)]
        if sv.size:
            vals.append(sv)
    if not vals:
        return np.geomspace(1e-6, 1.0, 256)
    all_sv = np.concatenate(vals)
    lo = max(float(all_sv.min()), eps)
    hi = float(all_sv.max())
    if hi <= lo:
        hi = lo * 10.0
    return np.geomspace(lo, hi, 256)


def predict_spc_curve_np(xs: np.ndarray, sigma: float, beta: float) -> np.ndarray:
    """Predict SPC(xs) from sigma and beta (numpy)."""
    xs = np.asarray(xs, dtype=np.float64)
    sigma = float(sigma)
    beta = float(beta)
    y2 = (xs / max(sigma, 1e-30)) ** 2
    t = invert_spike_mapping_y2_to_t_np(y2, beta)
    spc = predict_spectral_projection_coefficient_from_squared_true_signal(t, beta)
    return np.clip(spc, 0.0, 1.0)

def make_gif_from_layer_property_time_series(
    layer_property_time_series: dict[int, GPTLayerProperty],
    plot_fn: Callable[[plt.Axes, GPTLayerProperty, str, mcolors.Colormap, int], List[Any]],
    title: np.str_,
    output_dir: Path | None = None,
):
    """
    layer_property_time_series: maps from checkpoint idx to GPTLayerProperty
    """
    # 1) Create figures for each checkpoint
    frames = {
        step: create_subplot_grid(layer_property=lp,
                                   plot_fn=plot_fn,
                                   title=f"{title}, Checkpoint {step}")
        for step, lp in layer_property_time_series.items()
    }

    # 2) Measure per-subplot bounds across all frames (first infer scales from first frame)
    for fig in frames.values():
        fig.canvas.draw()
    xmins, xmaxs, ymins, ymaxs = get_global_axis_bounds(list(frames.values()))

    # 3) Apply global bounds to each frame and save PNGs
    out_dir = Path("research_logs/visualizations") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_paths: List[str] = []
    for step, fig in frames.items():
        for i, ax in enumerate(fig.axes):
            ax.set_xlim(xmins[i], xmaxs[i])
            ax.set_ylim(ymins[i], ymaxs[i])
        png = out_dir / f"{str(title).replace(' ', '_')}_{step:06d}.png"
        fig.savefig(png, dpi=150, bbox_inches='tight')
        png_paths.append(str(png))

    # 4) Write GIF
    images = [imageio.imread(p) for p in png_paths]
    gif_path = out_dir / f"{str(title).replace(' ', '_')}.gif"
    imageio.mimsave(gif_path, images, fps=12, loop=0)
    for p in png_paths:
        Path(p).unlink(missing_ok=True)
    return gif_path


def get_global_axis_bounds(frames: List[plt.Figure]):
    """Aggregate per-subplot global (xmin,xmax,ymin,ymax) across frames.

    Infers log/linear scales from the first frame's axes and enforces
    log-safe minima for log-scaled axes.
    """
    assert frames, "No frames provided"
    first_axes = frames[0].axes
    x_is_log = [ax.get_xscale() == 'log' for ax in first_axes]
    y_is_log = [ax.get_yscale() == 'log' for ax in first_axes]
    n = len(first_axes)
    eps = 1e-8
    xmins = [float('inf')] * n; xmaxs = [float('-inf')] * n
    ymins = [float('inf')] * n; ymaxs = [float('-inf')] * n
    for fig in frames:
        for i, ax in enumerate(fig.axes):
            bb = ax.dataLim
            xmins[i] = min(xmins[i], bb.xmin)
            xmaxs[i] = max(xmaxs[i], bb.xmax)
            ymins[i] = min(ymins[i], bb.ymin)
            ymaxs[i] = max(ymaxs[i], bb.ymax)
    for i in range(n):
        if x_is_log[i]:
            xmins[i] = max(xmins[i], eps)
        if y_is_log[i]:
            ymins[i] = max(ymins[i], eps)
    return xmins, xmaxs, ymins, ymaxs
