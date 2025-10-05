#!/usr/bin/env python3
"""
Build and plot sigma time series from serialized checkpoints.

Reads checkpoints from research_logs/checkpoints/medium_20251004 and
creates a GPTLayerProperty mapping (param_type, layer) -> Dict[int, float],
where the dict maps checkpoint step index -> sigma for that layer. This allows
unevenly spaced checkpoints to render with the actual step on the x-axis.

Then renders a 2x3 grid (Attention Q/K/V/O, MLP Input/Output) with 16 lines
per panel (one per GPT block) showing sigma vs. training step.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, List
import re
import torch
import numpy as np

from .logging_utilities import deserialize_model_checkpoint, categorize_parameter
from .core_visualization import create_subplot_grid, PARAM_TYPES
import matplotlib.pyplot as plt


ParamKey = Tuple[str, int]
GPTLayerProperty = Dict[ParamKey, Any]


def _list_checkpoints(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    files = sorted(dir_path.glob('step_*.pt'), key=lambda p: int(re.findall(r'(\d+)', p.stem)[0]))
    return files


def build_sigma_time_series(checkpoint_root: Path) -> GPTLayerProperty:
    """Aggregate sigma values across checkpoints into time series per layer/key."""
    series: Dict[ParamKey, Dict[int, float]] = {}
    ckpts = _list_checkpoints(checkpoint_root)
    if not ckpts:
        return {}

    # Ensure stable keys across all layers/param types for consistent length
    # Pre-initialize keys so all lists align to the same length with NaNs.
    # We'll discover keys as we parse the first checkpoint.

    for idx, ckpt_path in enumerate(ckpts):
        data = torch.load(ckpt_path, map_location='cpu')
        step = int(data['step'])
        sigma_map: Dict[str, float] = data['muon_sigma']

        for pname, sigma in sigma_map.items():
            ptype, layer = categorize_parameter(pname)
            if layer < 0:
                continue
            # Expand Attention fused weight into Q/K/V/O panels
            if ptype.lower() == 'attention' or ptype == 'attention':
                keys = [(f'Attention {c}', layer) for c in ('Q', 'K', 'V', 'O')]
            elif ptype.lower() == 'mlp_input' or ptype == 'MLP Input':
                keys = [('MLP Input', layer)]
            elif ptype.lower() == 'mlp_output' or ptype == 'MLP Output':
                keys = [('MLP Output', layer)]
            else:
                # Skip non-hidden params
                continue

            for key in keys:
                if key not in series:
                    series[key] = {}
                series[key][step] = float(sigma)

    return series


def plot_sigma_time_series(ax, prop: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    """Render sigma vs. time for a given param_type."""
    ax.set_title(param_type)
    ax.set_xlabel('Training step')
    ax.set_ylabel('sigma')
    ax.grid(True, alpha=0.3)

    denom = max(1, max_layers - 1)
    for (p_type, layer), values in sorted(prop.items(), key=lambda x: x[0][1]):
        if p_type != param_type:
            continue
        # values is Dict[step, sigma]
        steps = sorted(int(s) for s in values.keys())
        if not steps:
            continue
        x = np.asarray(steps, dtype=float)
        y = np.asarray([float(values[s]) for s in steps], dtype=float)
        color = viridis(layer / denom)
        ax.plot(x, y, lw=1.5, color=color)

    return []


def main():
    checkpoint_dir = Path('research_logs/checkpoints/medium_20251004')
    out_path = Path('research_logs/plots/sigma_time_series.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prop_all = build_sigma_time_series(checkpoint_dir)
    if not prop_all:
        print(f'No checkpoints found under {checkpoint_dir}')
        return

    create_subplot_grid(
        PARAM_TYPES,
        figsize=(20, 10),
        property_map=prop_all,
        plot_fn=plot_sigma_time_series,
        title='Sigma vs. time (checkpoint index)',
        output_path=out_path,
        wants_colorbar=True,
    )
    print(f'Wrote figure: {out_path}')


if __name__ == '__main__':
    main()
