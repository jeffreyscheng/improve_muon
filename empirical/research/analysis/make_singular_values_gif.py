#!/usr/bin/env python3
"""
Token-parallel gradient noise evolution GIF generation.
Follows the same distributed approach as minimal_medium.py for proper GPU utilization.

Usage:
    torchrun --standalone --nproc_per_node=8 make_singular_values_gif.py checkpoint_6a9d8bce 30
"""

import os
import sys
import glob
import re
import json
import time
from pathlib import Path
from collections import defaultdict

# Memory optimization like training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch._inductor.config.coordinate_descent_tuning = False
import torch.distributed as dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Import all dependencies at top level
from empirical.research.training.training_core import setup_distributed_training, get_window_size_blocks, Hyperparameters, create_gpt_with_muon
from empirical.research.analysis.offline_logging import deserialize_model_checkpoint, split_qkv_weight, compute_singular_values, compute_stable_rank, categorize_parameter
from empirical.research.analysis.map import get_weight_matrix_iterator, get_research_log_path


PARAM_TYPES = ['Attention Q', 'Attention K', 'Attention V', 'Attention O', 'MLP Input', 'MLP Output']

def extract_step_from_checkpoint_path(checkpoint_path: Path) -> int:
    """Extract step number from checkpoint filename."""
    step_match = re.search(r'step_(\d+)', str(checkpoint_path))
    return int(step_match.group(1)) if step_match else 0

def load_cached_results(sv_cache_dir: Path, sr_cache_dir: Path) -> dict:
    """Load cached results combining both singular values and stable rank data."""
    sv_results = {}
    
    for cache_file in sv_cache_dir.glob("step_*.csv"):
        step = int(re.search(r'step_(\d+)\.csv', cache_file.name).group(1))
        df = pd.read_csv(cache_file)
        step_data = {}
        
        for _, row in df.iterrows():
            key = (row['param_type'], int(row['layer_num']))
            step_data[key] = {
                'means': np.array(json.loads(row['means'])),
                'stds': np.array(json.loads(row['stds'])),
                # tau is per (step, param_type); duplicated across layers in cache
                'tau': float(row['tau']) if 'tau' in row and not pd.isna(row['tau']) else None,
                'alpha': float(row['alpha']) if 'alpha' in row and not pd.isna(row['alpha']) else None,
                'beta':  float(row['beta'])  if 'beta'  in row and not pd.isna(row['beta'])  else None,
            }
        sv_results[step] = step_data
    
    for cache_file in sr_cache_dir.glob("step_*.csv"):
        step = int(re.search(r'step_(\d+)\.csv', cache_file.name).group(1))
        df = pd.read_csv(cache_file)
        
        for _, row in df.iterrows():
            key = (row['param_type'], int(row['layer_num']))
            sv_results[step][key].update({
                'weight_stable_rank': row['weight_stable_rank'],
                'gradient_stable_rank_mean': row['gradient_stable_rank_mean'],
                'gradient_stable_rank_std': row['gradient_stable_rank_std']
            })
    
    return sv_results

def save_results_to_cache(results: dict, sv_cache_dir: Path, sr_cache_dir: Path):
    """Save both singular values and stable rank results to cache."""
    sv_cache_dir.mkdir(parents=True, exist_ok=True)
    sr_cache_dir.mkdir(parents=True, exist_ok=True)
    
    for step, step_data in results.items():
        sv_cache_data = []
        sr_cache_data = []
        
        for (param_type, layer_num), layer_data in step_data.items():
            sv_cache_data.append({
                'param_type': param_type,
                'layer_num': layer_num,
                'means': json.dumps(layer_data['means'].tolist()),
                'stds': json.dumps(layer_data['stds'].tolist()),
                'tau': layer_data.get('tau', None),
                'alpha': layer_data.get('alpha', None),
                'beta':  layer_data.get('beta',  None),
            })
            
            sr_cache_data.append({
                'param_type': param_type,
                'layer_num': layer_num,
                'weight_stable_rank': layer_data['weight_stable_rank'],
                'gradient_stable_rank_mean': layer_data['gradient_stable_rank_mean'],
                'gradient_stable_rank_std': layer_data['gradient_stable_rank_std']
            })
        
        pd.DataFrame(sv_cache_data).to_csv(sv_cache_dir / f"step_{step}.csv", index=False)
        pd.DataFrame(sr_cache_data).to_csv(sr_cache_dir / f"step_{step}.csv", index=False)
    

def setup_model_from_checkpoint(checkpoint_file: str, device: torch.device):
    """Load and setup model from checkpoint matching training's model creation."""
    from empirical.research.training.architecture import GPT
    
    rank = dist.get_rank()
    if rank == 0:
        print(f"Rank {rank}: Loading checkpoint {checkpoint_file}")
    
    # Create fresh model exactly like training does
    args = Hyperparameters()
    model = GPT(args.vocab_size, 16, 8, 1024, max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    
    # Broadcast parameters like training does
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    
    # Load checkpoint weights
    checkpoint_data = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint_data['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # Compile model like training does
    model = torch.compile(model, dynamic=False)
    
    if rank == 0:
        print(f"Rank {rank}: Model loaded and compiled")
    
    return model

def create_real_data_loader(args, rank: int, world_size: int):
    """Create real training data loader using FineWeb dataset."""
    from empirical.research.training.training_core import distributed_data_generator
    return distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)


def extract_model_analysis_distributed(model, use_gradients: bool = True):
    """Extract analysis from model parameters using distributed GPU SVD."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Get all parameters (model is replicated on all ranks like training)
    all_params = list(get_weight_matrix_iterator(model, only_hidden=True))
    
    # Distribute SVD computation work across ranks, not the parameters themselves
    all_param_splits = []
    for param_name, param in all_params:
        tensor = param.grad if use_gradients else param
        if use_gradients and tensor is None:
            continue
        
        split_tensors = split_qkv_weight(param_name, tensor)
        for split_name, split_tensor in split_tensors.items():
            all_param_splits.append((split_name, split_tensor))
    
    # Distribute SVD work across ranks
    splits_per_rank = len(all_param_splits) // world_size
    start_idx = rank * splits_per_rank
    end_idx = start_idx + splits_per_rank if rank < world_size - 1 else len(all_param_splits)
    my_splits = all_param_splits[start_idx:end_idx]
    
    if rank == 0:
        print(f"    Distributed {len(all_param_splits)} SVD computations across {world_size} ranks ({len(my_splits)} per rank)")
    
    # Process my assigned SVD computations
    local_results = {}
    for split_name, split_tensor in my_splits:
        # Use GPU SVD instead of CPU numpy
        tensor_f32 = split_tensor.detach().to(torch.float32)
        try:
            _, s, _ = torch.linalg.svd(tensor_f32, full_matrices=False)
            sv = s.cpu().numpy()
            stable_rank = compute_stable_rank(sv)
            local_results[split_name] = {'sv': sv, 'stable_rank': stable_rank}
        except Exception:
            pass
    
    # Gather all results on rank 0
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)
    
    if rank == 0:
        # Merge results from all ranks
        merged_results = {}
        for rank_results in all_results:
            merged_results.update(rank_results)
        return merged_results
    
    return {}

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

def aggregate_singular_values_and_stable_ranks(all_gradient_results, weight_results):
    """Aggregate singular values and stable rank statistics across minibatches."""
    param_data = {}
    
    all_param_names = set()
    for mb_grads in all_gradient_results:
        all_param_names.update(mb_grads.keys())
    
    for param_name in all_param_names:
        param_type, layer_num = categorize_parameter(param_name)
        
        sv_arrays = [mb_grads[param_name]['sv'] for mb_grads in all_gradient_results if param_name in mb_grads]
        stable_ranks = [mb_grads[param_name]['stable_rank'] for mb_grads in all_gradient_results if param_name in mb_grads]
        
        min_len = min(len(sv) for sv in sv_arrays)
        aligned_svs = np.array([sv[:min_len] for sv in sv_arrays])
        
        key = (param_type, layer_num)
        param_data[key] = {
            'means': np.mean(aligned_svs, axis=0),
            'stds': np.std(aligned_svs, axis=0),
            'weight_stable_rank': weight_results[param_name]['stable_rank'],
            'gradient_stable_rank_mean': np.mean(stable_ranks),
            'gradient_stable_rank_std': np.std(stable_ranks)
        }
    
    # ----- fit alpha, beta per param_type (pooled across layers) and attach -----
    for ptype in PARAM_TYPES:
        xs, ys, gs = [], [], []
        keys = [(pt, ln) for (pt, ln) in param_data.keys() if pt == ptype]
        for key in keys:
            m = param_data[key]['means']
            s = param_data[key]['stds']
            g = _local_gaps_from_singulars(m)
            xs.append(m); ys.append(s); gs.append(g)
        if not xs:
            continue
        x_all = np.concatenate(xs); y_all = np.concatenate(ys); g_all = np.concatenate(gs)
        alpha_hat, beta_hat = _fit_alpha_beta(x_all, y_all, g_all)
        for key in keys:
            param_data[key]['alpha'] = alpha_hat
            param_data[key]['beta']  = beta_hat
            # legacy tau derived from typical gap (for backward compat only)
            g_typ = float(np.median(_local_gaps_from_singulars(param_data[key]['means'])))
            param_data[key]['tau'] = (beta_hat / g_typ) if g_typ > 0 else None

    return param_data

def compute_analysis_for_step(step: int, checkpoint_file: str, num_minibatches: int, rank: int, world_size: int, device: torch.device):
    """Compute singular values and stable rank analysis for a single step."""
    
    model = setup_model_from_checkpoint(checkpoint_file, device)
    args = Hyperparameters()
    
    if rank == 0:
        print(f"Rank {rank}: Starting weight analysis")
        weight_start = time.time()
    weight_results = extract_model_analysis_distributed(model, use_gradients=False)
    torch.cuda.empty_cache()
    if rank == 0:
        print(f"Rank {rank}: Weight analysis took {time.time() - weight_start:.1f}s")
    
    # Create real data loader like training
    data_loader = create_real_data_loader(args, rank, world_size)
    all_gradient_results = []
    
    for mb_idx in range(num_minibatches):
        if rank == 0:
            print(f"  Minibatch {mb_idx+1}/{num_minibatches}")
        
        inputs, targets = next(data_loader)
        
        model.zero_grad()
        model.train()
        
        window_size_blocks = get_window_size_blocks(step, args.num_iterations).to(device)
        if rank == 0:
            print(f"    Starting forward pass")
        loss = model(inputs, targets, window_size_blocks)
        if rank == 0:
            print(f"    Starting backward pass")
        loss.backward()
        
        if rank == 0:
            print(f"    Starting gradient sync")
        # Synchronize gradients across ranks
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"    Gradient sync complete")
        
        if rank == 0:
            print(f"    Rank {rank}: Extracting gradients for mb {mb_idx}")
            mb_start = time.time()
        mb_results = extract_model_analysis_distributed(model, use_gradients=True)
        if rank == 0:
            all_gradient_results.append(mb_results)
            print(f"    Rank {rank}: Gradient extraction took {time.time() - mb_start:.1f}s")
        
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
    if rank == 0:
        print(f"Rank {rank}: Aggregating results for {len(all_gradient_results)} minibatches")
        agg_start = time.time()
        result = aggregate_singular_values_and_stable_ranks(all_gradient_results, weight_results)
        print(f"Rank {rank}: Aggregation took {time.time() - agg_start:.1f}s")
        return step, result
    return step, {}

def find_all_checkpoints() -> list[tuple[int, str]]:
    """Find all checkpoint files and return sorted list of (step, filepath) tuples."""
    checkpoints_dir = Path("research_logs/checkpoints")
    checkpoint_pattern = str(checkpoints_dir / "*/model_step_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    unique_steps = {}
    for file in checkpoint_files:
        step = extract_step_from_checkpoint_path(Path(file))
        unique_steps[step] = file
    
    return [(step, unique_steps[step]) for step in sorted(unique_steps.keys())]

def setup_output_directories(run_name: str) -> tuple[Path, Path, Path, Path, Path]:
    """Setup output directories and return paths."""
    sv_output_dir = get_research_log_path("singular_values_distribution", run_name, "")
    sr_output_dir = get_research_log_path("stable_rank_distribution", run_name, "")
    sv_output_dir.mkdir(parents=True, exist_ok=True)
    sr_output_dir.mkdir(parents=True, exist_ok=True)
    
    gif_path = sv_output_dir / "gradient_noise_evolution.gif"
    sv_cache_dir = sv_output_dir / "cache"
    sr_cache_dir = sr_output_dir / "cache"
    
    return sv_output_dir, sr_output_dir, gif_path, sv_cache_dir, sr_cache_dir

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
            
        x_margin = (axis_ranges[param_type]['x_max'] / axis_ranges[param_type]['x_min']) ** 0.1
        y_margin = (axis_ranges[param_type]['y_max'] / axis_ranges[param_type]['y_min']) ** 0.1
        axis_ranges[param_type]['x_min'] /= x_margin
        axis_ranges[param_type]['x_max'] *= x_margin
        axis_ranges[param_type]['y_min'] /= y_margin
        axis_ranges[param_type]['y_max'] *= y_margin
    
    return axis_ranges

def compute_gap_axis_ranges(all_results: dict) -> dict:
    """Stationary bounds for gap vs singular value scatter across frames."""
    axis = {pt: {'x_min': np.inf, 'x_max': 0.0, 'y_min': np.inf, 'y_max': 0.0} for pt in PARAM_TYPES}
    for step_data in all_results.values():
        for (pt, ln), layer in step_data.items():
            if pt not in axis: continue
            s = np.asarray(layer['means'])
            g = _local_gaps_from_singulars(s)
            m = (s > 0) & (g > 0)
            if not np.any(m): continue
            axis[pt]['x_min'] = min(axis[pt]['x_min'], float(s[m].min()))
            axis[pt]['x_max'] = max(axis[pt]['x_max'], float(s[m].max()))
            axis[pt]['y_min'] = min(axis[pt]['y_min'], float(g[m].min()))
            axis[pt]['y_max'] = max(axis[pt]['y_max'], float(g[m].max()))
    for pt in PARAM_TYPES:
        if not np.isfinite(axis[pt]['x_min']) or not np.isfinite(axis[pt]['y_min']):
            axis[pt] = {'x_min':1e-9,'x_max':1.0,'y_min':1e-9,'y_max':1.0}
            continue
        axis[pt]['x_min'] *= 0.9; axis[pt]['x_max'] *= 1.1
        axis[pt]['y_min'] *= 0.9; axis[pt]['y_max'] *= 1.1
    return axis

def compute_hist_bins(all_results: dict, nbins: int = 40) -> dict:
    """Stationary log-spaced bins for histograms per param_type across frames."""
    bounds = {pt: {'xmin': np.inf, 'xmax': 0.0} for pt in PARAM_TYPES}
    for step_data in all_results.values():
        for (pt, ln), layer in step_data.items():
            if pt not in bounds: continue
            s = np.asarray(layer['means']); s = s[s > 0]
            if s.size == 0: continue
            bounds[pt]['xmin'] = min(bounds[pt]['xmin'], float(s.min()))
            bounds[pt]['xmax'] = max(bounds[pt]['xmax'], float(s.max()))
    bins = {}
    for pt in PARAM_TYPES:
        xmin = bounds[pt]['xmin']; xmax = bounds[pt]['xmax']
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin <= 0 or xmax <= xmin:
            bins[pt] = np.logspace(-9, 0, nbins)
        else:
            bins[pt] = np.logspace(np.log10(xmin*0.9), np.log10(xmax*1.1), nbins)
    return bins

def compute_c_axis_ranges(all_results: dict) -> dict:
    """Stationary bounds for c(s) vs s (y fixed to [0,1])."""
    axis = {pt: {'x_min': np.inf, 'x_max': 0.0, 'y_min': 0.0, 'y_max': 1.0} for pt in PARAM_TYPES}
    for step_data in all_results.values():
        for (pt, ln), layer in step_data.items():
            if pt not in axis: continue
            s = np.asarray(layer['means']); s = s[s > 0]
            if s.size == 0: continue
            axis[pt]['x_min'] = min(axis[pt]['x_min'], float(s.min()))
            axis[pt]['x_max'] = max(axis[pt]['x_max'], float(s.max()))
    for pt in PARAM_TYPES:
        if not np.isfinite(axis[pt]['x_min']):
            axis[pt]['x_min'], axis[pt]['x_max'] = 1e-9, 1.0
        else:
            axis[pt]['x_min'] *= 0.9; axis[pt]['x_max'] *= 1.1
    return axis

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

def create_frame(step: int, param_data: dict, axis_ranges: dict, output_dir: Path):
    """Create a single frame for the GIF."""
    param_types = PARAM_TYPES
    
    def get_frame_data(param_type):
        return [(layer_num, data) for (p_type, layer_num), data in param_data.items() if p_type == param_type and layer_num >= 0]
    
    def plot_singular_values(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Mean Singular Value')
        ax.set_ylabel('Std Singular Value')
        ax.set_title(f'{param_type}')
        ax.grid(True, alpha=0.3)
        
        ranges = axis_ranges[param_type]
        ax.set_xlim(ranges['x_min'], ranges['x_max'])
        ax.set_ylim(ranges['y_min'], ranges['y_max'])
        
        for layer_num, data in layer_data_list:
            color = viridis(layer_num / (max_layers - 1))
            ax.scatter(data['means'], data['stds'], alpha=0.1, s=20, c=[color], label=f'Layer {layer_num}')
    
    frame_path = output_dir / f"frame_{step:06d}.png"
    create_subplot_grid(param_types, (20, 10), get_frame_data, plot_singular_values, f'Gradient Noise Evolution - Step {step}', frame_path)
    return str(frame_path)

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
            ax.text(0.02, 0.98, f"α={alpha_hat:.2e}\nβ={beta_hat:.2e}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9)
    frame_path = output_dir / f"frame_fit_{step:06d}.png"
    create_subplot_grid(param_types, (20, 10), get_frame_data, plot_std_mean_fit, f'Std–Mean with Fit - Step {step}', frame_path)
    return str(frame_path)

def create_gap_vs_sv_frame(step: int, param_data: dict, output_dir: Path, gap_ranges: dict):
    """Gap_i vs s_i scatter (both on log scales) with stationary axes."""
    param_types = PARAM_TYPES
    def get_frame_data(param_type):
        return [(layer_num, data) for (p_type, layer_num), data in param_data.items() if p_type == param_type and layer_num >= 0]
    def plot_gap_sv(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Mean Singular Value s_i'); ax.set_ylabel('Local gap_i')
        ax.set_title(f'{param_type}: gap vs s')
        ax.grid(True, alpha=0.3)
        rng = gap_ranges[param_type]; ax.set_xlim(rng['x_min'], rng['x_max']); ax.set_ylim(rng['y_min'], rng['y_max'])
        alpha_hat = None; beta_hat = None
        for layer_num, data in layer_data_list:
            s = np.asarray(data['means']); g = _local_gaps_from_singulars(s)
            m = (s > 0) & (g > 0)
            if not np.any(m): continue
            color = viridis(layer_num / (max_layers - 1))
            ax.scatter(s[m], g[m], alpha=0.15, s=12, c=[color])
            alpha_hat = data.get('alpha', alpha_hat); beta_hat = data.get('beta', beta_hat)
        if alpha_hat is not None and beta_hat is not None:
            ax.text(0.02, 0.98, f"α={alpha_hat:.2e}\nβ={beta_hat:.2e}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9)
    frame_path = output_dir / f"frame_gap_{step:06d}.png"
    create_subplot_grid(param_types, (20, 10), get_frame_data, plot_gap_sv, f'Gap vs Singular Value - Step {step}', frame_path)
    return str(frame_path)

def create_sv_hist_frame(step: int, param_data: dict, output_dir: Path, bins_by_type: dict | None = None):
    """Histogram of singular-value means per param_type (log-x) with stationary bins."""
    param_types = PARAM_TYPES
    def get_frame_data(param_type):
        # return a flat list of singular means arrays across layers
        return [data['means'] for (pt, ln), data in param_data.items() if pt == param_type and ln >= 0]
    def plot_hist(ax, param_type, list_of_means, viridis, max_layers):
        ax.set_xscale('log')
        ax.set_xlabel('Mean Singular Value s'); ax.set_ylabel('Count')
        ax.set_title(f'{param_type} density')
        ax.grid(True, alpha=0.3)
        if not list_of_means:
            return
        vals = np.concatenate([np.asarray(m) for m in list_of_means])
        vals = vals[vals > 0]
        if vals.size == 0:
            return
        bins = bins_by_type[param_type] if bins_by_type is not None else np.logspace(np.log10(vals.min()), np.log10(vals.max()), 40)
        ax.hist(vals, bins=bins)
    frame_path = output_dir / f"frame_hist_{step:06d}.png"
    create_subplot_grid(param_types, (20, 10), get_frame_data, plot_hist, f'Singular Value Density - Step {step}', frame_path)
    return str(frame_path)

def create_c_vs_sv_frame(step: int, param_data: dict, output_dir: Path, c_ranges: dict | None = None):
    """Scatter of c_i = gap_i^2 / (gap_i^2 + (β/α)^2) vs s_i per param_type, stationary axes."""
    param_types = PARAM_TYPES
    def get_frame_data(param_type):
        return [(layer_num, data) for (p_type, layer_num), data in param_data.items() if p_type == param_type and layer_num >= 0]
    def plot_c_sv(ax, param_type, layer_data_list, viridis, max_layers):
        ax.set_xscale('log')
        ax.set_xlabel('Mean Singular Value s_i'); ax.set_ylabel('c_i = gap_i^2 / (gap_i^2 + (β/α)^2)')
        ax.set_title(f'{param_type}: c vs s')
        ax.grid(True, alpha=0.3)
        if c_ranges is not None:
            rng = c_ranges[param_type]; ax.set_xlim(rng['x_min'], rng['x_max']); ax.set_ylim(rng['y_min'], rng['y_max'])
        alpha_hat = None; beta_hat = None
        xmin, xmax = np.inf, 0.0
        for layer_num, data in layer_data_list:
            s = np.asarray(data['means']); g = _local_gaps_from_singulars(s)
            m = (s > 0)
            if not np.any(m): continue
            alpha_hat = data.get('alpha', alpha_hat); beta_hat = data.get('beta', beta_hat)
            lam = (beta_hat/alpha_hat)**2 if (alpha_hat is not None and alpha_hat>0 and beta_hat is not None) else 0.0
            c = (g[m]**2) / (g[m]**2 + lam) if lam > 0 else np.ones_like(g[m])
            color = viridis(layer_num / (max_layers - 1))
            ax.scatter(s[m], c, alpha=0.15, s=12, c=[color])
            xmin = min(xmin, s[m].min()); xmax = max(xmax, s[m].max())
        if alpha_hat is not None and beta_hat is not None:
            ax.text(0.02, 0.98, f"α={alpha_hat:.2e}\nβ={beta_hat:.2e}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9)
        if c_ranges is None and np.isfinite(xmin):
            ax.set_xlim(xmin*0.9, xmax*1.1); ax.set_ylim(0.0, 1.0)
    frame_path = output_dir / f"frame_c_{step:06d}.png"
    create_subplot_grid(param_types, (20, 10), get_frame_data, plot_c_sv, f'c(s) using gap & α,β - Step {step}', frame_path)
    return str(frame_path)


def main():
    _, rank, world_size, device, master_process = setup_distributed_training()
    
    # Initialize global print function for training_core patch
    from empirical.research.training.training_core import _global_print0
    import empirical.research.training.training_core as training_core
    training_core._global_print0 = lambda s, console=False: print(s) if rank == 0 else None
    
    # Parse arguments
    run_name = sys.argv[1] if len(sys.argv) >= 2 else "gradient_noise_run"
    fps = int(sys.argv[2]) if len(sys.argv) >= 3 else 36
    testing = len(sys.argv) >= 4 and sys.argv[3] == "--testing"
    force_recompute = "--force" in sys.argv
    num_minibatches = 8
    
    
    # Find all checkpoints (we might still plot from CSVs if none exist)
    selected_checkpoints = find_all_checkpoints()
    
    # NOTE: Checkpoint 0 often has NaN/None gradients that break axis range computation
    # for attention output parameters. Starting from checkpoint 1 avoids this issue.
    if testing:
        # Use 12 evenly-spaced checkpoints starting from checkpoint 1
        available_checkpoints = selected_checkpoints[1:]  # Skip checkpoint 0
        if len(available_checkpoints) >= 12:
            indices = np.linspace(0, len(available_checkpoints) - 1, 12, dtype=int)
            selected_checkpoints = [available_checkpoints[i] for i in indices]
        else:
            selected_checkpoints = available_checkpoints
        fps = 12  # Testing mode uses 12 fps
    
    # Setup directories (safe on all ranks) and load cache on all ranks
    sv_output_dir, sr_output_dir, gif_path, sv_cache_dir, sr_cache_dir = setup_output_directories(run_name)
    cached_results = load_cached_results(sv_cache_dir, sr_cache_dir)
    
    # Process checkpoints
    all_results = {}
    new_results = {}

    # ---------- CSV-only fast path (all ranks) ----------
    cached_steps = set(cached_results.keys())
    selected_steps = set(step for step, _ in selected_checkpoints) if selected_checkpoints else set()
    if not force_recompute and cached_results and (not selected_steps or selected_steps.issubset(cached_steps)):
        steps_to_use = sorted(selected_steps) if selected_steps else sorted(cached_steps)
        all_results = {step: cached_results[step] for step in steps_to_use if step in cached_results}
        if master_process:
            print(f"[cache] Using {len(all_results)} cached step(s); skipping checkpoint computation.")
    else:
        missing = sorted(selected_steps - cached_steps) if selected_steps else []
        if master_process and missing:
            print(f"[cache] Missing {len(missing)} step(s) in cache; computing those from checkpoints.")
        if master_process and force_recompute:
            print(f"[cache] Force recomputation enabled; ignoring cache.")
    # ----------------------------------------

    for i, (step, checkpoint_file) in enumerate(selected_checkpoints):
        if not force_recompute and (step in all_results or step in cached_results):
            all_results[step] = all_results.get(step, cached_results[step])
            continue
        
        if master_process:
            print(f"Processing step {step} ({i+1}/{len(selected_checkpoints)})")
            start_time = time.time()
        _, param_data = compute_analysis_for_step(step, checkpoint_file, num_minibatches, rank, world_size, device)
        if master_process:
            print(f"Step {step} completed in {time.time() - start_time:.1f}s")
        
        if master_process:
            all_results[step] = param_data
            new_results[step] = param_data
    
    # If nothing to plot (no cache + no computed steps), exit cleanly
    if not all_results:
        if master_process:
            print("[cache] No checkpoints processed and no cached CSVs found — nothing to plot.")
        dist.destroy_process_group()
        return

    # Save new results and create visualizations
    if master_process:
        if new_results:
            save_results_to_cache(new_results, sv_cache_dir, sr_cache_dir)
        
        # Precompute stationary bounds / bins across all frames (from all_results only)
        axis_ranges     = compute_axis_ranges(all_results)
        gap_axis_ranges = compute_gap_axis_ranges(all_results)
        hist_bins       = compute_hist_bins(all_results)
        c_axis_ranges   = compute_c_axis_ranges(all_results)
        frame_paths = []
        for step in sorted(all_results.keys()):
            frame_path = create_frame(step, all_results[step], axis_ranges, sv_output_dir)
            frame_paths.append(frame_path)
        
        create_gif_from_frames(frame_paths, gif_path, fps)
        create_stable_rank_plots(all_results, sr_output_dir)

        # Create GIF: std vs mean with fitted curve y = sqrt((alpha x)^2 + tau^2)
        frame_paths_fit = []
        for step in sorted(all_results.keys()):
            frame_paths_fit.append(create_frame_with_fit(step, all_results[step], axis_ranges, sv_output_dir))
        create_gif_from_frames(frame_paths_fit, sv_output_dir / "std_mean_with_fit.gif", fps)

        # Create GIF: gap vs singular value
        frame_paths_gap = []
        for step in sorted(all_results.keys()):
            frame_paths_gap.append(create_gap_vs_sv_frame(step, all_results[step], sv_output_dir, gap_axis_ranges))
        create_gif_from_frames(frame_paths_gap, sv_output_dir / "gap_vs_sv.gif", fps)

        # Create GIF: histogram of singular value density
        frame_paths_hist = []
        for step in sorted(all_results.keys()):
            frame_paths_hist.append(create_sv_hist_frame(step, all_results[step], sv_output_dir, bins_by_type=hist_bins))
        create_gif_from_frames(frame_paths_hist, sv_output_dir / "sv_hist.gif", fps)

        # Create GIF: c(s) = gap^2/(gap^2 + tau^2) vs s
        frame_paths_c = []
        for step in sorted(all_results.keys()):
            frame_paths_c.append(create_c_vs_sv_frame(step, all_results[step], sv_output_dir, c_ranges=c_axis_ranges))
        create_gif_from_frames(frame_paths_c, sv_output_dir / "c_vs_sv.gif", fps)
    
    dist.destroy_process_group()

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
            if param_type in param_types:
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
    

if __name__ == "__main__":
    main()