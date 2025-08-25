from pathlib import Path
from typing import Tuple, Iterator, TypeAlias
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import time
import copy
from torch.nn import Parameter, Module
from .map import get_weight_matrices, get_research_log_path

def is_logging_step_piecewise_log(step: int, total_steps: int) -> bool:
    """Determine if we should serialize the model at this step.
    
    Precomputes all logging steps to avoid gaps from dynamically changing intervals.
    Starts with logging every step, rapidly decreases so that by step 25
    we're no longer logging every minibatch, and gradually approaches 
    logging every ~500 steps by the end of training.
    
    Args:
        step: Current training step
        total_steps: Total number of training steps
        
    Returns:
        Boolean indicating whether to serialize the model
    """
    # Cache the logging steps for this total_steps to avoid recomputation
    if not hasattr(is_logging_step_piecewise_log, '_cache'):
        is_logging_step_piecewise_log._cache = {}
    
    if total_steps not in is_logging_step_piecewise_log._cache:
        logging_steps = set()
        
        # Always log first and last steps
        logging_steps.add(0)
        logging_steps.add(total_steps)
        
        # Generate logging steps by walking through and adding when interval changes
        current_step = 1
        while current_step < total_steps:
            # Calculate interval using same logic as before
            if current_step < 25:
                # More aggressive early falloff (quadratic growth)
                log_interval = max(1, int((current_step / 25) ** 2 * 10) + 1)
            else:
                # Progress through remaining training (0 to 1)
                remaining_progress = (current_step - 25) / (total_steps - 25)
                # Start at interval ~10 and grow to ~500
                log_interval = max(10, int(10 + remaining_progress * 490))
            
            # Add current step to logging set
            logging_steps.add(current_step)
            
            # Move to next logging step
            current_step += log_interval
        
        # Cache the result
        is_logging_step_piecewise_log._cache[total_steps] = logging_steps
    
    return step in is_logging_step_piecewise_log._cache[total_steps]

def deserialize_model_checkpoint(checkpoint_path: Path):
    """
    Load a model checkpoint and return the model in evaluation mode.
    """
    from empirical.research.training.architecture import GPT
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    arch = checkpoint['architecture']
    
    # Recreate model with saved architecture
    model = GPT(
        vocab_size=arch['vocab_size'],
        num_layers=arch['num_layers'],
        num_heads=arch['num_heads'],
        model_dim=arch['model_dim'],
        max_seq_len=arch['max_seq_len']
    )
    
    # Load saved weights (strip _orig_mod prefix from torch.compile)
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, checkpoint['metadata']

def procrustes_align_gpu(X, Y):
    """Align matrix X to matrix Y using orthogonal Procrustes rotation."""
    dtype = X.dtype
    X_flat, Y_flat = X.reshape(X.shape[0], -1), Y.reshape(Y.shape[0], -1)
    
    # Convert to float32 for SVD
    X_flat_f32, Y_flat_f32 = X_flat.to(torch.float32), Y_flat.to(torch.float32)
    
    # Perform SVD in float32
    U, _, Vh = torch.linalg.svd((X_flat_f32.T @ Y_flat_f32), full_matrices=False)
    
    # Convert back to original dtype and perform the alignment
    return (X_flat @ (U.to(dtype) @ Vh.to(dtype))).reshape(X.shape)

def split_qkv_weight(name: str, param: Parameter) -> dict:
    """Split QKV weights into Q, K, V components if applicable."""
    if 'qkvo_w' in name and param.shape[0] >= 4:
        layer_match = name.split('.')
        layer_num = next((int(p) for p in layer_match if p.isdigit()), -1)
        
        return {
            f"layer_{layer_num}_attn_Q": param[0],
            f"layer_{layer_num}_attn_K": param[1],
            f"layer_{layer_num}_attn_V": param[2],
            f"layer_{layer_num}_attn_out": param[3],
        }
    else:
        # Extract layer number and clean weight name
        parts = name.split('.')
        layer_num = next((int(p) for p in parts if p.isdigit()), -1)
        
        if 'fc_w' in name:
            weight_name = 'mlp_fc'
        elif 'proj_w' in name:
            weight_name = 'mlp_proj'
        elif 'qkvo_w' in name:
            weight_name = 'attn_out'
        else:
            weight_name = name.split('.')[-1]
            
        return {f"layer_{layer_num}_{weight_name}": param}

def compute_gradient_singular_values(model, data_loader, num_minibatches: int, run_name: str):
    """
    Compute singular value analysis of gradients across minibatches.
    
    Args:
        model: The model to analyze
        data_loader: Data loader for minibatches
        num_minibatches: Number of minibatches to process
        run_name: Name for logging outputs
    """
    from empirical.research.training.training_core import get_window_size_blocks, Hyperparameters
    
    args = Hyperparameters()
    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    master_process = rank == 0
    
    def print0(msg):
        if master_process:
            print(msg)
    
    # Collect gradients across minibatches
    print0(f"Computing gradients for {num_minibatches} minibatches...")
    
    all_gradients = []
    gradient_sums = {}
    
    model.train()
    
    for batch_idx in range(num_minibatches):
        model.zero_grad(set_to_none=True)
        
        try:
            inputs, targets = next(data_loader)
            window_size_blocks = get_window_size_blocks(0, args.num_iterations)
            
            loss = model(inputs, targets, window_size_blocks)
            loss.backward()
            
            # Extract gradients using weight iterator
            batch_gradients = {}
            
            layer_properties = get_weight_matrices(model, only_hidden=True)
            for (param_type, layer_num), param in layer_properties.items():
                if param.grad is not None:
                    grad = param.grad.clone().detach()
                    
                    # Average across GPUs
                    if dist.is_initialized():
                        dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                    
                    # Store gradient with proper key
                    key_name = f"{param_type}_layer_{layer_num}"
                    batch_gradients[key_name] = grad.to(torch.bfloat16)
                    
                    # Accumulate for average
                    if key_name not in gradient_sums:
                        gradient_sums[key_name] = torch.zeros_like(grad, dtype=torch.bfloat16)
                    gradient_sums[key_name] += grad.to(torch.bfloat16)
            
            all_gradients.append(batch_gradients)
            
            if master_process and (batch_idx + 1) % 10 == 0:
                print0(f"Processed {batch_idx + 1}/{num_minibatches} minibatches")
                
        except Exception as e:
            print0(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Compute average gradients
    avg_gradients = {name: grad_sum / num_minibatches for name, grad_sum in gradient_sums.items()}
    
    print0("Computing SVDs and alignment...")
    
    # Compute SVD of average gradients
    avg_svds = {}
    for name, grad in avg_gradients.items():
        if grad.ndim >= 2:
            try:
                U, S, Vh = torch.linalg.svd(grad.to(torch.float32), full_matrices=False)
                avg_svds[name] = (U.to(torch.bfloat16), S.to(torch.bfloat16), Vh.to(torch.bfloat16))
            except Exception as e:
                print0(f"SVD failed for {name}: {e}")
    
    # Process individual minibatch gradients and collect results
    results = []
    
    for batch_idx, batch_grads in enumerate(all_gradients):
        for name, grad in batch_grads.items():
            if name not in avg_svds or grad.ndim < 2:
                continue
                
            try:
                # Align with average gradient
                avg_grad = avg_gradients[name]
                aligned_grad = procrustes_align_gpu(grad.to(torch.float32), avg_grad.to(torch.float32))
                
                # Compute SVD
                _, S_i, _ = torch.linalg.svd(aligned_grad, full_matrices=False)
                S_avg = avg_svds[name][1]
                
                # Extract layer number and weight type from our naming scheme
                # name format: "{param_type}_layer_{layer_num}"
                parts = name.split('_layer_')
                weight_name = parts[0] if len(parts) > 0 else name
                layer_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
                
                # Record singular values
                min_size = min(len(S_i), len(S_avg))
                for sv_idx in range(min_size):
                    results.append({
                        'layer_number': layer_num,
                        'weight_name': weight_name,
                        'singular_value_index': sv_idx,
                        'minibatch_index': batch_idx,
                        'G_avg_singular_value': S_avg[sv_idx].item(),
                        'G_i_singular_value': S_i[sv_idx].item(),
                        'frobenius_norm': torch.norm(avg_grad).item(),
                    })
                    
            except Exception as e:
                print0(f"Error processing {name} in batch {batch_idx}: {e}")
    
    # Aggregate results and compute statistics
    if master_process and results:
        df = pd.DataFrame(results)
        
        # Group by layer, weight, and singular value index to compute variance
        grouped = df.groupby(['layer_number', 'weight_name', 'singular_value_index'])
        
        summary_results = []
        for (layer, weight, sv_idx), group in grouped:
            sv_values = group['G_i_singular_value'].values
            summary_results.append({
                'layer_number': layer,
                'weight_name': weight,
                'singular_value_index': sv_idx,
                'G_avg_singular_value': group['G_avg_singular_value'].iloc[0],
                'singular_value_variance': float(np.var(sv_values)),
                'singular_value_mean': float(np.mean(sv_values)),
                'count': len(sv_values),
                'frobenius_norm': group['frobenius_norm'].iloc[0],
            })
        
        # Save results
        summary_df = pd.DataFrame(summary_results)
        output_path = get_research_log_path("singular_gradient_noise", run_name, "singular_gradient_noise.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        
        print0(f"Saved singular gradient noise analysis to {output_path}")
        print0(f"Processed {len(results)} singular value measurements across {num_minibatches} minibatches")
        
        return summary_df
    
    return None

def create_singular_value_evolution_gif(run_name: str, fps: int = 15):
    """
    Create a GIF showing the evolution of singular values across checkpoints.
    
    Args:
        run_name: Name of the run to analyze
        fps: Frames per second for the GIF
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    import imageio.v2 as imageio
    import glob
    from collections import defaultdict
    from tqdm import tqdm
    
    # Setup paths
    base_path = get_research_log_path("singular_gradient_noise", run_name, "")
    output_dir = get_research_log_path("singular_gradient_noise", run_name, "evolution_frames")
    gif_path = get_research_log_path("singular_gradient_noise", run_name, "sv_evolution.gif")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files (assuming they're named with checkpoint numbers)
    csv_pattern = str(base_path / "*" / "singular_gradient_noise.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {base_path}")
        return None
    
    # Sort by checkpoint number (extract from path)
    def extract_checkpoint_num(path):
        parts = Path(path).parts
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0
    
    csv_files = sorted(csv_files, key=extract_checkpoint_num)
    print(f"Found {len(csv_files)} CSV files")
    
    # Compute axis bounds across all checkpoints
    print("Computing axis bounds...")
    bounds = defaultdict(lambda: {'x_min': float('inf'), 'x_max': float('-inf'), 
                                 'y_min': float('inf'), 'y_max': float('-inf')})
    all_weight_types = set()
    
    for csv_file in tqdm(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
                
            df['sv_std'] = np.sqrt(df['singular_value_variance'])
            weight_types = df['weight_name'].unique()
            all_weight_types.update(weight_types)
            
            for weight_type in weight_types:
                weight_df = df[df['weight_name'] == weight_type]
                
                sv_mean = weight_df['singular_value_mean'].replace([np.inf, -np.inf], np.nan).dropna()
                sv_std = weight_df['sv_std'].replace([np.inf, -np.inf], np.nan).dropna()
                
                if not sv_mean.empty and not sv_std.empty:
                    sv_mean_pos = sv_mean[sv_mean > 0]
                    sv_std_pos = sv_std[sv_std > 0]
                    
                    if not sv_mean_pos.empty:
                        bounds[weight_type]['x_min'] = min(bounds[weight_type]['x_min'], sv_mean_pos.min())
                        bounds[weight_type]['x_max'] = max(bounds[weight_type]['x_max'], sv_mean.max())
                    
                    if not sv_std_pos.empty:
                        bounds[weight_type]['y_min'] = min(bounds[weight_type]['y_min'], sv_std_pos.min())
                        bounds[weight_type]['y_max'] = max(bounds[weight_type]['y_max'], sv_std.max())
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Add padding to bounds
    for weight_type in bounds:
        bounds[weight_type]['x_min'] = max(1e-10, bounds[weight_type]['x_min'] * 0.9)
        bounds[weight_type]['x_max'] *= 1.1
        bounds[weight_type]['y_min'] = max(1e-10, bounds[weight_type]['y_min'] * 0.9)
        bounds[weight_type]['y_max'] *= 1.1
    
    all_weight_types = sorted(list(all_weight_types))
    print(f"Found {len(all_weight_types)} unique weight types")
    
    # Create frames
    frame_paths = []
    for i, csv_file in enumerate(tqdm(csv_files, desc="Creating frames")):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
                
            df['sv_std'] = np.sqrt(df['singular_value_variance'])
            
            # Extract checkpoint number
            checkpoint_num = extract_checkpoint_num(csv_file)
            
            # Create subplot grid
            num_types = len(all_weight_types)
            num_cols = int(np.ceil(np.sqrt(num_types)))
            num_rows = int(np.ceil(num_types / num_cols))
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 3*num_rows))
            if num_rows == 1 and num_cols == 1:
                axes = [axes]
            elif num_rows == 1 or num_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Color mapping by layer
            if 'layer_number' in df.columns:
                layer_min, layer_max = df['layer_number'].min(), df['layer_number'].max()
                norm = Normalize(vmin=layer_min, vmax=layer_max)
                cmap = plt.cm.viridis
            else:
                norm, cmap = None, None
            
            # Plot each weight type
            for j, weight_type in enumerate(all_weight_types):
                if j >= len(axes):
                    break
                    
                ax = axes[j]
                ax.set_xscale('log')
                ax.set_yscale('log')
                
                # Set consistent bounds
                if weight_type in bounds:
                    b = bounds[weight_type]
                    ax.set_xlim(b['x_min'], b['x_max'])
                    ax.set_ylim(b['y_min'], b['y_max'])
                
                ax.set_title(weight_type, fontsize=10)
                ax.set_xlabel('Mean SV')
                ax.set_ylabel('Std SV')
                
                # Plot data if available
                weight_df = df[df['weight_name'] == weight_type]
                if not weight_df.empty:
                    if norm and cmap and 'layer_number' in weight_df.columns:
                        scatter = ax.scatter(weight_df['singular_value_mean'], weight_df['sv_std'],
                                           alpha=0.6, c=weight_df['layer_number'], cmap=cmap, norm=norm, s=20)
                    else:
                        ax.scatter(weight_df['singular_value_mean'], weight_df['sv_std'],
                                 alpha=0.6, s=20)
                    
                    layer_count = weight_df['layer_number'].nunique() if 'layer_number' in weight_df.columns else len(weight_df)
                    ax.text(0.05, 0.95, f'N={layer_count}', transform=ax.transAxes, 
                           fontsize=8, verticalalignment='top')
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                           ha='center', va='center', alpha=0.5)
            
            # Hide unused subplots
            for j in range(len(all_weight_types), len(axes)):
                axes[j].set_visible(False)
            
            plt.suptitle(f'Singular Value Evolution - Checkpoint {checkpoint_num}', fontsize=14)
            plt.tight_layout()
            
            # Save frame
            frame_path = output_dir / f"frame_{i:04d}.png"
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            frame_paths.append(str(frame_path))
            
        except Exception as e:
            print(f"Error creating frame for {csv_file}: {e}")
    
    # Create GIF
    if frame_paths:
        print(f"Creating GIF with {len(frame_paths)} frames...")
        with imageio.get_writer(str(gif_path), mode='I', fps=fps, loop=0) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        
        print(f"GIF saved to {gif_path}")
        return str(gif_path)
    else:
        print("No frames created")
        return None

def compute_singular_values(matrix: np.ndarray) -> np.ndarray:
    """Compute singular values of a matrix using SVD."""
    try:
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        return s
    except np.linalg.LinAlgError:
        return np.array([])

def categorize_parameter(param_name: str) -> tuple[str, int]:
    """Extract parameter type and layer number from parameter name."""
    parts = param_name.split('_')
    
    if len(parts) >= 3 and parts[0] == 'layer' and parts[1].isdigit():
        layer_num = int(parts[1])
        param_suffix = '_'.join(parts[2:])
        
        # Map parameter suffixes to readable names
        param_type_map = {
            'attn_Q': 'Attention Q',
            'attn_K': 'Attention K', 
            'attn_V': 'Attention V',
            'attn_out': 'Attention O',
            'mlp_fc': 'MLP Input',
            'mlp_proj': 'MLP Output'
        }
        
        param_type = param_type_map.get(param_suffix, param_suffix)
        return param_type, layer_num
    
    return param_name, -1

def compute_stable_rank(singular_values: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute stable rank as sum of singular values squared divided by max singular value squared."""
    if len(singular_values) == 0:
        return 0.0
    
    # Filter out very small singular values
    sv_filtered = singular_values[singular_values > epsilon]
    if len(sv_filtered) == 0:
        return 0.0
    
    # Stable rank = ||A||_F^2 / ||A||_2^2 = sum(s_i^2) / max(s_i)^2
    return float(np.sum(sv_filtered**2) / (sv_filtered[0]**2))