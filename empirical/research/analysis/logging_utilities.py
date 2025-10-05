#!/usr/bin/env python3
"""
Unified logging utilities for gradient analysis.

Consolidates online and offline logging functionality including:
- Model checkpoint serialization/deserialization
- Logging step determination
- Singular value computation and logging
- GIF creation for visualization
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.nn import Parameter
import torch.distributed as dist


def is_logging_step_piecewise_log(step: int, total_steps: int) -> bool:
    """
    Determine if current step should be logged using piecewise logarithmic sampling.
    
    Samples more frequently early in training, less frequently later.
    """
    if step == 0:
        return True
    
    # Early training: every 10 steps for first 100 steps
    if step <= 100:
        return step % 10 == 0
    
    # Mid training: every 100 steps up to 1000
    if step <= 1000:
        return step % 100 == 0
        
    # Late training: logarithmic sampling
    log_step = int(np.log10(step))
    interval = 10 ** max(1, log_step - 1)
    return step % interval == 0


def serialize_model_checkpoint(
    model,
    _optimizer_unused,
    other_state: Dict[str, Any],
    run_name: str,
    checkpoint_dir: Path,
):
    """
    Callback-style model checkpoint serializer.

    Expected signature for use with training.run_loggers:
      serialize_model_checkpoint(model, optimizer, other_state, checkpoint_dir)

    - model: PyTorch model to serialize
    - optimizer: unused placeholder to match callback interface
    - other_state: dict containing at least {'run_name': str, 'step': int}
    - checkpoint_dir: base directory to save checkpoints (Path or str)

    Writes checkpoints as {checkpoint_dir}/{run_name}/step_{step:06d}.pt on rank 0.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return None

    # Expect strict schema for other_state
    step = int(other_state["step"])  # KeyError if missing
    model_args = other_state["model_args"] if "model_args" in other_state else {}

    run_dir = Path(checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = run_dir / f"step_{step:06d}.pt"

    # Collect per-parameter Muon sigma if available (attached by training_core.optimize_step)
    muon_sigma: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if hasattr(param, "_muon_sigma"):
            try:
                muon_sigma[name] = float(getattr(param, "_muon_sigma"))
            except Exception:
                pass

    checkpoint = {
        'model': model.state_dict(),
        'step': step,
        'model_args': model_args,
        'timestamp': time.time(),
        'muon_sigma': muon_sigma,
    }
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def deserialize_model_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load model checkpoint with metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary containing model state, step, and metadata
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Validate checkpoint contents
    required_keys = ['model', 'step', 'model_args']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Missing required key '{key}' in checkpoint")
    
    return checkpoint


def categorize_parameter(param_name: str) -> Tuple[str, int]:
    """
    Extract parameter type and layer number from parameter name.
    
    Args:
        param_name: Full parameter name (e.g., "blocks.5.attn.qkvo_w")
        
    Returns:
        Tuple of (param_type, layer_number)
    """
    # Handle compiled model names
    if param_name.startswith("_orig_mod."):
        param_name = param_name[10:]
    
    if not param_name.startswith("blocks."):
        return "other", -1
    
    parts = param_name.split('.')
    if len(parts) < 3:
        return "other", -1
        
    layer_num = int(parts[1])
    
    param_type_map = {
        "attn.qkvo_w": "attention",
        "mlp.fc_w": "mlp_input", 
        "mlp.proj_w": "mlp_output"
    }
    
    param_suffix = '.'.join(parts[2:])
    param_type = param_type_map.get(param_suffix, "other")
    
    return param_type, layer_num


def calculate_singular_values(key: Tuple[str, int], weight: Parameter | np.ndarray, run_name: str):
    """
    Calculate and log singular values for a parameter.
    
    Args:
        key: (param_type, layer_number) tuple
        weight: Parameter tensor or numpy array
        run_name: Name of the run for logging
    """
    param_type, layer_num = key
    
    if isinstance(weight, Parameter):
        matrix = weight.data.cpu().numpy()
    else:
        matrix = np.asarray(weight)
    
    if matrix.ndim != 2:
        return
    
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    
    # Create logging record
    record = {
        'run_name': run_name,
        'param_type': param_type,
        'layer': layer_num,
        'singular_values': singular_values.tolist(),
        'stable_rank': float(np.sum(singular_values**2) / singular_values[0]**2),
        'condition_number': float(singular_values[0] / singular_values[-1]),
        'timestamp': time.time()
    }
    
    return record


def calculate_weight_norm(key: Tuple[str, int], weight: Parameter | np.ndarray, run_name: str):
    """
    Calculate various norms for a parameter.
    
    Args:
        key: (param_type, layer_number) tuple  
        weight: Parameter tensor or numpy array
        run_name: Name of the run for logging
    """
    param_type, layer_num = key
    
    if isinstance(weight, Parameter):
        tensor = weight.data
    else:
        tensor = torch.from_numpy(weight) if isinstance(weight, np.ndarray) else weight
    
    # Calculate various norms
    record = {
        'run_name': run_name,
        'param_type': param_type,
        'layer': layer_num,
        'frobenius_norm': float(torch.norm(tensor, p='fro')),
        'spectral_norm': float(torch.norm(tensor, p=2)),
        'nuclear_norm': float(torch.norm(tensor, p='nuc')),
        'l1_norm': float(torch.norm(tensor, p=1)),
        'timestamp': time.time()
    }
    
    return record


def dummy_logging(
    model,
    run_name: str,
    step: int,
    rank: int = 0,
    world_size: int = 1
):
    """
    Dummy logging function for testing.
    
    Args:
        model: PyTorch model
        run_name: Name of the run
        step: Current training step
        rank: Process rank
        world_size: Total number of processes
    """
    if rank == 0:
        print(f"Dummy logging at step {step} for run {run_name}")
