from pathlib import Path
from typing import Tuple, Iterator, TypeAlias
import numpy as np
import torch
import time
from torch.nn import Parameter, Module
from empirical.research.analysis.map import *


def serialize_model_checkpoint(
    model,
    optimizer,
    other_state,
):
    """
    Serialize the complete model state for reproducible forward passes.
    Saves model state_dict, architecture config, and metadata.
    """
    run_name = other_state["run_name"]
    step = other_state.get("step", 0)
    
    # Extract model architecture info
    if hasattr(model, 'embed'):
        vocab_size = model.embed.num_embeddings
        model_dim = model.embed.embedding_dim
        num_layers = len(model.blocks)
        num_heads = model.blocks[0].attn.num_heads if model.blocks[0].attn else 0
        max_seq_len = model.blocks[0].attn.rotary.cos.size(0) if model.blocks[0].attn else 0
    else:
        # Fallback for wrapped models
        vocab_size = model_dim = num_layers = num_heads = max_seq_len = None
    
    # Create checkpoint data
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'architecture': {
            'vocab_size': vocab_size,
            'num_layers': num_layers, 
            'num_heads': num_heads,
            'model_dim': model_dim,
            'max_seq_len': max_seq_len,
        },
        'metadata': {
            'step': step,
            'run_name': run_name,
            'timestamp': torch.tensor(time.time() if 'time' in globals() else 0),
        }
    }
    
    # Save checkpoint
    log_path = get_research_log_path("checkpoints", run_name, "model_step_{step:06d}.pt", step=step)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint_data, log_path)
    print(f"Model checkpoint saved to {log_path}")


def dummy_logging(
    model,
    optimizer,
    other_state,
):
    """
    Just log the norm of each weight.
    """
    run_name = other_state["run_name"]
    apply_fn_to_all_layers(model, calculate_weight_norm, run_name)

def svd_logging(
    model,
    optimizer,
    other_state,
):
    """
    Log the singular values of each weight matrix.
    """
    run_name = other_state["run_name"]
    apply_fn_to_all_layers(model, calculate_singular_values, run_name)

def calculate_singular_values(key: tuple[str, int], weight: Parameter | np.ndarray, run_name: str):
    param_type, layer_num = key
    
    # Only compute SVD for 2D matrices
    if weight.dim() != 2:
        return
    
    # Compute SVD and get singular values
    with torch.no_grad():
        _, singular_values, _ = torch.svd(weight)
    
    # Convert to numpy for saving
    singular_values_np = singular_values.cpu().numpy()
    
    log_path = get_research_log_path("svd", run_name, "singular_values_{param_type}_layer_{layer_num}.npy", param_type=param_type, layer_num=layer_num)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array
    np.save(log_path, singular_values_np)

def calculate_weight_norm(key: tuple[str, int], weight: Parameter | np.ndarray, run_name: str):
    param_type, layer_num = key
    log_path = get_research_log_path("norm_test", "dummy", "norm_{param_type}_layer_{layer_num}.txt", param_type=param_type, layer_num=layer_num)
    with open(log_path, "w") as f:
        f.write(f"{weight.norm()}\n")

