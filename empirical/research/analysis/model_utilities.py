#!/usr/bin/env python3
"""
Unified model parameter processing utilities.

Consolidates model parameter extraction, layer property management,
and distributed communication functions from the original map.py.
"""

from pathlib import Path
from typing import Tuple, Iterator, TypeAlias, Callable, Iterable, List
import numpy as np
import torch
import torch.distributed as dist
import time
from torch.nn import Parameter, Module

# Type alias for layer properties
GPTLayerProperty: TypeAlias = dict[tuple[str, int], Parameter | np.ndarray | torch.Tensor]


def _split_attention_tensor(tensor: torch.Tensor, layer_num: int) -> Iterator[tuple[tuple[str, int], torch.Tensor]]:
    """Split qkvo_w tensor into Q, K, V, O components."""
    for i, component in enumerate(["Q", "K", "V", "O"]):
        yield (f"Attention {component}", layer_num), tensor[i]


def _extract_layer_info(name: str) -> tuple[str, int] | None:
    """Extract param type and layer number from parameter name."""
    # Handle compiled model names that have _orig_mod prefix
    if name.startswith("_orig_mod."):
        name = name[10:]  # Remove "_orig_mod." prefix
    
    if not name.startswith("blocks."):
        return None
    
    parts = name.split('.')
    layer_num = int(parts[1])
    
    param_type_map = {
        "attn.qkvo_w": "Attention",  # Will be split later
        "mlp.fc_w": "MLP Input", 
        "mlp.proj_w": "MLP Output"
    }
    
    for pattern, param_type in param_type_map.items():
        if pattern in name:
            return param_type, layer_num
    
    return None


def process_model_parameters(model, only_hidden: bool, tensor_extractor, result_processor) -> GPTLayerProperty:
    """Process model parameters using functional approach with generators."""
    
    def filtered_params():
        """Generator of filtered parameters."""
        for name, param in model.named_parameters():
            if only_hidden and not ("blocks." in name and param.ndim >= 2 and "embed" not in name):
                continue
            tensor = tensor_extractor(name, param)
            if tensor is not None:
                yield name, param, tensor
                
    
    def parameter_entries():
        """Generator of (key, tensor) pairs."""
        for name, param, tensor in filtered_params():
            layer_info = _extract_layer_info(name)
            if layer_info is None:
                continue
            
            param_type, layer_num = layer_info
            
            # Handle attention tensor splitting
            if param_type == "Attention" and tensor.ndim >= 3:
                yield from _split_attention_tensor(tensor, layer_num)
            else:
                yield (param_type, layer_num), tensor
    
    return result_processor(parameter_entries())


def get_weight_matrices(model, only_hidden: bool = True) -> GPTLayerProperty:
    """Extract weight matrices from model."""
    return process_model_parameters(
        model, 
        only_hidden,
        tensor_extractor=lambda name, param: param.data,
        result_processor=lambda entries: dict(entries)
    )


def get_research_log_path(run_name: str) -> Path:
    """Get path to research logs directory."""
    return Path("research_logs") / run_name


def combine_layer_properties(fn: Callable, *layer_properties: GPTLayerProperty) -> GPTLayerProperty:
    """
    Combine multiple layer properties using a function.
    
    Args:
        fn: Function that takes (*tensors) and returns a tensor
        *layer_properties: Variable number of layer property dicts
        
    Returns:
        Combined layer properties
    """
    if not layer_properties:
        return {}
        
    # Get common keys
    common_keys = set(layer_properties[0].keys())
    for props in layer_properties[1:]:
        common_keys &= set(props.keys())
    
    result = {}
    for key in common_keys:
        tensors = [props[key] for props in layer_properties]
        result[key] = fn(*tensors)
    
    return result


def gather_layer_properties_to_rank_zero(*props: GPTLayerProperty):
    """Gather layer properties from all ranks to rank 0."""
    if not dist.is_initialized():
        return props[0] if props else {}
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        # Rank 0 collects from all other ranks
        all_props = [props[0]] if props else [{}]
        
        for src_rank in range(1, world_size):
            # Receive from each rank
            received = {}
            for key in props[0].keys() if props else []:
                tensor = props[0][key]
                gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
                dist.gather(tensor, gathered_tensors if rank == 0 else None, dst=0)
                
        return all_props[0] if all_props else {}
    else:
        # Other ranks send their data
        for key, tensor in (props[0].items() if props else []):
            dist.gather(tensor, None, dst=0)
        return {}


def get_accumulated_gradient_matrices(model, args, step: int, num_minibatches: int) -> GPTLayerProperty:
    """
    Compute accumulated gradient matrices for analysis.
    
    This function runs forward/backward passes to accumulate gradients
    across multiple minibatches for gradient analysis.
    
    Args:
        model: The model to compute gradients for
        args: Hyperparameters for data loading
        step: Current training step (for logging)
        num_minibatches: Number of minibatches to accumulate
        
    Returns:
        GPTLayerProperty containing per-minibatch gradient tensors
    """
    from empirical.research.training.training_core import distributed_data_generator
    import torch.distributed as dist
    
    # Get data generator
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    data_generator = distributed_data_generator(args, rank, world_size)
    
    # Storage for per-minibatch gradients
    per_minibatch_grads = {}
    
    model.eval()  # Set to eval mode for consistent analysis
    
    with torch.no_grad():
        for minibatch_idx in range(num_minibatches):
            try:
                inputs, targets = next(data_generator)
            except StopIteration:
                # Re-initialize data generator if exhausted
                data_generator = distributed_data_generator(args, rank, world_size)
                inputs, targets = next(data_generator)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            with torch.enable_grad():
                model.train()  # Brief switch to train mode for gradient computation
                outputs = model(inputs)
                
                # Compute loss (simple cross-entropy)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
                
                # Backward pass
                loss.backward()
                
                model.eval()  # Back to eval mode
            
            # Collect gradients for this minibatch
            minibatch_grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None and param.ndim >= 2 and "embed" not in name:
                    # Extract layer info
                    layer_info = _extract_layer_info(name)
                    if layer_info is None:
                        continue
                    
                    param_type, layer_num = layer_info
                    
                    # Handle attention tensor splitting
                    if param_type == "Attention" and param.grad.ndim >= 3:
                        for i, component in enumerate(["Q", "K", "V", "O"]):
                            key = (f"Attention {component}", layer_num)
                            grad_tensor = param.grad[i].clone().detach()
                            if key not in minibatch_grads:
                                minibatch_grads[key] = []
                            minibatch_grads[key].append(grad_tensor)
                    else:
                        key = (param_type, layer_num)
                        grad_tensor = param.grad.clone().detach()
                        if key not in minibatch_grads:
                            minibatch_grads[key] = []
                        minibatch_grads[key].append(grad_tensor)
            
            # Store this minibatch's gradients
            for key, grad_list in minibatch_grads.items():
                if key not in per_minibatch_grads:
                    per_minibatch_grads[key] = []
                per_minibatch_grads[key].extend(grad_list)
    
    # Stack gradients into batched tensors
    result = {}
    for key, grad_list in per_minibatch_grads.items():
        if grad_list:
            result[key] = torch.stack(grad_list, dim=0)  # Shape: [num_minibatches, ...]
    
    return result