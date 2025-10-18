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


# empirical/research/analysis/model_utilities.py

from typing import Dict, Tuple, Any, Optional
import torch.distributed as dist
from empirical.research.training.training_core import distributed_data_generator, get_window_size_blocks
import io, pickle

def _merge_dicts(dicts):
    merged = {}
    for d in dicts:
        if not d:
            continue
        # later ranks overwrite on duplicate keys (your sharding should prevent conflicts)
        merged.update(d)
    return merged

def gather_layer_properties_to_rank_zero(local_props: Dict[Tuple[str, int], Dict[str, Any]]
                                        ) -> Optional[Dict[Tuple[str, int], Dict[str, Any]]]:
    """
    Gather arbitrary (picklable) Python objects from all ranks to rank 0.

    Rank 0 returns the merged dict. Non-zero ranks return None.
    All ranks must call this function.
    """
    if not dist.is_initialized():
        # single process fallback
        return local_props

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Fast path: use gather_object when available
    if hasattr(dist, "gather_object"):
        obj_list = [None] * world_size if rank == 0 else None
        # Every rank sends its local_props to dst=0
        dist.gather_object(local_props, obj_list, dst=0)
        if rank == 0:
            return _merge_dicts(obj_list)
        return None

    # ===== Fallback for very old PyTorch: tensor-based gather =====
    # Serialize to bytes -> ByteTensor on CPU, gather sizes then payloads.

    buf = io.BytesIO()
    pickle.dump(local_props, buf, protocol=pickle.HIGHEST_PROTOCOL)
    byte_arr = bytearray(buf.getbuffer())
    local_len = torch.tensor([len(byte_arr)], dtype=torch.int64, device="cpu")

    # Gather lengths to rank 0
    if rank == 0:
        lens = [torch.empty_like(local_len) for _ in range(world_size)]
    else:
        lens = None
    dist.gather(local_len, gather_list=lens, dst=0)

    # Send payloads
    if rank == 0:
        recv_bufs = []
        for i in range(world_size):
            recv_bufs.append(torch.empty(int(lens[i].item()), dtype=torch.uint8, device="cpu"))
        # Rank 0 copies its own bytes
        recv_bufs[0].copy_(torch.tensor(list(byte_arr), dtype=torch.uint8))
        # Receive from others
        for src in range(1, world_size):
            dist.recv(recv_bufs[src], src=src)
        # Deserialize and merge
        objs = []
        for t in recv_bufs:
            b = bytes(memoryview(t.numpy()))
            objs.append(pickle.loads(b))
        return _merge_dicts(objs)
    else:
        payload = torch.tensor(list(byte_arr), dtype=torch.uint8, device="cpu")
        dist.send(payload, dst=0)
        return None


def gather_microgradients_across_ranks(per_minibatch_grads: Dict[Tuple[str, int], torch.Tensor]) -> Dict[Tuple[str, int], torch.Tensor]:
    """Gather per-layer microgradients across ranks and concatenate along batch dim.

    - On single process, returns input unchanged.
    - With distributed, gathers dicts to rank 0, concatenates along dim 0 for each key,
      and broadcasts the merged dict back to all ranks.
    """
    if not dist.is_initialized():
        return per_minibatch_grads

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Gather objects to rank 0
    obj_list = [None] * world_size if rank == 0 else None
    dist.gather_object(per_minibatch_grads, obj_list, dst=0)

    if rank == 0:
        # Merge and concatenate along batch dim
        merged: Dict[Tuple[str, int], List[torch.Tensor]] = {}
        for d in obj_list:
            for k, t in d.items():
                merged.setdefault(k, []).append(t)
        concat: Dict[Tuple[str, int], torch.Tensor] = {}
        for k, tensors in merged.items():
            # Ensure tensors are on same device (move to CPU for broadcast)
            tensors_cpu = [ti.detach().cpu() for ti in tensors]
            concat[k] = torch.cat(tensors_cpu, dim=0)
        payload = [concat]
    else:
        payload = [None]

    # Broadcast merged dict to all ranks
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def get_accumulated_gradient_matrices(model, args, step: int, num_minibatches: int, assigned_params: set = None) -> GPTLayerProperty:
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
    # Get data generator
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    data_generator = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
    
    # Storage for per-minibatch gradients
    per_minibatch_grads = {}
    
    # Original behavior: eval mode outside; enable grads only for forward/backward block
    model.eval()  # Set to eval mode for consistent analysis
    with torch.no_grad():
        for minibatch_idx in range(num_minibatches):
            try:
                inputs, targets = next(data_generator)
            except StopIteration:
                # Re-initialize data generator if exhausted
                data_generator = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
                inputs, targets = next(data_generator)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass with grads enabled in a tight scope
            with torch.enable_grad():
                model.train()  # Briefly switch to train for grad computation
                window_size_blocks = get_window_size_blocks(step, args.num_iterations).to(inputs.device)
                loss = model(inputs.to(torch.int32), targets, window_size_blocks)
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
                            # Accumulate for all keys (no sharding during accumulation)
                            grad_tensor = param.grad[i].clone().detach()
                            if key not in minibatch_grads:
                                minibatch_grads[key] = []
                            minibatch_grads[key].append(grad_tensor)
                    else:
                        key = (param_type, layer_num)
                        # Accumulate for all keys (no sharding during accumulation)
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

    # (debug logs removed)

    return result
