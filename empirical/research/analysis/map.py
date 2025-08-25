from pathlib import Path
from typing import Tuple, Iterator, TypeAlias, Callable
import numpy as np
import torch
import torch.distributed as dist
import time
from torch.nn import Parameter, Module

"""
Utilities
"""

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
                if not only_hidden:
                    yield (name, -1), tensor
                continue
            
            param_type, layer_num = layer_info
            
            if param_type == "Attention" and "qkvo_w" in name:
                yield from _split_attention_tensor(tensor, layer_num)
            else:
                yield (param_type, layer_num), tensor
    
    return {key: result_processor(key, tensor) for key, tensor in parameter_entries()}

def get_weight_matrices(model, only_hidden: bool = True) -> GPTLayerProperty:
    """
    Get all weight matrices in the model as a GPTLayerProperty dict.
    If only_hidden=True, only returns hidden matrices (from blocks, 2D+, no embeddings)
    following the same logic as Muon optimizer parameter selection.
    """
    def extract_weight(name, param):
        return param.data
        
    def process_weight(param_key, tensor):
        return tensor
    
    return process_model_parameters(model, only_hidden, extract_weight, process_weight)

def apply_fn_to_all_layers(layer_properties: GPTLayerProperty, fn) -> GPTLayerProperty:
    """
    Apply a function to all layers in a GPTLayerProperty dict.
    
    Args:
        layer_properties: GPTLayerProperty dict to process
        fn: Function to apply to each (key, tensor) pair
    
    Returns:
        GPTLayerProperty dict with fn applied to each entry
    """
    return {
        key: fn(key, tensor)
        for key, tensor in layer_properties.items()
    }

def get_accumulated_gradient_matrices(model, momentum_buffers: dict, momentum_value: float) -> GPTLayerProperty:
    """
    Extract gradients from model parameters, properly handling split parameters like qkvo_w.
    Returns accumulated_gradients dict with the same schema as get_weight_matrices output:
    - Key: (param_type, layer_num) where param_type is "Attention Q", "Attention K", etc.
    - Value: torch.Tensor gradient
    
    This function mirrors the logic of get_weight_matrices but operates on gradients.
    """
    def extract_gradient(name, param):
        return param.grad.float() if param.grad is not None else None
        
    def process_gradient(param_key, tensor):
        # Momentum accumulation logic (only written once!)
        if param_key not in momentum_buffers:
            momentum_buffers[param_key] = torch.zeros_like(tensor, dtype=torch.float32)
        momentum_buffers[param_key] = momentum_value * momentum_buffers[param_key] + (1 - momentum_value) * tensor
        return momentum_value * momentum_buffers[param_key] + (1 - momentum_value) * tensor
    
    return process_model_parameters(model, True, extract_gradient, process_gradient)


def get_research_log_path(metric_name: str, run_name: str, pattern: str, **kwargs) -> Path:
    return Path("research_logs") / metric_name / run_name / pattern.format(**kwargs)


"""
Functional Programming Infrastructure for GPTLayerProperty
"""

def combine_layer_properties(fn: Callable, *layer_properties: GPTLayerProperty) -> GPTLayerProperty:
    """
    Combine multiple GPTLayerProperty objects using fn.
    
    Args:
        fn: Function that takes (*tensors) and returns a tensor
        *layer_properties: Variable number of GPTLayerProperty objects
        
    Returns:
        GPTLayerProperty with same schema, values = fn(*[prop[k] for prop in props]) for each k
        Only operates on keys present in ALL input layer_properties.
    """
    if not layer_properties:
        return {}
    
    # Get intersection of all keys
    common_keys = set(layer_properties[0].keys())
    for prop in layer_properties[1:]:
        common_keys &= set(prop.keys())
    
    return {
        key: fn(*[prop[key] for prop in layer_properties])
        for key in common_keys
    }


def apply_fn_to_all_layers_multi(fn: Callable, *layer_properties: GPTLayerProperty) -> GPTLayerProperty:
    """
    Apply fn to corresponding tensors across multiple GPTLayerProperty objects.
    Like combine_layer_properties but passes key as first argument to fn.
    
    Args:
        fn: Function that takes (key, *tensors) and returns a tensor
        *layer_properties: Variable number of GPTLayerProperty objects
        
    Returns:
        GPTLayerProperty with same schema, values = fn(key, *[prop[key] for prop in props]) for each k
    """
    if not layer_properties:
        return {}
    
    common_keys = set(layer_properties[0].keys())
    for prop in layer_properties[1:]:
        common_keys &= set(prop.keys())
    
    return {
        key: fn(key, *[prop[key] for prop in layer_properties])
        for key in common_keys
    }


def filter_layer_properties(layer_properties: GPTLayerProperty, predicate: Callable) -> GPTLayerProperty:
    """
    Filter a GPTLayerProperty based on a predicate function.
    
    Args:
        layer_properties: GPTLayerProperty to filter
        predicate: Function that takes (key, tensor) and returns bool
        
    Returns:
        Filtered GPTLayerProperty containing only entries where predicate(key, tensor) is True
    """
    return {
        key: tensor
        for key, tensor in layer_properties.items()
        if predicate(key, tensor)
    }


def gather_layer_properties_to_rank_zero(*sharded_properties: GPTLayerProperty) -> list[GPTLayerProperty]:
    """
    Gather sharded GPTLayerProperty objects from all ranks to rank 0.
    
    Args:
        *sharded_properties: Variable number of sharded GPTLayerProperty objects
        
    Returns:
        List of full GPTLayerProperty objects (only populated on rank 0, empty on other ranks)
    """
    if not dist.is_initialized():
        return list(sharded_properties)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    gathered_properties = []
    
    for sharded_prop in sharded_properties:
        # Gather all sharded properties from all ranks
        all_sharded_props = [None] * world_size
        dist.all_gather_object(all_sharded_props, sharded_prop)
        
        if rank == 0:
            # Merge all sharded properties into one full property
            full_property = {}
            for rank_prop in all_sharded_props:
                if rank_prop is not None:
                    full_property.update(rank_prop)
            gathered_properties.append(full_property)
        else:
            gathered_properties.append({})
    
    return gathered_properties


def broadcast_layer_properties_from_rank_zero(*full_properties: GPTLayerProperty) -> list[GPTLayerProperty]:
    """
    Broadcast full GPTLayerProperty objects from rank 0 to all ranks.
    
    Args:
        *full_properties: Variable number of full GPTLayerProperty objects (only meaningful on rank 0)
        
    Returns:
        List of GPTLayerProperty objects (populated on all ranks)
    """
    if not dist.is_initialized():
        return list(full_properties)
    
    rank = dist.get_rank()
    
    broadcasted_properties = []
    
    for full_prop in full_properties:
        if rank == 0:
            # Broadcast from rank 0
            broadcast_prop = full_prop
        else:
            broadcast_prop = {}
        
        # Use all_gather_object as a broadcast mechanism
        broadcast_list = [broadcast_prop if rank == 0 else None]
        dist.broadcast_object_list(broadcast_list, src=0)
        
        broadcasted_properties.append(broadcast_list[0])
    
    return broadcasted_properties


"""
Statistical Transform Functions
"""

def apply_mean_across_batch_dim(layer_properties: GPTLayerProperty, dim: int = 0) -> GPTLayerProperty:
    """
    Apply mean reduction across specified dimension for all tensors.
    
    Args:
        layer_properties: GPTLayerProperty with tensors that have a batch dimension
        dim: Dimension to reduce (default 0 for batch dimension)
        
    Returns:
        GPTLayerProperty with mean applied across the specified dimension
    """
    return apply_fn_to_all_layers(layer_properties, lambda key, tensor: tensor.mean(dim=dim))


def apply_std_across_batch_dim(layer_properties: GPTLayerProperty, dim: int = 0, unbiased: bool = True) -> GPTLayerProperty:
    """
    Apply standard deviation reduction across specified dimension for all tensors.
    
    Args:
        layer_properties: GPTLayerProperty with tensors that have a batch dimension
        dim: Dimension to reduce (default 0 for batch dimension)
        unbiased: Whether to use unbiased std calculation
        
    Returns:
        GPTLayerProperty with std applied across the specified dimension
    """
    return apply_fn_to_all_layers(layer_properties, lambda key, tensor: tensor.std(dim=dim, unbiased=unbiased))


def apply_stable_rank(layer_properties: GPTLayerProperty) -> GPTLayerProperty:
    """
    Compute stable rank for each matrix in the GPTLayerProperty.
    Handles both single matrices (n×m) and batched matrices (batch×n×m).
    
    Args:
        layer_properties: GPTLayerProperty with matrix tensors
        
    Returns:
        GPTLayerProperty with stable ranks (scalars for single matrices, 1D tensors for batched matrices)
    """
    def compute_stable_rank_tensor(key, tensor):
        # Handle batched case (e.g., 8×n×m)
        if tensor.ndim == 3:
            stable_ranks = []
            for i in range(tensor.shape[0]):
                _, s, _ = torch.linalg.svd(tensor[i].float(), full_matrices=False)
                # Stable rank = ||A||_F^2 / ||A||_2^2 = sum(s_i^2) / max(s_i)^2
                stable_rank = (s**2).sum() / (s[0]**2) if s.numel() > 0 else torch.tensor(0.0)
                stable_ranks.append(stable_rank)
            return torch.stack(stable_ranks)
        
        # Handle single matrix case (n×m)
        elif tensor.ndim == 2:
            _, s, _ = torch.linalg.svd(tensor.float(), full_matrices=False)
            return (s**2).sum() / (s[0]**2) if s.numel() > 0 else torch.tensor(0.0)
        
        else:
            raise ValueError(f"Expected 2D or 3D tensor for stable rank computation, got {tensor.ndim}D for {key}")
    
    return apply_fn_to_all_layers(layer_properties, compute_stable_rank_tensor)


"""
SVD Transform Functions
"""

def apply_batched_svd(layer_properties: GPTLayerProperty) -> tuple[GPTLayerProperty, GPTLayerProperty, GPTLayerProperty]:
    """
    Apply SVD to each batched tensor (batch×n×m) in the GPTLayerProperty.
    
    Args:
        layer_properties: GPTLayerProperty with batched matrix tensors (e.g., 8×n×m)
        
    Returns:
        Tuple of (left_singular_vectors, singular_values, right_singular_vectors):
        - left_singular_vectors: GPTLayerProperty with 8×n×n tensors
        - singular_values: GPTLayerProperty with 8×min(n,m) tensors  
        - right_singular_vectors: GPTLayerProperty with 8×m×m tensors
    """
    def compute_batched_svd(key, tensor):
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor for batched SVD, got {tensor.ndim}D for {key}")
        
        batch_size = tensor.shape[0]
        U_list, S_list, Vh_list = [], [], []
        
        for i in range(batch_size):
            U_i, S_i, Vh_i = torch.linalg.svd(tensor[i].float(), full_matrices=False)
            U_list.append(U_i)
            S_list.append(S_i)
            Vh_list.append(Vh_i)
        
        U = torch.stack(U_list, dim=0)  # batch×n×n
        S = torch.stack(S_list, dim=0)  # batch×min(n,m)
        Vh = torch.stack(Vh_list, dim=0)  # batch×m×m
        
        return U, S, Vh
    
    U_prop, S_prop, Vh_prop = {}, {}, {}
    
    for key, tensor in layer_properties.items():
        U, S, Vh = compute_batched_svd(key, tensor)
        U_prop[key] = U
        S_prop[key] = S
        Vh_prop[key] = Vh
    
    return U_prop, S_prop, Vh_prop


def apply_svd(layer_properties: GPTLayerProperty) -> tuple[GPTLayerProperty, GPTLayerProperty, GPTLayerProperty]:
    """
    Apply SVD to each matrix (n×m) in the GPTLayerProperty.
    
    Args:
        layer_properties: GPTLayerProperty with matrix tensors (n×m)
        
    Returns:
        Tuple of (left_singular_vectors, singular_values, right_singular_vectors):
        - left_singular_vectors: GPTLayerProperty with n×n tensors
        - singular_values: GPTLayerProperty with min(n,m) tensors  
        - right_singular_vectors: GPTLayerProperty with m×m tensors
    """
    def compute_svd(key, tensor):
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor for SVD, got {tensor.ndim}D for {key}")
        
        U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
        return U, S, Vh
    
    U_prop, S_prop, Vh_prop = {}, {}, {}
    
    for key, tensor in layer_properties.items():
        U, S, Vh = compute_svd(key, tensor)
        U_prop[key] = U
        S_prop[key] = S
        Vh_prop[key] = Vh
    
    return U_prop, S_prop, Vh_prop


"""
Basis Similarity and Spectral Analysis Functions
"""

def compute_basis_cosine_similarity(
    batched_basis: torch.Tensor,  # batch×k×k
    reference_basis: torch.Tensor  # k×k
) -> torch.Tensor:
    """
    Compute cosine similarity between each batched basis and reference basis.
    
    This computes |<u_i, u_ref>| for each column vector u_i in the batched basis
    against the corresponding column vector u_ref in the reference basis.
    
    Args:
        batched_basis: Batched orthonormal basis matrices (batch×k×k)
        reference_basis: Reference orthonormal basis matrix (k×k)
        
    Returns:
        Tensor of shape (batch×k) containing cosine similarities
    """
    # Ensure same device and dtype
    batched_basis = batched_basis.to(reference_basis.device).to(reference_basis.dtype)
    
    # Compute inner products: batch×k×k @ k×k -> batch×k×k
    # We want the diagonal elements, which are the inner products of corresponding columns
    inner_products = torch.bmm(batched_basis.transpose(-2, -1), reference_basis.unsqueeze(0).expand(batched_basis.shape[0], -1, -1))
    
    # Extract diagonal elements (inner products of corresponding basis vectors)
    cosines = torch.diagonal(inner_products, dim1=-2, dim2=-1)  # batch×k
    
    # Return absolute values (cosine similarity)
    return torch.abs(cosines)


def compute_spectral_projection_coefficients(
    left_cosines: torch.Tensor,   # batch×n
    right_cosines: torch.Tensor   # batch×m
) -> torch.Tensor:
    """
    Compute spectral projection coefficients as element-wise products.
    
    For each minibatch i and singular value index j (up to min(n,m)):
    coeff[i,j] = |left_cosines[i,j]| * |right_cosines[i,j]|
    
    Args:
        left_cosines: Left basis cosine similarities (batch×n)
        right_cosines: Right basis cosine similarities (batch×m)
        
    Returns:
        Tensor of shape (batch×min(n,m)) containing spectral projection coefficients
    """
    # Take minimum rank to get valid spectral coefficients
    min_rank = min(left_cosines.shape[-1], right_cosines.shape[-1])
    
    # Truncate to minimum rank and compute element-wise product
    left_truncated = left_cosines[..., :min_rank].abs()
    right_truncated = right_cosines[..., :min_rank].abs()
    
    return left_truncated * right_truncated


def compute_basis_similarities_and_spectral_coeffs(
    per_gradient_U: GPTLayerProperty,      # batch×n×n
    per_gradient_Vh: GPTLayerProperty,     # batch×m×m  
    average_gradient_U: GPTLayerProperty,  # n×n
    average_gradient_Vh: GPTLayerProperty  # m×m
) -> tuple[GPTLayerProperty, GPTLayerProperty, GPTLayerProperty]:
    """
    Compute basis similarities and spectral projection coefficients.
    
    Args:
        per_gradient_U: Batched left singular vectors from per-minibatch gradients
        per_gradient_Vh: Batched right singular vectors from per-minibatch gradients
        average_gradient_U: Left singular vectors from average gradient
        average_gradient_Vh: Right singular vectors from average gradient
        
    Returns:
        Tuple of (left_cosines, right_cosines, spectral_coeffs):
        - left_cosines: GPTLayerProperty with batch×n tensors
        - right_cosines: GPTLayerProperty with batch×m tensors
        - spectral_coeffs: GPTLayerProperty with batch×min(n,m) tensors
    """
    # Compute left cosine similarities
    left_cosines = combine_layer_properties(
        compute_basis_cosine_similarity,
        per_gradient_U,
        average_gradient_U
    )
    
    # Compute right cosine similarities (need to transpose Vh to get column vectors)
    right_cosines = combine_layer_properties(
        lambda batched_vh, ref_vh: compute_basis_cosine_similarity(
            batched_vh.transpose(-2, -1),  # Convert to column-major
            ref_vh.transpose(-2, -1)       # Convert to column-major
        ),
        per_gradient_Vh,
        average_gradient_Vh
    )
    
    # Compute spectral projection coefficients
    spectral_coeffs = combine_layer_properties(
        compute_spectral_projection_coefficients,
        left_cosines,
        right_cosines
    )
    
    return left_cosines, right_cosines, spectral_coeffs


"""
Gradient Standard Deviation Computation
"""

def compute_gradient_singular_value_std(
    per_minibatch_singular_values: GPTLayerProperty,  # batch×min(n,m)
    average_gradient_singular_values: GPTLayerProperty  # min(n,m)
) -> GPTLayerProperty:
    """
    Compute standard deviation of singular values across minibatches.
    
    Args:
        per_minibatch_singular_values: Batched singular values (batch×min(n,m))
        average_gradient_singular_values: Average gradient singular values (min(n,m))
        
    Returns:
        GPTLayerProperty with standard deviations (batch×min(n,m))
    """
    def compute_std(batched_sv, avg_sv):
        # Expand average to match batch dimension for broadcasting
        avg_expanded = avg_sv.unsqueeze(0).expand_as(batched_sv)  # batch×min(n,m)
        
        # Compute standard deviation across batch dimension
        return batched_sv.std(dim=0, unbiased=True)
    
    return combine_layer_properties(compute_std, per_minibatch_singular_values, average_gradient_singular_values)


"""
Core Gradient Computation Functions
"""

def compute_sharded_gradients(
    model: torch.nn.Module,
    data_loader,
    num_minibatches: int,
    momentum_buffers: dict,
    momentum_value: float,
    step: int
) -> GPTLayerProperty:
    """
    Compute per-minibatch gradients with momentum accumulation.
    Each rank processes all minibatches but only stores gradients for parameters assigned to it.
    
    Args:
        model: The model to compute gradients for
        data_loader: Data loader for minibatches
        num_minibatches: Number of minibatches to process (typically 8)
        momentum_buffers: Dictionary to store momentum state
        momentum_value: Momentum coefficient for accumulation
        step: Current training step (for window size computation)
        
    Returns:
        GPTLayerProperty where each value is a num_minibatches×n×m tensor
        Keys are sharded across ranks (each rank only has a subset).
    """
    from empirical.research.training.training_core import get_window_size_blocks, Hyperparameters
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = next(model.parameters()).device
    
    args = Hyperparameters()
    
    # Get all parameters that we'll need for sharding
    all_layer_properties = get_weight_matrices(model, only_hidden=True)
    all_param_keys = list(all_layer_properties.keys())
    
    # Shard parameters across ranks
    params_per_rank = len(all_param_keys) // world_size
    start_idx = rank * params_per_rank
    end_idx = start_idx + params_per_rank if rank < world_size - 1 else len(all_param_keys)
    my_param_keys = set(all_param_keys[start_idx:end_idx])
    
    if rank == 0:
        print(f"    Rank {rank}: Processing {len(my_param_keys)} parameters out of {len(all_param_keys)} total")
    
    # Collect gradients for my assigned parameters across all minibatches
    per_minibatch_gradients = {}
    
    for mb_idx in range(num_minibatches):
        if rank == 0:
            print(f"    Processing minibatch {mb_idx + 1}/{num_minibatches}")
        
        # Forward and backward pass
        model.zero_grad()
        model.train()
        
        inputs, targets = next(data_loader)
        window_size_blocks = get_window_size_blocks(step, args.num_iterations).to(device)
        
        loss = model(inputs, targets, window_size_blocks)
        loss.backward()
        
        # Synchronize gradients across ranks (like in training)
        for param in model.parameters():
            if param.grad is not None:
                if dist.is_initialized():
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
        # Update momentum buffers and get accumulated gradients
        accumulated_gradients = get_accumulated_gradient_matrices(model, momentum_buffers, momentum_value)
        
        # Store gradients for my assigned parameters
        for param_key, grad_tensor in accumulated_gradients.items():
            if param_key in my_param_keys:
                if param_key not in per_minibatch_gradients:
                    # Initialize tensor to store all minibatches: num_minibatches×n×m
                    per_minibatch_gradients[param_key] = torch.zeros(
                        (num_minibatches,) + grad_tensor.shape,
                        dtype=grad_tensor.dtype,
                        device=grad_tensor.device
                    )
                
                # Store this minibatch's gradient
                per_minibatch_gradients[param_key][mb_idx] = grad_tensor.detach()
        
        # Clear gradients to save memory
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
    if rank == 0:
        print(f"    Rank {rank}: Collected gradients for {len(per_minibatch_gradients)} parameters")
    
    return per_minibatch_gradients


def convert_to_record_format(
    average_gradient_singular_values: GPTLayerProperty,
    per_minibatch_gradient_singular_values: GPTLayerProperty, 
    gradient_singular_value_standard_deviations: GPTLayerProperty,
    spectral_projection_coefficients: GPTLayerProperty,
    per_minibatch_gradient_stable_rank: GPTLayerProperty,
    weight_matrix_stable_rank: GPTLayerProperty
) -> dict:
    """
    Convert functional computation results back to the record format expected by CSV output.
    
    Returns:
        Dictionary with param_key -> data_dict mapping for CSV serialization
    """
    records = {}
    
    # Get all keys (should be the same across all inputs on rank 0)
    all_keys = set(average_gradient_singular_values.keys())
    
    for param_key in all_keys:
        param_type, layer_num = param_key
        
        records[param_key] = {
            'gradient_singular_values': per_minibatch_gradient_singular_values[param_key].cpu().numpy(),
            'weight_stable_rank': weight_matrix_stable_rank[param_key].cpu().item(),
            'gradient_stable_rank_mean': per_minibatch_gradient_stable_rank[param_key].cpu().mean().item(),
            'gradient_stable_rank_std': per_minibatch_gradient_stable_rank[param_key].cpu().std().item(),
            'c_with_mean_truth': spectral_projection_coefficients[param_key].cpu().numpy()
        }
    
    return records