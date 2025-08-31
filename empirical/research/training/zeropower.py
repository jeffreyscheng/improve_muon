# =============================================================================
# ZEROPOWER BACKEND SELECTION - CAN BE OVERRIDDEN BY CLI
# =============================================================================
DEFAULT_ZEROPOWER_METHOD = "newton_schulz"  # Options: "newton_schulz", "svd_polar", "classic_newton_schulz", "tanh_matrix"
DEFAULT_ZEROPOWER_HYPERPARAMS = {}

import sys
import argparse
import json
with open(sys.argv[0]) as f:
    code = f.read()

import os
import uuid
import time
import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import itertools

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
if torch.cuda.is_available():
    torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention
torch._inductor.config.coordinate_descent_tuning = True
# torch._dynamo.config.compiled_autograd = True  # Disabled due to FlexAttention incompatibility

# =============================================================================
# NEWTON-SCHULZ QUINTIC COEFFICIENTS
# =============================================================================

# Optimized Newton-Schulz quintic coefficients for fast msign computation
# These coefficients are optimized to compute msign(G)=UV^T using odd quintic polynomials
# See: https://leloykun.github.io/ponder/muon-opt-coeffs/
NEWTON_SCHULZ_QUINTIC_COEFFICIENTS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]

# =============================================================================
# ZEROPOWER BACKEND IMPLEMENTATIONS
# =============================================================================

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in NEWTON_SCHULZ_QUINTIC_COEFFICIENTS:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_svd_polar(G: Tensor) -> Tensor:
    """
    SVD-based polar decomposition for orthogonalization.
    """
    try:
        # Handle non-square matrices like Newton-Schulz does
        was_transposed = False
        X = G.float()
        if G.size(-2) > G.size(-1):
            X = X.mT
            was_transposed = True
        
        # Use modern linalg.svd instead of deprecated torch.svd
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # Note: torch.linalg.svd returns Vh (V hermitian), not V
        result = U @ Vh
        
        if was_transposed:
            result = result.mT
            
        return result.to(G.dtype)
    except Exception:
        # Fallback to Newton-Schulz if SVD fails
        return zeropower_via_newtonschulz5(G)

def zeropower_via_classic_newton_schulz(G: Tensor, num_iters: int = 15) -> Tensor:
    """
    Classic Newton-Schulz iteration using f(x) = 1.5x - 0.5x^3.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Classic Newton-Schulz iteration: f(x) = 1.5x - 0.5x^3
    # For non-square matrices: X^3 = X @ X.T @ X
    for _ in range(num_iters):
        X = 1.5 * X - 0.5 * X @ X.mT @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_tanh_matrix(G: Tensor, alpha: float = 10.0) -> Tensor:
    """
    Matrix tanh approximation (simplified version to avoid hangs).
    Falls back to Newton-Schulz for safety.
    """
    # For now, fallback to Newton-Schulz to avoid the hanging issues we discovered
    # This can be improved with a proper safe matrix tanh implementation
    return zeropower_via_newtonschulz5(G)

def zeropower_via_perfect_cutoff(G: Tensor, cutoff: float = 1e-3) -> Tensor:
    """
    Perfect cutoff method: SVD + hard threshold at cutoff, then reconstruct.
    f(x) = 1 if x > cutoff else 0
    Returns U @ f(S) @ V^T where f(S) applies the cutoff function.
    torch.compile compatible version.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize like other backends to avoid numerical issues
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Compute SVD - convert to float for better numerical stability
    X_float = X.float()
    U, S, Vh = torch.linalg.svd(X_float, full_matrices=False)
    
    # Apply perfect cutoff function: f(x) = 1 if x > cutoff else 0
    S_cutoff = torch.where(S > cutoff, torch.ones_like(S), torch.zeros_like(S))
    
    # Ensure at least one singular value survives using torch operations only
    # If sum is 0, add 1.0 to the largest singular value position
    has_survivors = torch.sum(S_cutoff) > 0
    max_idx = torch.argmax(S)
    # Create one-hot vector for the largest singular value
    max_one_hot = torch.nn.functional.one_hot(max_idx, num_classes=S.size(-1)).float()
    # Use where to conditionally add the largest singular value if no survivors
    S_cutoff = torch.where(has_survivors, S_cutoff, max_one_hot)
    
    # Reconstruct: U @ f(S) @ V^T
    result = U @ torch.diag_embed(S_cutoff) @ Vh
    
    if G.size(-2) > G.size(-1):
        result = result.mT
        
    return result.to(G.dtype)

# =============================================================================
# ZEROPOWER BACKEND REGISTRY AND SELECTOR
# =============================================================================

ZEROPOWER_BACKENDS = {
    "newton_schulz": zeropower_via_newtonschulz5,
    "svd_polar": zeropower_via_svd_polar,
    "classic_newton_schulz": zeropower_via_classic_newton_schulz,
    "tanh_matrix": zeropower_via_tanh_matrix,
    "perfect_cutoff": zeropower_via_perfect_cutoff,
}

def get_zeropower_function(method: str, hyperparams: dict):
    """Get the zeropower function with hyperparameters applied."""
    if method not in ZEROPOWER_BACKENDS:
        available = ", ".join(ZEROPOWER_BACKENDS.keys())
        raise ValueError(f"Unknown zeropower method '{method}'. Available: {available}")
    
    base_func = ZEROPOWER_BACKENDS[method]
    
    # Create a wrapper that applies hyperparameters
    def zeropower_with_hyperparams(G: Tensor) -> Tensor:
        return base_func(G, **hyperparams)
    
    return zeropower_with_hyperparams

# Global variables to be set by CLI parsing
ZEROPOWER_METHOD = DEFAULT_ZEROPOWER_METHOD
ZEROPOWER_HYPERPARAMS = DEFAULT_ZEROPOWER_HYPERPARAMS
zeropower_func = None  # Will be set after CLI parsing

# =============================================================================
# CONFIGURABLE MUON OPTIMIZER
# =============================================================================

def make_update_function(zeropower_function):
    """Create an update function with a specific zeropower function."""
    @torch.compile
    def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
        assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
        grad = grad.float()
        momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
        v = zeropower_function(momentum * momentum_buffer + (1 - momentum) * grad)

        acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
        acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
        acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
        mantissa.copy_(acc_m_u32.to(torch.uint16))
    return update
