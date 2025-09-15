"""
Core mathematical functions for gradient analysis.

This module contains all the fundamental mathematical operations needed across
the analysis pipeline. It provides both numpy and torch implementations where
needed, with consistent interfaces.
"""

import math
from typing import Union, Tuple, Dict
import numpy as np
import torch


def compute_stable_rank(singular_values: Union[np.ndarray, torch.Tensor], epsilon: float = 1e-8) -> float:
    """
    Compute stable rank from singular values.
    
    Stable rank = ||A||_F^2 / ||A||_2^2 = sum(s^2) / s_max^2
    where s are the singular values.
    
    Args:
        singular_values: Singular values (sorted descending)
        epsilon: Threshold for considering singular values as zero
        
    Returns:
        Stable rank as a float
    """
    # Convert to numpy if needed
    if isinstance(singular_values, torch.Tensor):
        sv = singular_values.detach().cpu().numpy()
    else:
        sv = np.asarray(singular_values)
    
    # Filter out small singular values
    sv_filtered = sv[sv > epsilon]
    
    if len(sv_filtered) == 0:
        return 0.0
    
    # Stable rank formula
    return float(np.sum(sv_filtered**2) / (sv_filtered[0]**2))


def matrix_shape_beta(shape: Union[Tuple[int, int], torch.Size]) -> float:
    """
    Compute beta = min(n,m)/max(n,m) from a (n,m) shape tuple.
    Useful as an aspect ratio parameter in rectangular matrix analyses.
    
    Args:
        shape: Matrix shape (height, width)
        
    Returns:
        Beta parameter as float
    """
    n, m = int(shape[0]), int(shape[1])
    a, b = (n, m) if n < m else (m, n)
    return float(a) / float(b)


## Removed MP-specific density helpers (mp_pdf_singular_*). Finite-size Wishart overlays
## now come from tabulated CDFs (see wishart_tables.py).


def safe_svd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD with proper memory management (handles both single and batched).
    
    Args:
        tensor: Input tensor of shape [H, W] or [B, H, W]
        
    Returns:
        U, s, Vh tensors with proper cloning for CUDA graph safety
    """
    with torch.no_grad():
        # Cast to float32 if needed for SVD compatibility
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        U, s, Vh = torch.linalg.svd(tensor, full_matrices=False)
        return U.clone(), s.clone(), Vh.clone()


def compute_basis_cosine_similarity(
    batched_basis: torch.Tensor, 
    reference_basis: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarities between batched and reference bases.
    
    Args:
        batched_basis: [B, H, K] or [B, K, W] basis vectors
        reference_basis: [H, K] or [K, W] reference basis
        
    Returns:
        Cosine similarities [B, K]
    """
    if batched_basis.dim() == 3 and reference_basis.dim() == 2:
        # Compute cosine similarity for each basis vector
        if batched_basis.shape[1] == reference_basis.shape[0]:
            # Left singular vectors: [B, H, K] @ [H, K] -> [B, K]
            cosines = torch.einsum('bhk,hk->bk', batched_basis, reference_basis)
        else:
            # Right singular vectors: [B, K, W] @ [K, W] -> [B, K]
            cosines = torch.einsum('bkw,kw->bk', batched_basis, reference_basis)
    else:
        raise ValueError(f"Unsupported shapes: {batched_basis.shape}, {reference_basis.shape}")
    
    return torch.abs(cosines)  # Take absolute value


def compute_spectral_projection_coefficients_from_cosines(
    left_cosines: torch.Tensor, 
    right_cosines: torch.Tensor
) -> torch.Tensor:
    """
    Compute spectral projection coefficients from basis cosine similarities.
    
    Args:
        left_cosines: [B, K] left singular vector similarities
        right_cosines: [B, K] right singular vector similarities
        
    Returns:
        Spectral projection coefficients [B, K]
    """
    # Truncate to minimum rank for non-square matrices
    min_rank = min(left_cosines.shape[-1], right_cosines.shape[-1])
    left_truncated = left_cosines[..., :min_rank]
    right_truncated = right_cosines[..., :min_rank]
    
    return left_truncated * right_truncated


def trapz_cdf_from_pdf(pdf: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute CDF from PDF using trapezoid rule.
    
    Args:
        pdf: PDF values [..., G]
        x: Grid points [G]
        
    Returns:
        CDF values [..., G] starting at 0
    """
    dx = torch.diff(x)                               # [G-1]
    mid = 0.5 * (pdf[..., 1:] + pdf[..., :-1])      # [..., G-1]
    F = torch.cumsum(mid * dx, dim=-1)              # [..., G-1]
    F = torch.nn.functional.pad(F, (1, 0))          # [..., G]
    # Numerical guard to keep within [0,1]
    return torch.clamp(F, 0.0, 1.0)


# Convenience functions for common operations
def stable_rank_from_tensor(tensor: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute stable rank directly from a matrix (computes SVD internally)."""
    if isinstance(tensor, torch.Tensor):
        with torch.no_grad():
            # Cast to float32 if needed for SVD compatibility
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            s = torch.linalg.svdvals(tensor)
            return compute_stable_rank(s)
    else:
        s = np.linalg.svd(tensor, compute_uv=False)
        return compute_stable_rank(s)


def compute_innovation_spectrum(per_minibatch_grad: torch.Tensor, mean_grad: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute spectrum of the innovation matrix (per_batch - mean)."""
    
    # Compute innovation matrix directly
    innovation = per_minibatch_grad - mean_grad.unsqueeze(0)
    
    with torch.no_grad():
        # Compute singular values of each innovation matrix  
        _, s, _ = safe_svd(innovation)
        
        # Get matrix shape parameters
        B, H, W = innovation.shape
        beta = matrix_shape_beta((H, W))
        
        return s.clone()