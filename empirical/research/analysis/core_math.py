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


def compute_spc_from_svds(
    mb_svd: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mean_svd: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Compute spectral projection coefficients via Procrustes alignment.

    Args:
        mb_svd: (U_b, S_b, Vh_b) from batched SVD of per-minibatch gradients
                U_b: [B, H, K], S_b: [B, K], Vh_b: [B, K, W]
        mean_svd: (U_m, S_m, Vh_m) from SVD of mean gradient
                U_m: [H, K], S_m: [K], Vh_m: [K, W]

    Returns:
        SPC tensor [B, K] where each entry is the product of principal
        correlations (cosines) between aligned left/right singular subspaces.
    """
    U_b, _, Vh_b = mb_svd
    U_m, _, Vh_m = mean_svd

    # Truncate to common rank K
    K = min(U_b.shape[-1], U_m.shape[-1], Vh_b.shape[-2], Vh_m.shape[-2])
    U_b = U_b[..., :K]            # [B, H, K]
    U_m = U_m[..., :K]            # [H, K]
    Vh_b = Vh_b[..., :K, :]       # [B, K, W]
    Vh_m = Vh_m[..., :K, :]       # [K, W]

    B = U_b.shape[0]

    # Left subspace alignment: C_u = U_b^T @ U_m  -> [B, K, K]
    C_u = U_b.transpose(-2, -1) @ U_m      # [B, K, H] @ [H, K] -> [B, K, K]
    # Right subspace alignment: C_v = V_b^T @ V_m -> [B, K, K]
    V_b = Vh_b.transpose(-2, -1)           # [B, W, K]
    V_m = Vh_m.transpose(-2, -1)           # [W, K]
    C_v = V_b.transpose(-2, -1) @ V_m      # ([B, K, W] @ [W, K]) -> [B, K, K]

    # Singular values of C_u and C_v are principal correlations (cosines)
    sigma_u = torch.linalg.svdvals(C_u)    # [B, K]
    sigma_v = torch.linalg.svdvals(C_v)    # [B, K]

    # Sort descending and take elementwise product for SPC
    sigma_u_sorted, _ = torch.sort(sigma_u, dim=-1, descending=True)
    sigma_v_sorted, _ = torch.sort(sigma_v, dim=-1, descending=True)
    Kc = min(sigma_u_sorted.shape[-1], sigma_v_sorted.shape[-1])
    spc = sigma_u_sorted[..., :Kc] * sigma_v_sorted[..., :Kc]
    return spc


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
