"""
Core mathematical functions for gradient analysis.

This module contains all the fundamental mathematical operations needed across
the analysis pipeline. It provides both numpy and torch implementations where
needed, with consistent interfaces.
"""

import math
import logging
from typing import Union, Tuple, Dict, List
import time
import torch.distributed as dist
from empirical.research.analysis.logging_utilities import log_from_rank
import numpy as np
try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
except Exception:
    _scipy_curve_fit = None
import torch

from empirical.research.analysis.model_utilities import GPTLayerProperty


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


# Backward-compatible alias used elsewhere in the codebase
## No alias needed; use matrix_shape_beta directly


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


def compute_spectral_echo_by_permutation_alignment(
    mb_svd: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mean_svd: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Compute spectral echo via Procrustes alignment.

    Args:
        mb_svd: (U_b, S_b, Vh_b) from batched SVD of per-minibatch gradients
                U_b: [B, H, K], S_b: [B, K], Vh_b: [B, K, W]
        mean_svd: (U_m, S_m, Vh_m) from SVD of mean gradient
                U_m: [H, K], S_m: [K], Vh_m: [K, W]

    Returns:
        spectral_echo tensor [B, K] where each entry corresponds to the product of
        cosine agreements between matched singular directions (via a
        permutation alignment) of (U_b,V_b) to (U_m,V_m).
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

    # Pairwise cosine agreements between singular directions
    # Left: cosU[b, j, i] = |<u_b_j, u_m_i>|
    cosU = torch.einsum('bhk,hq->bkq', U_b, U_m).abs()  # [B, K_b, K_m]
    # Right: cosV[b, j, i] = |<v_b_j, v_m_i>|
    V_b = Vh_b.transpose(-2, -1)  # [B, W, K_b]
    V_m = Vh_m.transpose(-2, -1)  # [W, K_m]
    cosV = torch.einsum('bwk,wq->bkq', V_b, V_m).abs()  # [B, K_b, K_m]
    # Combined similarity matrix per batch: M[b, j, i]
    M = cosU * cosV  # [B, K, K]

    B, Kb, Km = M.shape
    Kc = min(Kb, Km)
    echo_out = U_b.new_zeros((B, Kc))

    # Greedy permutation alignment per batch: maximize sum of matches
    for b in range(B):
        Mb = M[b]
        # Track available rows (batch directions j)
        used = torch.zeros(Kb, dtype=torch.bool, device=Mb.device)
        # Order columns (mean directions i) by their best available match descending
        best_per_col, _ = Mb.max(dim=0)  # [K]
        order = torch.argsort(best_per_col, descending=True)
        assigned = torch.full((Km,), -1, dtype=torch.long, device=Mb.device)
        for i in order.tolist():
            scores = Mb[:, i].clone()
            scores[used] = -1.0  # mask used rows
            j = int(torch.argmax(scores).item())
            if scores[j] < 0:
                continue
            assigned[i] = j
            used[j] = True
        # Build spectral echo vector for first Kc columns of mean
        for i in range(Kc):
            j = int(assigned[i].item())
            if j >= 0:
                echo_out[b, i] = Mb[j, i]
            else:
                echo_out[b, i] = 0.0
    return echo_out


def compute_spectral_echo_and_alignment(
    mb_svd: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mean_svd: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute spectral echo and the alignment indices j*(i) per batch.

    Returns:
        echo_out: [B, Kc]
        assign_idx: [B, Kc] long, where assign_idx[b, i] = j index in minibatch aligning to mean i, or -1.
    """
    U_b, _, Vh_b = mb_svd
    U_m, _, Vh_m = mean_svd

    K = min(U_b.shape[-1], U_m.shape[-1], Vh_b.shape[-2], Vh_m.shape[-2])
    U_b = U_b[..., :K]
    U_m = U_m[..., :K]
    Vh_b = Vh_b[..., :K, :]
    Vh_m = Vh_m[..., :K, :]

    B = U_b.shape[0]
    cosU = torch.einsum('bhk,hq->bkq', U_b, U_m).abs()
    V_b = Vh_b.transpose(-2, -1)
    V_m = Vh_m.transpose(-2, -1)
    cosV = torch.einsum('bwk,wq->bkq', V_b, V_m).abs()
    M = cosU * cosV  # [B, K, K]

    B, Kb, Km = M.shape
    Kc = min(Kb, Km)
    echo_out = U_b.new_zeros((B, Kc))
    assign_idx = torch.full((B, Kc), -1, dtype=torch.long, device=U_b.device)

    for b in range(B):
        Mb = M[b]
        used = torch.zeros(Kb, dtype=torch.bool, device=Mb.device)
        best_per_col, _ = Mb.max(dim=0)
        order = torch.argsort(best_per_col, descending=True)
        assigned = torch.full((Km,), -1, dtype=torch.long, device=Mb.device)
        for i in order.tolist():
            scores = Mb[:, i].clone()
            scores[used] = -1.0
            j = int(torch.argmax(scores).item())
            if scores[j] < 0:
                continue
            assigned[i] = j
            used[j] = True
        for i in range(Kc):
            j = int(assigned[i].item())
            if j >= 0:
                echo_out[b, i] = Mb[j, i]
                assign_idx[b, i] = j
            else:
                echo_out[b, i] = 0.0
                assign_idx[b, i] = -1
    return echo_out, assign_idx

def compute_reverb_echo_and_alignment(
    mb_svd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fast & robust alignment:
      - Build per-replica similarity M_b = |U_b^T U_ref| * |V_b^T V_ref|
      - Average M_b across replicas (excluding the reference) -> M_avg
      - Solve ONE LAP on the top-K block of M_avg to get a global permutation
      - Reuse that permutation for all replicas; then do per-replica sign correction
      - Return median-of-replica echoes and the assignment indices

    Env knobs:
      REVERB_MAX_KC: int cap on Kc (default 256)
      REVERB_JITTER: float jitter magnitude added to M before LAP (default 1e-12)
    """
    import time
    import numpy as np

    # Optional SciPy LAP (preferred)
    try:
        from scipy.optimize import linear_sum_assignment as _scipy_lsa
    except Exception:
        _scipy_lsa = None

    def _greedy_perm(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One-to-one greedy: columns by descending best score; pick best available row."""
        K = M.shape[0]
        # tiny deterministic jitter to break ties
        rng = np.random.RandomState(0)
        J = M + (1e-12 * rng.standard_normal(M.shape))
        col_order = np.argsort(-J.max(axis=0))
        used = np.zeros(K, dtype=bool)
        row_for_col = -np.ones(K, dtype=int)
        for i in col_order:
            j = int(np.argmax(J[:, i]))
            if not used[j]:
                row_for_col[i] = j
                used[j] = True
        # fill any leftovers arbitrarily
        free = np.where(~used)[0].tolist()
        for i in range(K):
            if row_for_col[i] < 0:
                row_for_col[i] = free.pop()
        row_ind = row_for_col
        col_ind = np.arange(K, dtype=int)
        return row_ind, col_ind

    # ---- unpack & basic guards ----
    U_b, _, Vh_b = mb_svd  # U_b: [B,H,K], s: [B,K], Vh_b: [B,K,W]
    with torch.no_grad():
        if U_b.dtype == torch.bfloat16:
            U_b = U_b.float()
        if Vh_b.dtype == torch.bfloat16:
            Vh_b = Vh_b.float()
        V_b = Vh_b.transpose(-2, -1)  # [B,W,K]

        B, H, K = U_b.shape
        _, W, Kv = V_b.shape
        Kc = int(min(K, Kv))

        # cap Kc aggressively; tiny singular directions are noisy & expensive
        import os
        K_cap = int(os.getenv("REVERB_MAX_KC", "256"))
        Kc = min(Kc, max(1, K_cap))

        U_b = U_b[:, :, :Kc]        # [B,H,Kc]
        V_b = V_b[:, :, :Kc]        # [B,W,Kc]

        # reference replica
        ref = 0
        U_ref = U_b[ref]            # [H,Kc]
        V_ref = V_b[ref]            # [W,Kc]

        # ---- build M_b per replica against ref and form a consensus M_avg ----
        # M_b = |U_b^T U_ref| * |V_b^T V_ref|  -> shape [Kc,Kc]
        # exclude b==ref from the average to avoid trivial identity dominance
        Ms = []
        for b in range(B):
            if b == ref:
                continue
            MU = (U_b[b].transpose(0, 1) @ U_ref).abs()  # [Kc,Kc]
            MV = (V_b[b].transpose(0, 1) @ V_ref).abs()  # [Kc,Kc]
            Ms.append((MU * MV).cpu().numpy())
        if not Ms:
            # degenerate B=1: identity alignment
            assign_idx = torch.arange(Kc, device=U_b.device, dtype=torch.long).view(1, Kc).repeat(1, 1)
            # aligned copies are trivial; echoes are all ones
            spectral_echo = torch.ones(Kc, device=U_b.device, dtype=U_b.dtype)
            return spectral_echo, assign_idx

        M_avg = np.mean(np.stack(Ms, axis=0), axis=0)  # [Kc,Kc]

        # trim to the leading block (already Kc-capped), add jitter, and solve one LAP
        jitter = float(os.getenv("REVERB_JITTER", "1e-12"))
        M_block = M_avg.copy()
        if jitter > 0.0:
            # deterministic jitter
            rng = np.random.RandomState(0)
            M_block = M_block + jitter * rng.standard_normal(M_block.shape)

        # maximize similarity -> min cost = max(M)-M
        Mmax = float(M_block.max()) if M_block.size else 0.0
        cost = (Mmax - M_block)

        t_lap0 = time.time()
        try:
            if _scipy_lsa is not None:
                r_ind, c_ind = _scipy_lsa(cost)   # returns row_ind, col_ind
            else:
                r_ind, c_ind = _greedy_perm(M_block)
        except Exception:
            r_ind, c_ind = _greedy_perm(M_block)
        lap_ms = (time.time() - t_lap0) * 1000.0

        # We want j*(i): for each column i (ref index), which row j (b index) to take.
        # SciPy returns pairs (row, col) that minimize cost; ensure full 0..Kc-1 coverage.
        # Build inverse map: inv[col] = row
        inv = np.empty(Kc, dtype=np.int64)
        inv[c_ind] = r_ind
        perm = torch.from_numpy(inv).to(U_b.device, dtype=torch.long)  # [Kc]

        # alignment indices for every replica are identical permutation (consensus)
        assign_idx = perm.view(1, -1).repeat(B, 1)  # [B,Kc]

        # ---- sign correction per replica (keep your logic) ----
        # After permuting columns to reference order, correct signs so overlaps are positive.
        for b in range(B):
            idx = assign_idx[b]  # [Kc]
            u_overlap = torch.sum(U_b[b].gather(1, idx.view(1, -1).expand(H, -1)) * U_ref, dim=0)
            v_overlap = torch.sum(V_b[b].gather(1, idx.view(1, -1).expand(W, -1)) * V_ref, dim=0)
            sgn = torch.sign(u_overlap * v_overlap)
            sgn[sgn == 0] = 1.0
            U_b[b].index_copy_(1, idx, U_b[b][:, idx] * sgn)
            V_b[b].index_copy_(1, idx, V_b[b][:, idx] * sgn)

        # ---- gather aligned bases and compute echoes via triple–OLS (your solver) ----
        aligned_U = torch.stack([U_b[b].index_select(1, assign_idx[b]) for b in range(B)], dim=0)  # [B,H,Kc]
        aligned_V = torch.stack([V_b[b].index_select(1, assign_idx[b]) for b in range(B)], dim=0)  # [B,W,Kc]

    # compute echoes; median across replicas for robustness
    t_reverb = time.time()
    echoes = solve_for_spectral_echo_using_reverb(aligned_U, aligned_V)  # [B,Kc]
    spectral_echo = echoes.median(dim=0).values.clamp(0.0, 1.0)          # [Kc]
    t_end = time.time()

    # logging (matches your style)
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    log_from_rank(f"reverb-align(consensus): B={B}, Kc={Kc}, lap_ms={lap_ms:.1f}, "
                  f"solve_ms={(t_end - t_reverb)*1000.0:.1f}", rank)

    return spectral_echo, assign_idx

def gather_aligned_singulars(minibatch_singular_values: torch.Tensor,
                             alignment_indices: torch.Tensor) -> torch.Tensor:
    """Gather per-batch singular values aligned to mean indices via alignment indices.

    Args:
        minibatch_singular_values: [B, K] singulars per batch
        alignment_indices: [B, Kc] long indices j*(i)
    Returns:
        aligned_sv: [B, Kc] with s[b, j*(i)] (zeros where assignment is -1)
    """
    with torch.no_grad():
        if (alignment_indices < 0).any():
            raise RuntimeError("Negative alignment index encountered; alignment must fully assign columns.")
        B, K = minibatch_singular_values.shape
        _, Kc = alignment_indices.shape
        gather = torch.gather(minibatch_singular_values, dim=1, index=alignment_indices)
        return gather


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

def estimate_gradient_noise_sigma2(
    per_minibatch_gradient: torch.Tensor,
    mean_gradient: torch.Tensor
) -> float:
    """Unbiased per-entry variance estimate from per-minibatch gradients.

    sigma^2_hat = (1/((B-1) m n)) * sum_i ||G_i - G_bar||_F^2
    """
    with torch.no_grad():
        B, m, n = per_minibatch_gradient.shape
        diffs = per_minibatch_gradient - mean_gradient.unsqueeze(0)
        # Frobenius norm squared per minibatch, then sum over batches
        frob2_per_mb = torch.sum(diffs * diffs, dim=(-2, -1))  # [B]
        total_frob2 = torch.sum(frob2_per_mb)
        return float(total_frob2 / max(1, (B - 1) * m * n))
 
def _logspace_weights_np(s: np.ndarray, nbins: int = 32) -> np.ndarray:
    log_s = np.log(s)
    edges = np.linspace(log_s.min(), log_s.max(), nbins + 1)
    idx = np.clip(np.digitize(log_s, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins).astype(np.float64)
    w = 1.0 / np.maximum(counts[idx], 1.0)
    return w * (len(w) / w.sum())


def _spectral_echo_model_np(s: np.ndarray, tau2: float) -> np.ndarray:
    return 1.0 / (1.0 + (tau2 / (s * s)))


def fit_empirical_phase_constant_tau2(
    minibatch_singular_values: torch.Tensor,
    spectral_echo: torch.Tensor,
    eps: float = 1e-12,
    nbins: int = 32,
) -> float:
    """Fit tau^2 in echo(s)=1/(1+tau^2/s^2) with REQUIRED SciPy nonlinear LS.

    - `minibatch_singular_values`: [B, K]
    - `spectral_echo`: [Kc] (per-direction, aggregated across replicas)
    We pair the first Kc singulars per replica with the Kc echoes, tile echoes across B,
    and fit a single tau^2.

    Raises:
        RuntimeError on any missing SciPy or fitting failure.
        ValueError on shape mismatch.
    """
    if _scipy_curve_fit is None:
        raise RuntimeError("SciPy is required for fit_empirical_phase_constant_tau2 but is not available.")

    with torch.no_grad():
        if minibatch_singular_values.ndim != 2:
            raise ValueError(f"minibatch_singular_values must be [B,K], got {tuple(minibatch_singular_values.shape)}")
        if spectral_echo.ndim != 1:
            raise ValueError(f"spectral_echo must be [Kc], got {tuple(spectral_echo.shape)}")

        B, K = minibatch_singular_values.shape
        Kc = int(spectral_echo.numel())
        if Kc > K:
            raise ValueError(f"spectral_echo length Kc={Kc} exceeds K={K}.")

        # Pair only the first Kc directions with the Kc echoes
        s_use = minibatch_singular_values[:, :Kc]             # [B, Kc]
        echo_use = spectral_echo.view(1, Kc).expand(B, Kc)    # [B, Kc]

        s = s_use.reshape(-1).detach().cpu().numpy().astype(np.float64)       # [B*Kc]
        echo = echo_use.reshape(-1).detach().cpu().numpy().astype(np.float64) # [B*Kc]
        if s.size == 0:
            raise RuntimeError("No samples to fit tau^2 (B*Kc == 0).")

        # Guard against zeros to keep the model well-defined
        s = np.clip(s, eps, None)
        echo = np.clip(echo, eps, 1.0 - eps)

        # Reweight in log-space to avoid over-emphasizing dense low-s regions
        w = _logspace_weights_np(s, nbins=nbins)             # length = B*Kc
        sigma = 1.0 / np.sqrt(np.maximum(w, eps))

        # Moment-based positive initialization
        tau2_init = float(np.mean((s * s) * (1.0 / echo - 1.0)))
        tau2_init = max(tau2_init, eps)

        (tau2_hat,), _ = _scipy_curve_fit(
            _spectral_echo_model_np,
            xdata=s,
            ydata=echo,
            p0=(tau2_init,),
            bounds=(eps, 1e40),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20000,
        )

        tau2_hat = float(tau2_hat)
        if not np.isfinite(tau2_hat) or tau2_hat < 0.0:
            raise RuntimeError(f"Invalid tau^2 estimate: {tau2_hat}")

        return tau2_hat


def fit_empirical_noise_to_phase_slope_kappa(
    gradient_noise_sigma2: GPTLayerProperty,
    empirical_phase_constant_tau2: GPTLayerProperty
) -> float:
    """Fit kappa in tau^2 ≈ kappa * sigma^2 using per-layer points.

    Input dicts map (param_type, layer) -> scalar. We perform a
    through-origin least squares fit across available layers.
    """
    with torch.no_grad():
        # Intersect keys to align x,y
        keys = [k for k in empirical_phase_constant_tau2.keys() if k in gradient_noise_sigma2]
        if not keys:
            return float('nan')
        sigma2s = torch.tensor([float(gradient_noise_sigma2[k]) for k in keys], dtype=torch.float64).reshape(-1, 1)
        tau2s = torch.tensor([float(empirical_phase_constant_tau2[k]) for k in keys], dtype=torch.float64).reshape(-1, 1)
        sol = torch.linalg.lstsq(sigma2s, tau2s).solution  # [1,1]
        return float(sol.squeeze())

def precompute_theoretical_noise_to_phase_slope_kappa(
    aspect_ratio_beta: float
) -> float:
    """
    """
    pass

def aspect_ratio_beta(matrix: torch.Tensor) -> float:
    return float(matrix_shape_beta(matrix.shape))

def solve_for_spectral_echo_using_reverb(
    left_bases_U: torch.Tensor,   # (r, H, Kc)  aligned left singular bases per replica
    right_bases_V: torch.Tensor,  # (r, W, Kc)  aligned right singular bases per replica
    noise_quantile: float = 0.10, # lower-tail quantile to set per-direction noise floor
    tau_mult: float = 4.0,        # multiplier on the noise floor for denominator guarding
    weight_power: float = 2.0,    # use |Z_ab|**weight_power as weights
) -> torch.Tensor:                # returns echoes with shape (r, Kc)
    """
    Triple–OLS with robust weights and denominator guarding (no NaNs).
    Uses per-direction spectral reverb Z_ab = <U_a,U_b>*<V_a,V_b>, weights w_ab = |Z_ab|^p,
    and divides by Z_ab + sign(Z_ab)*tau_dir to avoid small-denominator blowups.
    """
    # Shapes
    r, _, Kc = left_bases_U.shape

    # --- small-B guard: need r>=3 for triples; else pairwise fallback
    if r < 3:
        ZU = torch.einsum('aik,bik->abk', left_bases_U, left_bases_U)  # (r,r,Kc)
        ZV = torch.einsum('aik,bik->abk', right_bases_V, right_bases_V)  # (r,r,Kc)
        Z  = ZU * ZV                                                    # (r,r,Kc)
        mask = ~torch.eye(r, dtype=torch.bool, device=Z.device)
        Z_off = Z[mask].reshape(r*(r-1), Kc).abs().clamp_min(0.0)
        echoes = torch.sqrt(Z_off.median(dim=0).values.clamp_min(0.0))  # (Kc,)
        return echoes.unsqueeze(0).expand(r, -1)

    # Spectral reverb across replicas and directions: Z[a,b,k] = <U_a,U_b> * <V_a,V_b>
    ZU = torch.einsum('aik,bik->abk', left_bases_U, left_bases_U)       # (r,r,Kc)
    ZV = torch.einsum('aik,bik->abk', right_bases_V, right_bases_V)     # (r,r,Kc)
    spectral_reverb_Z = ZU * ZV                                         # (r,r,Kc)

    absZ = spectral_reverb_Z.abs()                                      # (r,r,Kc)

    # Quantile over (a,b) entries for each k; flatten (a,b) first (no contiguity assumptions)
    absZ_flat = absZ.reshape(-1, Kc)                                    # (r*r, Kc)
    tau_dir = tau_mult * torch.quantile(absZ_flat, noise_quantile, dim=0)  # (Kc,)

    # Build numerators/denominators for triples r_{a,b->p}
    idx = torch.arange(r, device=left_bases_U.device)
    Z_ap = spectral_reverb_Z[:, idx, :].permute(1, 0, 2).unsqueeze(2)   # (p,a,1,Kc) = Z[a,p,:]
    Z_pb = spectral_reverb_Z[idx, :, :].unsqueeze(2)                     # (p,1,b,Kc) = Z[p,b,:]
    numer = Z_ap * Z_pb                                                  # (p,a,b,Kc)

    denom = spectral_reverb_Z.unsqueeze(0)                               # (1,a,b,Kc)
    sgn = torch.where(denom >= 0, 1.0, -1.0)                             # (1,a,b,Kc)
    denom_safe = denom + sgn * tau_dir.view(1, 1, 1, Kc)                 # (1,a,b,Kc)

    # Valid triple mask: a!=b, a!=p, b!=p
    a_ne_b = idx.view(1, -1, 1) != idx.view(1, 1, -1)                   # (1,a,b)
    a_ne_p = idx.view(-1, 1, 1) != idx.view(1, -1, 1)                   # (p,a,1)
    b_ne_p = idx.view(-1, 1, 1) != idx.view(1, 1, -1)                   # (p,1,b)
    valid = (a_ne_b & a_ne_p & b_ne_p).unsqueeze(-1)                     # (p,a,b,1)

    # Robust weights w_ab = |Z_ab|**weight_power
    weights = absZ.pow(weight_power).unsqueeze(0)                        # (1,a,b,Kc)

    # Weighted Triple–OLS over (a,b)
    triple_vals = torch.where(valid, numer / denom_safe, 0.0)            # (p,a,b,Kc)
    weighted_sum = (weights * triple_vals).sum(dim=(1, 2))               # (p,Kc)
    weight_sum   = (weights * valid).sum(dim=(1, 2)).clamp_min(1e-12)    # (p,Kc)
    s_hat = weighted_sum / weight_sum                                    # (p,Kc) ~ ζ_p^2

    return torch.sqrt(s_hat.clamp_min(0.0))                              # echoes (r,Kc)
