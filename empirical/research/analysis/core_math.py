"""
Core mathematical functions for gradient analysis.

This module contains all the fundamental mathematical operations needed across
the analysis pipeline. It provides both numpy and torch implementations where
needed, with consistent interfaces.
"""

 
from typing import Union, Tuple, Dict, List
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
    Fast, robust SVD for [H,W] or [B,H,W].

    - Promotes bf16 -> f32 for numerical stability.
    - Ensures contiguity for linalg kernels.
    - JIT offloads CPU tensors to CUDA for compute (if available), then
      returns results on CPU to keep GPU memory footprint low across layers.
    """
    with torch.no_grad():
        x = tensor
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.contiguous()

        need_offload = (x.device.type != 'cuda') and torch.cuda.is_available()
        if need_offload:
            x = x.cuda(non_blocking=True)

        U, s, Vh = torch.linalg.svd(x, full_matrices=False)

        # Move back to CPU if we offloaded to keep memory steady over many layers
        if need_offload:
            U = U.cpu()
            s = s.cpu()
            Vh = Vh.cpu()
        return U, s, Vh


########################
# Removed legacy echo/alignment helpers
########################


########################
# Core convenience functions
########################


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


########################
# Noise estimation and tau^2 fitting
########################

def estimate_gradient_noise_sigma2(
    per_replicate_gradient: torch.Tensor,
    mean_gradient: torch.Tensor
) -> float:
    """Unbiased per-entry variance estimate from per-replica gradients.

    sigma^2_hat = (1/((R-1) m n)) * sum_i ||G_i - G_bar||_F^2
    """
    with torch.no_grad():
        B, m, n = per_replicate_gradient.shape
        diffs = per_replicate_gradient - mean_gradient.unsqueeze(0)
        # Frobenius norm squared per replicate, then sum over replicates
        frob2_per_rep = torch.sum(diffs * diffs, dim=(-2, -1))  # [B]
        total_frob2 = torch.sum(frob2_per_rep)
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

    - `minibatch_singular_values` (replica singulars): [R, K]
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

########################
# Spectral reverb solver (tuning-free debiased variant)
########################

def solve_for_spectral_echo_using_reverb(
    left_bases_U: torch.Tensor,   # (r, H, Kc)  aligned left singular bases per replica
    right_bases_V: torch.Tensor,  # (r, W, Kc)  aligned right singular bases per replica
    noise_quantile: float = 0.02, # UNUSED in this debiased, tuning-free variant
    tau_mult: float = 1.5,        # UNUSED in this debiased, tuning-free variant
    weight_power: float = 2.0,    # UNUSED in this debiased, tuning-free variant
) -> torch.Tensor:                # returns echoes with shape (r, Kc)
    """
    Debiased spectral-echo estimator (no tunable floors, no weights).

    Model: independent replicas of \hat{G} = G + E (isotropic), alignment already applied.
    We estimate the per-direction overlap statistics of Z_{abk} directly from data and
    apply a Fieller/delta-method bias correction to the triple ratio before the sqrt.

    Notation:
        Z_{abk} = <U_a[:,k], U_b[:,k]> * <V_a[:,k], V_b[:,k]>
        r_{a,b->p,k} = (Z_{apk} * Z_{pbk}) / Z_{abk}

    Steps per direction k:
        1) Compute all off-diagonal Z_{abk} across replicas (a != b).
        2) Estimate μ_Z(k) = mean(Z_{abk}) and v_Z(k) = var(Z_{abk}) from those pairs.
        3) For each pivot p, average r_{a,b->p,k} over valid (a,b) without any floor/weights:
               rbar_{p,k} = mean_{a!=b, a!=p, b!=p} r_{a,b->p,k}
        4) Bias-correct the mean ratio using Fieller (second order):
               rtilde_{p,k} = rbar_{p,k} / (1 + v_Z(k) / μ_Z(k)^2)
        5) Echo per-pivot: ζ̂_{p,k} = sqrt(max(rtilde_{p,k}, 0))

    Returns:
        echoes: (r, Kc) tensor with per-pivot echoes; upstream can take median over pivots.
    """
    with torch.no_grad():
        U = left_bases_U
        V = right_bases_V
        if U.dtype == torch.bfloat16: U = U.float()
        if V.dtype == torch.bfloat16: V = V.float()

        r, H, Kc = U.shape
        _, W, Kc2 = V.shape
        assert Kc == Kc2, "Left/Right Kc mismatch"

        # --- small-r guard: need r >= 3 for triples; else pairwise fallback
        if r < 3:
            ZU = torch.einsum('aik,bik->abk', U, U)     # (r,r,Kc)
            ZV = torch.einsum('aik,bik->abk', V, V)     # (r,r,Kc)
            Z  = ZU * ZV                                # (r,r,Kc)
            mask = ~torch.eye(r, dtype=torch.bool, device=Z.device)
            Z_off = Z[mask].reshape(r*(r-1), Kc).abs().clamp_min(0.0)
            echoes = torch.sqrt(Z_off.median(dim=0).values.clamp_min(0.0))  # (Kc,)
            return echoes.unsqueeze(0).expand(r, -1)

        # --- Build Z_{abk} across replicas and directions
        # Z[a,b,k] = <U_a[:,k],U_b[:,k]> * <V_a[:,k],V_b[:,k}>
        ZU = torch.einsum('aik,bik->abk', U, U)         # (r,r,Kc)
        ZV = torch.einsum('aik,bik->abk', V, V)         # (r,r,Kc)
        Z  = ZU * ZV                                    # (r,r,Kc)

        # --- Estimate μ_Z(k) and v_Z(k) from off-diagonals (a != b), per direction k
        diag_mask = torch.eye(r, dtype=torch.bool, device=Z.device).unsqueeze(-1)  # (r,r,1)
        Z_off = Z.masked_select(~diag_mask).view(r*(r-1), Kc)  # (r*(r-1), Kc)
        # Unbiased sample mean and variance (per direction)
        muZ = Z_off.mean(dim=0)                                       # (Kc,)
        # Use unbiased var; guard tiny negatives from numerical noise
        if Z_off.size(0) > 1:
            vZ  = Z_off.var(dim=0, unbiased=True).clamp_min(0.0)     # (Kc,)
        else:
            vZ  = torch.zeros_like(muZ)

        # Avoid division by zero in correction factor
        eps = torch.finfo(Z.dtype).eps
        muZ_safe = torch.where(muZ.abs() < eps, torch.sign(muZ) * eps + (muZ == 0).float() * eps, muZ)
        corr = 1.0 + (vZ / (muZ_safe * muZ_safe))                    # (Kc,)

        # --- Form all triple ratios r_{a,b->p,k} without any floors/weights
        # Use vectorized construction:
        #   numer[p,a,b,k] = Z[a,p,k] * Z[p,b,k]
        #   denom[a,b,k]   = Z[a,b,k]
        idx = torch.arange(r, device=Z.device)
        Z_ap = Z[:, idx, :].permute(1, 0, 2).unsqueeze(2)   # (p,a,1,Kc) = Z[a,p,k]
        Z_pb = Z[idx, :, :].unsqueeze(2)                    # (p,1,b,Kc) = Z[p,b,k]
        numer = Z_ap * Z_pb                                 # (p,a,b,Kc)
        denom = Z.unsqueeze(0)                              # (1,a,b,Kc)

        # Valid triple mask: a!=b, a!=p, b!=p
        a_ne_b = idx.view(1, -1, 1) != idx.view(1, 1, -1)   # (1,a,b)
        a_ne_p = idx.view(-1, 1, 1) != idx.view(1, -1, 1)   # (p,a,1)
        b_ne_p = idx.view(-1, 1, 1) != idx.view(1, 1, -1)   # (p,1,b)
        valid  = (a_ne_b & a_ne_p & b_ne_p).unsqueeze(-1)   # (p,a,b,1)

        # Purely numerical protection against exact zeros (machine-precision only; NOT a tunable floor)
        tiny = eps
        denom_safe = torch.where(denom.abs() < tiny,
                                 denom + torch.sign(denom) * tiny,
                                 denom)

        triples = torch.where(valid, numer / denom_safe, torch.zeros_like(numer))  # (p,a,b,Kc)

        # Average across valid (a,b) for each pivot p, per direction k
        valid_f = valid.float()
        cnt = valid_f.sum(dim=(1, 2)).clamp_min(1.0)       # (p,1)
        rbar = (triples.sum(dim=(1, 2)) / cnt).squeeze(1)  # (p,Kc)

        # --- Fieller/delta-method debiasing: divide by (1 + vZ / muZ^2)
        # Broadcast corr (Kc,) over pivots p
        rtilde = rbar / corr.unsqueeze(0)                  # (p,Kc)

        # --- Echo per pivot: sqrt of positive part
        echoes = torch.sqrt(rtilde.clamp_min(0.0))         # (p,Kc) == (r,Kc)

        return echoes

def get_spectral_echoes_from_empirical_gradients(empirical_gradients: torch.Tensor) -> torch.Tensor:
    """
    Args:
        empirical_gradients: (R, N, M) tensor of empirical gradients for R replicates.
    Returns:
        echoes_zeta: (R, D) tensor of spectral echoes, where D = min(N, M).
    """
    aligned_U, _, aligned_V = get_aligned_svds(empirical_gradients)  # U:(R,N,D), V:(R,M,D)
    num_replicates_R, _, dim_D = aligned_U.shape
    U_stacked = aligned_U.permute(2, 0, 1)  # (D,R,N)
    V_stacked = aligned_V.permute(2, 0, 1)  # (D,R,M)

    # Build replica Gram matrices per direction: G[d] = U[d] @ U[d]^T and V[d] @ V[d]^T
    # Einsum: sum over feature dimension (i), keep both replica indices (r,s)
    gram_U = torch.einsum('dri,dsi->drs', U_stacked, U_stacked)        # (D,R,R)
    gram_V = torch.einsum('dri,dsi->drs', V_stacked, V_stacked)        # (D,R,R)
    reverb_tensor_Z = gram_U * gram_V                                    # (D,R,R)
    reverb_tensor_Z = reverb_tensor_Z * (1 - torch.eye(num_replicates_R, device=reverb_tensor_Z.device, dtype=reverb_tensor_Z.dtype).unsqueeze(0))

    zz = reverb_tensor_Z @ reverb_tensor_Z                               # (D,R,R)
    numerator = (zz * reverb_tensor_Z.transpose(-1, -2)).sum(dim=-1)     # (D,R)
    denominator = (reverb_tensor_Z * reverb_tensor_Z).sum(dim=(-2, -1)).clamp_min(torch.finfo(reverb_tensor_Z.dtype).eps)  # (D,)
    echoes_sq = numerator / denominator.unsqueeze(-1)                     # (D,R)
    echoes_zeta = torch.sqrt(echoes_sq.clamp_min(0.0)).transpose(0, 1)    # (R,D)
    return echoes_zeta

def get_aligned_svds(empirical_gradients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Greedy-only alignment of SVDs across replicates to a fixed reference (replica 0).
    No Hungarian, no SciPy. Uses safe_svd.

    Args:
        empirical_gradients: (R, N, M) bf16/fp32
    Returns:
        aligned_U: (R, N, D)
        aligned_S_vals: (R, D)
        aligned_V: (R, M, D)
    """
    # Promote dtype; compute batched SVDs
    X = empirical_gradients.float() if empirical_gradients.dtype == torch.bfloat16 else empirical_gradients
    U_all, S_vals, Vh_all = safe_svd(X)              # U:(R,N,D), S:(R,D), Vh:(R,D,M)
    V_all = Vh_all.transpose(-2, -1)                 # (R,M,D)
    R, N, D = U_all.shape

    U_ref, V_ref = U_all[0], V_all[0]

    def _greedy_perm(sim: torch.Tensor) -> torch.Tensor:
        """sim: (D,D) >=0; returns perm idx p s.t. columns map j->row p[j]."""
        K = sim.shape[0]
        # tiny deterministic jitter to avoid exact ties
        eps = 1e-12 if sim.dtype in (torch.float32, torch.float64) else 1e-6
        jitter = eps * torch.linspace(0, 1, steps=sim.numel(), device=sim.device, dtype=sim.dtype).reshape_as(sim)
        scores = sim + jitter
        col_order = torch.argsort(scores.max(dim=0).values, descending=True)
        used = torch.zeros(K, dtype=torch.bool, device=sim.device)
        row_for_col = torch.full((K,), -1, dtype=torch.long, device=sim.device)
        for j in col_order:
            i = torch.argmax(scores[:, j])
            if not used[i]:
                row_for_col[j] = i
                used[i] = True
        free = torch.nonzero(~used, as_tuple=False).flatten()
        kf = 0
        for j in range(K):
            if row_for_col[j] < 0:
                row_for_col[j] = free[kf]; kf += 1
        return row_for_col

    aligned_U = torch.empty_like(U_all)
    aligned_V = torch.empty_like(V_all)
    aligned_S_vals = torch.empty_like(S_vals)
    aligned_U[0], aligned_V[0], aligned_S_vals[0] = U_ref, V_ref, S_vals[0]

    for i in range(1, R):
        # similarity per column = |<u_i,u_ref>|*|<v_i,v_ref>|
        sim_U = (U_all[i].transpose(0, 1) @ U_ref).abs()    # (D,D)
        sim_V = (V_all[i].transpose(0, 1) @ V_ref).abs()    # (D,D)
        sim = sim_U * sim_V
        perm = _greedy_perm(sim)

        U_perm = U_all[i][:, perm]
        V_perm = V_all[i][:, perm]
        S_perm = S_vals[i][perm]

        # joint sign fix
        u_overlap = (U_perm * U_ref).sum(dim=0)
        v_overlap = (V_perm * V_ref).sum(dim=0)
        sgn = torch.sign(u_overlap * v_overlap)
        sgn[sgn == 0] = 1.0

        aligned_U[i] = U_perm * sgn.unsqueeze(0)
        aligned_V[i] = V_perm * sgn.unsqueeze(0)
        aligned_S_vals[i] = S_perm

    return aligned_U, aligned_S_vals, aligned_V
