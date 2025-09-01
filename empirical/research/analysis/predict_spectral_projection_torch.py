import math
from typing import Tuple, Union

import torch

# We keep tiny epsilons to avoid hitting MP support endpoints numerically
_EPS = 1e-6

def matrix_shape_beta(shape: Union[Tuple[int, int], torch.Size]) -> float:
    """
    Compute beta = min(n,m)/max(n,m) from a (n,m) shape tuple or torch.Size.
    Pure-Python, safe to call from both torch and numpy codepaths.
    """
    n, m = int(shape[0]), int(shape[1])
    a, b = (n, m) if n < m else (m, n)
    return float(a) / float(b)

@torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=False)
def _mp_quantiles_lambda(probs: torch.Tensor, beta: float, grid_size: int = 2048, device: torch.device | None = None) -> torch.Tensor:
    """
    Quantiles of the Marchenko–Pastur law for eigenvalues λ = s^2/σ^2.
    Returns tensor of shape [P] with λ-quantiles for given probs ∈ (0,1).
    Computed by numerically inverting the CDF on a fixed grid (compile-friendly).
    """
    if device is None:
        device = probs.device
    sqrtb = float(math.sqrt(beta))
    lam_m = (1.0 - sqrtb) ** 2
    lam_p = (1.0 + sqrtb) ** 2
    # avoid singular endpoints; keep shapes static
    eps = max(1e-12, 1e-6 * lam_p)
    x0 = lam_m + eps
    x1 = lam_p - eps
    # fixed-size grid (static for compilation)
    grid = torch.linspace(x0, x1, grid_size, device=device, dtype=torch.float32)  # [G]
    # MP pdf on λ (safe near edges because of eps)
    num = torch.sqrt(torch.clamp((lam_p - grid) * (grid - lam_m), min=0.0))       # [G]
    den = (2.0 * math.pi * beta) * torch.clamp(grid, min=1e-30)                   # [G]
    pdf = num / den                                                                # [G]
    # CDF via trapezoid rule
    dx = (x1 - x0) / (grid_size - 1)
    inc = 0.5 * (pdf[1:] + pdf[:-1]) * dx                                         # [G-1]
    cdf = torch.cumsum(inc, dim=0)
    cdf = torch.cat([torch.zeros(1, device=device, dtype=pdf.dtype), cdf], dim=0) # [G]
    cdf = cdf / torch.clamp(cdf[-1], min=1e-30)
    # inverse CDF by interpolation
    # probs: [P]  -> idx in [1..G-1]
    idx_hi = torch.searchsorted(cdf, probs)                                        # [P]
    idx_hi = torch.clamp(idx_hi, 1, grid_size - 1)
    idx_lo = idx_hi - 1
    c_lo = torch.take(cdf, idx_lo)
    c_hi = torch.take(cdf, idx_hi)
    x_lo = torch.take(grid, idx_lo)
    x_hi = torch.take(grid, idx_hi)
    t = (probs - c_lo) / torch.clamp(c_hi - c_lo, min=1e-30)
    lam_q = x_lo + t * (x_hi - x_lo)                                              # [P]
    return lam_q

@torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=False)
def _empirical_quantiles_from_sorted(s2_asc: torch.Tensor, probs: torch.Tensor, k_eff: torch.Tensor | None = None) -> torch.Tensor:
    """
    Empirical quantiles for each batch row from an ASCENDING-sorted tensor of s^2.
    s2_asc: [B, K] ascending; probs: [P]; k_eff: [B] (effective count to use, ≤ K).
    Returns [B, P] with linear interpolation between adjacent order stats.
    """
    B, K = s2_asc.shape
    P = probs.numel()
    if k_eff is None:
        k_eff = torch.full((B,), K, device=s2_asc.device, dtype=torch.int64)
    else:
        k_eff = torch.clamp(k_eff.to(torch.int64), 1, K)
    # positions in [0, k_eff-1]
    pos = probs.unsqueeze(0) * (k_eff.unsqueeze(1).to(s2_asc.dtype) - 1.0)        # [B,P]
    lo = torch.floor(pos)
    hi = torch.ceil(pos)
    w = (pos - lo).to(s2_asc.dtype)
    lo_idx = torch.clamp(lo.to(torch.int64), 0, K - 1)                            # [B,P]
    hi_idx = torch.clamp(hi.to(torch.int64), 0, K - 1)
    # gather via take_along_dim (batch-safe)
    q_lo = torch.take_along_dim(s2_asc, lo_idx, dim=-1)                           # [B,P]
    q_hi = torch.take_along_dim(s2_asc, hi_idx, dim=-1)
    q = (1.0 - w) * q_lo + w * q_hi
    return q                                                                       # [B,P]


def _score_fixed_mask(
    svals_sq: torch.Tensor,  # [B, K], squared singular values
    sigma: torch.Tensor,     # [B]    or [B, 1]
    beta: float,
    mask: torch.Tensor       # [B, K] boolean: which entries are "inliers"
) -> torch.Tensor:
    """
    Score F_I(sigma) with a FIXED inlier mask I.
    Monotone increasing in sigma for any fixed mask.
    Returns shape [B].
    """
    # broadcast sigma to [B, 1]
    sigma = sigma.unsqueeze(-1)
    u2 = (svals_sq / (sigma * sigma)).clamp(min=0)  # [B, K]

    sqrtb = math.sqrt(beta)
    lam_m = (1.0 - sqrtb) ** 2
    lam_p = (1.0 + sqrtb) ** 2

    # clamp u2 strictly inside MP support to avoid infinities
    u2 = u2.clamp(min=lam_m + _EPS, max=lam_p - _EPS)

    # term = u2/(u2 - lam_m) - u2/(lam_p - u2)
    term1 = u2 / (u2 - lam_m)
    term2 = u2 / (lam_p - u2)
    score_terms = term1 - term2  # [B, K]

    # sum only over inliers
    score = (score_terms * mask.float()).sum(dim=-1)  # [B]
    return score


@torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=False)
def _estimate_noise_level_torch_compiled(
    svals_desc: torch.Tensor,  # [B, K] singular values, DESCENDING order
    beta: float,
    p_lo: float = 0.05,
    p_hi: float = 0.30,
    num_p: int = 6,
    tail_rel: float = 0.00,    # trim anything above λ_+(1+tail_rel) after first pass
) -> torch.Tensor:
    """
    Batched, spike-robust MP fit using LEFT-CDF quantile matching + one trim.
    1) Choose a set of left quantiles p in [p_lo, p_hi], match empirical s^2 quantiles to MP λ-quantiles to get σ_p.
    2) Take median over p -> σ0. Trim right-tail by λ_+ threshold on whitened s^2, re-fit quantiles -> σ1.
    Returns σ̂ of shape [B]. Fully vectorized and compile-friendly.
    """
    device = svals_desc.device
    B, K = svals_desc.shape
    s = svals_desc.to(dtype=torch.float32)
    # sort ascending on s^2 for stable quantiles
    s2_asc = torch.sort(s * s, dim=-1, descending=False).values                   # [B,K]
    # build left quantile grid p
    if num_p <= 1:
        probs = torch.tensor([(p_lo + p_hi) * 0.5], device=device, dtype=torch.float32)
    else:
        probs = torch.linspace(p_lo, p_hi, num_p, device=device, dtype=torch.float32)  # [P]
    # MP λ-quantiles (independent of σ)
    lam_q = _mp_quantiles_lambda(probs, beta, device=device)                       # [P]
    lam_q = torch.clamp(lam_q, min=1e-12)
    # ---- Pass 1: raw left-quantile fit ----
    q_emp = _empirical_quantiles_from_sorted(s2_asc, probs)                        # [B,P]
    sigma_p = torch.sqrt(q_emp / lam_q.unsqueeze(0))                               # [B,P]
    sigma0 = torch.median(sigma_p, dim=-1).values                                   # [B]
    # ---- Trim tail once, then re-fit ----
    sqrtb = float(math.sqrt(beta))
    lam_plus = (1.0 + sqrtb) ** 2
    thresh = (lam_plus * (1.0 + tail_rel)) * (sigma0 * sigma0).unsqueeze(-1)       # [B,1]
    inlier_count = (s2_asc <= thresh).to(torch.int64).sum(dim=-1)                  # [B]
    inlier_count = torch.clamp(inlier_count, min=2, max=K)
    q_emp_trim = _empirical_quantiles_from_sorted(s2_asc, probs, k_eff=inlier_count)  # [B,P]
    sigma_p2 = torch.sqrt(q_emp_trim / lam_q.unsqueeze(0))                         # [B,P]
    sigma1 = torch.median(sigma_p2, dim=-1).values                                  # [B]
    return sigma1


def estimate_noise_level_torch(
    innovation_svals_desc: torch.Tensor,  # [B, K] DESC
    beta: float,
    **kwargs,
) -> torch.Tensor:
    """
    Convenience wrapper (keeps public API small). Returns [B] sigma_hat.
    Accepts optional kwargs: p_lo, p_hi, num_p, tail_rel.
    """
    return _estimate_noise_level_torch_compiled(innovation_svals_desc, beta, **kwargs)


@torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=False)
def get_denoised_squared_singular_value_torch(y: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Solve y^2 = t + (1+β) + β/t for t (per-entry), with t >= sqrt(β).
    y: [...], whitened empirical singular values
    Returns t with NaN inside bulk (y <= 1+sqrtβ).
    """
    y = y.to(dtype=torch.float32)
    sqrtb = math.sqrt(beta)
    mask = y > (1.0 + sqrtb)
    y2 = y * y
    A = y2 - (1.0 + beta)
    disc = A * A - 4.0 * beta
    disc = torch.clamp(disc, min=0.0)
    t = 0.5 * (A + torch.sqrt(disc))
    t = torch.clamp(t, min=sqrtb)
    out = torch.full_like(y, float("nan"))
    out = torch.where(mask, t, out)
    return out


@torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=False)
def estimate_spectral_projection_coefficients_torch(t: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Vectorized SPC formula. t may have NaNs for bulk entries.
    Returns SPC in [0,1], zeros where t is NaN.
    """
    t = t.to(dtype=torch.float32)
    mask = torch.isfinite(t)
    # avoid /0 and negatives under sqrt
    tm = torch.where(mask, t, torch.ones_like(t))
    tm2 = tm * tm
    num = torch.sqrt(torch.clamp((1.0 - beta / tm2) * (1.0 - 1.0 / tm2), min=0.0))
    den = torch.sqrt(torch.clamp((1.0 + beta / tm) * (1.0 + 1.0 / tm), min=1e-30))
    spc = num / den
    spc = torch.clamp(spc, min=0.0, max=1.0)
    spc = torch.where(mask, spc, torch.zeros_like(spc))
    return spc


def predict_spectral_projection_batched(
    per_minibatch_gradient: torch.Tensor,        # [B, n, m]
    per_minibatch_momentum_buffer: torch.Tensor  # [B, n, m]
) -> torch.Tensor:
    """
    GPU-friendly, batched SPC prediction for an entire minibatch stack on one device.
    Returns SPC tensor of shape [B, min(n,m)] (zeros inside MP bulk).
    """
    assert per_minibatch_gradient.ndim == 3 and per_minibatch_momentum_buffer.ndim == 3
    B, n, m = per_minibatch_gradient.shape
    beta = matrix_shape_beta((n, m))

    innovation = (per_minibatch_gradient - per_minibatch_momentum_buffer).to(dtype=torch.float32)
    # Batched singular values [B, K]
    s = torch.linalg.svdvals(innovation)
    s, _ = torch.sort(s, dim=-1, descending=True)

    # Clone OUTSIDE compiled function so we don't hold onto cudagraph-managed storage
    sigma_hat = estimate_noise_level_torch(s, beta=beta).clone()      # [B]
    y = s / sigma_hat.unsqueeze(-1).clamp_min(1e-30)                  # [B,K]
    # Clone OUTSIDE compiled function for the same reason
    t = get_denoised_squared_singular_value_torch(y, beta=beta).clone()  # [B,K]
    # Clone OUTSIDE compiled function to decouple from cudagraph output buffers
    spc = estimate_spectral_projection_coefficients_torch(t, beta=beta).clone()  # [B,K]
    return spc

