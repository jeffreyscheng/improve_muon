import math
from typing import Tuple, Union

import functools
import torch
from empirical.research.analysis.core_math import mp_pdf_singular_torch, matrix_shape_beta

# We keep tiny epsilons to avoid hitting MP support endpoints numerically
_EPS = 1e-6

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


# ---------- MP helpers (torch) ----------

def _trapz_cdf_from_pdf(pdf: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute CDF from pdf on a shared grid via trapezoid rule.
    pdf: [..., G], x: [G]
    Returns CDF with shape [..., G], starting at 0.
    """
    dx = torch.diff(x)                               # [G-1]
    mid = 0.5 * (pdf[..., 1:] + pdf[..., :-1])      # [..., G-1]
    F = torch.cumsum(mid * dx, dim=-1)              # [..., G-1]
    F = torch.nn.functional.pad(F, (1, 0))          # [..., G]
    # numerical guard to keep within [0,1]
    return torch.clamp(F, 0.0, 1.0)


@torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=False)
def _estimate_noise_level_one_sided_cdf(
    svals_desc: torch.Tensor,   # [B, K], DESC
    beta: float,
    grid_points: int = 256,
    cand_points: int = 33,
    refine_rounds: int = 2,
) -> torch.Tensor:
    """
    One-sided CDF fit:
      minimize mean_g ReLU(F_MP(s_g; sigma) - F_emp(s_g))^2
    over sigma. Robust to spikes because underprediction is not penalized.
    Fully batched and compile-friendly (fixed grid sizes).
    Returns [B] sigma_hat.
    """
    device = svals_desc.device
    B, K = svals_desc.shape
    s_desc = svals_desc.to(torch.float32)
    # Ascending order for cheap empirical CDF
    s = torch.flip(s_desc, dims=[-1])                # [B, K] ASC
    # Global grid bounds across batch to keep shapes static
    s_min_pos = torch.clamp_min(s[..., 0].min(), 1e-20)
    s_max = s[..., -1].max()
    # Modest expansion to accommodate edge placement
    s_lo = s_min_pos
    s_hi = s_max * 1.1
    # Log-spaced grid for numerical stability
    s_grid = torch.logspace(torch.log10(s_lo), torch.log10(s_hi),
                            steps=grid_points, device=device, dtype=torch.float32)  # [G]

    # Empirical CDF at grid points: fraction of samples ≤ s_g
    # Broadcast compare: (B,K,1) <= (G) -> (B,G)
    F_emp = (s.unsqueeze(-1) <= s_grid).float().mean(dim=-2)  # [B, G]

    # Base scale ~ right edge / (1+sqrtβ)
    one_plus_sqrtb = 1.0 + math.sqrt(beta)
    smax = s_desc.max(dim=-1).values                            # [B]
    base = (smax / one_plus_sqrtb).clamp_min(1e-12)            # [B]

    # Coarse candidate ratios around base (log-spaced)
    def _ratios(n, span_lo, span_hi):
        return torch.logspace(math.log10(span_lo), math.log10(span_hi),
                              steps=n, device=device, dtype=torch.float32)  # [n]

    ratios = _ratios(cand_points, 0.125, 8.0)                   # [C]
    sigma_c = (base.unsqueeze(-1) * ratios).contiguous()        # [B, C]

    # Fixed number of refinement rounds around current best
    for _ in range(refine_rounds):
        # MP pdf for each (B, C) at grid s_grid -> [B, C, G]
        pdf = mp_pdf_singular_torch(
            s_grid.view(1, 1, -1).expand(B, sigma_c.shape[1], -1),
            beta=beta,
            sigma=sigma_c.unsqueeze(-1),
        )  # [B,C,G]
        F_mp = _trapz_cdf_from_pdf(pdf, s_grid)                 # [B,C,G]
        # One-sided squared error
        err = torch.relu(F_mp - F_emp.unsqueeze(1)) ** 2        # [B,C,G]
        loss = err.mean(dim=-1)                                  # [B,C]
        # Best per batch
        best_idx = loss.argmin(dim=-1)                           # [B]
        best_sigma = sigma_c.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)  # [B]
        # Build a tighter candidate set around the winner
        ratios = _ratios(cand_points, 0.5, 2.0)                  # narrower
        sigma_c = (best_sigma.unsqueeze(-1) * ratios).contiguous()

    # Final pick from the last refinement grid
    pdf = mp_pdf_singular_torch(
        s_grid.view(1, 1, -1).expand(B, sigma_c.shape[1], -1),
        beta=beta,
        sigma=sigma_c.unsqueeze(-1),
    )
    F_mp = _trapz_cdf_from_pdf(pdf, s_grid)                      # [B,C,G]
    err = torch.relu(F_mp - F_emp.unsqueeze(1)) ** 2             # [B,C,G]
    loss = err.mean(dim=-1)                                      # [B,C]
    best_idx = loss.argmin(dim=-1)
    sigma_hat = sigma_c.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)  # [B]
    return sigma_hat


def estimate_noise_level_torch(
    innovation_svals_desc: torch.Tensor,  # [B, K] DESC
    beta: float,
    grid_points: int = 256,
    cand_points: int = 33,
    refine_rounds: int = 2,
) -> torch.Tensor:
    """
    One-sided CDF fit wrapper. Returns [B] sigma_hat.
    """
    return _estimate_noise_level_one_sided_cdf(
        innovation_svals_desc, beta,
        grid_points=grid_points,
        cand_points=cand_points,
        refine_rounds=refine_rounds,
    )


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

