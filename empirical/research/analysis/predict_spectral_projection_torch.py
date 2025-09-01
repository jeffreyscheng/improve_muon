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
    outer_iters: int = 3,
    bisect_iters: int = 20,
) -> torch.Tensor:
    """
    Batched hard-mask fixed-point with inner bisection on GPU.
    Returns sigma_hat of shape [B].
    """
    device = svals_desc.device
    B, K = svals_desc.shape
    svals = svals_desc.to(dtype=torch.float32)
    s2 = svals * svals  # [B,K]

    sqrtb = math.sqrt(beta)
    one_plus_sqrtb = 1.0 + sqrtb

    # Initial guess close to edge-of-bulk from the largest s
    smax = svals.max(dim=-1).values  # [B]
    sigma = (smax / one_plus_sqrtb) * 0.9  # [B]

    # Ensure we have at least one inlier (edge = sigma*(1+sqrtb))
    # Do a few fixed expansions (kept small & fixed to remain compile-friendly).
    for _ in range(6):
        edge = sigma * one_plus_sqrtb  # [B]
        inliers = (svals <= edge.unsqueeze(-1))  # [B,K]
        has_any = inliers.any(dim=-1)           # [B]
        sigma = torch.where(has_any, sigma, sigma * 2.0)

    # Outer fixed-point: recompute mask from current sigma, then
    # solve F_I(sigma)=0 on that fixed mask via bisection.
    for _ in range(outer_iters):
        edge = sigma * one_plus_sqrtb  # [B]
        mask = (svals <= edge.unsqueeze(-1))    # [B,K]
        # if mask is empty for a batch item, fall back to include the smallest value
        # (make sure bisection has something to work with)
        any_inlier = mask.any(dim=-1, keepdim=True)
        mask = torch.where(any_inlier, mask, torch.nn.functional.one_hot(
            torch.full((B,), K - 1, device=device), num_classes=K
        ).bool())

        # s_I_max used to anchor the bracket
        neg_inf = torch.full_like(svals, -float("inf"))
        s_masked = torch.where(mask, svals, neg_inf)
        s_I_max = s_masked.max(dim=-1).values  # [B]

        base = (s_I_max / one_plus_sqrtb).clamp(min=1e-12)  # [B]
        sigma_L = base * 0.5
        sigma_R = base * 4.0

        # Make sure the bracket encloses a root: F_L <= 0, F_R >= 0 (monotone increasing)
        for _ in range(6):
            F_L = _score_fixed_mask(s2, sigma_L, beta, mask)
            F_R = _score_fixed_mask(s2, sigma_R, beta, mask)
            # If F_L > 0, move left bound further left
            sigma_L = torch.where(F_L > 0, sigma_L * 0.5, sigma_L)
            # If F_R < 0, move right bound further right
            sigma_R = torch.where(F_R < 0, sigma_R * 2.0, sigma_R)

        # Bisection with fixed steps (branchless via where)
        for _ in range(bisect_iters):
            sigma_M = 0.5 * (sigma_L + sigma_R)
            F_M = _score_fixed_mask(s2, sigma_M, beta, mask)
            go_right = (F_M <= 0)  # if <= 0, move left up to mid; else move right down to mid
            sigma_L = torch.where(go_right, sigma_M, sigma_L)
            sigma_R = torch.where(go_right, sigma_R, sigma_M)

        sigma = 0.5 * (sigma_L + sigma_R)

    return sigma  # [B]


def estimate_noise_level_torch(
    innovation_svals_desc: torch.Tensor,  # [B, K] DESC
    beta: float,
    outer_iters: int = 3,
    bisect_iters: int = 20,
) -> torch.Tensor:
    """
    Convenience wrapper (keeps public API small). Returns [B] sigma_hat.
    """
    return _estimate_noise_level_torch_compiled(
        innovation_svals_desc, beta, outer_iters=outer_iters, bisect_iters=bisect_iters
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


# --- Optional NumPy convenience wrappers for plotting paths ---
def estimate_noise_level_numpy(innovation_spectrum_desc_np, beta: float) -> float:
    import numpy as _np
    t = torch.as_tensor(innovation_spectrum_desc_np, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    if t.ndim == 1:
        t = t.unsqueeze(0)
    with torch.no_grad():
        sig = estimate_noise_level_torch(t, beta=beta)
    return float(sig.squeeze().cpu().item())

def get_denoised_squared_singular_value_numpy(y_np, beta: float):
    import numpy as _np
    t = torch.as_tensor(y_np, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = get_denoised_squared_singular_value_torch(t, beta=beta)
    return out.cpu().numpy()

def estimate_spectral_projection_coefficients_numpy(t_np, beta: float):
    import numpy as _np
    t = torch.as_tensor(t_np, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = estimate_spectral_projection_coefficients_torch(t, beta=beta)
    return out.cpu().numpy()