"""
Anisotropy metrics for per-worker minibatch gradient noise.

Implements the metrics proposed in the Noise Anisotropy Test notes:
 - Center across workers
 - Diagonal-only spread (kappa_diag)
 - Random sketch covariance condition number (kappa_sketch)
 - John's sphericity statistic (U)
 - Lanczos/Power-based condition with Ledoit–Wolf style shrinkage (kappa_LW)
 - Optional: Matrix-normal flip–flop per layer (kappa_U, kappa_V, kappa_MN)

All functions operate on a batched gradient tensor `Ghats` of shape [B, H, W],
where B is the number of workers/minibatches collected at the same step for a
single layer. They return torch tensors (scalars) for seamless integration in
the property pipeline.
"""

from typing import Tuple
import math
import torch
import numpy as np


def center_across_workers(Ghats: torch.Tensor) -> torch.Tensor:
    """Center gradients across worker/minibatch dimension.

    Args:
        Ghats: [B, H, W] per-minibatch gradients for a layer.

    Returns:
        E: [B, H, W] residuals with worker-mean removed.
    """
    if Ghats.ndim < 2:
        return Ghats
    Gbar = Ghats.mean(dim=0, keepdim=True)
    return Ghats - Gbar


def diagonal_spread(E: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Diagonal-only spread proxy for anisotropy.

    Computes coordinate-wise sample variances across workers and returns
    kappa_diag = max(v_ij) / max(min(v_ij), eps).

    Args:
        E: [B, H, W] residuals centered over workers
        eps: numerical guard for minimum variance

    Returns:
        torch.scalar tensor with kappa_diag
    """
    if E.ndim < 2:
        return torch.tensor(1.0, device=E.device, dtype=torch.float32)
    # Sample variance across workers (unbiased)
    v = torch.var(E, dim=0, unbiased=True)
    vmax = torch.max(v)
    vmin = torch.clamp(torch.min(v), min=eps)
    return (vmax / vmin).to(dtype=torch.float32)


def _sketch_once(E_flat: torch.Tensor, s: int, rng: torch.Generator) -> torch.Tensor:
    """Compute a single Gaussian sketch Y = E_flat @ Omega where Omega ~ N(0,1)^{p x s}.

    Implemented column-by-column to avoid materializing Omega when p is large.

    Args:
        E_flat: [B, p] worker-residuals flattened per worker
        s: sketch dimension
        rng: torch.Generator for reproducibility

    Returns:
        Y: [B, s]
    """
    B, p = E_flat.shape
    device = E_flat.device
    dtype = torch.float32
    Y = torch.empty((B, s), device=device, dtype=dtype)
    # Column-wise generation to cap memory to O(p)
    for j in range(s):
        r = torch.randn(p, device=device, dtype=dtype, generator=rng)
        Y[:, j] = E_flat @ r
    return Y


def sketch_condition(E: torch.Tensor, s: int = 64, repeats: int = 3, seed: int = 1234) -> torch.Tensor:
    """Random-sketch covariance condition number proxy.

    Draw Gaussian sketch matrices Omega (p x s), project worker residuals, and
    compute the s x s sketched covariance S_s = (1/(B-1)) * sum_k y_k y_k^T = (1/(B-1)) * Y^T Y.
    Return median over a few repetitions of cond(S_s).

    Args:
        E: [B, H, W]
        s: sketch dimension (64–256 reasonable)
        repeats: number of independent sketches, take median
        seed: RNG seed base

    Returns:
        torch.scalar tensor with kappa_sketch
    """
    if E.ndim < 2:
        return torch.tensor(1.0, device=E.device, dtype=torch.float32)
    B = E.shape[0]
    if B <= 1:
        return torch.tensor(1.0, device=E.device, dtype=torch.float32)
    c = B - 1
    E_flat = E.reshape(B, -1).to(dtype=torch.float32)
    kappas = []
    for r in range(repeats):
        rng = torch.Generator(device=E.device)
        rng.manual_seed(seed + r)
        Y = _sketch_once(E_flat, s, rng)  # [B, s]
        # S_s = (1/c) * Y^T Y (s x s)
        S_s = (Y.transpose(0, 1) @ Y) / float(c)
        # Symmetrize for safety
        S_s = 0.5 * (S_s + S_s.transpose(0, 1))
        # Eigenvalues (symmetric PSD); clamp min eigenvalue
        evals = torch.linalg.eigvalsh(S_s).real
        lam_max = torch.max(evals)
        lam_min = torch.clamp(torch.min(evals), min=1e-12)
        kappas.append((lam_max / lam_min).item())
    # Median for robustness
    return torch.tensor(float(sorted(kappas)[len(kappas)//2]), device=E.device, dtype=torch.float32)


def john_sphericity(E: torch.Tensor) -> torch.Tensor:
    """John's sphericity statistic U = nu / mu^2 - 1 in operator view.

    mu = (1/p) * (1/c) * sum_k ||e_k||^2
    nu = (1/p) * (1/c^2) * sum_{k,l} <e_k, e_l>^2 = (1/p)*(1/c^2)*||E_flat E_flat^T||_F^2

    Args:
        E: [B, H, W]

    Returns:
        torch.scalar tensor with U
    """
    if E.ndim < 2:
        return torch.tensor(0.0, device=E.device, dtype=torch.float32)
    B = E.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=E.device, dtype=torch.float32)
    c = B - 1
    p = int(E[0].numel())
    E_flat = E.reshape(B, -1).to(dtype=torch.float32)
    norms2 = torch.sum(E_flat * E_flat, dim=1)
    mu = (norms2.sum() / float(c)) / float(p)
    # Gram G = E_flat E_flat^T -> [B,B]; sum of squares gives Frobenius^2
    G = E_flat @ E_flat.transpose(0, 1)
    nu = (G.pow(2).sum() / float(c * c)) / float(p)
    if mu <= 0:
        return torch.tensor(0.0, device=E.device, dtype=torch.float32)
    U = nu / (mu * mu) - 1.0
    return U.to(dtype=torch.float32)


def _S_operator_mv(E_flat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply sample covariance operator S(v) = (1/c) sum_k e_k (e_k^T v).

    Args:
        E_flat: [B, p]
        v: [p]
    Returns:
        S(v): [p]
    """
    B = E_flat.shape[0]
    c = max(1, B - 1)
    alpha = E_flat @ v  # [B]
    return (E_flat.transpose(0, 1) @ alpha) / float(c)


def _power_iteration_max_eig(E_flat: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Power iteration to estimate largest eigenvalue of S.

    Args:
        E_flat: [B, p]
        iters: number of iterations

    Returns:
        estimate of lambda_max(S) as torch scalar
    """
    p = E_flat.shape[1]
    device = E_flat.device
    v = torch.randn(p, device=device, dtype=torch.float32)
    v = v / (torch.norm(v) + 1e-30)
    lam = torch.tensor(0.0, device=device, dtype=torch.float32)
    for _ in range(max(1, iters)):
        w = _S_operator_mv(E_flat, v)  # [p]
        norm_w = torch.norm(w)
        if norm_w <= 1e-30:
            break
        v = w / norm_w
        lam = torch.dot(v, _S_operator_mv(E_flat, v))
    return lam


def lanczos_condition_shrunk(E: torch.Tensor, alpha: float = 0.1, iters: int = 20) -> torch.Tensor:
    """Condition estimate of shrunk covariance operator S_alpha.

    S_alpha = (1-α) S + α * tau * I, with tau = (1/p) * (1/c) * sum_k ||e_k||^2.
    Since S is PSD with rank <= B-1, lambda_min(S_alpha) = α*tau, and
    lambda_max(S_alpha) = (1-α) * lambda_max(S) + α * tau.
    Thus kappa = lambda_max(S_alpha) / lambda_min(S_alpha) =
                 ((1-α)*lambda_max(S) + α*tau) / (α*tau).

    Args:
        E: [B, H, W]
        alpha: shrinkage parameter in [0,1)
        iters: power-iteration steps to estimate lambda_max(S)

    Returns:
        torch.scalar tensor with kappa_LW
    """
    if E.ndim < 2:
        return torch.tensor(1.0, device=E.device, dtype=torch.float32)
    B = E.shape[0]
    if B <= 1:
        return torch.tensor(1.0, device=E.device, dtype=torch.float32)
    c = B - 1
    p = int(E[0].numel())
    E_flat = E.reshape(B, -1).to(dtype=torch.float32)
    # tau is mu from sphericity
    norms2 = torch.sum(E_flat * E_flat, dim=1)
    tau = (norms2.sum() / float(c)) / float(p)
    tau = torch.clamp(tau, min=1e-30)
    # Estimate lambda_max(S)
    lam_max_S = _power_iteration_max_eig(E_flat, iters=iters)
    lam_max_S = torch.clamp(lam_max_S, min=0.0)
    # Condition of shrunk operator
    num = (1.0 - float(alpha)) * lam_max_S + float(alpha) * tau
    den = float(alpha) * tau
    if den <= 0:
        return torch.tensor(1.0, device=E.device, dtype=torch.float32)
    return (num / den).to(dtype=torch.float32)


def matrix_normal_flipflop(
    E: torch.Tensor,
    iters: int = 4,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flip–flop MLE for matrix-normal noise covariance factors with ridge.

    E_k ~ MN(0, Σ_U, Σ_V) so Cov(vec(E_k)) = Σ_V ⊗ Σ_U.
    Iteratively update:
        Σ_U ← (1/(B*m)) * Σ_k E_k Σ_V^{-1} E_k^T + ε I
        Σ_V ← (1/(B*n)) * Σ_k E_k^T Σ_U^{-1} E_k + ε I

    Report condition numbers for Σ_U (row) and Σ_V (col), and their product.

    Args:
        E: [B, n, m]
        iters: number of flip–flop iterations
        eps: ridge added to both Σ_U and Σ_V for stability

    Returns:
        (kappa_U, kappa_V, kappa_MN) as float tensors
    """
    if E.ndim != 3:
        return (torch.tensor(1.0, device=E.device, dtype=torch.float32),
                torch.tensor(1.0, device=E.device, dtype=torch.float32),
                torch.tensor(1.0, device=E.device, dtype=torch.float32))
    B, n, m = E.shape
    if B == 0:
        return (torch.tensor(1.0, device=E.device, dtype=torch.float32),
                torch.tensor(1.0, device=E.device, dtype=torch.float32),
                torch.tensor(1.0, device=E.device, dtype=torch.float32))
    device = E.device
    dtype = torch.float32
    # Initialize with identity
    SigmaU = torch.eye(n, device=device, dtype=dtype)
    SigmaV = torch.eye(m, device=device, dtype=dtype)
    # Precompute E list to avoid repeated views
    Ek = E.to(dtype=dtype)
    for _ in range(max(1, iters)):
        # Inverses with ridge
        invV = torch.inverse(SigmaV + eps * torch.eye(m, device=device, dtype=dtype))
        # Σ_U update
        SU = torch.zeros((n, n), device=device, dtype=dtype)
        for k in range(B):
            SU = SU + Ek[k] @ invV @ Ek[k].transpose(0, 1)
        SU = SU / float(B * m)
        # Ridge
        SigmaU = SU + eps * torch.eye(n, device=device, dtype=dtype)
        # Σ_V update
        invU = torch.inverse(SigmaU + eps * torch.eye(n, device=device, dtype=dtype))
        SV = torch.zeros((m, m), device=device, dtype=dtype)
        for k in range(B):
            SV = SV + Ek[k].transpose(0, 1) @ invU @ Ek[k]
        SV = SV / float(B * n)
        SigmaV = SV + eps * torch.eye(m, device=device, dtype=dtype)
    # Condition numbers
    evalsU = torch.linalg.eigvalsh(0.5 * (SigmaU + SigmaU.transpose(0, 1))).real
    evalsV = torch.linalg.eigvalsh(0.5 * (SigmaV + SigmaV.transpose(0, 1))).real
    lamU_max = torch.max(evalsU)
    lamU_min = torch.clamp(torch.min(evalsU), min=1e-12)
    lamV_max = torch.max(evalsV)
    lamV_min = torch.clamp(torch.min(evalsV), min=1e-12)
    kappaU = (lamU_max / lamU_min).to(dtype=dtype)
    kappaV = (lamV_max / lamV_min).to(dtype=dtype)
    kappaMN = (kappaU * kappaV).to(dtype=dtype)
    return kappaU, kappaV, kappaMN


def sigma_eff2_from_centered_residuals(E: torch.Tensor) -> torch.Tensor:
    """Unbiased estimator of per-entry noise variance under matrix-normal noise.

    Given centered residuals across workers E = {R_k} with shape [W, n, m],
    returns sigma_eff^2 = (1/((W-1) n m)) * sum_k ||R_k||_F^2.

    Args:
        E: [W, n, m] centered residuals (worker mean removed)

    Returns:
        torch.scalar tensor sigma_eff^2
    """
    if E.ndim != 3:
        return torch.tensor(0.0, device=E.device, dtype=torch.float32)
    W, n, m = E.shape
    if W <= 1:
        return torch.tensor(0.0, device=E.device, dtype=torch.float32)
    sse = (E.to(dtype=torch.float32).pow(2)).sum()
    denom = float((W - 1) * n * m)
    return (sse / denom).to(dtype=torch.float32)


def anisotropy_dispersion_tau2(E: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Compute anisotropy–dispersion scalar tau^2 from worker-centered residuals.

    Implements the formula based on minimal sufficient statistics:
      - SSE = sum_k ||R_k||_F^2
      - S_C = sum_{k,l} ||R_k^T R_l||_F^2 = ||sum_k R_k R_k^T||_F^2 = ||sum_k R_k^T R_k||_F^2

    Then with W workers, layer shape (n, m):
      a = b = SSE / ((W-1) n m)
      A = (1/n) * S_C / ((W-1)^2 m^2)
      B = (1/m) * S_C / ((W-1)^2 n^2)
      tau^2 = (2/n) * (A / a^2) + (2/m) * (B / b^2)

    Args:
        E: [W, n, m] centered residuals across workers
        eps: numerical guard for denominator stability

    Returns:
        torch.scalar tensor tau^2 (nonnegative)
    """
    if E.ndim != 3:
        return torch.tensor(0.0, device=E.device, dtype=torch.float32)
    W, n, m = E.shape
    if W <= 1:
        return torch.tensor(0.0, device=E.device, dtype=torch.float32)

    Ef = E.to(dtype=torch.float32)
    # SSE
    sse = (Ef * Ef).sum()
    # Accumulate either sum_k R_k R_k^T (n x n) or sum_k R_k^T R_k (m x m) based on smaller dim
    if n <= m:
        S = torch.zeros((n, n), device=E.device, dtype=torch.float32)
        for k in range(W):
            Rk = Ef[k]
            S = S + (Rk @ Rk.transpose(0, 1))
    else:
        S = torch.zeros((m, m), device=E.device, dtype=torch.float32)
        for k in range(W):
            Rk = Ef[k]
            S = S + (Rk.transpose(0, 1) @ Rk)
    SC = (S * S).sum()

    denomW = float(W - 1)
    if denomW <= 0:
        return torch.tensor(0.0, device=E.device, dtype=torch.float32)
    n_f = float(n)
    m_f = float(m)

    a = sse / (denomW * n_f * m_f)
    # A and B per definitions
    A = (1.0 / n_f) * (SC / (denomW * denomW * m_f * m_f))
    B = (1.0 / m_f) * (SC / (denomW * denomW * n_f * n_f))

    a2 = torch.clamp(a * a, min=eps)
    # b == a
    term1 = (2.0 / n_f) * (A / a2)
    term2 = (2.0 / m_f) * (B / a2)
    tau2 = term1 + term2
    return torch.clamp(tau2, min=0.0).to(dtype=torch.float32)


def _spc_iso_from_y(y: torch.Tensor, beta: float) -> torch.Tensor:
    """Isotropic SPC f_iso(y, beta) operating on torch tensors.

    Implements the mapping described via the spiked model:
      - Threshold y_star = 1 + sqrt(beta)
      - t solves t^2 - (y^2 - 1 - beta)*t + beta = 0 → take positive root
      - spc = sqrt(((1 - β/t^2)(1 - 1/t^2)) / ((1 + β/t)(1 + 1/t))) for y > y_star, else 0
    """
    dtype = torch.float32
    device = y.device
    y = y.to(dtype)
    b = torch.tensor(float(beta), dtype=dtype, device=device)
    ystar = 1.0 + torch.sqrt(torch.clamp(b, min=0.0))
    y2 = y * y
    A = y2 - (1.0 + b)
    disc = torch.clamp(A * A - 4.0 * b, min=0.0)
    t = 0.5 * (A + torch.sqrt(disc))  # t = x^2
    # Guard
    t = torch.clamp(t, min=1e-12)
    num = (1.0 - b / (t * t)) * (1.0 - 1.0 / (t * t))
    den = (1.0 + b / t) * (1.0 + 1.0 / t)
    val = torch.sqrt(torch.clamp(num / torch.clamp(den, min=1e-30), min=0.0))
    val = torch.where(y > ystar, val, torch.zeros_like(val))
    return torch.clamp(val, 0.0, 1.0)


def predicted_spc_soft(
    s: torch.Tensor,
    noise_sigma: float,
    tau2: float,
    beta: float,
    worker_count: int = 1,
    m_big: int = 1,
    g_accum: int = 1,
    gh_points: int = 100,
    edge_scale_X: float = 1.0,
) -> torch.Tensor:
    """Compute softened SPC via log-normal anisotropy averaging.

    spc(s | σ_eff^2, τ^2, β) = E_Θ[ f_iso( s / (σ_eff * sqrt(Θ)), β ) ],
    with Θ = exp( sqrt(v) Z - v/2 ), v = log(1 + τ^2), Z ~ N(0,1).

    Uses Gauss–Hermite quadrature for the standard normal by transforming
    hermgauss nodes/weights associated with exp(-x^2) weight.

    Args:
        s: singular values tensor (1D)
        noise_sigma: σ_eff (scalar)
        tau2: τ^2 (scalar)
        beta: aspect ratio ≤ 1
        gh_points: number of Gauss–Hermite nodes (e.g., 9 or 11)

    Returns:
        torch.Tensor of same shape as s with values in [0,1].
    """
    s = s.to(dtype=torch.float32)
    device = s.device
    sigma = max(float(noise_sigma), 1e-30)
    if sigma <= 0.0:
        return torch.zeros_like(s, dtype=torch.float32)
    # Effective mean noise scale for averaged gradient and MP normalization
    W = max(int(worker_count), 1)
    G = max(int(g_accum), 1)
    mnorm = max(int(m_big), 1)
    # y_base = s / (sigma_eff_mean * sqrt(m_big)) = (s / sigma) * sqrt(W*G / m_big)
    scale = math.sqrt((W * G) / float(mnorm))
    v = float(np.log1p(max(float(tau2), 0.0)))
    X = max(float(edge_scale_X), 1e-30)
    if v <= 0.0:
        # No anisotropy → isotropic rule
        y = ((s / sigma) * scale) / X
        return _spc_iso_from_y(y, beta)
    # Gauss–Hermite nodes/weights for exp(-x^2)
    x, w = np.polynomial.hermite.hermgauss(gh_points)
    # Convert to standard normal expectation: z = sqrt(2) x, weights /= sqrt(pi)
    z = (np.sqrt(2.0) * x).astype(np.float32)
    w_norm = (w / np.sqrt(np.pi)).astype(np.float32)
    z_t = torch.from_numpy(z).to(device=device)
    w_t = torch.from_numpy(w_norm).to(device=device)
    # y_j factor: exp(-0.5*sqrt(v) z + v/4)
    y_factor = torch.exp(-0.5 * math.sqrt(v) * z_t + 0.25 * v)
    y0 = (((s / sigma) * scale) / X).unsqueeze(-1)  # [N, 1]
    yj = y0 * y_factor  # broadcast over nodes → [N, M]
    fij = _spc_iso_from_y(yj, beta)  # [N, M]
    spc_avg = (fij * w_t).sum(dim=-1)
    return torch.clamp(spc_avg, 0.0, 1.0)


def predicted_spc_soft_plugin(
    s: torch.Tensor,
    noise_sigma: float,
    tau2: float,
    beta: float,
    worker_count: int,
    m_big: int,
    edge_scale_X: float,
    g_accum: int = 1,
    gh_points: int = 100,
    gl_points: int = 7,
) -> torch.Tensor:
    """Alias to predicted_spc_soft using the provided edge-scale correction X.

    Kept for API compatibility; plug-in averaging is removed per spec.
    """
    return predicted_spc_soft(
        s,
        noise_sigma,
        tau2,
        beta,
        worker_count=worker_count,
        m_big=m_big,
        g_accum=g_accum,
        gh_points=gh_points,
        edge_scale_X=edge_scale_X,
    )
