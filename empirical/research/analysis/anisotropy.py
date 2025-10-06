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

