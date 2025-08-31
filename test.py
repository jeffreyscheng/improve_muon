#!/usr/bin/env python3
"""
Testbed: spiked random matrix -> estimate noise scale, invert spikes, predict SPC.
- Generates \hat{G} = S + E with a few planted spikes in S and MP noise E.
- Fits sigma via your fixed-point bulk-inlier MLE.
- Inverts y (whitened empirical singulars) to t (whitened true signal strength).
- Compares predicted SPC(t) against ground-truth SPC from planted spikes.
- Visualizes: spectrum + MP density, SPC vs y, and y->t inversion error.

Run:
  python test_spiked_fit.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ----------------------------
# Your API (kept verbatim)
# ----------------------------
def matrix_shape_beta(matrix_shape: tuple[int, int]):
    return min(matrix_shape) / max(matrix_shape)

def get_denoised_squared_singular_value(y: np.ndarray, beta: float):
    y = np.asarray(y, dtype=float)
    sqrtb = np.sqrt(beta)
    mask = y > (1.0 + sqrtb)
    y2 = y * y
    A = y2 - (1.0 + beta)
    disc = A * A - 4.0 * beta
    disc = np.where(disc < 0.0, 0.0, disc)
    t = 0.5 * (A + np.sqrt(disc))
    t = np.where(t < sqrtb, sqrtb, t)  # guard
    out = np.full_like(y, np.nan)
    out[mask] = t[mask]
    return out

def estimate_spectral_projection_coefficients(t: np.ndarray, beta: float):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = np.isfinite(t)
    if not np.any(mask):
        return out
    tm = t[mask]
    tm2 = tm * tm
    num = np.sqrt(np.clip((1.0 - beta / tm2) * (1.0 - 1.0 / tm2), 0.0, None))
    den = np.sqrt((1.0 + beta / tm) * (1.0 + 1.0 / tm))
    spc = num / np.maximum(den, 1e-30)
    out[mask] = np.clip(spc, 0.0, 1.0)
    return out

def fit_sigma_to_bulk_inliers(inliers: np.ndarray, beta: float) -> float:
    s = np.asarray(inliers, float); sqrtb = np.sqrt(beta)
    lam_m, lam_p = (1 - sqrtb)**2, (1 + sqrtb)**2
    smax = s.max(); sigma0 = smax / (1 + sqrtb)
    def nll(sig):
        u2 = (s / sig)**2
        u2 = np.clip(u2, lam_m + 1e-12, lam_p - 1e-12)
        logq = 0.5*np.log(lam_p - u2) + 0.5*np.log(u2 - lam_m) - np.log(np.pi*beta) - 0.5*np.log(u2)
        return -(logq.sum() - s.size*np.log(sig))
    res = minimize_scalar(nll, bounds=(0.5*sigma0, 4.0*sigma0), method="bounded", options={"maxiter": 48})
    return float(res.x)

def estimate_noise_level(innovation_spectrum: np.ndarray, beta: float) -> float:
    assert np.all(np.diff(innovation_spectrum) <= 0.0)
    spectrum = innovation_spectrum[::-1]  # ascending
    sigma = None
    seen = set()
    k = len(spectrum) // 2
    while k not in seen:
        seen.add(k)
        inliers = spectrum[:k]
        sigma = fit_sigma_to_bulk_inliers(inliers, beta)
        edge = sigma * (1 + np.sqrt(beta))
        k = int(np.searchsorted(spectrum, edge, side='left'))
    return sigma

def predict_spectral_projection(
    per_minibatch_gradient: np.ndarray,
    per_minibatch_momentum_buffer: np.ndarray
):
    innovation = per_minibatch_gradient - per_minibatch_momentum_buffer
    innovation_spectrum = np.linalg.svdvals(innovation)
    innovation_spectrum.sort(); innovation_spectrum = innovation_spectrum[::-1]
    beta = matrix_shape_beta(innovation.shape)
    sigma_hat = estimate_noise_level(innovation_spectrum, beta=beta)
    y = innovation_spectrum / max(sigma_hat, 1e-30)
    t_hat = get_denoised_squared_singular_value(y, beta)
    spc_hat = estimate_spectral_projection_coefficients(t_hat, beta=beta)
    return sigma_hat, y, t_hat, spc_hat

# ----------------------------
# Helpers: spiked model + MP
# ----------------------------
def make_spiked_matrix(n: int, m: int, sigma: float, spike_singulars: list[float], rng: np.random.Generator):
    """
    E ~ (sigma/sqrt(m)) * N(0,1)^{n×m};  S = U diag(spike_singulars) V^T, with U,V random orthonormal columns.
    Returns: Ghat = S + E, and (S, E).
    """
    k = len(spike_singulars)
    # Random orthonormal columns
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((m, m)))
    U = U[:, :k]; V = V[:, :k]
    S = (U * spike_singulars) @ V.T
    E = (sigma / np.sqrt(m)) * rng.standard_normal((n, m))
    return S + E, S, E

def mp_pdf_singular(s: np.ndarray, beta: float, sigma: float):
    """
    MP density for singular values (scale sigma). q_sigma(s) = (1/sigma) q_1(s/sigma).
    q_1(u) = sqrt((λ+ - u^2)(u^2 - λ-)) / (π β u) on u∈(sqrt(λ-), sqrt(λ+)).
    """
    s = np.asarray(s, float)
    lam_m = (1 - np.sqrt(beta))**2
    lam_p = (1 + np.sqrt(beta))**2
    u = s / sigma
    lam = u*u
    inside = (lam > lam_m) & (lam < lam_p) & (u > 0)
    out = np.zeros_like(s)
    if np.any(inside):
        num = np.sqrt((lam_p - lam[inside]) * (lam[inside] - lam_m))
        out[inside] = (num / (np.pi * beta * u[inside])) * (1.0 / sigma)
    return out

def spc_true_from_t(t: np.ndarray, beta: float):
    """Same as estimate_spectral_projection_coefficients, but expects exact t."""
    t = np.asarray(t, float)
    tm2 = t*t
    num = np.sqrt(np.clip((1.0 - beta / tm2) * (1.0 - 1.0 / tm2), 0.0, None))
    den = np.sqrt((1.0 + beta / t) * (1.0 + 1.0 / t))
    return np.clip(num / np.maximum(den, 1e-30), 0.0, 1.0)

# ----------------------------
# Experiment
# ----------------------------
def main():
    rng = np.random.default_rng(123)
    # Shape + beta
    n, m = 1024, 1024          # try (1024, 4096) to see rectangular effects
    beta = matrix_shape_beta((n, m))
    # Noise + spikes (in *raw* units)
    sigma_true = 0.08          # noise scale
    # Choose spike strengths so that t = (theta/sigma)^2 crosses BBP at t>sqrt(beta)
    # e.g., theta = sigma * sqrt(t), t in {2, 4, 9}  (note: t here is *x^2* in standard derivations)
    t_true = np.array([2.0, 4.0, 9.0])
    spike_singulars = (np.sqrt(t_true) * sigma_true).tolist()

    # Build \hat{G}
    Ghat, S, E = make_spiked_matrix(n, m, sigma_true, spike_singulars, rng)
    s_emp = np.linalg.svdvals(Ghat)
    s_emp.sort(); s_emp = s_emp[::-1]

    # Pretend momentum buffer is zero (so innovation==current gradient)
    sigma_hat, y, t_hat, spc_hat = predict_spectral_projection(Ghat, np.zeros_like(Ghat))

    # Ground truth SPC for planted spikes:
    spc_true = spc_true_from_t(t_true, beta)

    # --------- Print diagnostics ----------
    print(f"shape (n,m)=({n},{m}), beta={beta:.4f}")
    print(f"true sigma={sigma_true:.5f},  estimated sigma={sigma_hat:.5f},  rel.err={(sigma_hat-sigma_true)/sigma_true:+.2%}")
    edge_true = sigma_true * (1 + np.sqrt(beta))
    edge_hat  = sigma_hat  * (1 + np.sqrt(beta))
    print(f"bulk edge: true τ={edge_true:.5f}, est τ̂={edge_hat:.5f}")

    # Identify empirical spikes (above estimated edge)
    is_spike = s_emp > edge_hat
    y_spikes = y[is_spike]
    spc_pred_spikes = spc_hat[is_spike]

    # --------- Plots ----------
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))

    # (1) Spectrum + MP density
    ax = axs[0]
    ax.set_title("Spectrum with MP density")
    ax.hist(s_emp, bins=120, density=True, alpha=0.35, label="empirical s")
    xs = np.linspace(0, s_emp.max()*1.05, 2048)
    ax.plot(xs, mp_pdf_singular(xs, beta, sigma_true), lw=2, label="MP (true σ)")
    ax.plot(xs, mp_pdf_singular(xs, beta, sigma_hat),  lw=2, ls="--", label="MP (σ̂)")
    ax.axvline(edge_true, color="C2", lw=1.5, label="τ (true)")
    ax.axvline(edge_hat,  color="C3", lw=1.5, ls="--", label="τ̂ (est)")
    ax.set_xlabel("singular value s"); ax.set_ylabel("density")
    ax.legend(loc="upper right")

    # (2) SPC vs whitened singular y
    ax = axs[1]
    ax.set_title("SPC(y): predicted vs planted spike levels")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("whitened singular y = s/σ̂"); ax.set_ylabel("SPC")
    # predicted curve across y-grid
    ygrid = np.logspace(np.log10(max(1e-2, y[~np.isnan(y)].min() if np.any(np.isfinite(y)) else 1e-2)),
                        np.log10(max(2.5*(1+np.sqrt(beta)), y.max() if np.any(np.isfinite(y)) else 10.0)), 400)
    tgrid = get_denoised_squared_singular_value(ygrid, beta)
    spcgrid = estimate_spectral_projection_coefficients(tgrid, beta)
    ax.plot(ygrid, spcgrid, color="k", lw=2, label="predicted SPC(y)")
    # empirical spike points
    if y_spikes.size:
        ax.scatter(y_spikes, spc_pred_spikes, s=25, alpha=0.8, label="empirical spikes (pred)")
    # true horizontal SPC levels for planted t
    for j, tj in enumerate(t_true):
        ax.hlines(spc_true[j], xmin=1+np.sqrt(beta), xmax=ygrid.max(), colors=f"C{j}", linestyles="--",
                  label=f"true SPC (t={tj:.1f})" if j==0 else None)
    ax.legend(loc="lower right")

    # (3) y->t inversion quality on spikes
    ax = axs[2]
    ax.set_title("Recovered t̂ vs true t (spikes only)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true t"); ax.set_ylabel("estimated t̂")
    t_true = np.sort(t_true)[::-1]
    if y_spikes.size:
        # Pair top-L spikes by order (works when t_j distinct)
        L = min(len(t_true), y_spikes.size)
        ax.plot([min(t_true.min(), 0.7), max(t_true.max(), 20)],
                [min(t_true.min(), 0.7), max(t_true.max(), 20)], "k:", lw=1)
        ax.scatter(t_true[:L], t_hat[is_spike][:L], s=40, c="C1", label="top-L spikes")
        for j in range(L):
            ax.annotate(str(j), (t_true[j], t_hat[is_spike][j]))
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "No detected spikes (y ≤ 1+√β)", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
