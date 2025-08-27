import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq

def predict_spectral_projection(
    per_minibatch_gradient: np.ndarray,
    per_minibatch_momentum_buffer: np.ndarray
):
    """
    Predict spectral projection coefficients from per-minibatch gradient singular values.
    """
    innovation = per_minibatch_gradient - per_minibatch_momentum_buffer
    innovation_spectrum = np.linalg.svdvals(innovation)
    beta = matrix_shape_beta(innovation.shape)
    noise_level_sigma = estimate_noise_level(innovation_spectrum, beta=beta)
    whitened_empirical_spectrum = innovation_spectrum / max(noise_level_sigma, 1e-30)
    squared_signal_spectrum = get_denoised_squared_singular_value(whitened_empirical_spectrum, beta)
    return estimate_spectral_projection_coefficients(t=squared_signal_spectrum, beta=beta)


def matrix_shape_beta(shape):
    n, m = shape
    a, b = (n, m) if n < m else (m, n)
    return float(a) / float(b)


def get_denoised_squared_singular_value(y: np.ndarray, beta: float):
    """
    Solves y^2 = t + (1 + β) + β/t for t
    with the quadratic formula

    Args:
        y: whitened empirical singular value
        beta: matrix shape ratio
    """
    y = np.asarray(y, dtype=float)
    sqrtb = np.sqrt(beta)
    # inside-bulk → no spike; mark as NaN (caller maps to SPC=0)
    mask = y > (1.0 + sqrtb)
    y2 = y * y
    A = y2 - (1.0 + beta)
    disc = A * A - 4.0 * beta
    # numerical guard
    disc = np.where(disc < 0.0, 0.0, disc)
    t = 0.5 * (A + np.sqrt(disc))
    # guard: t ≥ √β
    t = np.where(t < sqrtb, sqrtb, t)
    out = np.full_like(y, np.nan)
    out[mask] = t[mask]
    return out

def estimate_spectral_projection_coefficients(t: np.ndarray, beta: float):
    """
    Args:
        t: the square of the true signal singular value
        beta: matrix shape ratio
    """
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


def estimate_noise_level(innovation_spectrum: np.ndarray, beta: float) -> float:
    """
    Given current gradient g_t = S_t + E for signal S_t and error E
    and previous accumulated gradient G_{t-1},
    we define innovation R_t = g_t - G_{t-1}

    The point of accumulation is that the signal moves slowly, so S_t≈G_{t-1}.
    Then we have R_t = (S_t - G_{t-1}) + E, where (S_t - G_{t-1}) is small (low stable-rank)

    So the singular spectrum of R_t should be mostly Marchenko-Pastur with a few spikes.

    We find the noise level \hat{\sigma} by using fixed point iteration to the condition that:
    - inliers are exactly those below the edge {i: s_i\leq \tau(\hat{sigma})}
    - \hat{\sigma} is the MLE of \sigam on those inliers.
    """
    assert np.all(np.diff(innovation_spectrum) <= 0.0)
    spectrum = innovation_spectrum[::-1]
    sigma = None

    bulk_sizes_attempted = []
    bulk_size = len(spectrum) // 2
    while bulk_size not in bulk_sizes_attempted:
        inliers = spectrum[:bulk_size]
        bulk_sizes_attempted.append(bulk_size)
        sigma = fit_sigma_to_bulk_inliers(inliers, beta)
        edge = sigma * (1 + np.sqrt(beta))
        bulk_size = np.searchsorted(spectrum, edge, side='left')
    return sigma
        


def fit_sigma_to_bulk_inliers(inliers: np.ndarray, beta: float) -> float:
    s = np.asarray(inliers, float); sqrtb = np.sqrt(beta)
    lam_m, lam_p = (1 - sqrtb)**2, (1 + sqrtb)**2
    def score(sig):
        u2 = (s / sig)**2
        u2 = np.clip(u2, lam_m + 1e-12, lam_p - 1e-12)
        return np.sum(u2/(u2 - lam_m) - u2/(lam_p - u2))
    smax = s.max(); sigma0 = smax / (1 + sqrtb)
    return float(brentq(score, 0.5*sigma0, 4.0*sigma0, maxiter=48))