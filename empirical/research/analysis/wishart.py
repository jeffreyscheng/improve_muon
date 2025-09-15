import torch
import numpy as np
from empirical.research.analysis.core_math import matrix_shape_beta

# Strict finite-size Wishart helpers (no fallbacks)

SV_TABLES: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None
CURRENT_SHAPE: tuple[int, int] | None = None


def load_sv_quantile_tables_npz(path: str) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path, allow_pickle=True)
    tables: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    base_keys = {k[:-2] for k in blob.files if k.endswith(("_u", "_q"))}
    for base in sorted(base_keys):
        p, n = map(int, base.split("x"))
        u = np.asarray(blob[f"{base}_u"], dtype=np.float64)
        q = np.asarray(blob[f"{base}_q"], dtype=np.float64)
        tables[(p, n)] = (u, q)
    return tables


def set_sv_tables_from_npz(path: str) -> None:
    global SV_TABLES
    SV_TABLES = load_sv_quantile_tables_npz(path)


def select_table(tables: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], p: int, n: int):
    key = (p, n)
    if key in tables:
        return tables[key]
    key_swapped = (n, p)
    if key_swapped in tables:
        return tables[key_swapped]
    raise KeyError(f"No quantile table for shape {(p,n)}")


def predict_counts_from_tabulated(bin_edges: np.ndarray, table: tuple[np.ndarray, np.ndarray], sigma: float, total: int) -> np.ndarray:
    u1, q1 = table
    a = np.clip(bin_edges[:-1] / max(float(sigma), 1e-30), q1[0], q1[-1])
    b = np.clip(bin_edges[1:]  / max(float(sigma), 1e-30), q1[0], q1[-1])
    cdf_a = np.interp(a, q1, u1)
    cdf_b = np.interp(b, q1, u1)
    return total * np.maximum(cdf_b - cdf_a, 0.0)


def set_current_shape(shape: tuple[int, int]) -> None:
    global CURRENT_SHAPE
    CURRENT_SHAPE = (int(shape[0]), int(shape[1]))


def fit_sigma_with_wishart(spectrum: torch.Tensor) -> float:
    """Bottom-tail scale fit using the globally selected table for CURRENT_SHAPE.

    Requires: set_sv_tables_from_npz(...) and set_current_shape(...) have been called.
    """
    if SV_TABLES is None or CURRENT_SHAPE is None:
        raise RuntimeError("SV_TABLES and CURRENT_SHAPE must be set before calling fit_sigma_with_wishart.")
    u_tbl, q_tbl = select_table(SV_TABLES, CURRENT_SHAPE[0], CURRENT_SHAPE[1])

    s = spectrum.detach().float().cpu().numpy().reshape(-1)
    s = np.sort(np.clip(s, 0.0, None))
    m = s.size
    if m < 8:
        raise ValueError("Spectrum too short for bottom-tail fit (need >= 8 values).")

    def Q1(u):
        uu = np.clip(u, u_tbl[0], u_tbl[-1])
        return np.interp(uu, u_tbl, q_tbl)

    kmax = min(80, max(12, m // 3))
    kmin = min(max(10, 8), kmax)
    idx_all = np.arange(1, kmax + 1, dtype=np.float64)
    u_all = (idx_all - 0.5) / m

    best = {"score": np.inf, "sigma": np.nan}
    for k in range(kmin, kmax + 1):
        x_full = s[:k]
        q_full = Q1(u_all[:k])
        for t in range(0, min(8, k - 4) + 1):
            x = x_full[t:k]
            q = q_full[t:k]
            denom = float(np.dot(q, q))
            if denom <= 0:
                continue
            sigma_hat = float(np.dot(q, x) / denom)
            r = x - sigma_hat * q
            RSS = float(np.dot(r, r))
            score = RSS / (k - t)
            if score < best["score"]:
                best.update({"score": score, "sigma": sigma_hat})
    if not np.isfinite(best["sigma"]):
        raise ValueError("Failed to fit sigma (non-finite result).")
    return float(best["sigma"]) 


def aspect_ratio_beta(matrix: torch.Tensor) -> float:
    return float(matrix_shape_beta(matrix.shape))


def squared_true_signal_from_quadratic_formula(
    spectrum: torch.Tensor,
    noise_sigma: float,
    aspect_ratio_beta: float
) -> torch.Tensor:
    """Solve y^2 = t + (1+β) + β/t for t >= 0, where y = s/σ.

    Returns t (same broadcastable shape as spectrum).
    """
    s = spectrum.to(dtype=torch.float32)
    sigma = torch.tensor(max(float(noise_sigma), 1e-30), dtype=s.dtype, device=s.device)
    beta = torch.tensor(float(aspect_ratio_beta), dtype=s.dtype, device=s.device)
    y2 = (s / sigma) ** 2
    A = torch.ones_like(y2)
    B = -(y2 - (1.0 + beta))
    C = beta * torch.ones_like(y2)
    disc = B * B - 4.0 * A * C
    disc = torch.clamp(disc, min=0.0)
    t = 0.5 * ( -B + torch.sqrt(disc) )  # positive root
    return t


def predict_spectral_projection_coefficient_from_squared_true_signal(
    squared_true_signal_t,
    aspect_ratio_beta: float
):
    """
    Compute SPC prediction as a piecewise function of t = s^2 (s true signal singular value):
      - 0, if t < sqrt(beta)
      - sqrt(((1 - β/t^2)(1 - 1/t^2))/((1 + β/t)(1 + 1/t))), otherwise

    Accepts either torch.Tensor or numpy.ndarray and returns matching type.
    """
    beta_scalar = float(aspect_ratio_beta)

    if hasattr(squared_true_signal_t, "detach"):  # torch path
        t = torch.clamp(squared_true_signal_t.to(torch.float32), min=1e-12)
        beta = torch.tensor(beta_scalar, dtype=t.dtype, device=t.device)
        thresh = torch.sqrt(torch.clamp(beta, min=0.0))
        # compute only on stable region t >= thresh to avoid overflow in 1/t^2
        mask = t >= thresh
        val = torch.zeros_like(t)
        if mask.any():
            tt = t[mask]
            num = (1.0 - beta / (tt * tt)) * (1.0 - 1.0 / (tt * tt))
            den = (1.0 + beta / tt) * (1.0 + 1.0 / tt)
            out = torch.sqrt(torch.clamp(num / torch.clamp(den, min=1e-30), min=0.0))
            val[mask] = out
        return val
    else:  # numpy path
        import numpy as _np
        t = _np.asarray(squared_true_signal_t, dtype=_np.float64)
        t = _np.clip(t, 1e-12, None)
        beta = float(beta_scalar)
        thresh = _np.sqrt(max(beta, 0.0))
        val = _np.zeros_like(t)
        mask = t >= thresh
        if mask.any():
            tt = t[mask]
            num = (1.0 - beta / (tt * tt)) * (1.0 - 1.0 / (tt * tt))
            den = (1.0 + beta / tt) * (1.0 + 1.0 / tt)
            out = _np.sqrt(_np.clip(num / _np.clip(den, 1e-30, None), 0.0, None))
            val[mask] = out
        return val
