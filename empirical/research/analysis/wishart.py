import torch
import numpy as np
import pandas as pd
from pathlib import Path
from empirical.research.analysis.core_math import matrix_shape_beta

# Strict finite-size Wishart helpers (no global tables)

# In-memory cache: (min_dim, max_dim) -> pandas DataFrame with columns
# ['singular_value', 'cumulative_probability'] for σ=1
_CDF_CACHE: dict[tuple[int, int], pd.DataFrame] = {}


def _normalize_shape(shape: tuple[int, int]) -> tuple[int, int]:
    p, n = int(shape[0]), int(shape[1])
    a, b = (p, n) if p <= n else (n, p)
    return a, b


def wishart_cdf_path_for_shape(shape: tuple[int, int], base_dir: str | Path = "wishart_cdfs") -> Path:
    a, b = _normalize_shape(shape)
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{a}x{b}.csv"


def get_wishart_cdf(shape: tuple[int, int], base_dir: str | Path = "wishart_cdfs") -> pd.DataFrame:
    """Load σ=1 Wishart CDF table for shape as a DataFrame with columns:
    ['singular_value', 'cumulative_probability'].

    Uses an in-memory cache; reads from CSV if not cached. The CSV filename is
    '<min_dim>x<max_dim>.csv' under `base_dir`. Does not auto-precompute.
    """
    key = _normalize_shape(shape)
    if key in _CDF_CACHE:
        return _CDF_CACHE[key]
    path = wishart_cdf_path_for_shape(key, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing Wishart CDF CSV for shape {key}: {path}. "
                                f"Run precompute_quantile_table_for_shape({key}) to generate it.")
    df = pd.read_csv(path)
    # Ensure required columns and sorted by singular_value
    if not {"singular_value", "cumulative_probability"}.issubset(df.columns):
        raise ValueError(f"Invalid CDF CSV schema at {path}; expected columns 'singular_value,cumulative_probability'.")
    df = df.sort_values("singular_value").reset_index(drop=True)
    _CDF_CACHE[key] = df
    return df


def predict_counts_from_tabulated(bin_edges: np.ndarray, cdf_df: pd.DataFrame, sigma: float, total: int) -> np.ndarray:
    """Predict expected counts per histogram bin from a σ-scaled finite-size CDF table.

    bin_edges: array of edges in the observed scale.
    cdf_df: DataFrame with 'singular_value' (Q1 grid) and 'cumulative_probability' (u grid).
    sigma: noise scale.
    total: total number of samples contributing to the histogram.
    """
    q1 = np.asarray(cdf_df["singular_value"], dtype=np.float64)
    u1 = np.asarray(cdf_df["cumulative_probability"], dtype=np.float64)
    s = max(float(sigma), 1e-30)
    a = np.clip(bin_edges[:-1] / s, q1[0], q1[-1])
    b = np.clip(bin_edges[1:]  / s, q1[0], q1[-1])
    cdf_a = np.interp(a, q1, u1)
    cdf_b = np.interp(b, q1, u1)
    return total * np.maximum(cdf_b - cdf_a, 0.0)


def F_noise_sigma(s: np.ndarray, cdf_df: pd.DataFrame, sigma: float) -> np.ndarray:
    """Finite-size noise CDF at scale sigma using σ=1 CDF table (CSV-backed).

    Fσ(s) = F1(s/σ) with clamping to the table's domain.
    """
    q1 = np.asarray(cdf_df["singular_value"], dtype=np.float64)
    u1 = np.asarray(cdf_df["cumulative_probability"], dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    x = np.clip(s / max(float(sigma), 1e-30), q1[0], q1[-1])
    return np.interp(x, q1, u1)


# --------------------------- Table precompute (σ=1) ---------------------------
def precompute_quantile_table_for_shape(
    shape: tuple[int, int],
    draws: int = 20000,
    L: int = 4096,
    u_min: float = 1e-6,
    u_max: float = 0.995,
    *,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    base_dir: str | Path = "wishart_cdfs",
) -> pd.DataFrame:
    """Precompute σ=1 Wishart singular-value CDF for a given shape and save CSV.

    - Uses batched eigendecomposition of the Gram matrix via torch.
    - Always normalizes shape as (min_dim, max_dim) for filename.
    - Returns a DataFrame with columns ['singular_value','cumulative_probability'] and writes it to disk.

    Parameters:
      draws: number of independent noise matrices to sample (higher → smoother CDF).
      L: number of CDF grid points to output (log-heavy in the bottom tail).
      batch_size: number of draws per batch for batched eigvals (defaults to min(draws, 512)).
      device: optional torch device (defaults to 'cuda' if available else 'cpu').
      base_dir: destination folder for CSV files.
    """
    p_raw, n_raw = int(shape[0]), int(shape[1])
    p, n = _normalize_shape((p_raw, n_raw))

    if batch_size is None:
        batch_size = int(min(max(64, draws // 32), 1024))
        batch_size = max(32, min(batch_size, draws))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Collect singular values across all draws in streaming batches
    num_batches = (int(draws) + batch_size - 1) // batch_size
    svals_list: list[np.ndarray] = []
    for b in range(num_batches):
        cur = int(min(batch_size, draws - b * batch_size))
        if cur <= 0:
            break
        E = torch.randn(cur, p_raw, n_raw, device=device, dtype=torch.float32)
        # Form Gram on the smaller side based on raw dims; eigvalsh is batched and robust
        if p_raw <= n_raw:
            G = E @ E.transpose(-1, -2)  # (cur, p, p)
        else:
            G = E.transpose(-1, -2) @ E  # (cur, n, n)
        ev = torch.linalg.eigvalsh(G)  # (cur, m)
        ev = torch.clamp(ev, min=0.0)
        svals = torch.sqrt(ev).reshape(-1).to("cpu", dtype=torch.float64).numpy()
        svals_list.append(svals)
        # free memory early
        del E, G, ev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    s_all = np.concatenate(svals_list) if svals_list else np.empty(0, dtype=np.float64)
    if s_all.size == 0:
        raise RuntimeError("No singular values generated; check parameters.")
    s_all.sort()
    N = s_all.size

    # Log-dense near zero, linear above
    L_log = max(16, L // 3)
    u_log = np.geomspace(u_min, 0.01, num=L_log, endpoint=False)
    u_lin = np.linspace(0.01, u_max, num=L - L_log)
    u = np.concatenate([u_log, u_lin])
    idx = np.clip((u * (N - 1)).astype(int), 0, N - 1)
    q = s_all[idx].astype(np.float64)

    df = pd.DataFrame({
        "singular_value": q,
        "cumulative_probability": u,
    })

    out_path = wishart_cdf_path_for_shape((p, n), base_dir)
    df.to_csv(out_path, index=False)
    # cache
    _CDF_CACHE[(p, n)] = df.copy()
    return df


def fit_sigma_with_wishart(spectrum: torch.Tensor, shape: tuple[int, int], base_dir: str | Path = "wishart_cdfs") -> float:
    """Bottom-tail scale fit using on-disk σ=1 CDF table for the provided shape.

    spectrum: 1D tensor of nonnegative singular values (observed scale).
    shape: matrix shape to select the appropriate finite-size CDF.
    """
    cdf_df = get_wishart_cdf(shape, base_dir)
    q_tbl = np.asarray(cdf_df["singular_value"], dtype=np.float64)
    u_tbl = np.asarray(cdf_df["cumulative_probability"], dtype=np.float64)

    s = spectrum.detach().float().cpu().numpy().reshape(-1)
    s = np.sort(np.clip(s, 0.0, None))
    m = s.size
    if m < 8:
        raise ValueError("Spectrum too short for bottom-tail fit (need >= 8 values).")

    def Q1(u):
        uu = np.clip(u, u_tbl[0], u_tbl[-1])
        return np.interp(uu, u_tbl, q_tbl)

    kmax = int(0.95 * m)
    kmin = int(0.0 * m)
    idx_all = np.arange(1, kmax + 1, dtype=np.float64)
    u_all = (idx_all - 0.5) / m

    best = {"score": np.inf, "sigma": np.nan}
    for k in range(kmin, kmax + 1):
        if k < 5:
            continue
        x_full = s[:k]
        q_full = Q1(u_all[:k])
        # allow small trims to avoid small-sample artifacts
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
        t = np.asarray(squared_true_signal_t, dtype=np.float64)
        t = np.clip(t, 1e-12, None)
        beta = float(beta_scalar)
        thresh = np.sqrt(max(beta, 0.0))
        val = np.zeros_like(t)
        mask = t >= thresh
        if mask.any():
            tt = t[mask]
            num = (1.0 - beta / (tt * tt)) * (1.0 - 1.0 / (tt * tt))
            den = (1.0 + beta / tt) * (1.0 + 1.0 / tt)
            out = np.sqrt(np.clip(num / np.clip(den, 1e-30, None), 0.0, None))
            val[mask] = out
        return val
