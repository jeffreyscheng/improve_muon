#!/usr/bin/env python3
"""
Profile Wishart CDF precomputation runtime across batching/method/compile settings.

Examples:
  python -m empirical.research.analysis.profile_wishart_precompute --shape 1024 4096 \
         --draws 512 --batches 1 2 4 8 16 32 64 --method svd --compile --device auto

  python -m empirical.research.analysis.profile_wishart_precompute --shape 1024 4096 \
         --draws 512 --batches 1 2 4 8 16 --method gram --dtype float16
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch

from .wishart import precompute_quantile_table_for_shape, wishart_cdf_path_for_shape


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=2, type=int, default=[1024, 4096], help="Matrix shape p n")
    ap.add_argument("--draws", type=int, default=1024, help="Total number of draws to sample")
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128], help="Batch sizes to test")
    ap.add_argument("--method", choices=["svd", "gram"], default="svd", help="Computation path: svdvals or Gram eigvals")
    ap.add_argument("--compile", action="store_true", help="Enable torch.compile for batch kernel wrapper")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat each configuration and report median")
    ap.add_argument("--device", default="auto", help="cuda|cpu|mps|auto")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Sampling dtype for E")
    ap.add_argument("--no-write", action="store_true", help="Do not write CSVs (compute only)")
    return ap.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def map_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def time_config(shape, draws, batch_size, method, compile_flag, device, dtype, write_csv):
    t0 = time.perf_counter()
    df = precompute_quantile_table_for_shape(
        shape=shape,
        draws=draws,
        batch_size=batch_size,
        use_svdvals=(method == "svd"),
        torch_compile=compile_flag,
        device=device,
        dtype=dtype,
    )
    t1 = time.perf_counter()
    # Optionally avoid disk write effects in timing; delete file if not desired
    if not write_csv:
        try:
            path = wishart_cdf_path_for_shape(shape)
            if path.exists():
                path.unlink()
        except Exception:
            pass
    return t1 - t0, df


def main():
    args = parse_args()
    device = select_device(args.device)
    dtype = map_dtype(args.dtype)
    shape = (int(args.shape[0]), int(args.shape[1]))

    # Warm-up to avoid first-call overhead in cuSOLVER and RNG
    with torch.no_grad():
        bs = max(2, min(8, args.draws))
        X = torch.randn(bs, shape[0], shape[1], device=device, dtype=torch.float32)
        torch.linalg.svdvals(X)
        if device.type == "cuda":
            torch.cuda.synchronize()

    print(f"Profiling precompute on device={device}, shape={shape}, draws={args.draws}, dtype={dtype}, method={args.method}, compile={args.compile}")
    print("batch_size, seconds_total, seconds_per_draw, seconds_per_batch")

    for bs in args.batches:
        times = []
        for _ in range(max(1, args.repeat)):
            dt, _ = time_config(shape, args.draws, bs, args.method, args.compile, device, dtype, not args.no_write)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(dt)
        med = float(np.median(times))
        per_draw = med / max(args.draws, 1)
        per_batch = med / max((args.draws + bs - 1) // bs, 1)
        print(f"{bs}, {med:.4f}, {per_draw:.6f}, {per_batch:.6f}")


if __name__ == "__main__":
    main()

