#!/usr/bin/env python3
"""
Multi-GPU precompute of σ=1 Wishart CDF tables.

Spawns one worker per GPU (or CPU if none) to generate singular values in parallel,
then aggregates and writes the final CSV to wishart_cdfs/<min>x<max>.csv.

Example:
  torchrun --standalone --nproc_per_node=8 -m empirical.research.analysis.precompute_wishart_cdf_distributed \
      --shape 1024 4096 --draws 20000 --batch-size 32

Or without torchrun using Python multiprocessing (falls back automatically):
  python -m empirical.research.analysis.precompute_wishart_cdf_distributed --shape 1024 4096 --draws 20000 --workers 4
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp

from .wishart import wishart_cdf_path_for_shape


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=2, type=int, required=True, help="Matrix shape p n")
    ap.add_argument("--draws", type=int, required=True, help="Total number of draws across all workers")
    ap.add_argument("--batch-size", type=int, default=None, help="Per-worker batch size (auto if not set)")
    ap.add_argument("--workers", type=int, default=None, help="Number of worker processes (defaults to num CUDA devices or 1)")
    ap.add_argument("--tmp-dir", type=str, default="wishart_cdfs/tmp", help="Temporary directory for partial outputs")
    ap.add_argument("--out-dir", type=str, default="wishart_cdfs", help="Output directory for final CSV")
    ap.add_argument("--L", type=int, default=4096, help="Number of CDF grid points")
    ap.add_argument("--u-min", type=float, default=1e-6, help="Lower u for log grid part")
    ap.add_argument("--u-split", type=float, default=0.01, help="Split between log and linear u grids")
    ap.add_argument("--u-max", type=float, default=0.995, help="Upper u for the grid")
    return ap.parse_args()


def _worker(rank: int, device: torch.device, shape: tuple[int, int], draws: int, batch_size: int, out_file: Path):
    torch.manual_seed(1337 + rank)
    p, n = shape
    bs = batch_size
    num_batches = (draws + bs - 1) // bs
    svals_list = []
    with torch.no_grad():
        for b in range(num_batches):
            cur = int(min(bs, draws - b * bs))
            if cur <= 0:
                break
            E = torch.randn(cur, p, n, device=device, dtype=torch.bfloat16)
            X = E.to(torch.float32)
            s = torch.linalg.svdvals(X)  # (cur, min(p,n))
            svals_list.append(s.reshape(-1).to("cpu", dtype=torch.float64).numpy())
    s_all = np.concatenate(svals_list) if svals_list else np.empty(0, dtype=np.float64)
    np.save(out_file, s_all)


def main():
    args = parse_args()
    shape = (int(args.shape[0]), int(args.shape[1]))
    total_draws = int(args.draws)

    # Select devices and workers
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        devs = [torch.device(f"cuda:{i}") for i in range(num_devices)]
    else:
        num_devices = 0
        devs = [torch.device("cpu")]

    workers = int(args.workers) if args.workers is not None else (num_devices if num_devices > 0 else 1)
    workers = max(1, min(workers, (num_devices if num_devices > 0 else workers)))
    if workers > len(devs):
        # cycle devices if asking for more workers than devices
        devs = (devs * ((workers + len(devs) - 1) // len(devs)))[:workers]
    else:
        devs = devs[:workers]

    # Auto batch size heuristic mirrors wishart.precompute
    if args.batch_size is None:
        bs = int(min(max(64, total_draws // 32), 1024))
        bs = max(32, min(bs, total_draws))
    else:
        bs = int(args.batch_size)

    # Partition draws
    per = total_draws // workers
    rem = total_draws % workers
    draws_by_rank = [per + (1 if i < rem else 0) for i in range(workers)]

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_files = [tmp_dir / f"part_{i:03d}.npy" for i in range(workers)]

    # Launch workers
    procs = []
    for i in range(workers):
        d = devs[i]
        p = mp.Process(target=_worker, args=(i, d, shape, draws_by_rank[i], bs, part_files[i]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Aggregate singular values and write final CSV
    svals = [np.load(f) for f in part_files if f.exists()]
    s_all = np.concatenate(svals) if svals else np.empty(0, dtype=np.float64)
    for f in part_files:
        try:
            f.unlink()
        except Exception:
            pass
    if s_all.size == 0:
        raise RuntimeError("No singular values produced.")
    s_all.sort()

    L = int(args.L)
    u_min = float(args.u_min)
    u_split = float(args.u_split)
    u_max = float(args.u_max)

    L_log = max(16, L // 3)
    u_log = np.geomspace(u_min, u_split, num=L_log, endpoint=False)
    u_lin = np.linspace(u_split, u_max, num=L - L_log)
    u = np.concatenate([u_log, u_lin])
    idx = np.clip((u * (s_all.size - 1)).astype(int), 0, s_all.size - 1)
    q = s_all[idx].astype(np.float64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = wishart_cdf_path_for_shape(shape, out_dir)
    df = pd.DataFrame({"singular_value": q, "cumulative_probability": u})
    df.to_csv(out_path, index=False)
    print(f"Wrote CDF CSV to {out_path} with draws={total_draws}, workers={workers}, batch_size={bs}")


if __name__ == "__main__":
    main()

