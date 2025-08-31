# Early kernel-cache bootstrap for Triton/TorchInductor under DDP.
# This runs as soon as any `empirical.*` module is imported (e.g. when you use
# `python -m empirical.research.*`), so the env vars are set before torch/triton initialize.
import os, socket, pathlib

def _set_once(name: str, value: str) -> None:
    # Don't override if the user explicitly set it.
    os.environ.setdefault(name, value)

# Identify this worker
host = socket.gethostname()
rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"

# Default: per-rank caches (most robust; avoids cross-process file races)
root = pathlib.Path("~/.cache/improve_muon").expanduser() / "runs" / host / f"rank{rank}"
triton_dir = root / "triton"
inductor_dir = root / "inductor"

# Optional: allow opting back into shared caches (only if you really want it)
if os.environ.get("IMPROVE_MUON_SHARED_TRITON_CACHE") == "1":
    triton_dir = pathlib.Path("~/.cache/improve_muon/triton").expanduser() / host
if os.environ.get("IMPROVE_MUON_SHARED_INDUCTOR_CACHE") == "1":
    inductor_dir = pathlib.Path("~/.cache/improve_muon/inductor").expanduser() / host

# Create dirs and set env exactly once
triton_dir.mkdir(parents=True, exist_ok=True)
inductor_dir.mkdir(parents=True, exist_ok=True)
_set_once("TRITON_CACHE_DIR", str(triton_dir))
_set_once("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))