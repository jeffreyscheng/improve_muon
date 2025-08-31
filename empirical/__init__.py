# Early kernel-cache bootstrap for Triton/TorchInductor under DDP.
# This runs as soon as any `empirical.*` module is imported (e.g. when you use
# `python -m empirical.research.*`), so the env vars are set before torch/triton initialize.
import os, pathlib

def _set_once(name: str, value: str) -> None:
    # Don't override if the user explicitly set it.
    os.environ.setdefault(name, value)

# Cache roots: host-agnostic, rank-scoped (reusable across nodes).
# If you ever run two jobs concurrently, set IMPROVE_MUON_RUN_TAG to separate them.
CACHE_ROOT = os.path.expanduser("~/.cache/improve_muon")
RUN_TAG    = os.environ.get("IMPROVE_MUON_RUN_TAG", "shared")
RUN_ROOT   = os.path.join(CACHE_ROOT, "runs", RUN_TAG)
os.makedirs(RUN_ROOT, exist_ok=True)

# Rank-scoped dirs (avoid races across local ranks)
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
RANK_ROOT  = os.path.join(RUN_ROOT, f"rank{LOCAL_RANK}")
INDUCTOR_CACHE = os.path.join(RANK_ROOT, "inductor")
os.makedirs(INDUCTOR_CACHE, exist_ok=True)
_set_once("TORCHINDUCTOR_CACHE_DIR", INDUCTOR_CACHE)

# Optional: allow opting back into shared caches (only if you really want it)
if os.environ.get("IMPROVE_MUON_SHARED_TRITON_CACHE") == "1":
    triton_dir = pathlib.Path("~/.cache/improve_muon/triton").expanduser() / RUN_TAG
    triton_dir.mkdir(parents=True, exist_ok=True)
    _set_once("TRITON_CACHE_DIR", str(triton_dir))
else:
    # Default: per-rank Triton cache (most robust; avoids cross-process file races)
    triton_dir = pathlib.Path(RANK_ROOT) / "triton"
    triton_dir.mkdir(parents=True, exist_ok=True)
    _set_once("TRITON_CACHE_DIR", str(triton_dir))