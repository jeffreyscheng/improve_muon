Hungarian Runtime: Hypotheses And Diagnostics

Observations
- First owner-layer processed: param=('Attention O', 1); SVD took ~20.6 s; then Hungarian logged once: "hungarian: n=1024, iters=0, 52.5 ms".
- Process stalled for minutes inside _hungarian_linear_sum_assignment on rank 1, per stacktrace, after the first log.
- For owners, B ≈ 8×A = 64 microgradients; Kc ≈ 1024 for 1024×1024 layers; so we perform Hungarian on 1024×1024 cost matrices up to 64 times per layer.

Scale Estimates
- Hungarian (O(n^3)) at n=1024 implies ~1.07e9 primitive ops per invocation; pure-Python loops exacerbate cost.
- 64 replicas × 11 owned keys (from logs) × 1e9 basic ops ~ impractical on CPU, especially across 8 ranks contending for CPU.

Primary Hypotheses
- Algorithmic complexity in Python
  - The custom Hungarian implementation is Python/NumPy loop–heavy. For n=1024, the O(n^3) path in Python is too slow, even if one instance logged 52 ms (likely a trivial case).
  - The first case (b=0) compares reference to itself; cost is near-diagonal with trivial zeros. Subsequent cases (b>0) require many augmentation steps, causing the algorithm to balloon.

- Exact-zero comparisons cause degenerate iterations
  - find_a_zero checks cost[i, j] == 0. After step-6 adjustments (adding/subtracting min_uncovered), floating-point noise may prevent zeros from being exactly zero, drastically increasing the number of iterations needed to create uncovered zeros.
  - With cost = max_val - M and M derived from dot products, many near-duplicates and ties can exacerbate this behavior.

- Degenerate/tie-heavy cost structure
  - For nearly orthonormal bases with many similar directions, M can have many nearly-equal maxima per row/column (especially at depth-1 attention-O). This can produce cost matrices with multiple identical minima per row/column, triggering worst-case alternating patterns in the star/prime/cover steps.

- CPU contention across ranks
  - Each owner rank runs large 1024×1024 Hungarian solves concurrently across 8 processes, saturating shared CPU cores and cache. Even a compiled implementation would slow; a Python one will crawl.

- Repeated conversions and memory pressure
  - For each replica b, we compute MU and MV on GPU, then convert 1024×1024 arrays to NumPy (CPU) for Hungarian. The host copies are ~8 MB per matrix, twice per replica, 64 replicas: ~1 GB of array traffic per layer just for alignment cost construction. This bandwidth plus GIL-bound loops can further stall throughput.

Secondary Hypotheses
- Reference choice amplifies work
  - Using b=0 as reference yields trivial b=0 alignment but may make other b have less structured M, worsening tie patterns and runtime. A better reference (e.g., medoid by similarity) might reduce costs per replica.

- Overly large Kc
  - Using the full Kc=1024 for alignment may be unnecessary; many modes contribute little to echo estimation. Aligning all modes multiplies runtime with little benefit.

Diagnostics To Collect (Non-invasive)
- Per-replica timing for Hungarian calls
  - Already partially covered by logs: add “b index” and cumulative time per b to confirm that b=1..B-1 explode while b=0 is trivial.

- Cost-matrix structure stats
  - Number of exact zeros after row/col reduction; min uncovered value during step-6; iteration count per b; proportion of near-zero entries (|cost| < 1e-12) to gauge degeneracy.

- Kc and B sensitivity
  - Log Kc actually used (post truncation) and B at the alignment step per key; correlate runtime with Kc×B.

Likely Remedies (For Future Implementation)
- Use a compiled Hungarian (SciPy’s linear_sum_assignment) or an optimized PyTorch/CUDA implementation.
- Introduce a small epsilon for zero-tests and reductions to prevent pathological loops (replace equality checks with |x| < eps).
- Reduce alignment dimensionality Kc (e.g., top-k by singular value, or energy threshold) prior to solving assignments.
- Switch to a faster approximate alignment (e.g., greedy best-match with backtracking) when Kc is large, reserving Hungarian for smaller Kc.
- Choose a stronger reference (e.g., medoid by similarity across replicas) to simplify cost matrices for most b.
- Pre-normalize and compute all similarities on GPU, keep in torch, and (if possible) run a GPU-side assignment algorithm to avoid CPU copies and GIL-bound loops.

Summary
The stall is most plausibly due to the O(n^3), Python-implemented Hungarian operating on 1024×1024 cost matrices across ~64 replicas and multiple keys per owner rank, compounded by exact-zero checks and tie-heavy cost structures. CPU contention and repeated GPU→CPU transfers likely worsen runtime. The safest path forward is to replace the Hungarian with a compiled/accelerated solver and/or reduce Kc, alongside tolerance-based zero handling.

