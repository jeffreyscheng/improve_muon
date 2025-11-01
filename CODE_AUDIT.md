Code Audit: empirical/research/analysis/run_gradient_analysis.py

Scope
- Focused on distributed correctness, sharding, and performance on 8xH100 after the latest refactor to use get_aligned_svds and get_spectral_echoes_from_empirical_gradients. Prior to refactor, this script ran; issues below are limited to changes introduced by the refactor or nearby code paths it now exercises.

Findings

1) Heavy CPU batched SVD in get_aligned_svds
- What changed: The pipeline now calls get_aligned_svds(per_replicate_gradient) directly. per_replicate_gradient is stored on CPU (to save GPU memory) by get_accumulated_gradient_matrices. The refactored alignment calls torch.linalg.svd on the 3D CPU tensor of shape (R, N, M).
- Risk: CPU batched SVD over many large matrices (e.g., R≈64; N=1024 or 4096; M=1024) may become a bottleneck and negate GPU advantages. Previously, safe_svd offloaded to CUDA opportunistically and returned results to CPU.
- Impact: End-to-end wall-clock for each owner rank could increase substantially, especially at 4096×1024 MLP Input.
- Recommendation: Add a safe offload inside get_aligned_svds similar to safe_svd: promote bf16→f32, move the batched (or chunked) per-replica matrices to a CUDA stream for SVD, then bring U,S,V back to CPU. Consider chunking in R (e.g., 8–16 replicas per SVD batch) to bound peak HBM usage.

2) Memory amplification via diag_embed
- What changed: get_aligned_svds returns S as a full diagonal matrix (R, D, D); the runner then takes diag with torch.diagonal(...).
- Risk: (R, D, D) is O(D²) memory per layer per owner while the pipeline only needs (R, D) singular values.
- Impact: Higher CPU memory footprint and serialization pressure; unnecessary copies.
- Recommendation: Return the aligned singular values directly as shape (R, D) and, only if needed elsewhere, create diag on demand. The runner already consumes (R, D).

3) Alignment solver: greedy vs Hungarian
- What changed: get_aligned_svds uses a greedy permutation with deterministic jitter. The theoretical note suggests Hungarian/LAP. This is an accuracy trade-off, not a distributed bug.
- Impact: Potentially slightly worse alignment in near-degenerate spectra; could mildly affect echo curves. No distributed correctness issue.
- Recommendation: Optionally call SciPy linear_sum_assignment when available, fallback to greedy; keep current default if runtime is a concern.

4) Compute locality and PCIe traffic
- Context: Owners gather per-replica gradients to CPU; the new pipeline performs SVDs/echo entirely on CPU. This avoids repeated GPU reallocations but sacrifices GPU compute.
- Impact: The CPU SVD path (1) combined with large (R, N, M) tensors likely makes analysis significantly slower than the prior safe_svd offload behavior.
- Recommendation: Use GPU for SVD/Gram steps, but keep per-layer outputs (U,S,V reduced products) on CPU right away to limit HBM residency across many layers. If HBM pressure is a concern, process layers serially and free as you go (already the case), but also keep batched size in R to a small chunk.

5) CSV payload size and I/O
- What changed: replicate_singular_values now contains aligned singulars [R, D] as before; spectral_echo remains [D]. The code writes singular values as JSON float16 arrays.
- Impact: Similar to pre-refactor; potentially large files but manageable. No correctness issue.
- Recommendation: None needed for correctness; optional compression or fewer retained directions (cap D for very rectangular layers) if logs become large.

6) Distributed sharding and synchronization
- Sharding: shard_param_keys slices a stable list of parameter keys; ownership map is consistent across ranks. This matches pre-refactor behavior; no issue.
- Gather: get_accumulated_gradient_matrices gathers per-key grad tensors to the owner via dist.gather; owners store on CPU; barriers after each accumulation maintain ordering. This path is unchanged.
- Streaming + barrier: After pipeline execution, stream_write_analysis_results writes per-rank CSV and then a barrier synchronizes ranks. OK.
- Rank-0 aggregation: gather_layer_properties_to_rank_zero merges pruned props to rank 0; the pruned fields used by plots were updated to include the new keys. OK.

7) Naming mismatch (resolved)
- The CSV column is now spectral_echo (previously spectral_echo_from_reverb); downstream readers should consume spectral_echo.

8) Data types and numeric stability
- bf16 inputs: get_aligned_svds doesn’t currently promote bf16→f32 before SVD; if per_replicate_gradient is bf16, SVD on CPU could be unsupported/unstable. Pre-refactor safe_svd handled this.
- Recommendation: Inside get_aligned_svds, promote bf16 to float32 prior to SVD for robustness.

Summary of proposed fixes (non-breaking)
- Add CUDA offload + optional chunking in get_aligned_svds; return aligned singulars as (R, D) instead of diag matrices.
- Promote bf16→f32 before SVD.
- Optionally use Hungarian when SciPy available; keep greedy fallback.
- Optionally rename CSV column to spectral_echo for clarity.

With these addressed, the distributed behavior remains correct, ownership sharding is sensible, and performance should align with or improve upon the pre-refactor baseline.
