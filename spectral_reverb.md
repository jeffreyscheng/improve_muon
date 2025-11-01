Spectral Reverb Refactor Plan

Purpose
- Replace the mean-gradient SVD alignment echo with a reverb-based estimator that does not assume access to the true gradient.
- Increase replicate count for echo estimation: when running A accumulations on 8 GPUs, use all 8×A microgradients (no averaging) because estimator variance scales with replicate count.
- Preserve downstream interfaces (property keys and visualizations) so existing analyses continue to work unchanged, with a single CSV field rename for clarity.

Key References (current implementation)
- Gradient collection: `empirical/research/analysis/model_utilities.py:get_accumulated_gradient_matrices`
- Analysis execution and specs: `empirical/research/analysis/run_gradient_analysis.py:ANALYSIS_SPECS`, `PropertyPipeline`
- Current spectral echo via mean alignment: `empirical/research/analysis/core_math.py:compute_spectral_echo_and_alignment`, `gather_aligned_singulars`
- New theory and solver: `theoretical/06_how_to_hear_the_spectral_echo.md`, `core_math.solve_for_spectral_echo_using_reverb`

Current Data Flow (relevant nodes)
1) `get_accumulated_gradient_matrices` → per-layer `per_minibatch_gradient: [B, H, W]` (B = number of locally collected microgradients)
2) `ANALYSIS_SPECS` builds:
   - `mean_gradient = mean(per_minibatch_gradient, dim=0)`
   - `minibatch_gradient_svd = safe_svd(per_minibatch_gradient)`
   - `mean_gradient_svd = safe_svd(mean_gradient)`
   - `spectral_echo_and_alignment = compute_spectral_echo_and_alignment(minibatch_gradient_svd, mean_gradient_svd)`
   - `spectral_echo = spectral_echo_and_alignment[0]`
   - `aligned_minibatch_singular_values = gather_aligned_singulars(minibatch_singular_values, spectral_echo_and_alignment[1])`
   - `gradient_noise_sigma2` and `empirical_phase_constant_tau2` consume the above
3) Results stream to CSV and feed existing visualizations.

What To Keep (unchanged components)
- `get_accumulated_gradient_matrices` return type and per-layer keying.
- `safe_svd`, property pipeline architecture, and sharding logic for non-echo properties.
- Property names and downstream consumers:
  - Keep `spectral_echo` property name and shape.
  - Keep `aligned_minibatch_singular_values` derived via `gather_aligned_singulars`.
- Visualization code and the kappa-fitting workflow (`fit_empirical_phase_constant_tau2`, log–log kappa fit, etc.).

What To Modify
1) Microgradient replicate handling (unconditionally use 8×A)
   - Goal: Use 8×A microgradients, not their average, for echo estimation.
   - Change the accumulation stage to collect the same parameter keys on all ranks (no sharding during gradient accumulation) so each key has A local micrograds per rank.
   - New helper: `gather_microgradients_across_ranks(per_minibatch_grads) -> GPTLayerProperty` that concatenates `[B_local, H, W]` along the batch dimension to form `B_total = 8×A` on rank 0.
   - After gather, perform echo-specific computations on rank 0 (see below); keep other non-echo nodes sharded as today to limit memory where possible.

2) Spectral echo property wiring
   - Replace the dependency on `mean_gradient_svd` within `ANALYSIS_SPECS` for echo.
   - New property node (same output interface as today):
     - `PropertySpec("spectral_echo_and_alignment", ["minibatch_gradient_svd"], <new_fn>)`
     - `<new_fn>` computes reverb-based echo and returns `(spectral_echo, alignment_indices)`.

What To Replace (logic-level changes)
- Remove use of `compute_spectral_echo_and_alignment(mb_svd, mean_svd)` for echo and alignment.
- Replace with a reverb-based path that:
  1) Aligns singular directions across replicas (microgradients) without using a mean-gradient reference.
  2) Calls `solve_for_spectral_echo_using_reverb` on aligned bases to estimate per-direction echo.

What To Author (new functionality to implement)
1) Cross-replica singular alignment (Hungarian; fail fast)
   - Function: `align_svd_across_replicas(mb_svd) -> (aligned_U, aligned_V, alignment_indices)`
     - Inputs: `mb_svd = (U_b, S_b, Vh_b)` from `safe_svd(per_minibatch_gradient)` where `U_b: [B, H, K]`, `Vh_b: [B, K, W]`.
     - Similarity between replica b1 and b2 for directions k1, k2: `M[k1,k2] = |<u_{b1}[:,k1], u_{b2}[:,k2]>| * |<v_{b1}[:,k1], v_{b2}[:,k2]>|`.
     - Choose a reference replica p (e.g., with maximal median similarity to others) and compute permutation that aligns each b to p via Hungarian assignment on M.
     - Produce:
       - `alignment_indices: [B, Kc]` where `Kc = min_b K_b` mapping each replica’s local direction to the reference index.
       - `aligned_U: [B, H, Kc]` and `aligned_V: [B, W, Kc]`, optionally sign-adjusted for positive correlation with the reference directions.
     - Implement Hungarian internally (no dependency fallbacks). If alignment cannot be computed, raise an error (prefer fail-fast over silent degradation).

2) Reverb echo wrapper
   - Function: `compute_reverb_spectral_echo(aligned_U, aligned_V) -> spectral_echo`
     - Adapter to call `solve_for_spectral_echo_using_reverb` which expects aligned left/right bases per replica and returns echoes.
     - If `solve_for_spectral_echo_using_reverb` returns `[B, Kc]` (replica-specific echoes), aggregate across replicas (median or mean) to produce a single `[Kc]` echo curve used downstream.
     - Clamp to [0,1] for numerical safety.

3) Echo-and-alignment property function
   - Function: `compute_reverb_echo_and_alignment(mb_svd) -> (spectral_echo: [Kc], alignment_indices: [B, Kc])`
     - Steps:
       1) `aligned_U, aligned_V, alignment_indices = align_svd_across_replicas(mb_svd)`
       2) `spectral_echo = compute_reverb_spectral_echo(aligned_U, aligned_V)`
       3) Return `(spectral_echo, alignment_indices)`
   - This maintains compatibility with `gather_aligned_singulars(minibatch_singular_values, alignment_indices)`.

Interface and Schema Compatibility
- Property keys and shapes remain the same for: `spectral_echo`, `aligned_minibatch_singular_values`, `gradient_noise_sigma2`, `empirical_phase_constant_tau2`.
- CSV schema change: rename the echo column to `spectral_echo_from_reverb` (formerly `spectral_echo_from_8x_mean_gradient`). Downstream visualizations will read the new field.
- Visualizations continue to consume the same properties; no plot code changes.

Execution Model Changes (no flags; fix default behavior)
- Always accumulate microgradients for all parameter keys on all ranks during analysis (no sharding at accumulation time).
- Gather microgradients across ranks to rank 0 and concatenate along the batch dimension to form 8×A replicates per key.
- Compute reverb-based echo on rank 0 using the gathered replicates. Keep non-echo nodes sharded where feasible to manage memory.

Edge Cases and Numerical Guards
- Rectangular matrices: work with `U: [H, K]`, `V: [W, K]` directly; truncate directions to `Kc` across replicas before alignment.
- Variable rank across replicas: enforce `Kc = min(K_b)`. Discard excess directions consistently across all replicas.
- Numerical robustness: retain `solve_for_spectral_echo_using_reverb` guarding: quantile-based floor, tau multiplier, and weighted triple–OLS. Clamp outputs to [0,1].
- SciPy availability: prefer Hungarian solver when present; otherwise use greedy fallback to maintain progress.

Validation Plan
- Consistency: Ensure CSVs and GIFs generate with unchanged schemas and file names.
- Sanity checks: Compare old vs new echo on a small subset; expect new method to differ when mean-gradient proxy is poor.
- Variance: Show reduced variance in echo estimates when enabling 8×A replicates versus A-only.
- Kappa fits: Validate `tau^2` fits and per-type kappa remain numerically stable and interpretable.

Rollout Steps
1) Add flags to the analysis runner and plumb them into `compute_analysis_for_step`.
2) Implement `gather_microgradients_across_ranks` and integrate gather when replicate mode is on.
3) Author `align_svd_across_replicas`, `compute_reverb_spectral_echo`, and `compute_reverb_echo_and_alignment`.
4) Update `ANALYSIS_SPECS` to swap in the new `spectral_echo_and_alignment` node depending only on `minibatch_gradient_svd`.
5) Smoke test a few layers/checkpoints; then run full analysis.

Open Questions / Decisions
- Replica echo aggregation: mean vs median across replicas. Proposed: median for robustness.
- If memory becomes a concern, consider: (a) rank-0 echo only (already planned), (b) chunked gather by parameter-type, (c) reduce Kc further via energy thresholding.
