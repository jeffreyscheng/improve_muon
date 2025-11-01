Dead code inventory for empirical/research/analysis

Scope: Only code under empirical/research/analysis. This marks unused or superseded functions/imports so we can prune safely later. Listings are based on repository-wide ripgrep usage at time of change.

core_math.py
- compute_spectral_echo_by_permutation_alignment: superseded by get_aligned_svds + get_spectral_echoes_from_empirical_gradients; no external references.
- compute_spectral_echo_and_alignment: superseded by new alignment/echo; unused.
- compute_reverb_echo_and_alignment: superseded; unused after refactor of run_gradient_analysis.
- gather_aligned_singulars: superseded by aligned S from get_aligned_svds; unused after refactor.
- trapz_cdf_from_pdf: no references in repo.
- compute_innovation_spectrum: no references in repo.
- aspect_ratio_beta (function wrapper): duplicate of matrix_shape_beta; unused.
- precompute_theoretical_noise_to_phase_slope_kappa: stub (pass); unused.

run_gradient_analysis.py
- gradients_stable_rank_from_svd: no longer used after switching to gradients_stable_rank_from_singulars.
- singular_value_std and related mean_singular_values plumbing: not used; corresponding PropertySpec removed.

core_visualization.py
- Unused imports: pandas (pd), matplotlib.ticker.LogLocator, LogFormatterMathtext, NullLocator.
  All current visualizations work without these.

logging_utilities.py
- calculate_singular_values: no references in repo.
- calculate_weight_norm: no references in repo.
- dummy_logging: no references in repo.

constants.py
- FIELD_NAMES includes legacy entries (mean_gradient_svd, alignment_indices) that are not used by current pruner; kept for backward compatibility of older logs only.

Notes
- If we decide to remove the above, confirm no downstream notebooks or ad‑hoc scripts depend on them outside this repo.
- SciPy-dependent fit_empirical_phase_constant_tau2 remains in use via the pipeline; do not remove.
