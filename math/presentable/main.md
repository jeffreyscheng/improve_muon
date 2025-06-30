# Muon Theory Overview

The Muon optimizer orthogonalizes gradients to approximate steepest descent in the spectral norm.
The key results are collected in this folder:

* **[Orthogonalization Theory](orthogonalization_theory.md)** — derives the equivalence between whitening and the polar factor and analyses the bias and variance of several Muon estimators.
* **[Bias--Variance Geometry](bias_variance_geometry.md)** — explains how batch size, momentum, and Richardson extrapolation affect estimator quality and outlines practical diagnostics.
* **[Matrix-Sign Approximations](matrix_sign_approximation.md)** — develops analytic spectral filters that implement the polar factor without an explicit SVD.

Current experiments focus on confirming these analytic predictions and on turning the matrix-sign approximations into efficient GPU kernels.
