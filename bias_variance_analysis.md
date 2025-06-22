# Bias-Variance Analysis of Muon Optimizer

## Setup and Mathematical Framework

### Notation
- $\ell(\theta, x)$: per-token loss function
- $\theta$: model parameters  
- $x \sim X$: training example
- $L(\theta) = \mathbb{E}_{x \sim X}[\ell(\theta, x)]$: overall loss
- $G = \nabla_\theta L(\theta)$: true gradient
- $g_x = \nabla_\theta \ell(\theta, x)$: per-example gradient
- $H(\theta) = \nabla^2_\theta L(\theta)$: Hessian matrix
- $\text{Ortho}(A) = UV^T$ where $A = USV^T$: matrix orthogonalization (zero-power)

### Key Assumption
We use the Gauss-Newton approximation: $H = GG^T$

### Fundamental Equivalence
**Claim:** Under $H = GG^T$, the preconditioned gradient $H^{-1/2}G \equiv \text{Ortho}(G)$.

**Proof:** Let $G = U\Sigma V^T$ be the SVD. Then:
$$H = GG^T = U\Sigma^2 U^T \implies H^{-1/2} = U\Sigma^{-1} U^T$$
$$H^{-1/2}G = U\Sigma^{-1} U^T \cdot U\Sigma V^T = UV^T = \text{Ortho}(G)$$

## Optimizer Definitions

**Uon (no momentum):**
$$\text{Uon}(g_1, \ldots, g_k) = \text{Ortho}\left(\frac{1}{k}\sum_{i=1}^k g_i\right)$$

**Muon (with momentum):**
$$\text{Muon}(g_1^t, \ldots, g_k^t, \ldots) = \text{Ortho}\left(\frac{1}{k}\sum_{i=1}^k \sum_{s=0}^t \gamma^{t-s} g_i^s\right)$$

## Higham Expansion for Bias-Variance Analysis

For $\text{Ortho}(G + E)$ where $E$ is a perturbation:

### Linear Approximation
$$\text{Ortho}(G + E) = \text{Ortho}(G) + P_{\perp}E\text{Ortho}(G)^{\dagger} + O(\|E\|^2)$$

where $P_{\perp} = I - \text{Ortho}(G)\text{Ortho}(G)^T$ and $\text{Ortho}(G)^{\dagger}$ is the Moore-Penrose pseudoinverse.

### Complete Expansion
$$\text{Ortho}(G + E) = \text{Ortho}(G) + L_1(E) + Q_2(E) + C_3(E) + O(\|E\|^4)$$

- $L_1(E)$: linear term (zero expectation)
- $Q_2(E)$: quadratic terms → $O(1/k)$ bias  
- $C_3(E)$: cubic terms → $O(1/k^{3/2})$ bias

## Bias Analysis

### Uon Bias
Let $\bar{g} = \frac{1}{k}\sum_{i=1}^k g_i$ and $E = \bar{g} - G$.

Since $\mathbb{E}[E] = 0$, the linear term vanishes:
$$\text{Bias}[\text{Uon}] = \mathbb{E}[Q_2(E)] = O\left(\frac{\text{tr}(\Sigma)}{k}\right)$$

where $\Sigma = \text{Cov}[g_i]$ is the gradient covariance matrix.

### Muon Bias
For momentum-averaged gradients with parameter $\gamma$:
$$\mathbb{E}[E_{\text{Muon}}] = \frac{\gamma}{1-\gamma} \cdot G \neq 0$$

This introduces bias at the linear level:
$$\text{Bias}[\text{Muon}] = O\left(\frac{\gamma\|G\|}{1-\gamma}\right) + O\left(\frac{\text{tr}(\Sigma_{\text{eff}})}{k}\right)$$

## Variance Analysis

### Uon Variance
Dominated by the linear term:
$$\text{Var}[\text{Uon}] = O\left(\frac{\|P_{\perp}\Sigma\text{Ortho}(G)^{\dagger}\|_F}{k}\right)$$

### Muon Variance
With effective covariance $\Sigma_{\text{eff}} = \frac{1+\gamma^2}{(1-\gamma)^2}\Sigma$:
$$\text{Var}[\text{Muon}] = O\left(\frac{1+\gamma^2}{(1-\gamma)^2} \cdot \frac{\|P_{\perp}\Sigma\text{Ortho}(G)^{\dagger}\|_F}{k}\right)$$

## Nonlinear Effects and Deviations

### Why Plots Deviate from Linearity

The gradient residual vs. $(1/n - 1/k)$ relationship becomes nonlinear when:

1. **Small $k$ regime**: $\|E\| \approx \|G\|$, higher-order terms dominate
2. **Rank deficiency**: $\bar{g}$ has different rank structure than $G$  
3. **Numerical issues**: Newton-Schulz iteration convergence problems

### Transition Point
Nonlinearity begins roughly when:
$$k_{\text{transition}} \approx \frac{\text{tr}(\Sigma)}{\|G\|^2}$$

### Modeling Strategies

**Polynomial model:**
$$\mathbb{E}[\|\text{residual}\|] = a_1 \left(\frac{1}{n} - \frac{1}{k}\right) + a_2 \left(\frac{1}{n} - \frac{1}{k}\right)^{3/2} + a_3 \left(\frac{1}{n} - \frac{1}{k}\right)^2$$

**Finite-sample correction:**
$$\mathbb{E}[\|\text{residual}\|] = \frac{a_1}{1 + c/k} \left(\frac{1}{n} - \frac{1}{k}\right) + \frac{a_2}{k^{3/2}}$$

## Summary

| Estimator | Bias | Variance |
|-----------|------|----------|
| **Uon** | $O(\text{tr}(\Sigma)/k)$ | $O(\|P_{\perp}\Sigma\text{Ortho}(G)^{\dagger}\|_F/k)$ |
| **Muon** | $O(\gamma\|G\|/(1-\gamma)) + O(\text{tr}(\Sigma_{\text{eff}})/k)$ | $O((1+\gamma^2)\|P_{\perp}\Sigma\text{Ortho}(G)^{\dagger}\|_F/((1-\gamma)^2 k))$ |

### Key Insights

1. **Linear regime** (large $k$): Both bias and variance scale as $O(1/k)$
2. **Momentum effects**: Muon has additional $O(\gamma)$ bias and amplified variance
3. **Nonlinear regime** (small $k$): Higher-order terms and numerical effects dominate
4. **Practical approach**: Use linear regime for extrapolation and bias correction

### Experimental Validation

The jackknife experiments empirically validate the $O(1/k)$ theoretical scaling in the linear regime, providing foundation for Richardson extrapolation techniques to reduce bias in gradient-based optimization.