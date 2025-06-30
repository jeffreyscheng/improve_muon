# Bias--Variance Geometry of Gradient Orthogonalization

We study how minibatch size, momentum, and Richardson extrapolation influence bias and variance when estimating $\widetilde G = H^{-\tfrac12}G$.
Here $g_i=G+\delta_i$ are per-token gradients with $\mathbb E[\delta_i]=0$ and $\operatorname{Cov}(\delta_i)=\Sigma$.

---

## Averaging Before or After Projection

Form $\bar g = \tfrac1n\sum_{i=1}^n g_i$ and consider $f(v)=v/\|v\|$.
A second--order Taylor expansion around $g$ gives
$$
\mathbb E[f(\bar g)] 
  = z - \tfrac{\operatorname{tr}(P_{\perp}\Sigma)}{2\,\|g\|^{3} n}\,z + O(n^{-3/2}),
\qquad z = \tfrac{g}{\|g\|},
$$
where $P_{\perp}=I-zz^{\top}$.
Normalising *after* averaging thus has $O(n^{-1})$ bias, whereas normalising each $g_i$ first and then averaging retains an $O(1)$ bias.
For matrices the same phenomenon appears with the projector onto the tangent plane at $\operatorname{Ortho}(G)$.

---

## Micro-Batch and Momentum Effects

Let each micro-batch contain $m$ tokens and let $n$ such batches be averaged.
Then
\begin{align}
\operatorname{Bias} &\propto \tfrac{1}{m},\\
\operatorname{Var}  &\propto \tfrac{1}{n m}.
\end{align}
Applying momentum with parameter $\gamma$ scales the covariance by $(1-\gamma)/(1+\gamma)$ and introduces a staleness bias $O(\eta\, \tfrac{\gamma}{1-\gamma}\,\|H^{1/2} G\|)$.

---

## Richardson Extrapolation

Bias that behaves like $C/k$ can be removed via
$$
R = 2\,g_{2k}-g_{k},
$$
which leaves $O(k^{-2})$ bias but doubles the variance.

---

## Diagnostics Without Huge Batches

Collect eight gradients $g_i$ and compute
\begin{align}
\widehat\sigma^{2} &= \tfrac{n_0}{2{\binom 82}}\sum_{i<j}\|g_i-g_j\|_{F}^{2},\\
\text{bias}_{(m)} &\approx \|\bar g_{(m)}-\bar g_{(2m)}\|_{F},\\
\text{SNR} &= \text{bias}_{(m)}/\sigma_{\text{var}}.
\end{align}
Plotting subset means against $|1/n-1/k|$ estimates the bias slope $\|b\|$.

Techniques such as gradient clipping, sign-based updates, or low precision reduce variance at the cost of additional bias; the trade-offs can be quantified with the same expansions.

