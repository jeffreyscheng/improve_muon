# Orthogonalization Theory

The population gradient $G \in \mathbb R^{p\times p}$ admits the singular value decomposition
$$
G = U S V^{\top},\qquad S = \mathrm{diag}(s_1,\ldots,s_p),\; s_i>0.
$$
Under the empirical--Fisher assumption $H = G G^{\top}$ the Fisher--whitened gradient is
$$
\widetilde G = H^{-\tfrac12} G = U V^{\top}.
$$
Hence whitening coincides with taking the polar unitary factor
$$
\mathrm{Ortho}(G) = G (G^{\top} G)^{-\tfrac12} = U V^{\top}.
$$

Let $E$ be a small perturbation with $\mathbb E[E]=0$ and $\mathrm{Cov}(E)=\Sigma/k$.
A second--order expansion yields
$$
\mathrm{Ortho}(G+E)
  = \mathrm{Ortho}(G)
  + P_{\!\perp}(E) (G^{\top} G)^{-\tfrac12}
  - \tfrac12\,\mathrm{Ortho}(G)
     (G^{\top} G)^{-1} E^{\top} E
  + O(\|E\|^{3}),
$$
where $P_{\!\perp}(E)=E-\tfrac12\,\mathrm{Ortho}(G)(\mathrm{Ortho}(G)^{\top}E+E^{\top}\mathrm{Ortho}(G))$ is the tangent projection.

---

## Muon Estimators

Given per--sample gradients $g_i=G+\varepsilon_i$ with $\mathbb E[\varepsilon_i]=0$ and $\mathrm{Cov}[\mathrm{vec}\varepsilon_i]=\Sigma$, define
$$
\begin{aligned}
\bar g &= \tfrac1k \sum_{i=1}^k g_i, \\
\hat G_{\text{Muon}} &= \mathrm{Ortho}(\bar g), \\
\hat G_{\text{Soupy}} &= \tfrac1k \sum_{i=1}^k \mathrm{Ortho}(g_i), \\
\hat G_{\text{deb}} &= \dfrac{\mathrm{Ortho}(\bar g)}{1-\tfrac12\hat\gamma},\\
\hat\gamma &= \dfrac{\mathrm{tr}\hat\Sigma}{k\,\|\bar g\|_F^{2}},\\[-4pt]
\hat\Sigma &= \tfrac1k \sum_i (g_i-\bar g)(g_i-\bar g)^{\top}.
\end{aligned}
$$

With $g=\|G\|_F$ and $P_{\!\perp}(G)$ the projection onto the tangent plane, their leading bias and variance are
$$
\begin{aligned}
\mathbb E[\hat G_{\text{Muon}}]-\widetilde G &
  = -\tfrac{\mathrm{tr}\Sigma}{2k\,g^{3}}\,P_{\!\perp}(G)+O(k^{-2}),\\
\mathrm{Var}[\mathrm{vec}\hat G_{\text{Muon}}] &
  = \tfrac{\mathrm{tr}\Sigma}{k\,g^{2}}\,\mathrm{Proj}_{\text{tan}}+O(k^{-3/2}),\\[4pt]
\mathbb E[\hat G_{\text{Soupy}}]-\widetilde G &
  = -\tfrac{\mathrm{tr}\Sigma}{2 g^{3}}\,P_{\!\perp}(G)+O(\sigma^{3}),\\
\mathrm{Var}[\mathrm{vec}\hat G_{\text{Soupy}}] &
  = \tfrac{\mathrm{tr}\Sigma}{k\,g^{2}}\,\mathrm{Proj}_{\text{tan}}+O(k^{-1}\sigma^{3}),\\[4pt]
\mathbb E[\hat G_{\text{deb}}]-\widetilde G &
  = O(k^{-2}),\\
\mathrm{Var}[\mathrm{vec}\hat G_{\text{deb}}] &
  = \mathrm{Var}[\mathrm{vec}\hat G_{\text{Muon}}]\bigl(1+O(k^{-1})\bigr).
\end{aligned}
$$

In practice $p\sim10^{3\text{--}4}$ so $p^{-1/2}$ makes even small $k$ nearly unbiased.

---

| Estimator | Leading Bias | Leading Variance | Notes |
|-----------|--------------|------------------|-------|
| Muon | $O(k^{-1} p^{-1/2})$ | $O(k^{-1} p^{-1})$ | Consistent |
| SoupyMuon | $O(p^{-1/2})$ | $O(k^{-1} p^{-1})$ | Asymptotic bias |
| Debiased-Muon | $O(k^{-2} p^{-1/2})$ | $O(k^{-1} p^{-1})$ | Cheap correction |

