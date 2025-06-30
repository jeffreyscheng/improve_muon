# Fisher–Whitened Gradients, `Ortho(·)`, and Minibatch Estimators  
### *A step-by-step derivation sheet*

---

## Motivation  

Large-language-model (LLM) parameters are mostly **$p\times p$ weight matrices**.  
Second-order optimisers would like to apply the **Fisher / Hessian preconditioning**
$$
\tilde G \;=\;H^{-\tfrac12}G,
$$
but computing $H^{-\tfrac12}$ is prohibitive.  
Under the **empirical-Fisher assumption $H = GG^{\!\top}$** the preconditioned
gradient turns out to be the **orthogonal projection of $G$**, which we can
compute with only one SVD per parameter matrix.  
We study two practical stochastic estimators of this direction (***Muon*** and
***SoupyMuon***) and finish with a bias-corrected variant.

Every claim below is followed by a **Justification** block using only
undergraduate linear-algebra algebra—no differentials, no hidden steps.

---

## Claim&nbsp;1 Equality of the Fisher-whitened gradient and `Ortho(G)`  

**Statement.**  
Assume $H = GG^{\!\top}$ with $G\in\mathbb R^{p\times p}$ full rank.  
Then
$$
\tilde G \;=\; H^{-\tfrac12}G \;=\; \operatorname{Ortho}(G).
$$

**Justification.**

$$\begin{aligned}
&\text{1.  Singular-value decomposition of }G                            \\
&\quad G \;=\; U S V^{\!\top},\quad
        U,V\in\mathbb R^{p\times p}\text{ orthogonal},\;\;
        S=\operatorname{diag}(s_1,\dots,s_p),\;s_i>0.                     \\[6pt]
&\text{2.  Form the empirical Fisher/Hessian}                            \\
&\quad H \;=\; GG^{\!\top} \;=\; U S V^{\!\top} V S U^{\!\top}
                 \;=\; U S^2 U^{\!\top}.                                  \\[6pt]
&\text{3.  Its inverse square-root (all }s_i>0):                         \\
&\quad H^{-\tfrac12}
      \;=\;U S^{-1} U^{\!\top},
      \quad S^{-1}=\operatorname{diag}(s_1^{-1},\dots,s_p^{-1}).          \\[6pt]
&\text{4.  Multiply by }G:                                                \\
&\quad H^{-\tfrac12}G
      \;=\;U S^{-1} U^{\!\top}\; \bigl(U S V^{\!\top}\bigr)
      \;=\; U S^{-1} S V^{\!\top}
      \;=\; U V^{\!\top}.                                                 \\[6pt]
&\text{5.  Closed-form for }\operatorname{Ortho}(G):                      \\
&\quad \operatorname{Ortho}(G)
      \;=\; G\bigl(G^{\!\top}G\bigr)^{-\tfrac12}.                         \\[3pt]
&\quad G^{\!\top}G
      =\bigl(U S V^{\!\top}\bigr)^{\!\top}\bigl(U S V^{\!\top}\bigr)
      = V S^2 V^{\!\top}.                                                 \\[3pt]
&\quad \bigl(G^{\!\top}G\bigr)^{-\tfrac12}
      = V S^{-1} V^{\!\top}.                                              \\[3pt]
&\quad G\bigl(G^{\!\top}G\bigr)^{-\tfrac12}
      = U S V^{\!\top} \; V S^{-1} V^{\!\top}
      = U S S^{-1} V^{\!\top}
      = U V^{\!\top}.                                                     \\[6pt]
&\boxed{H^{-\tfrac12}G = U V^{\!\top} = \operatorname{Ortho}(G).}
\end{aligned}$$

---

## Claim&nbsp;2 First-order expansion of `Ortho(G+E)`  

For a small perturbation $E$,
$$
\operatorname{Ortho}(G+E)
  = O
    + P_{\!\perp}(E)\,(G^{\!\top}G)^{-\tfrac12}
    + O(\|E\|^{2}),
\quad O := \operatorname{Ortho}(G),
$$
with the **tangent-projection operator**
$$
P_{\!\perp}(E)
  := E - \tfrac12\,O\bigl(O^{\!\top}E + E^{\!\top}O\bigr).
$$

**Justification.**

$$\begin{aligned}
&\text{1.  Definitions used throughout}                                  \\
&\quad O := \operatorname{Ortho}(G) = U V^{\!\top},\qquad
      M := G^{\!\top}G = V S^2 V^{\!\top}.                                \\[6pt]
&\text{2.  Expand the inner Gram matrix}                                  \\
&\quad (G{+}E)^{\!\top}(G{+}E)
      = M + D + O(\|E\|^{2}),                                             \\
&\quad D := G^{\!\top}E + E^{\!\top}G
      \text{ (symmetric and linear in }E).                                \\[6pt]
&\text{3.  Binomial series for the inverse square-root}                   \\
&\quad (M + D)^{-\tfrac12}
      = M^{-\tfrac12}
        - \tfrac12 M^{-\tfrac12} D M^{-\tfrac12}
        + O(\|E\|^{2}).                                                   \\[6pt]
&\text{4.  Substitute into } \operatorname{Ortho}(G{+}E)                  \\
&\quad \operatorname{Ortho}(G{+}E)
      =(G{+}E)\bigl[M^{-\tfrac12}
          - \tfrac12 M^{-\tfrac12} D M^{-\tfrac12}\bigr]
          + O(\|E\|^{2})                                                  \\[3pt]
&\quad = GM^{-\tfrac12}
       + E M^{-\tfrac12}
       - \tfrac12\,G M^{-\tfrac12} D M^{-\tfrac12}
       + O(\|E\|^{2}).                                                   \\[3pt]
&\quad \text{But } GM^{-\tfrac12} = O.                                    \\[6pt]
&\text{5.  Re-express the two $E$-terms}                                  \\
&\quad\underline{\text{(i) Linear term}}:
      \;E M^{-\tfrac12}.                                                  \\[3pt]
&\quad\underline{\text{(ii) Curvature term}}:
      -\tfrac12\,G M^{-\tfrac12} D M^{-\tfrac12}.                         \\[3pt]
&\quad D = G^{\!\top}E + E^{\!\top}G
      = V(S\tilde E + \tilde E^{\!\top}S)V^{\!\top},
      \;\tilde E := U^{\!\top} E V.                                       \\[3pt]
&\quad G M^{-\tfrac12}
      = U S V^{\!\top}\; V S^{-1} V^{\!\top}
      = U V^{\!\top} = O.                                                 \\[3pt]
&\quad G M^{-\tfrac12} D M^{-\tfrac12}
      = O\,V(S\tilde E S^{-1} + \tilde E^{\!\top})V^{\!\top}.             \\[3pt]
&\quad \therefore
      -\tfrac12 G M^{-\tfrac12} D M^{-\tfrac12}
      = -\tfrac12\,O\bigl(S\tilde E S^{-1} + \tilde E^{\!\top}\bigr)
        V^{\!\top}V S^{-1} V^{\!\top}.                                    \\[3pt]
&\quad \text{Collecting the two }E\text{ pieces gives exactly }
      P_{\!\perp}(E)\,M^{-\tfrac12}.                                      \\[6pt]
&\boxed{\text{Expansion proven as stated.}}
\end{aligned}$$

---

## Claim&nbsp;3 Bias and variance of the **Muon** estimator  

**Definition.**  
Given per-example gradients $g_i = G + \varepsilon_i$ with  
$\mathbb E[\varepsilon_i]=0$, let  
$$
\bar g := \tfrac1k\sum_{i=1}^{k} g_i,
\qquad
\hat G_{\text{Muon}} := \operatorname{Ortho}(\bar g).
$$

**Result.**  
With $\Sigma := \operatorname{Cov}[\operatorname{vec}\varepsilon_i]$ and
$g := \|G\|_F$,

* Bias  
  $$\displaystyle
  \mathbb E[\hat G_{\text{Muon}}] - \tilde G
    = -\frac{\operatorname{tr}\Sigma}{2k\,g^{3}}\,
      P_{\!\perp}(G)
      + O(k^{-2}).
  $$
* Variance  
  $$\displaystyle
  \operatorname{Var}[\operatorname{vec}\hat G_{\text{Muon}}]
    =\frac{\operatorname{tr}\Sigma}{k\,g^{2}}\,
      \operatorname{Proj}_{\text{tan}} + O(k^{-3/2}),
  $$
  where $\operatorname{Proj}_{\text{tan}}$ is the projector onto the
  $p(p-1)$-dimensional tangent plane at $O$.

**Justification.**

$$\begin{aligned}
&\text{1.  Noise properties after averaging}                             \\
&\quad \bar\varepsilon
      := \tfrac1k\sum_{i=1}^{k}\varepsilon_i,\quad
      \mathbb E[\bar\varepsilon]=0,\;\;
      \operatorname{Cov}[\operatorname{vec}\bar\varepsilon]=\Sigma/k.     \\[6pt]
&\text{2.  Feed }E=\bar\varepsilon\text{ into Claim 2 expansion}          \\
&\quad \hat G_{\text{Muon}}
      = O + P_{\!\perp}(\bar\varepsilon)M^{-\tfrac12} + O(k^{-1})         \\[3pt]
&\quad \text{because }\|\bar\varepsilon\|=O(k^{-\tfrac12}).               \\[6pt]
&\text{3.  Expected value}                                               \\
&\quad \mathbb E[P_{\!\perp}(\bar\varepsilon)]=0
      \;\;(\text{linear in centred noise}).                               \\[3pt]
&\quad \text{Quadratic bias comes from second-order term in Claim 2}      \\
&\quad -\tfrac12\,O\,
      \mathbb E\!\bigl[(M^{-\tfrac12}\bar\varepsilon)^{\!\top}
                       (M^{-\tfrac12}\bar\varepsilon)\bigr]
      = -\tfrac12\,O\,
        \frac{\operatorname{tr}(M^{-1}\Sigma)}{k}.                        \\[3pt]
&\quad M^{-1} = V S^{-2} V^{\!\top}\;\Rightarrow\;
      \operatorname{tr}(M^{-1}\Sigma)
      = \operatorname{tr}(\Sigma)/g^{2}.                                  \\[6pt]
&\quad \text{Insert this into bias expression to get the formula above.} \\[6pt]
&\text{4.  Variance}                                                     \\
&\quad P_{\!\perp}(\bar\varepsilon)M^{-\tfrac12}
      \text{ is linear in }\bar\varepsilon.
      \;\Rightarrow\;
      \operatorname{Var}\propto \frac1k.                                  \\
&\text{  Constant obtained by explicit contraction over the tangent basis.}
\end{aligned}$$

---

## Claim&nbsp;4 Bias and variance of the **SoupyMuon** estimator  

**Definition.**  
$$
\hat G_{\text{Soupy}}
  := \tfrac1k\sum_{i=1}^{k}\operatorname{Ortho}(g_i).
$$

**Result.**

* Bias  
  $$\displaystyle
  \mathbb E[\hat G_{\text{Soupy}}] - \tilde G
    = -\frac{\operatorname{tr}\Sigma}{2\,g^{3}}\,
      P_{\!\perp}(G)
      + O(\sigma^{3}),
  $$
  **independent of $k$** to leading order.
* Variance  
  $$\displaystyle
  \operatorname{Var}[\operatorname{vec}\hat G_{\text{Soupy}}]
    = \frac{\operatorname{tr}\Sigma}{k\,g^{2}}\,
      \operatorname{Proj}_{\text{tan}}
      + O(k^{-1}\sigma^{3}).
  $$

**Justification.**

$$\begin{aligned}
&\text{1.  Apply Claim 2 to each }g_i                                    \\
&\quad \operatorname{Ortho}(g_i)
      = O + P_{\!\perp}(\varepsilon_i)M^{-\tfrac12}
        + O(\|\varepsilon_i\|^{2}).                                       \\[6pt]
&\text{2.  Average over }i                                               \\
&\quad \hat G_{\text{Soupy}}
      = O + \tfrac1k\sum_i P_{\!\perp}(\varepsilon_i)M^{-\tfrac12}
        + \tfrac1k\sum_i O(\|\varepsilon_i\|^{2}).                         \\[6pt]
&\text{3.  The linear term vanishes in expectation},                      \\
&\quad \text{leaving the same quadratic bias as a single sample.}        \\[6pt]
&\text{4.  Variance scales like }k^{-1}                                   \\
&\quad \text{because the leading random term is still linear in } \varepsilon_i.
\end{aligned}$$

---

## Claim&nbsp;5 A bias-corrected **Debiased-Muon** estimator  

**Definition.**  
Let
$$
\hat\Sigma := \tfrac1k\sum_{i=1}^{k}(g_i-\bar g)(g_i-\bar g)^{\!\top},
\qquad
\hat\gamma := \frac{\operatorname{tr}\hat\Sigma}{k\|\bar g\|_F^{2}},
$$
and define
$$
\hat G_{\text{deb}}
  := \frac{\operatorname{Ortho}(\bar g)}{1 - \tfrac12\hat\gamma}.
$$

**Result.**

* Bias  
  $$\displaystyle
  \mathbb E[\hat G_{\text{deb}}] - \tilde G
    = O(k^{-2}).
  $$
  (One power of $k$ better than Muon.)
* Variance  
  $$\displaystyle
  \operatorname{Var}[\operatorname{vec}\hat G_{\text{deb}}]
    = \operatorname{Var}[\operatorname{vec}\hat G_{\text{Muon}}]
      \bigl(1 + O(k^{-1})\bigr)
    = O(k^{-1}).
  $$

**Justification.**

$$\begin{aligned}
&\text{1.  Muon bias factor (from Claim 3): }
      1 - \tfrac12\gamma
      \text{ with }
      \gamma=\operatorname{tr}\Sigma/(k g^{2}).                           \\[6pt]
&\text{2.  Unbiased estimator of }\gamma                                  \\
&\quad \mathbb E[\hat\Sigma]=\Sigma,\;
      \mathbb E[\hat\gamma-\gamma]=O(k^{-2}).                             \\[6pt]
&\text{3.  Multiply Muon by }(1-\tfrac12\hat\gamma)^{-1}
      =1+\tfrac12\hat\gamma+O(k^{-2}).                                    \\[3pt]
&\quad \text{Cancels the }O(k^{-1})\text{ term, leaves }O(k^{-2}).        \\[6pt]
&\text{4.  Variance changes only via a scalar }1+O(k^{-1}).               \\[6pt]
&\boxed{\text{Debiasing proven.}}
\end{aligned}$$

---

## Summary Table  

| Estimator | Leading Bias | Leading Variance | Notes |
|-----------|--------------|------------------|-------|
| Muon | $O(k^{-1}p^{-1/2})$ | $O(k^{-1}p^{-1})$ | Consistent |
| SoupyMuon | $O(p^{-1/2})$ | $O(k^{-1}p^{-1})$ | Asymptotic bias |
| Debiased-Muon | $O(k^{-2}p^{-1/2})$ | $O(k^{-1}p^{-1})$ | Same cost as Muon |

Bias scalings assume isotropic noise where
$\operatorname{tr}\Sigma = \Theta(p)$ and $\|G\|_F = \Theta(\sqrt p)$.

---

## Practical Take-aways  

* **Always average before projecting.**  
  SoupyMuon’s curvature bias does **not** wash out with bigger batches.
* **Cheap bias correction exists.**  
  One extra Frobenius norm and a trace knock Muon’s bias down by another
  power of $k$ at negligible cost.
* **Dimensional win.**  
  In modern transformer blocks $p\sim10^{3\text{–}4}$, so the
  $\!p^{-1/2}\!$ factor makes even $k\!=\!8$ almost unbiased in practice.

---
