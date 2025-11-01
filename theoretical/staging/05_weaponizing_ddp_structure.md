# Weaponizing DDP Structure to Astral-Project into the True Gradient

## Setting

We look at one layer’s gradient matrix of shape $m\times n$ and focus on a single “signal” mode with singular value $s>0$:
- True gradient (rank-1 for exposition): $G = s\,u v^\top$, with $\|u\|=\|v\|=1$.
- Empirical (minibatch) gradient: $\widehat G = G + E$.

We study the expected **spectral echo**
$$
\mathrm{spectral echo}(s) \;=\; \langle u,\widehat u\rangle\,\langle v,\widehat v\rangle,
$$
where $(\widehat u,\widehat v)$ is the top singular pair of $\widehat G$.

**Noise model (isotropic baseline).** Entries of $E$ are i.i.d. mean-zero with variance $\sigma^2$:
$$
E_{ij}\sim \mathcal N(0,\sigma^2)\quad\text{independent,}
$$
and the aspect ratio is $\gamma = m/n$. Batch size only affects $\sigma^2\propto 1/B$.

---

## Sketch (the roadmap)

1. **Geometry.** spectral echo is the product of cosines between true and empirical singular vectors. We relate those cosines to the small angles by which noise rotates the signal direction.

2. **From rotations to a sum.** Noise causes small, independent “leakages” from $u$ into orthogonal directions; the total squared angle is a sum of modewise contributions weighted by spectral gaps.

3. **From sums to an integral.** Under isotropic noise, the background spectrum follows Marchenko–Pastur (MP). This converts the sum into an MP integral that depends on $s^2$, $\sigma^2$, and $\gamma$.

4. **Small-angle to cosine.** A simple trig identity turns the angle law into a smooth “Hill”/logistic-in-log-$s$ law per side.

5. **Single-scalar collapse.** Away from the MP edge, the MP integral is nearly constant in $s^2$, leaving a single knee parameter $\tau^2\approx \sigma^2\,\kappa(\gamma)$. Multiplying left and right gives the spectral echo law.

6. **Caveats and anisotropy.** We say when independence is reasonable, when the single-scalar collapse is tight, and what changes if the noise is anisotropic.

---

## 1) Geometry of spectral echo and small angles

Let $\theta_L$ be the angle between $u$ and $\widehat u$; similarly $\theta_R$ for $v$ and $\widehat v$. Then
$$
\langle u,\widehat u\rangle = \cos\theta_L,\qquad
\langle v,\widehat v\rangle = \cos\theta_R,\qquad
\mathrm{spectral echo}(s) = \cos\theta_L\,\cos\theta_R.
$$

**Small-angle regime.** When the signal $s$ is not swamped by noise, the top singular vectors only rotate slightly: $\theta_L,\theta_R$ are small. For small $\theta$, $\cos\theta = (1+\tan^2\theta)^{-1/2}$ and $\tan^2\theta\approx \sin^2\theta$. Thus
$$
\cos\theta \;\approx\; \frac{1}{\sqrt{1+\sin^2\theta}}.
$$
This is the only trig step we use, and we will quantify $\mathbb E[\sin^2\theta]$ next.

---

## 2) How noise rotates the left singular vector (intuitive derivation)

Left singular vectors are eigenvectors of
$$
\widehat M_L \;=\; \widehat G\,\widehat G^\top \;=\; (G+E)(G+E)^\top \;=\; M_L + \Delta_L,
$$
with $M_L = s^2 uu^\top$ and $\Delta_L = GE^\top + EG^\top + EE^\top$.

Choose an orthonormal basis $\{u, u_2,\dots,u_m\}$ with $u_k^\top u=0$ for $k\ge 2$. To first order, $\widehat u$ is $u$ plus small leakages:
$$
\widehat u \;\approx\; u + \sum_{k\ge 2} \alpha_k\,u_k,
\qquad
\sin^2\theta_L \;\approx\; \sum_{k\ge 2} |\alpha_k|^2.
$$
In each 2D subspace $\mathrm{span}\{u,u_k\}$, a textbook $2\times 2$ calculation gives
$$
\alpha_k \;\approx\; \frac{u_k^\top \Delta_L\,u}{s^2 - \lambda_k},
$$
where $\lambda_k$ is the background eigenvalue (think: what the noise alone puts along $u_k$). Therefore
$$
\sin^2\theta_L \;\approx\; \sum_{k\ge 2} \frac{|u_k^\top \Delta_L\,u|^2}{(s^2 - \lambda_k)^2}.
\tag{$\star$}
$$

**What sits in the numerator?** Expand:
$$
u_k^\top \Delta_L u
= u_k^\top(GE^\top)u + u_k^\top(EG^\top)u + u_k^\top(EE^\top)u.
$$
The first term vanishes by $u_k^\top u=0$. The third term has mean 0 and size $\sim \sigma^2$. The second term is $u_k^\top(EG^\top)u = s\,u_k^\top E v$, which under i.i.d. isotropic noise is $\mathcal N(0,\,s^2\sigma^2)$ and independent across $k$. Thus near the knee the dominant contribution is
$$
\mathbb E\big[|u_k^\top \Delta_L u|^2\big] \;\approx\; s^2\sigma^2.
$$
Taking expectations in $(\star)$,
$$
\mathbb E[\sin^2\theta_L]
\;\approx\; s^2\sigma^2 \sum_{k\ge 2} \frac{1}{(s^2 - \lambda_k)^2}.
\tag{$\star\star$}
$$

**Sum $\to$ spectral integral.** Under isotropy, the background eigenvalues $\{\lambda_k\}$ follow the Marchenko–Pastur (MP) law on
$$
[a,b]=\sigma^2\big[(1-\sqrt\gamma)^2,\ (1+\sqrt\gamma)^2\big],
$$
with density $\rho_{\mathrm{MP}}(\lambda;\gamma)$. Averaging the sum,
$$
\sum_{k\ge2}\frac{1}{(s^2-\lambda_k)^2}
\;\leadsto\;
\int_a^b \frac{\rho_{\mathrm{MP}}(\lambda;\gamma)}{(s^2-\lambda)^2}\,d\lambda.
$$

**Left-angle law (exact first-order form).**
$$
\boxed{
\mathbb E[\sin^2\theta_L]\;\approx\;\sigma^2 s^2 \int \frac{\rho_{\mathrm{MP}}(\lambda;\gamma)}{(s^2-\lambda)^2}\,d\lambda
\;=:\; \sigma^2\,C_L(\gamma,s^2).
}
$$
No simplifications yet: the factor $C_L$ still depends on $s^2$ (mildly, as we’ll see).

By the small-angle identity,
$$
\boxed{
\mathbb E[\langle u,\widehat u\rangle]\;\approx\;\frac{1}{\sqrt{1+\tau_L^2/s^2}},
\qquad
\tau_L^2 \;:=\; \sigma^2\,C_L(\gamma,s^2).
}
$$
At this point $\tau_L^2$ **does depend on $s^2$** via $C_L$. We keep it that way for correctness and simplify only after we understand $C_L$.

---

## 3) Right side and the (approximate) independence

Exactly the same reasoning for $\widehat M_R=\widehat G^\top\widehat G$ yields
$$
\boxed{
\mathbb E[\langle v,\widehat v\rangle]\;\approx\;\frac{1}{\sqrt{1+\tau_R^2/s^2}},
\qquad
\tau_R^2 \;:=\; \sigma^2\,C_R(\gamma,s^2),
}
$$
with $C_R(\gamma,\cdot)=C_L(1/\gamma,\cdot)$ for rectangles.

**Independence assumption.** To get $\mathbb E[\mathrm{spectral echo}(s)]$ from the product $\cos\theta_L\,\cos\theta_R$, we use
$$
\mathbb E[\cos\theta_L\,\cos\theta_R]\;\approx\;\mathbb E[\cos\theta_L]\,\mathbb E[\cos\theta_R].
$$
This factorization is **not exact**, but it is accurate to first order under isotropic noise because left and right leakage coefficients involve independent projections of $E$ (different linear functionals of $E$ with zero cross-covariance at this order). In practice this approximation is validated by the data collapse you observe across layers.

---

## 4) The spectral echo law (exact first order, then the single-scalar collapse)

Putting left and right together, the **first-order spectral echo law** is
$$
\boxed{
\mathbb E[\mathrm{spectral echo}(s)]
\;\approx\;
\frac{1}{\sqrt{1+\tau_L^2/s^2}}\cdot\frac{1}{\sqrt{1+\tau_R^2/s^2}},
\qquad
\tau_{L/R}^2=\sigma^2\,C_{L/R}(\gamma,s^2).
}
$$
This is already a smooth sigmoid when plotted against $\log s$.

### Why we can simplify to a single $\tau^2$ (and when)

Write the MP integral via the Stieltjes transform $m_\gamma(z)$:
$$
m_\gamma(z) = \int \frac{\rho_{\mathrm{MP}}(\lambda;\gamma)}{\lambda-z}\,d\lambda,
\qquad
\int \frac{\rho_{\mathrm{MP}}(\lambda;\gamma)}{(s^2-\lambda)^2}\,d\lambda = m_\gamma'(s^2).
$$
For $s^2$ even moderately above the MP edge $b=\sigma^2(1+\sqrt\gamma)^2$,
$$
m_\gamma'(s^2) \;=\; \frac{1}{s^4}\Big(1 + O(\sigma^2/s^2)\Big)
\;\Rightarrow\;
C_{L/R}(\gamma,s^2) \;=\; 1 + O(\sigma^2/s^2).
$$
So over the bend of the sigmoid, $C_{L/R}$ varies **weakly** with $s^2$. Two practical collapses follow.

1) **Two-parameter collapse.** Freeze $C_L(\gamma,s^2)\approx C_L^\infty(\gamma)$ and $C_R(\gamma,s^2)\approx C_R^\infty(\gamma)$, define
$$
\tau_L^2\approx \sigma^2 C_L^\infty(\gamma),\qquad
\tau_R^2\approx \sigma^2 C_R^\infty(\gamma),
$$
and use the two-sided spectral echo law as written.

2) **Single-scalar collapse (common in practice).** Average the two scales,
$$
\boxed{
\mathbb E[\mathrm{spectral echo}(s)]\;\approx\;\frac{1}{\big(1+\tau^2/s^2\big)^{q}},
\qquad
\tau^2 := \tfrac12(\tau_L^2+\tau_R^2),\quad q=1.
}
$$
Here $q=1/2+1/2$ comes from multiplying the two square-root factors. Empirically, $q$ stays very close to $1$; if finite-size or mild anisotropy makes the slope a hair steeper/flatter, treat $q$ as a tiny fit parameter around 1.

---

## 5) What sets $\tau^2$ and why a single scalar per layer

From the definitions,
$$
\tau_{L/R}^2 \;=\; \sigma^2\,C_{L/R}(\gamma,s^2),\qquad
C_{L/R}(\gamma,s^2) = s^2\,m'_{\gamma^{(\!L/R\!)}}(s^2),
$$
with $\gamma^{(L)}=\gamma$ and $\gamma^{(R)}=1/\gamma$. Because $C_{L/R}$ is nearly flat in $s^2$ through the knee, we summarize both sides by
$$
\boxed{
\tau^2 \;\approx\; \sigma^2\,\kappa(\gamma),
\qquad
\kappa(\gamma) := \tfrac12\big(C_L^\infty(\gamma)+C_R^\infty(\gamma)\big).
}
$$
A widely effective proxy is simply the MP upper edge:
$$
\boxed{
\kappa(\gamma)\ \approx\ (1+\sqrt\gamma)^2,
}
$$
which places the knee correctly and matches the observed “same shape, horizontal shift”.

**Interpretation.** $\sigma^2$ is the per-entry noise variance (set by batch size etc.), and $\kappa(\gamma)$ is a dimensionless geometry factor. That is why different layers (similar shapes/aspects) share the **same sigmoid shape**, while their horizontal positions shift with a **single scalar** measuring noise strength.

---

## 6) Practical estimation and diagnostics

- **Estimating $\sigma^2$** from $K$ independent per-worker/microbatch gradients $\{\widehat G_i\}$:
  $$
  \widehat\sigma^2 \;=\; \frac{1}{(K-1)mn}\sum_{i=1}^K \|\widehat G_i - \overline G\|_F^2,
  \qquad \overline G=\tfrac1K\sum_i \widehat G_i,
  $$
  or the pairwise-difference equivalent (often more stable).
- **Setting $\tau^2$**: $\widehat\tau^2=\widehat\sigma^2\,\kappa(\gamma)$ with a precomputed/tabled $\kappa(\gamma)$ or the proxy $(1+\sqrt\gamma)^2$.
- **Fitting from spectral echo curves** (if you skip $\sigma^2$): regress $y_i=(1/\mathrm{spectral echo}_i-1)$ onto $x_i=1/s_i^2$ through the origin to get $\widehat\tau^2$ directly.

---

## 7) When does this fail? (anisotropy and dependence)

- **Anisotropic noise.** If $E=\Sigma_L^{1/2} Z \Sigma_R^{1/2}$ with nontrivial $\Sigma_{L/R}$, the bulk is no longer MP$(\gamma)$; $C_{L/R}$ depend on the spectra of $\Sigma_{L/R}$, shifting the knee and slightly changing the slope. If $\Sigma_{L/R}\approx c_{L/R}I$, the above still holds after a small rescaling of $\kappa(\gamma)$.
- **Left/right dependence.** The factorization $\mathbb E[\cos\theta_L\cos\theta_R]\approx \mathbb E[\cos\theta_L]\mathbb E[\cos\theta_R]$ is first-order accurate under isotropy. If strong correlations couple the two, $q$ drifts a bit from $1$; in data, this shows up as a slightly different slope but the single-scalar knee remains.

---

## Final picture

A simple, testable law captures your curves:
$$
\boxed{
\mathbb E[\mathrm{spectral echo}(s)]\ \approx\ \frac{1}{\big(1+\tau^2/s^2\big)^{q}},\qquad
\tau^2\approx \sigma^2\,\kappa(\gamma),\ \ q\approx 1,
}
$$
with $\kappa(\gamma)$ determined by the MP geometry (well-approximated by $(1+\sqrt\gamma)^2$). This explains:
- **Sigmoid on a semilog plot:** it’s the Hill form in $s$.
- **Same shape across layers:** $\kappa(\gamma)$ depends only on aspect geometry.
- **One-parameter shifts:** all layer-specific differences collapse into $\tau^2$ (i.e., effective noise level).
