# Noise-Aware Spectral Projection from Rectangular Wishart (Finite-Size First Principles)

This note develops a fully finite-size method to (i) estimate the noise scale $\sigma$ in a noisy gradient matrix and (ii) separate noise “bulk” from signal “spikes”, (iii) select the spike count with an explicit model criterion, and (iv) predict spectral projection coefficients (SPC) using empirically calibrated, finite-$(p,n)$ mappings. Everything is built on the **rectangular Wishart** distribution of singular values for Gaussian noise at the **actual layer shape** $(p,n)$.

---

## 1) Noise-only reference: rectangular Gaussian and its singular values

Let $E\in\mathbb{R}^{p\times n}$ with entries $E_{ij}\sim\mathcal{N}(0,\sigma^2)$ i.i.d.  
The singular values of $E$ are the square roots of the eigenvalues of the smaller Gram matrix:
- If $p\le n$: $G = EE^\top \in\mathbb{R}^{p\times p}$ (Wishart with $n$ degrees of freedom).
- If $p > n$: $H = E^\top E \in\mathbb{R}^{n\times n}$ (Wishart with $p$ degrees of freedom).

Denote the singular values by $s_1\ge \dots \ge s_m\ge 0$, where $m=\min(p,n)$.  
Two crucial, finite-size facts:

1. **Scale invariance.** If $E=\sigma Z$ with $Z_{ij}\sim\mathcal{N}(0,1)$, then the singular values obey
   $$
   s_i(E)=\sigma\, s_i(Z).
   $$
   Thus the **shape** $(p,n)$ fixes the *shape* of the singular-value distribution at $\sigma=1$; general $\sigma$ just multiplies it.

2. **Finite-size deviations are real.** The noise-only spectrum depends noticeably on $(p,n)$ at practical sizes (e.g., 1024×1024, 4096×1024). Edges fluctuate; the bottom tail shows a hard edge; the top singular exhibits finite-$m$ variability. Relying on an asymptotic law can bias scale estimates and spike decisions. We therefore **precompute** the noise-only distribution at each $(p,n)$ we care about.

---

## 2) Precomputed quantile tables at $\sigma=1$

For each shape $(p,n)$, we Monte Carlo:
1. Draw $E^{(\ell)}\sim \mathcal{N}(0,1)^{p\times n}$ for $\ell=1,\dots,L$.
2. Compute singular values of each draw (via the small Gram matrix).
3. Aggregate all singular values across draws and tabulate the **empirical quantile function**
   $$
   Q_1(u)\quad\text{for}\quad u\in(0, u_{\max})\subset(0,1),
   $$
   where $Q_1(u)$ returns the $\sigma\!=\!1$ singular value at quantile level $u$.

At runtime, the **noise-only quantiles** at arbitrary $\sigma$ are given by the scale rule
$$
Q_\sigma(u)=\sigma\,Q_1(u),
$$
and the **noise-only CDF** is obtained by inverting the table:
$$
F_\sigma(s)=F_1\!\left(\frac{s}{\sigma}\right)\ \ \text{via}\ \ F_1(s)=\text{the CDF corresponding to }Q_1.
$$

This gives a **finite-$(p,n)$**, closed-form-free description of the noise singulars that we can interpolate cheaply and exactly for our layer.

---

## 3) Estimating the noise scale $\hat\sigma$ from the bottom tail

Given an observed singular spectrum $s_{(1)}\le \dots \le s_{(m)}$ (e.g., from an innovation matrix or a gradient matrix dominated by noise at the bottom), we fit a **single scale** on the bottom portion where spikes are unlikely:

- Map ranks to quantiles using the **full-$m$** normalization:
  $$
  u_i \approx \frac{i-0.5}{m}\quad\text{or Blom’s correction}\quad u_i \approx \frac{i-0.375}{m+0.25}.
  $$
- For a candidate bottom window $i\in\{t+1,\dots,k\}$ (with a tiny trim $t$ to avoid the very-hard edge), solve the 1-parameter regression
  $$
  \min_{\sigma>0} \sum_{i=t+1}^k \left(s_{(i)} - \sigma\,Q_1(u_i)\right)^2.
  $$
  The solution is
  $$
  \hat\sigma=\frac{\sum_i Q_1(u_i)\,s_{(i)}}{\sum_i Q_1(u_i)^2}.
  $$

We scan $(k,t)$ on small grids and pick the pair that minimizes a simple score penalizing tiny windows and excessive trimming. This yields a robust, **finite-size-aware** $\hat\sigma$.

---

## 4) Spike + noise as a finite-size mixture, with explicit model selection

Let the observed singular histogram (on a log-spaced grid) be counts $c_b$ in bins with edges $\{e_b\}$.  
We model the spectrum as a two-component mixture:

- **Noise component.** Proportion $\rho\in(0,1]$, with bin masses from the tabulated CDF:
  $$
  p_b^{\text{noise}}(\sigma)=F_\sigma(e_{b+1})-F_\sigma(e_b).
  $$
- **Spike component.** Proportion $(1-\rho)$, modeled as **point masses at the top-$r$ observed singulars** (equal weight). This is a pragmatic, finite-$m$ model for sharp outliers.

The **expected bin counts** are
$$
\mu_b(\sigma,\rho,r)
= N\left[\rho\cdot p_b^{\text{noise}}(\sigma) + (1-\rho)\cdot p_b^{\text{spike}}(r)\right],
\quad N=\sum_b c_b.
$$
We fit $(\sigma,\rho,r)$ by minimizing the **Poisson deviance**
$$
D=2\sum_b \left(\mu_b - c_b + c_b\log\frac{c_b}{\mu_b}\right),
$$
and select $r$ by **BIC**:
$$
\text{BIC}=D + k\log m,\qquad k=2+r,
$$
counting parameters for $\sigma$, $\rho$, and the $r$ spike locations (treated as parameters). This is a **spike-aware**, **finite-size** criterion.

---

## 5) From spikes to spectral projection coefficients (SPC) without asymptotics

Define the **spectral projection coefficient** for a rank-1 target $T=uv^\top$ against a basis $\{(u_i,v_i)\}$ by
$$
\operatorname{spc}_i = \langle u,u_i\rangle\,\langle v,v_i\rangle,
$$
which is exactly the optimal coefficient when projecting $T$ onto the atoms $\{u_i v_i^\top\}$ in Frobenius norm.

In practice we do not know $T$; we instead observe a noisy matrix $G=S+E$. After:
1) estimating $\hat\sigma$ and the spike count $\hat r$ with the **finite-size mixture**, and  
2) identifying the **super-noise** singular values (the top $\hat r$),

we want the **expected alignment** between the empirical singular directions of $G$ and the latent signal directions of $S$. Rather than invoking an asymptotic mapping, we keep the **finite-size** stance:

- **Empirical calibration (finite-$p,n$).** For each shape $(p,n)$ and a grid of noise scales $\sigma$ and spike strengths, simulate $S+E$, record
  - the empirical singular values $s$ (whitened as $y=s/\sigma$),
  - and the realized alignments $\langle u,u_i\rangle\langle v,v_i\rangle$ between empirical and true singular vectors.
- Tabulate the **conditional expectation** and **spread**:
  $$
  \mathbb{E}\!\left[\operatorname{spc}\mid (p,n),\,y\right],\qquad
  \operatorname{Var}\!\left[\operatorname{spc}\mid (p,n),\,y\right].
  $$
- At runtime, with $(\hat\sigma,\hat r)$ and the whitened empirical singulars $y_i=s_i/\hat\sigma$, read off a **finite-size predicted SPC** from the calibration table:
  $$
  \widehat{\operatorname{spc}}(y_i;\,p,n)\ \ \text{(optionally with uncertainty bands)}.
  $$

This keeps the entire pipeline **rectangular-Wishart grounded**:
- the **noise** side is handled exactly via $Q_1$ tables,
- the **spike** side is handled by **finite-size Monte Carlo calibration** of alignment vs. whitened singular magnitude $y$.

---

## 6) Why finite-size rectangular Wishart is preferable in practice

- **Correct edges at your $(p,n)$**: bottom hard edge variability and top-edge fluctuations are captured by the tables; no asymptotic edge formulas needed.
- **Reliable scale estimation**: $\hat\sigma$ comes from direct regressions against $Q_1$, not from asymptotic quantiles.  
- **Principled spike selection**: BIC on a finite-size mixture (noise CDF from tables + point spikes) performs well when the bottom tail is contaminated or the layer is square.  
- **SPC without asymptotics**: a simulation-based calibration yields alignment as a function of whitened strength $y$ at the exact shape, which is what matters for real layers.

---

## 7) Practical recipe (drop-in)

1. **Precompute** $Q_1(u)$ tables for the shapes you train (e.g., 4096×1024, 1024×1024, 1024×4096).  
2. For each observed spectrum:
   - **Estimate $\hat\sigma$** by bottom-tail regression to $Q_1$ with a tiny trim and a short $(k,t)$ scan.
   - **Fit spike+noise** by minimizing Poisson deviance on binned counts and choose $\hat r$ by BIC.
3. **Predict SPC**:
   - Whiten $y_i=s_i/\hat\sigma$.
   - Look up $\widehat{\operatorname{spc}}(y_i;\,p,n)$ from your finite-size calibration table (with optional uncertainty).
4. **(Optional) Odd-polynomial acceleration**:
   - If you approximate a scalar map $h(y)$ by odd polynomials and apply it spectrally via iterative odd powers, you can turn $\hat U\,h(\hat S)\,\hat V^\top$ into a **noise-aware** update rule.  
   - Because your calibration is in terms of the **whitened** argument $y$, you only need a **scalar rescaling** by $\hat\sigma$ at runtime; the odd-polynomial coefficients remain shape-specific but **sigma-agnostic**.

---

## 8) Notes on robustness

- Bottom-tail regressions should normalize ranks by **$m$**, not by the size of the chosen window, to align with the global noise quantiles.  
- A small trim (a few indices) is enough to avoid the very-hard edge; scanning $k$ avoids bias when weak spikes leak into the bottom tail.  
- The mixture fit is cheap: tabulated $F_\sigma$ makes expected counts a few vectorized interpolations; BIC selection over $r$ up to a modest cap (e.g., 64) is fast.  
- For SPC calibration, you can amortize costs: build shape-specific lookup tables once and reuse.

---

## 9) Summary

Everything above stays in **finite-$(p,n)$ rectangular Wishart**:
- Use **tabulated quantiles** $Q_1(u)$ to estimate $\sigma$ and to build a likelihood for the spectrum.  
- Use a **BIC-penalized spike+noise mixture** to decide how many spikes are present.  
- Predict **SPC** from a **finite-size, simulation-calibrated mapping** $\operatorname{spc}(y;\,p,n)$ with $y=s/\hat\sigma$.

This gives a cohesive, practical pipeline that matches the true finite behavior of your layers and plugs directly into fast odd-polynomial implementations for spectral operators.
