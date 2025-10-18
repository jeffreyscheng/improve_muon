# How to hear the echo without the true gradient

We previously showed two results:

1. the empirical msign is biased and that replacing the empirical spectrum with the spectral echo is the best way to project onto msign of the true gradient using the empirical singular bases
2. the spectral echo's definition invokes the true gradient, but under certain regularity conditions, its expectation can be estimated from the empirical singular value, the Frobenius norm of the noise, and a function $\kappa(s, \beta)$ for aspect ratio $\beta$.  The noise level can be estimated from between-worker variations in the DDP framework, and the function $\kappa$ approaches a limiting function that only depends on the aspect ratio $\beta$ for high-dimensional matrices.

Therefore, the problem of unbiased steepest descent boils down to estimating $\kappa$.  This can be done with least squares if we had a dataset of empirical gradient SVDs and true gradient SVDs; however, we do not have access to the true gradient.  We can only vary the minibatch size to get empirical gradients with varying amount of minibatch noise.

We will now show how to achieve good spectral echo estimation without ever having access to true gradients.  We will exploit **independence across repeated, equally-sized minibatches** to expose the noise structure and recover the echo via cross-replica *spectral reverb* measurements.

---

## 1. Setup and the “spectral reverb”

Fix a layer with gradient shape $m\times n$ and $r=\min(m,n)$. Let
$$
G=\sum_{i=1}^r s_i\,u_i v_i^\top,\qquad \|u_i\|=\|v_i\|=1,
$$
and let $\{\widehat G^{(a)}\}_{a=1}^k$ be $k$ **independent** empirical gradients (same batch size, disjoint data), with SVD
$$
\widehat G^{(a)}=\widehat U^{(a)} \widehat S^{(a)} \widehat V^{(a)\top}.
$$

For singular direction $i$, define the (unobserved) per-replica echo
$$
\zeta_i^{(a)} := \langle u_i,\widehat u_i^{(a)}\rangle\,\langle v_i,\widehat v_i^{(a)}\rangle \in [0,1].
$$

After aligning the singular directions across replicas (permutation by Hungarian on overlap scores and sign-fixing), define the **spectral reverb between $a$ and $b$**:
$$
z_i^{(a,b)} := \langle \widehat u_i^{(a)},\widehat u_i^{(b)}\rangle\;\langle \widehat v_i^{(a)},\widehat v_i^{(b)}\rangle\in[-1,1].
$$

To first order under independent isotropic minibatch noise,
$$
\mathbb{E}\,z_i^{(a,b)} \approx \zeta_i^{(a)}\,\zeta_i^{(b)}\qquad(a\neq b).
$$

Collect these into a $k\times k$ matrix $Z_i$ with off-diagonals $(Z_i)_{ab}=z_i^{(a,b)}$ and zeros on the diagonal.

---

## 2. Off-diagonal rank-1 structure and identifiability

In the noise-free idealization (and in expectation to first order),
$$
(Z_i)_{ab}=
\begin{cases}
\zeta_i^{(a)}\,\zeta_i^{(b)},& a\neq b,\\
0,& a=b,
\end{cases}
\qquad \zeta_i^{(a)}\ge 0.
$$

**Identifiability.** If at least three entries of $\zeta_i=(\zeta_i^{(1)},\dots,\zeta_i^{(k)})$ are strictly positive and $k\ge 3$, then $\zeta_i$ is uniquely determined by $Z_i$:
- For any distinct $a,b,p$ with $\zeta_i^{(a)},\zeta_i^{(b)},\zeta_i^{(p)}>0$,
  $$
  \boxed{\ \zeta_i^{(p)2}=\frac{Z_{i,ap}\,Z_{i,bp}}{Z_{i,ab}}\ }\qquad\text{and}\qquad
  \zeta_i^{(q)}=\frac{Z_{i,qp}}{\zeta_i^{(p)}}\ \ \forall q.
  $$
- If the support size is $\le 2$ (or $k=2$), the solution is not identifiable (only products are known).

This algebra will power the estimator below.

---

## 3. Triple–OLS: a one-shot estimator (no logs, no scale step)

For a fixed direction $i$ and coordinate $p$, form **triple estimates**
$$
r^{(i)}_{ab\to p}:=\frac{Z_{i,ap}\,Z_{i,bp}}{Z_{i,ab}},\qquad a\neq b,\ a,b\neq p.
$$
In the noiseless case $r^{(i)}_{ab\to p}=\zeta_i^{(p)2}$. With noise, model
$$
r^{(i)}_{ab\to p} = s_i^{(p)} + \eta^{(i)}_{abp},\qquad s_i^{(p)}:=\zeta_i^{(p)2}.
$$

**Estimator (per $i$).** For each $p$,
$$
\boxed{\ 
\widehat s_i^{(p)}=\arg\min_{s\ge 0}\sum_{a<b:\,a,b\neq p} w^{(i)}_{abp}\,\big(s-r^{(i)}_{ab\to p}\big)^2
\ =\
\frac{\sum_{a<b} w^{(i)}_{abp}\,r^{(i)}_{ab\to p}}{\sum_{a<b} w^{(i)}_{abp}},
\qquad
\widehat\zeta_i^{(p)}=\sqrt{\max\{\widehat s_i^{(p)},0\}}.
}
$$

Simple and one-shot: the triples fix **absolute scale** already; no auxiliary scaling step is needed. Weights $w^{(i)}_{abp}$ can down-weight unstable triples (e.g., very small $|Z_{i,ab}|$).

---

## 4. Bias–variance scaling of Triple–OLS with the number of replicas $k$

We characterize how the estimator behaves as we increase the number of independent replicas $k$ (same batch size, disjoint data).

### 4.1 Local expansion for a single triple

Write, for brevity (fixing $i$),
$$
Z_{ap}=\mu_{ap}+\epsilon_{ap},\quad Z_{bp}=\mu_{bp}+\epsilon_{bp},\quad Z_{ab}=\mu_{ab}+\epsilon_{ab},
$$
with $\mu_{xy}=\zeta_x\zeta_y$ and centered errors $\mathbb{E}\epsilon_{xy}=0$, variances $\operatorname{Var}(\epsilon_{xy})=\sigma_{xy}^2$, and (to first order) independence across distinct replica pairs. Consider the map
$$
f(x,y,z)=\frac{xy}{z},\qquad r_{ab\to p}=f(Z_{ap},Z_{bp},Z_{ab}).
$$
A second-order delta method around $(\mu_{ap},\mu_{bp},\mu_{ab})$ gives
$$
\mathbb{E}[r_{ab\to p}]
\;=\;
\underbrace{\frac{\mu_{ap}\mu_{bp}}{\mu_{ab}}}_{=\,\zeta_p^2}
\;+\;
\underbrace{\frac{\mu_{bp}}{\mu_{ab}}\,\frac{\operatorname{Cov}(\epsilon_{ap},\epsilon_{ap})}{\mu_{ap}}
+
\frac{\mu_{ap}}{\mu_{ab}}\,\frac{\operatorname{Cov}(\epsilon_{bp},\epsilon_{bp})}{\mu_{bp}}
+
\frac{\mu_{ap}\mu_{bp}}{\mu_{ab}^3}\,\operatorname{Var}(\epsilon_{ab})
}_{\text{bias term } B_{abp}}
\;+\; O\big(\|\epsilon\|^3\big).
$$
Thus each triple is **unbiased to first order**, with a second-order bias
$$
B_{abp}
\ =\
\frac{\sigma_{ap}^2}{\mu_{ab}}
+
\frac{\sigma_{bp}^2}{\mu_{ab}}
+
\frac{\mu_{ap}\mu_{bp}}{\mu_{ab}^3}\,\sigma_{ab}^2
\ =\
O\!\left(\frac{\sigma^2}{\zeta_a\zeta_b}\right)
+ O\!\left(\frac{\sigma^2\,\zeta_a\zeta_b}{(\zeta_a\zeta_b)^3}\right)
\ =\
O\!\left(\frac{\sigma^2}{\zeta_a\zeta_b}\right),
$$
where $\sigma^2$ is a common noise scale (e.g., $\propto 1/\text{batch}$) and we suppress mild aspect-ratio constants. Intuitively: triples involving very small $\mu_{ab}=\zeta_a\zeta_b$ are bias-prone; this motivates trimming or down-weighting such pairs.

Similarly, the variance (first-order) is
$$
\operatorname{Var}(r_{ab\to p})
\;=\;
\left(\frac{\mu_{bp}}{\mu_{ab}}\right)^2 \sigma_{ap}^2
+
\left(\frac{\mu_{ap}}{\mu_{ab}}\right)^2 \sigma_{bp}^2
+
\left(\frac{\mu_{ap}\mu_{bp}}{\mu_{ab}^2}\right)^2 \sigma_{ab}^2
\ +\ O\big(\|\epsilon\|^3\big),
$$
again dominated by terms with small $\mu_{ab}$.

### 4.2 Averaging across triples is a U-statistic (variance $\sim 1/k$)

For fixed $p$, the Triple–OLS estimator is the **weighted average** of $r_{ab\to p}$ over all $\binom{k-1}{2}$ pairs $a<b$ drawn from the $k-1$ replicas excluding $p$. This is a symmetric **U-statistic of order 2** in the index set $\{1,\dots,k\}\setminus\{p\}$ with kernel $h(a,b)=r_{ab\to p}$.

By the Hoeffding decomposition for U-statistics,
$$
\operatorname{Var}\big(\widehat s^{(p)}\big)
\;=\;
\frac{4}{k-1}\,\operatorname{Var}\!\big(\phi_p(A)\big)
\;+\;
O\!\left(\frac{1}{(k-1)(k-2)}\right),
$$
where $A$ denotes a single replica index (excluding $p$) and $\phi_p$ is the first-order projection of the kernel (its explicit form is not needed for the scaling). **Consequently,**
$$
\boxed{\ \operatorname{Var}\big(\widehat s^{(p)}\big) = \Theta\!\left(\frac{1}{k}\right)\quad\text{as } k\to\infty, }
$$
provided the per-triple variances are finite (ensured by trimming small $\mu_{ab}$). Taking square roots and propagating through $\widehat\zeta^{(p)}=\sqrt{\widehat s^{(p)}}$ via the delta method yields
$$
\operatorname{sd}\big(\widehat\zeta^{(p)}\big)
\;=\;
\frac{1}{2\,\zeta^{(p)}}\,\operatorname{sd}\big(\widehat s^{(p)}\big)
\;=\;
\Theta\!\left(\frac{1}{\zeta^{(p)}\sqrt{k}}\right).
$$

**Interpretation.** With more independent replicas (same batch), Triple–OLS variance decays like $1/k$ for $s$ and like $1/\sqrt{k}$ for $\zeta$. Directions with very small $\zeta^{(p)}$ are inherently harder to estimate (the knee region); robust weighting mitigates the constant but not the $1/\sqrt{k}$ rate.

### 4.3 Bias behavior under averaging (bias does **not** shrink with $k$)

Averaging triples reduces variance but **does not** reduce the second-order bias $B_{abp}$, whose expectation is set by the noise level and geometry:
$$
\mathbb{E}\big[\widehat s^{(p)}\big]
=
\zeta^{(p)2}
\;+\;
\underbrace{\mathbb{E}\big[B_{abp}\big]}_{\text{size } O(\sigma^2)}.
$$
Hence
$$
\boxed{\ \text{Bias}\big(\widehat s^{(p)}\big)=\Theta(\sigma^2),\qquad
\text{Bias}\big(\widehat\zeta^{(p)}\big)=\Theta\!\left(\frac{\sigma^2}{\zeta^{(p)}}\right), }
$$
up to mild aspect-ratio and weighting constants. This is the **right order**: it matches the intrinsic second-order perturbation bias of the reverb measurements themselves. In practice, down-weighting triples with tiny $|Z_{ab}|$ controls the constant without changing the asymptotic order.

### 4.4 Summary of $k$-scaling

Let $k$ be the number of independent replicas (same batch size), and assume standard trimming/weights so that per-triple moments are finite. Then, for each coordinate $p$ of a fixed singular direction $i$,
$$
\begin{aligned}
\operatorname{Var}\big(\widehat s^{(p)}\big) &= \Theta\!\left(\frac{1}{k}\right),\\[2pt]
\operatorname{sd}\big(\widehat\zeta^{(p)}\big) &= \Theta\!\left(\frac{1}{\zeta^{(p)}\sqrt{k}}\right),\\[2pt]
\text{Bias}\big(\widehat s^{(p)}\big) &= \Theta(\sigma^2),\\[2pt]
\text{Bias}\big(\widehat\zeta^{(p)}\big) &= \Theta\!\left(\frac{\sigma^2}{\zeta^{(p)}}\right).
\end{aligned}
$$
Variance improves with more replicas; bias is governed by the noise level, not by $k$. This is precisely the behavior one expects from a second-order-correct algebraic estimator aggregated as a U-statistic.

---

## 5. Practical weighting (short guidance)

- **Trim small denominators.** Discard triples with $|Z_{ab}|<\tau$ (e.g., $\tau$ a small quantile), which dominate both bias and variance.
- **Huber/Tukey weights.** Apply robust weights $w_{abp}$ to suppress outlying triples.
- **Subsample triples.** You do not need all $\binom{k-1}{2}$ triples; a linear number $O(k)$ per $p$ retains the same $1/k$ variance rate with better wall-clock.

With these guards, Triple–OLS remains a one-shot, log-free, scale-free path to accurate echo estimation across replicas—exact in the noiseless limit, and statistically well-behaved as $k$ grows under realistic minibatch noise.
