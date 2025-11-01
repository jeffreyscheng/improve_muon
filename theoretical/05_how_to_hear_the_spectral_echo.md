# How to hear the echo without the true gradient

We previously showed that the empirical msign is biased and that replacing the empirical spectrum with the spectral echo is the best way to project onto msign of the true gradient using the empirical singular bases; however, calculating the spectral echo from its definition requires access to the true gradient, and our entire motivation comes from only having access to empirical gradients.

We will now show how to achieve good spectral echo estimation without ever having access to true gradients.  We will exploit **independence across repeated, equally-sized replicas** (DDP workers × accumulation steps, i.e., disjoint data at the same batch size) to expose the noise structure and recover the echo via cross-replica *spectral reverb* measurements.

---

## 1. Setup and the “spectral reverb”

Fix a layer with gradient shape $m\times n$ and $r=\min(m,n)$. Let
$$
G=\sum_{i=1}^r s_i\,u_i v_i^\top,\qquad \|u_i\|=\|v_i\|=1,
$$
and let $\{\widehat G^{(a)}\}_{a=1}^k$ be $k$ **independent** empirical gradients (same batch size, disjoint data) evaluated on the same model checkpoint, with SVDs
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

To first order under independent isotropic minibatch noise (observed across independent replicas),
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

If at least three entries of $\zeta_i=(\zeta_i^{(1)},\dots,\zeta_i^{(k)})$ are strictly positive and $k\ge 3$, then $\zeta_i$ is uniquely determined by $Z_i$. For any distinct $a,b,p$ with $\zeta_i^{(a)},\zeta_i^{(b)},\zeta_i^{(p)}>0$,
$$
\zeta_i^{(p)2}=\frac{Z_{i,ap}\,Z_{i,bp}}{Z_{i,ab}},
\qquad
\zeta_i^{(q)}=\frac{Z_{i,qp}}{\zeta_i^{(p)}}\ \ \forall q.
$$

This algebra underlies the estimator below.

---

## 3. Triple–OLS: least squares objective, analytic solution, and division-free form

For fixed direction $i$ and target replica $p$, define the per-triple ratio
$$
r_{ab\to p} \;:=\; \frac{Z_i(a,p)\,Z_i(b,p)}{Z_i(a,b)}\qquad\text{for } a\neq b,\ a,b\neq p.
$$

Estimate $s_i^{(p)}:=\zeta_i^{(p)2}$ by minimizing the weighted constant-only least squares objective
$$
J_p(s)\;=\;\sum_{\substack{a<b\\a,b\neq p}} w_{ab}\,\big(s-r_{ab\to p}\big)^2.
$$

The analytic minimizer is the weighted mean
$$
\widehat s_i^{(p)}\;=\;\frac{\sum_{a<b,\ a,b\neq p} w_{ab}\, r_{ab\to p}}{\sum_{a<b,\ a,b\neq p} w_{ab}}.
$$

Choosing $w_{ab}=Z_i(a,b)^2$ down-weights noise-dominated pairs and yields a division-free expression in the data
$$
\widehat s_i^{(p)}\;=\;\frac{\sum_{a,b=1}^k Z_i(a,p)\,Z_i(p,b)\,Z_i(a,b)}{\sum_{a,b=1}^k Z_i(a,b)^2},
\qquad
\widehat \zeta_i^{(p)}\;=\;\sqrt{\max\{\widehat s_i^{(p)},0\}}.
$$

The vectorized matrix form for all $p$ at once is
$$
\widehat{\boldsymbol{s}}_i
\;=\;
\frac{\big((Z_i Z_i)\odot Z_i^\top\big)\,\mathbf{1}_k}{\|Z_i\|_F^2}\in\mathbb{R}^k,
\qquad
\widehat{\boldsymbol{\zeta}}_i\;=\;\sqrt{\max\{\widehat{\boldsymbol{s}}_i,0\}}\in\mathbb{R}^k,
$$
where $\mathbf{1}_k$ is the all-ones vector and $\odot$ is the Hadamard product. The diagonal of $Z_i$ is zero, so no masking is needed.

---

## 4. Implementation with explicit shapes and matrix multiplications

Assume $n<m$ and full rank $r=n$. For each replica $b\in\{1,\dots,k\}$, the SVD bases are
$$
U_b\in\mathbb{R}^{n\times n},\qquad V_b\in\mathbb{R}^{m\times n}.
$$

Alignment is performed once per replica. Select a reference replica $r$. For each $b$, build
$$
M_b \;=\; \big|U_b^\top U_r\big|\odot\big|V_b^\top V_r\big|\in\mathbb{R}^{n\times n},
$$
which uses two matrix multiplications $(n\times n)^\top(n\times n)$ and $(m\times n)^\top(m\times n)$, then solve a linear assignment to obtain a permutation matrix $\Pi_b\in\{0,1\}^{n\times n}$. Apply the permutation by updating $U_b\leftarrow U_b\Pi_b$ and $V_b\leftarrow V_b\Pi_b$. Fix signs using $t_{b,i}=\mathrm{sign}(\langle U_b[:,i],U_r[:,i]\rangle\langle V_b[:,i],V_r[:,i]\rangle)$ and update $U_b[:,i]\leftarrow t_{b,i}U_b[:,i]$, $V_b[:,i]\leftarrow t_{b,i}V_b[:,i]$.

For a fixed direction $i\in\{1,\dots,n\}$, stack the $i$-th columns across replicas as
$$
U^{(i)}=\big[U_1[:,i]\ \cdots\ U_k[:,i]\big]\in\mathbb{R}^{n\times k},\qquad
V^{(i)}=\big[V_1[:,i]\ \cdots\ V_k[:,i]\big]\in\mathbb{R}^{m\times k}.
$$

Form the replica-Gram matrices with two matrix multiplications
$$
G_U^{(i)}=(U^{(i)})^\top U^{(i)}\in\mathbb{R}^{k\times k},\qquad
G_V^{(i)}=(V^{(i)})^\top V^{(i)}\in\mathbb{R}^{k\times k}.
$$

Define the reverb matrix
$$
Z_i \;=\; G_U^{(i)}\odot G_V^{(i)}\in\mathbb{R}^{k\times k},
\qquad \mathrm{diag}(Z_i)\leftarrow 0.
$$

Compute $Y_i=Z_i Z_i\in\mathbb{R}^{k\times k}$ with one more matrix multiplication. The numerator vector is the column-sum of $(Y_i\odot Z_i^\top)$ and the denominator is $\|Z_i\|_F^2$. The echo vector is obtained by elementwise division and taking the nonnegative square root:
$$
\widehat{\boldsymbol{s}}_i
=
\frac{\big((Z_i Z_i)\odot Z_i^\top\big)\,\mathbf{1}_k}{\|Z_i\|_F^2},\qquad
\widehat{\boldsymbol{\zeta}}_i=\sqrt{\max\{\widehat{\boldsymbol{s}}_i,0\}}.
$$

This sequence uses, per direction $i$, two $k\times k$ Gram matrix multiplications and one $k\times k$ product $Z_i Z_i$, plus elementwise operations and reductions. With $k$ in the tens or low hundreds, the Gram multiplications dominate the cost; the echo solve is negligible by comparison.

---

## 5. Statistical scaling with the number of replicas

Write $Z_i(a,b)=\zeta_i^{(a)}\zeta_i^{(b)}+\epsilon_{ab}$ with $\mathbb{E}\epsilon_{ab}=0$ and independent replicas. The weighted Triple–OLS estimator with $w_{ab}=Z_i(a,b)^2$ is first-order unbiased,
$$
\mathbb{E}\,\widehat s_i^{(p)}=\zeta_i^{(p)2}+O(\sigma^2),
$$
and has variance that decays at rate $\Theta(1/k)$,
$$
\mathrm{Var}\big(\widehat s_i^{(p)}\big)=\Theta\!\left(\frac{1}{k}\right).
$$
Propagating through the square root gives
$$
\mathrm{sd}\big(\widehat\zeta_i^{(p)}\big)=\Theta\!\left(\frac{1}{\zeta_i^{(p)}\sqrt{k}}\right).
$$
The constants depend on aspect ratio and noise level but the rates are robust. The division-free form automatically suppresses pairs with tiny $|Z_i(a,b)|$ through the weights, so no per-pair thresholds are required.
