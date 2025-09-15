## From spectrum + $\hat\sigma$ to spectral projection coefficients

Given a gradient-like matrix $\hat G\in\mathbb{R}^{p\times n}$ with singular values $s_1\ge\cdots\ge s_m$ ($m=\min(p,n)$) and a **finite-size** noise-scale estimate $\hat\sigma$ (from the rectangular-Wishart bottom-tail fit), we can map each observed singular value $s$ to a predicted **spectral projection coefficient** (SPC) using the rectangular spiked-SVD mapping.

### 1) Shape ratio and bulk edge
Let
$$
\beta \;=\; \frac{\min(p,n)}{\max(p,n)},\qquad
\tau \;=\; \hat\sigma\,(1+\sqrt{\beta}) .
$$
Here $\tau$ is the **bulk edge in singular-value units** under noise-only at scale $\hat\sigma$.

### 2) Whiten the singular value
Define the **whitened** singular value
$$
y \;=\; \frac{s}{\hat\sigma}.
$$
If $y\le 1+\sqrt{\beta}$ (i.e., $s\le\tau$), treat it as **bulk** and set $\operatorname{spc}=0$.

### 3) Invert the spike mapping to get the (whitened) signal strength
For $y>1+\sqrt{\beta}$, the rectangular spiked model links the whitened observation $y$ to the
(whitened) true signal strength $x$ via
$$
y^2 \;=\; x^2 + (1+\beta) + \frac{\beta}{x^2}.
$$
Let $t := x^2$. Solve the quadratic
$$
t^2 - (y^2 - 1 - \beta)\,t + \beta \;=\; 0,
$$
and take the **larger** root (the physical branch $t>\sqrt{\beta}$):
$$
t \;=\; \frac{\,y^2 - 1 - \beta + \sqrt{(y^2 - 1 - \beta)^2 - 4\beta}\,}{2}.
$$

### 4) Convert $t$ to the spectral projection coefficient
The predicted **product alignment** of left/right singular vectors (our SPC) as a function of $t$ and $\beta$ is
$$
\operatorname{spc}(t;\beta)
\;=\;
\sqrt{\frac{1 - \beta/t^{2}}{1 + \beta/t}}
\;\times\;
\sqrt{\frac{1 - 1/t^{2}}{1 + 1/t}}.
$$
Combine Steps 2–4 to get $\operatorname{spc}(s;\hat\sigma,\beta)$.

### 5) Numerically stable, vectorized recipe

For each $s$:

1. $y \leftarrow s/\hat\sigma$.
2. If $y \le 1+\sqrt{\beta}$, set $\operatorname{spc}=0$ and continue.
3. $A \leftarrow y^2 - 1 - \beta$.
4. $D \leftarrow \max(A^2 - 4\beta,\,0)$.
5. $t \leftarrow \dfrac{A + \sqrt{D}}{2}$, then $t \leftarrow \max\!\big(t, \sqrt{\beta}+\varepsilon\big)$.
6. Compute
   $$
   \operatorname{spc}
   = \sqrt{\frac{(1 - \beta/t^{2})(1 - 1/t^{2})}{(1 + \beta/t)(1 + 1/t)}}\;,
   $$
   then clip to $[0,1]$.

Notes:
- This uses only $(p,n)$ via $\beta$ and your fitted $\hat\sigma$. No additional free parameters.
- Using $\tau=\hat\sigma(1+\sqrt{\beta})$ gives an intuitive bulk/spike boundary in raw units.
- Vectorize all steps over arrays of $s$ for speed.
