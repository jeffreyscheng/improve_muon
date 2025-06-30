# Motivation  

Our goal has been to understand **why** and **when** Muon’s orthogonalised update  
$$\operatorname{Ortho}(G)=G\,(G^{\top}G)^{-1/2}$$  
approximates the *natural-gradient* direction  
$$M^{-1/2}\nabla_\theta L(\theta),\qquad M=\mathbb E_x[G\,G^{\top}],$$  
and to explore ways of pushing Muon *inside* the expectation so that  
$$\mathbb E_x\bigl[\operatorname{Ortho}(G_x)\bigr]$$  
can be estimated efficiently.  
Along the way we compared “Ortho-of-sum’’ vs. “sum-of-Ortho”, studied momentum bias, curvature drift, and integrability.  
What follows is a self-contained record of every mathematical claim we produced, with **step-by-step justifications** written in raw KaTeX blocks.

---

## **Claim 1 — Unbiasedness of SGD**

> $$\mathbb E_{B}\!\bigl[g(\theta;B)\bigr]=\nabla_\theta L(\theta)$$  
> for the mini-batch estimator  
> $$g(\theta;B)=\tfrac1{|B|}\!\sum_{x\in B}\nabla_\theta \ell(\theta;x),\qquad  
> L(\theta)=\mathbb E_{x}[\ell(\theta;x)].$$  

**Justification**

$$\begin{aligned}
\mathbb E_{B}\!\bigl[g(\theta;B)\bigr]
&=\mathbb E_{x_1,\dots,x_{|B|}}
     \Bigl[\tfrac1{|B|}\sum_{i=1}^{|B|}\nabla_\theta\ell(\theta;x_i)\Bigr]\\
&=\tfrac1{|B|}\sum_{i=1}^{|B|}\mathbb E_{x_i}\!\bigl[\nabla_\theta\ell(\theta;x_i)\bigr]
     &&\text{(i.i.d.\ samples)}\\
&=\mathbb E_{x}\!\bigl[\nabla_\theta\ell(\theta;x)\bigr]\\
&=\nabla_\theta\,\mathbb E_{x}\!\bigl[\ell(\theta;x)\bigr]
     &&\text{(dominated conv.)}\\
&=\boxed{\nabla_\theta L(\theta).}
\end{aligned}$$  

---

## **Claim 2 — Polar relation inside Muon**

> If $G=U\,S\,V^{\top}$ is the thin SVD, then  
> $$\operatorname{Ortho}(G)=U\,V^{\top}=(G\,G^{\top})^{-1/2}G.$$  

**Justification**

$$\begin{aligned}
G\,G^{\top}
&=(U\,S\,V^{\top})(V\,S\,U^{\top})
 =U\,S^2\,U^{\top},\\[4pt]
\bigl(G\,G^{\top}\bigr)^{-1/2}
&=U\,S^{-1}\,U^{\top},\\[4pt]
\bigl(G\,G^{\top}\bigr)^{-1/2}G
&=U\,S^{-1}\,U^{\top}\,(U\,S\,V^{\top})
 =U\,V^{\top}.
\end{aligned}$$  

Hence $\operatorname{Ortho}(G)$ is a **left-whitened** gradient.

---

## **Claim 3 — Muon’s preconditioner equals $M^{-1/2}$ for the rank-1 case**

> When $m=1$ (gradient is a column vector)  
> $$\operatorname{Ortho}(g)=\dfrac{g}{\|g\|}=\nabla_g\|g\|,$$  
> so Muon coincides with half-whitening and is integrable.

**Justification**

$$\begin{aligned}
\|g\| 
&=\bigl(g^{\top}g\bigr)^{1/2},\\
\nabla_g\|g\|
&=\dfrac{1}{2}\bigl(g^{\top}g\bigr)^{-1/2}\,2g
 =\dfrac{g}{\|g\|}.
\end{aligned}$$  

---

## **Claim 4 — Non-integrability of $\operatorname{Ortho}$ for $m\ge2$**

> For matrices with two or more columns,  
> $$F(G)=\operatorname{Ortho}(G)$$  
> has a Jacobian with a skew part, hence no scalar potential exists.

**Justification**

Choose $G=I_m$ (square identity), and perturb directions  
$$dG^{(a)}=e_a e_b^{\top},\quad
  dG^{(b)}=e_b e_a^{\top},\qquad a\ne b.$$  

Compute the mixed inner products:

$$\begin{aligned}
\bigl\langle dG^{(b)},\,\partial_{dG^{(a)}}F\bigr\rangle_{F}
   &=+\tfrac12,\\
\bigl\langle dG^{(a)},\,\partial_{dG^{(b)}}F\bigr\rangle_{F}
   &=-\tfrac12.
\end{aligned}$$  

Since these differ, the Jacobian satisfies $J\ne J^{\top}$, violating the symmetry condition for conservative fields.  Therefore **no** $f$ with $\nabla_G f=F$ exists.

---

## **Claim 5 — Bias–variance trade-off for momentum EMA**

> For scalar model $g_t=μ+ε_t,\;ε_t\sim(0,σ^2)$ and  
> $$m_t=\gamma m_{t-1}+(1-\gamma)g_t,$$  
> the MSE is  
> $$\mathrm{MSE}(m_t)=μ^{2}\gamma^{2t}
    +σ^{2}\dfrac{1-\gamma}{1+\gamma}\bigl(1-\gamma^{2t}\bigr).$$  

**Justification**

*Bias*:

$$\begin{aligned}
\mathbb E[m_t]
&=(1-\gamma)\sum_{i=0}^{t-1}\gamma^{\,i}μ
 =μ\bigl(1-\gamma^{t}\bigr),\\
\text{Bias}^2&=μ^{2}\gamma^{2t}.
\end{aligned}$$  

*Variance*:

$$\begin{aligned}
Var(m_t)
&=(1-\gamma)^{2}\sum_{i=0}^{t-1}\gamma^{2i}Var(ε_{t-i})\\
&=(1-\gamma)^{2}σ^{2}\frac{1-\gamma^{2t}}{1-\gamma^{2}}
 =σ^{2}\frac{1-\gamma}{1+\gamma}\bigl(1-\gamma^{2t}\bigr).
\end{aligned}$$  

Add bias and variance to obtain the stated MSE.

---

## **Claim 6 — When momentum helps**

> Momentum improves MSE immediately if  
> $$\dfrac{μ^{2}}{σ^{2}}\le\dfrac{1-\gamma}{1+\gamma},$$  
> otherwise after  
> $$T^{*}
  =\dfrac{\ln\!\bigl[\tfrac{2\gamma}{(1+\gamma)\,\frac{μ^{2}}{σ^{2}}-(1-\gamma)}\bigr]}
         {2\ln\gamma}\quad\text{steps}. $$  

**Justification**

Set $\mathrm{MSE}(m_t)<\mathrm{MSE}(g_t)=σ^{2}$:

$$\begin{aligned}
μ^{2}\gamma^{2t}
+σ^{2}\frac{1-\gamma}{1+\gamma}\bigl(1-\gamma^{2t}\bigr)
&<σ^{2},\\
\frac{μ^{2}}{σ^{2}}\gamma^{2t}
&<\frac{1-\gamma^{2t}}{1+\gamma}.
\end{aligned}$$  

If $\frac{μ^{2}}{σ^{2}}\le\frac{1-\gamma}{1+\gamma}$ the RHS exceeds the LHS for all $t$.  
Otherwise solve equality for $t$ to get the threshold $T^{*}$.

---

## **Claim 7 — Fisher equals expected Hessian (log-likelihood)**

> $$I(\theta)=\mathbb E_X\!\bigl[\nabla_\theta\log p(X;\theta)\,\nabla_\theta\log p(X;\theta)^{\top}\bigr]
  =-\mathbb E_X\!\bigl[\nabla^{2}_\theta\log p(X;\theta)\bigr].$$  

**Justification**

Score function $s(X)=\nabla_\theta\log p(X;\theta)$ satisfies  

$$\begin{aligned}
0 
&=\nabla_\theta\int p(x;\theta)\,dx
 =\int p(x;\theta)\,s(x)^{\top}\,dx,\\
\nabla_\theta s(x)^{\top}
&=\nabla_\theta^{2}\log p(x;\theta).
\end{aligned}$$  

Take expectation:

$$\begin{aligned}
\mathbb E_X\!\bigl[s\,s^{\top}\bigr]
&=-\mathbb E_X\!\bigl[\nabla_\theta s^{\top}\bigr]
 =-\mathbb E_X\!\bigl[\nabla^{2}_\theta\log p(X;\theta)\bigr].
\end{aligned}$$  

---

## **Claim 8 — Asymptotics of “Ortho-of-sum’’**

> $$\operatorname{Ortho}\Bigl(\sum_{i=1}^{k}G_i\Bigr)\xrightarrow{k\to\infty}\operatorname{Ortho}(\mu)\neq M^{-1/2}\mu$$  
> unless $M\propto\mu\mu^{\top}$.

**Justification**

Law of large numbers: $\frac1k\sum_i G_i\to\mu$.  
By continuity of $\operatorname{Ortho}$, the limit is $\operatorname{Ortho}(\mu)$.  
Meanwhile $M^{-1/2}\mu=(\mathbb E[GG^{\top}])^{-1/2}\mu$.  
Equality holds iff $\operatorname{Ortho}(\mu)=M^{-1/2}\mu$, i.e.\ $(\mu\mu^{\top})^{-1/2}$ matches $M^{-1/2}$, which requires $M\propto\mu\mu^{\top}$.

---

## **Claim 9 — Bias explosion of Ortho-of-sum under anisotropic noise**

> With $G_i=\mu+ε_i,\;ε_i\sim\mathcal N(0,Σ)$,  
> $$\mathrm{Bias}\bigl[\operatorname{Ortho}(\textstyle\sum G_i)\bigr]
     =O(k\,Σ).$$  

**Justification**

A first-order Taylor expansion at $\mu$ gives  

$$\operatorname{Ortho}(\mu+Δ)\approx\operatorname{Ortho}(\mu)
  +J_{\mu}\,Δ,$$  

where $J_{\mu}$ is the Jacobian.  
Sum $k$ iid noises: variance grows like $k$, and expected linear term $\mathbb E[J_\mu\,\sum ε_i]=0$, but second-order bias term contains $\mathbb E[ΔΔ^{\top}]=k\,Σ$, so bias $\propto k$.

---

## **Claim 10 — Riemannian variance for curvature drift**

> $$\sigma_R^{2}
 =\tfrac1k\sum_{i=1}^{k}
   \bigl\|\log\bigl(\bar H^{-1/2}H_i\bar H^{-1/2}\bigr)\bigr\|_{F}^{2},
 \quad
 \bar H=\tfrac1k\sum_{i=1}^{k}H_i,$$  
> measures dispersion of SPD matrices.

**Justification**

$\mathbb S_{++}^{n}$ with affine-invariant metric has squared distance  

$$d^{2}(H_i,\bar H)=\|\log(\bar H^{-1/2}H_i\bar H^{-1/2})\|_{F}^{2}.$$  

Averaging these squared distances defines the variance $\sigma_R^{2}$; small values indicate that curvature changes slowly.

---

## **Claim 11 — Exact polar decomposition via Higham iteration**

> Starting with $X_0=G/\|G\|_F$ and  
> $$X_{k+1}=\tfrac12\,X_k\bigl(I+(X_k^{\top}X_k)^{-1}\bigr),$$  
> the sequence converges quadratically to $U$ in $G=UP$.

**Justification**

Higham shows (Theorem 8.1, *Functions of Matrices*) that the map  
$$\Phi(X)=\tfrac12\,X\bigl(I+(X^{\top}X)^{-1}\bigr)$$  
satisfies  

$$\|U-\Phi(X)\|\le C\,\|U-X\|^{2}$$  

in a neighbourhood of $U$, hence quadratic convergence; scaling by $\|G\|_F$ guarantees the starting point is inside that neighbourhood for fp16-fp32 ranges.

---

## **Claim 12 — Global whitening re-parameterisation**

> With nearly constant curvature $M$,  
> $$\phi=M^{1/2}\theta
   \;\Longrightarrow\;
   \nabla_\theta\ell=M^{-1/2}\nabla_\phi\ell,$$  
> reducing Muon to a small residual.

**Justification**

Chain rule:

$$\begin{aligned}
\nabla_\phi\ell
&=\Bigl(\frac{\partial\theta}{\partial\phi}\Bigr)^{\!\top}\nabla_\theta\ell
 =M^{-1/2}\nabla_\theta\ell,\\
\implies
\nabla_\theta\ell
&=M^{-1/2}\nabla_\phi\ell.
\end{aligned}$$  

Thus a standard gradient step in $\phi$-space implements half-whitening.

---

## **Claim 13 — “Ortho” reproduces gradient of $\|g\|$ only for rank 1**

> For $m=1$  
> $$\operatorname{Ortho}(g)=\nabla_g\|g\|,$$  
> but for $m\ge2$ the curl is non-zero.

**Justification**

Already shown in Claim 3 for $m=1$.  
For $m\ge2$ Jacobian skewness in Claim 4 implies non-zero curl, hence not a gradient field.

---

## **Claim 14 — Fisher equals Hessian expectation only for NLL**

> For generic loss $\ell$,  
> $$M(\theta)=\mathbb E[\nabla^{2}\ell]\quad\text{has no special name}.$$  

**Justification**

The identity in Claim 7 used $\ell(\theta;x)=-\log p(x;\theta)$ and relied on $\mathbb E[\nabla_\theta\log p]=0$.  For arbitrary $\ell$ that property fails, so $M$ is simply called the **population Hessian** or **expected curvature**.

---
