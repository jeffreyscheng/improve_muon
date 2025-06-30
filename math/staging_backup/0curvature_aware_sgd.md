# Muon, Natural Gradients & Curvature — A Practitioner-Focused Cheat-Sheet  

This document distills every mathematical result we produced, **plus a short “Why you care” context for each claim**.  All equations are in KaTeX blocks so you can paste straight into VS Code’s raw-markdown preview.

---

## Claim&nbsp;1  — Unbiasedness of SGD  

**Context (why you care)**  
Unbiasedness is the bedrock that lets you replace the expensive full-dataset gradient with cheap mini-batches without introducing systematic drift.  Everything else (momentum, Muon, natural-gradient tricks) assumes this property.

**Justification**

$$\begin{aligned}
g(\theta;B) &= \tfrac1{|B|}\!\sum_{x\in B}\nabla_\theta \ell(\theta;x), \\
\mathbb E_B[g(\theta;B)]
&=\tfrac1{|B|}\!\sum_{i=1}^{|B|}\mathbb E_{x_i}[\nabla_\theta \ell(\theta;x_i)] \\
&=\mathbb E_{x}[\nabla_\theta \ell(\theta;x)] \\
&=\nabla_\theta \mathbb E_x[\ell(\theta;x)] 
      \quad\text{(dominated convergence)} \\
&=\boxed{\nabla_\theta L(\theta).}
\end{aligned}$$  

---

## Claim&nbsp;2  — Orthogonal Factor as Left-Whitening  

**Context**  
Shows that Muon’s update is exactly a left-whitened gradient—a half-natural-gradient step.  That tells you *what curvature information Muon captures* and why it can accelerate training.

**Justification**

$$\begin{aligned}
G &= U\,S\,V^{\top}\quad (\text{thin SVD}),\\
G\,G^{\top} &= U\,S^2\,U^{\top},\\
(G\,G^{\top})^{-1/2} &= U\,S^{-1}\,U^{\top},\\
(G\,G^{\top})^{-1/2}G &= U\,S^{-1}\,U^{\top}(U\,S\,V^{\top}) = U\,V^{\top}.\\
\Rightarrow\;&\operatorname{Ortho}(G) = U\,V^{\top} = (G\,G^{\top})^{-1/2}G.
\end{aligned}$$  

---

## Claim&nbsp;3  — Rank-1 Case is Integrable  

**Context**  
For vector gradients (e.g. scalar parameters) Muon reduces to “normalize the gradient.”  If you ever implement Muon on biases or LayerNorm scales, you can skip the heavy machinery.

**Justification**

$$\begin{aligned}
\|g\| &= (g^{\top}g)^{1/2},\\
\nabla_g\|g\| &= \frac{1}{2}(g^{\top}g)^{-1/2}\,2g
              = \frac{g}{\|g\|} = \operatorname{Ortho}(g).
\end{aligned}$$  

---

## Claim&nbsp;4  — Non-Integrability for $m\ge2$  

**Context**  
Practitioners sometimes hope to “move” Muon into the computational graph (e.g. by prepending/ appending a scalar function).  
This claim explains **why that cannot work**: no scalar potential produces $\operatorname{Ortho}(G)$ for matrices with ≥2 columns.

**Justification**

Pick $G=I_m$ and perturbations  
$dG^{(a)} = e_a e_b^{\top},\;
 dG^{(b)} = e_b e_a^{\top},\; a\neq b.$

Compute mixed derivatives:

$$\begin{aligned}
\langle dG^{(b)},\,\partial_{dG^{(a)}}\operatorname{Ortho}\rangle_F &= +\tfrac12,\\
\langle dG^{(a)},\,\partial_{dG^{(b)}}\operatorname{Ortho}\rangle_F &= -\tfrac12.
\end{aligned}$$  

Since these are unequal, the Jacobian has a skew part; hence $\operatorname{Ortho}$ is **not** the gradient of any scalar field.

---

## Claim&nbsp;5  — Exact Bias, Variance & MSE of Momentum  

**Context**  
Lets you pick $\gamma$ rationally: if your gradient signal-to-noise ratio is low, crank $\gamma$ up; if SNR is high, large $\gamma$ can hurt in early iterations.

**Justification**

$$\begin{aligned}
\text{Bias}(m_t) &= μ\gamma^{t}-μ = -μ\gamma^{t},\\
\text{Bias}^2    &= μ^{2}\gamma^{2t},\\[6pt]
\Var(m_t) 
&= (1-\gamma)^2σ^2\sum_{i=0}^{t-1}\gamma^{2i}
  = σ^{2}\frac{1-\gamma}{1+\gamma}(1-\gamma^{2t}),\\[6pt]
\mathrm{MSE}(m_t) &= μ^{2}\gamma^{2t}
  + σ^{2}\frac{1-\gamma}{1+\gamma}(1-\gamma^{2t}).
\end{aligned}$$  

---

## Claim&nbsp;6  — When Momentum Beats Raw SGD  

**Context**  
Provides a *warm-up length* formula.  Good for scheduling: you can “burn-in” with smaller $\gamma$ and then increase it.

**Justification**

Set $\mathrm{MSE}(m_t)<σ^{2}$:

$$\begin{aligned}
μ^{2}\gamma^{2t} 
&< σ^{2}\frac{\gamma^{2t}-\gamma^{2t+1}}{1+\gamma},\\
\frac{μ^{2}}{σ^{2}}
&< \frac{1-\gamma}{1+\gamma}\frac{1-\gamma^{2t}}{\gamma^{2t}},\\
t &> T^{*} =
\dfrac{\ln\!\bigl[\tfrac{2\gamma}{(1+\gamma)\frac{μ^{2}}{σ^{2}}-(1-\gamma)}\bigr]}
      {2\ln\gamma}.
\end{aligned}$$  

If $\tfrac{μ^{2}}{σ^{2}}\le\tfrac{1-\gamma}{1+\gamma}$ the inequality holds for all $t$.

---

## Claim&nbsp;7  — Fisher = Expected Hessian for NLL  

**Context**  
Shows why natural-gradient methods often substitute Fisher for the Hessian—it is easier to estimate stochastically.

**Justification**

$$\begin{aligned}
s(x) &= \nabla_\theta\log p(x;\theta),\\
\nabla_\theta s(x)^{\top}
&= \nabla_\theta^{2}\log p(x;\theta),\\
\mathbb E[s\,s^{\top}]
&= -\mathbb E[\nabla_\theta s^{\top}]
 = -\mathbb E[\nabla^{2}_\theta\log p(x;\theta)].
\end{aligned}$$  

---

## Claim&nbsp;8  — Asymptotics of Ortho-of-Sum  

**Context**  
Explains why *single* Muon step on a mega-batch is biased unless the curvature is rank-1.  Motivates “sum-of-Ortho’’ variants.

**Justification**

LLN gives  
$$\frac1k\sum_{i=1}^k G_i \to \mu.$$  
Continuity of $\operatorname{Ortho}$ yields  
$$\operatorname{Ortho}\Bigl(\sum_i G_i\Bigr)\to \operatorname{Ortho}(\mu).$$  
But  
$$M^{-1/2}\mu = (\mathbb E[G\,G^{\top}])^{-1/2}\mu$$  
matches only if $M\propto\mu\mu^{\top}$.

---

## Claim&nbsp;9  — Bias Scale Difference (Ortho-of-Sum vs. Sum-of-Ortho)  

**Context**  
Quantifies why summing Ortho’d micro-grads can be much less biased when noise is anisotropic.

**Justification**

Second-order expansion at $\mu$ shows bias term  

$$\mathrm{Bias} \approx \frac{k}{2}\operatorname{Tr}\!\bigl(J_\mu H_\mu Σ\bigr),
\quad H_\mu=\nabla_G^2\operatorname{Ortho}$$  

for Ortho-of-Sum, hence $O(k\,Σ)$, whereas Sum-of-Ortho remains $O(Σ)$.

---

## Claim&nbsp;10  — Riemannian Variance $\sigma_R$  

**Context**  
A scalar you can log during training: when $\sigma_R$ is small your curvature is stable and global whitening will work; when it spikes, consider richer preconditioners.

**Justification**

On SPD manifold with affine-invariant metric, squared distance is  

$$d^{2}(H_i,\bar H)
  =\bigl\|\log(\bar H^{-1/2}H_i\bar H^{-1/2})\bigr\|_{F}^{2},$$  

so the sample variance $\sigma_R^{2}$ above is the mean squared distance to the Fréchet mean $\bar H$.

---

## Claim&nbsp;11  — Higham Polar Iteration  

**Context**  
A drop-in for Newton–Schulz: fewer iterations, no magic constants, stable in fp16.

**Justification**

Higham’s proof shows the map  

$$\Phi(X)=\tfrac12\,X\bigl(I+(X^{\top}X)^{-1}\bigr)$$  

is a Newton step for $X^{\top}X=I$; scaling by $\|G\|_F$ keeps $\|X_0\|_2\approx1$, guaranteeing convergence. Quadratic error contraction:

$$\|U-\Phi(X)\|_2\le C\|U-X\|_2^{2}.$$

---

## Claim&nbsp;12  — Global Whitening Change of Variables  

**Context**  
Cheap to implement: store $M^{1/2}$ as a per-layer weight transform, update it every few hundred steps, and Muon becomes almost an identity.

**Justification**

Chain rule for $\phi=M^{1/2}\theta$:

$$\nabla_\theta\ell 
  = \bigl(\partial\phi/\partial\theta\bigr)^{\!\top}\nabla_\phi\ell
  = M^{-1/2}\nabla_\phi\ell.$$

---

## Claim&nbsp;13  — Ortho Equals $\nabla\|g\|$ Only in Rank-1  

**Context**  
Confirms you can treat scalar/vector parameters with simple norm-normalisation but need Muon’s full machinery for matrices.

**Justification**

Rank-1 proof identical to Claim 3; non-integrability for $m\ge2$ in Claim 4.

---

## Claim&nbsp;14  — Generic Expected Hessian Has No Special Name  

**Context**  
Don’t waste time hunting literature for a fancy acronym; just call it the **population Hessian** or **expected curvature**.

**Justification**

Only in the negative-log-likelihood case does the score-function identity make the Fisher and expected Hessian the same.  For arbitrary loss, that identity breaks, so no classical terminology exists.
