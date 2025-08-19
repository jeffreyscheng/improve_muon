# Muon is Doubly Biased

For true gradient $G=\nabla_W L$ and minibatch gradient $\hat{G}=G+E$ under Gaussian noise model $E\sim N(0, \Sigma)$, we can straightforwardly show that $\hat{G}$ is an unbiased estimator of $G$ by linearity of expectation.  

However, $\text{msign}(\hat{G})$ is not an unbiased estimator of $\text{msign}(G)$.  If full-rank $G$ and $G_i$ differed only in their singular values, we would actually have $\text{msign}(\hat{G})=\text{msign}(G)$ since their singular values get snapped to one anyway.  But since there will be small perturbations such that the $j^\text{th}$ singular vectors do not exactly align $\hat{u}_j\neq u_j, \hat{v}_j\neq v_j$, we will be amplifying the wrong singular directions to unit length if we choose $\text{msign}(\hat{G})$ as our update direction.  For directions $u_iv_i^\top$ with extremely low signal (potentially below the noise floor of $\sigma(E)$), we should snapping the singular values to zero; for directions where $s_i>>\sigma_{\text{max}}(E)$, we should be snapping $s_i$ to 1 as before.  We assert that there is some optimal noise-aware projection that does better than msign.

We will first find an optimal projection by cheating; then we will show how to recover this optimal projection without cheating.

Suppose we had to make an update in the basis of $\hat{G}=\hat{U}\hat{S}\hat{V}^\top$ but had knowledge of the true gradient $\text{msign}(G)$ -- what would be the optimal singular value matrix $C$ to apply such that $\hat{U} C \hat{V}^\top$ is optimal in some sense?

Then once we know the form of $C$, we will discuss how to apply a spectral function $h:\mathbb{R}\rightarrow\mathbb{R}$ such that we can empirically approximate $h(\hat{S})\approx C$ without any knowledge of true gradient $G$, recovering the spectral output $h(\hat{G})=\hat{U} h(\hat{S})\hat{V}^\top$ as the noise-aware improvement upon Muon.

## Cheating

**Setup.** Let $\{u_i\}_{i=1}^r \subset \mathbb R^m$ and $\{v_i\}_{i=1}^r \subset \mathbb R^n$ be orthonormal:
$$
u_i^\top u_k=\delta_{ik},\qquad v_i^\top v_k=\delta_{ik}.
$$
Given a rank-1 target $T:=uv^\top$, solve
$$
\min_{c\in\mathbb R^r}\ \left\| \sum_{i=1}^r c_i\,u_i v_i^\top \;-\; uv^\top \right\|_F^2.
$$

**Claim.** The unique minimizer is
$$
\boxed{\,c_i^\star=\langle u,u_i\rangle\,\langle v,v_i\rangle \;=\; (u_i^\top u)\,(v_i^\top v).\,}
$$

---

## Proof (short)

1. **Orthonormal atoms.** The matrices $\{u_i v_i^\top\}$ are orthonormal in Frobenius inner product:
$$
\langle u_i v_i^\top,\ u_k v_k^\top\rangle_F
=\operatorname{tr}\!\big((u_i v_i^\top)^\top(u_k v_k^\top)\big)
=(u_i^\top u_k)(v_i^\top v_k)=\delta_{ik}.
$$

2. **Quadratic in $c$.** With $A(c):=\sum_i c_i u_i v_i^\top$,
$$
\|A(c)-uv^\top\|_F^2
=\sum_i c_i^2\ -\ 2\sum_i c_i\,\langle uv^\top,\ u_i v_i^\top\rangle_F\ +\ \|uv^\top\|_F^2,
$$
since cross-terms vanish by Step 1.

3. **Coordinatewise minimization.** Each $c_i$ appears as
$$
c_i^2 - 2c_i\,\langle uv^\top,\ u_i v_i^\top\rangle_F,
$$
so the minimizer is
$$
c_i^\star=\langle uv^\top,\ u_i v_i^\top\rangle_F.
$$

4. **Compute the coefficient.**
$$
\langle uv^\top,\ u_i v_i^\top\rangle_F
=\operatorname{tr}\!\big((uv^\top)^\top (u_i v_i^\top)\big)
=\operatorname{tr}\!\big(vu^\top u_i v_i^\top\big)
=(u_i^\top u)\,(v_i^\top v).
$$
This equals the claimed formula. $\square$

---

## How to empirically find $h$ without $G$

TODO

## Notes

- For $T=\sum_{r=1}^R u^{(r)}{v^{(r)}}^\top$,
$$
c_i^\star=\sum_{r=1}^R \langle u^{(r)},u_i\rangle\,\langle v^{(r)},v_i\rangle.
$$
- If $\{u_i\}$ or $\{v_i\}$ are not orthonormal (or indices are mismatched), first orthonormalize or solve the normal equations for the least-squares projection.
