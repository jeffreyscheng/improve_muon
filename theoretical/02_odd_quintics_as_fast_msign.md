# Odd Quintics Compute msign Really Fast

msign should be easy, but it's not.

Computing $\text{msign}(G)=UV^\top$ should be just one SVD and a matmul!  Or just a polar factorization!  We finished optimizing the LAPACK methods for SVD 30 years ago!

Unfortunately, a full SVD or UP factorization are simply too slow to use for LLM optimization.

If we allow ourselves to daydream for a bit, the msign looks like the SVD but with the singular spectrum $S$ replaced by the identity: $\text{msign}(G)=UV^\top=U\mathbb{I}_nV^\top$.  If we could just snap all of the singular values $\sigma_1, \sigma_2, ...\sigma_r$ to 1 without computing $USV^T$, our job would be done.  So to extract the msign without actually performing the full matrix factorization, we are on the lookout for matrix functions that preserve the left/right singular bases but manipulate the internal singular values.

To manifest this birthday wish properly, we need to formalize the property we want.

# Spectral Operators Perform Non-Invasive Surgery

Let analytic scalar function $f:\mathbb{R}\rightarrow\mathbb{R}$ have Taylor series at the origin $f(x)=\sum_k c_k x^k$.

If we define generalized matrix powers for rectangular matrix $A=USV^\top$ as:
$$
\begin{aligned}
A^1&=A\\
A^2&=AA^\top\\
A^3&=AA^\top A\\
&...\\
A^{2k}&=(AA^\top)^{k}\\
A^{2k+1}&=(AA^\top)^{k}A\\
&...
\end{aligned}
$$
... we can say that scalar function $f$ *induces* the matrix function $f^\diamond=\sum c_k A^k$.

Scalar function $f$ is interesting to us if the matrix function $f^\diamond$ defined by its Taylor expansion has the property that for input $A$ with SVD $A=USV^\top$, applying $f^\diamond$ is equivalent to applying $f$ directly to its singular values.  We call this the **spectral operator property**.

$$
\boxed{
\begin{aligned}
\textbf{Definition}\quad
& \text{Scalar function $f$ is a spectral operator}\\
& \text{if induced matrix function $f^\diamond$ satisfies}\\
& f^{\diamond}(A)=U f(S) V^\top\\
& \text{where $USV^\top$ is the SVD of the inputs and}\\
& \text{$f(S)$ applies $f$ elementwise to the singular values.}\\
\end{aligned}}
$$

If we had a basis of spectral operators, we would try to assemble those building blocks into a spectral function that snaps all of the singular values to 1.  Given $f_{\text{one}}(x)=1$, we'd like to just implement the induced matrix function $\text{msign}(A)=f_{\text{one}}^\diamond(A)$ and call it a day.  Spoiler: no such $f_{\text{one}}^\diamond$ exists, but this unfulfilled wish will still bring us where we want to go.

So where do we look for a basis of spectral operators?

# Spectral Operators have an Odd Basis

It's not crazy to think of odd polynomials as the right place to start.  For the simplest motivation, conside the non-square case $A\in\mathbb{R}^{n\times m}, n\neq m$.  Even matrix powers $A^{2k}$ belong to the vector space $\mathbb{R}^{n\times n}$, so any analytic operator $f^\diamond$ can't have even terms in its Taylor expansion at all.

Observe that for SVD $A=USV^\top$, the left and right singular bases of odd matrix powers are preserved:
$$
\begin{aligned}
A^1&=A&=\boxed{USV^\top}\\
A^3&=AA^\top A=USV^\top(V^\top SU)USV^\top&=\boxed{US^3V^\top}\\
&...&\\
A^{2k+1}&=(AA^\top)^{k}A&=\boxed{US^{2k+1}V^\top}\\
&...&\\
\end{aligned}
$$

Computing an odd matrix power is equivalent to modifying the original matrix's SVD in-place by taking the element-wise power of each singular value in $S$.

Each odd matrix power on its own is not that powerful, though. Fortunately:

$$
\boxed{
\begin{aligned}
\textbf{Theorem}\quad \text{Spectral operators $f:\mathbb{R}\rightarrow\mathbb{R}$ form a vector space.}\\
\end{aligned}}
$$

So all linear combinations of odd powers are fair game!

It turns out that there are no other spectral operators.  Linear combinations of odd powers are all we have.

- For functions on rectangular matrices $A\in\mathbb{R}^{n\times m}$, a Taylor expansion with even terms is not even well-defined!  So the set of induced matrix functions $\{f^\diamond:\mathbb{R}^n\rightarrow\mathbb{R}^m\}$ is exactly the set of odd analytic functions.
- Although functions of square matrices can have nonzero even Taylor expansion terms, it is also true that the set of spectral functions induced on square matrix spaces $\{f^\diamond:\mathbb{R}^n\rightarrow\mathbb{R}^n\}$ is just the set of odd analytic functions; the proof follows from applying the definition of spectral operators to $f(A)+f(-A)$.

# Approximating **sgn** the old way, but with a new twist

The constant function $f_{\text{one}}(x)=1$ is not odd!  But the dream is not dead: we now know that we have to find an analytic approximation of the non-smooth sgn function:

$$
\operatorname{sgn}(x)=
\begin{cases}
-1, & x<0,\\
0,  & x=0,\\
1,  & x>0.
\end{cases}
$$

There are many candidates for odd functions that look like **sgn**.  For example, $f(x)=\frac{\tanh(10000x)}{\tanh(10000)}$ is a cosmetically perfect candidate.  However, due to matrix tanh's definition involving exponentiation $\tanh(x)=\frac{e^{2x}-1}{e^{2x}+1}$, its numerical instability makes it unsuitable for computing msign.

The classic Newton-Schulz iterator for msign is $f_{\text{classic}}(x)=\frac{3}{2}x-x^3$.  This is because it is the unique odd cubic that passes through the origin and (1, 1).  Therefore, iterating $f_{\text{classic}}(f_{\text{classic}}(...f_{\text{classic}}(x)))$ converges to $\text{sgn}(x)$ in the unit interval.  If we want a matrix $A$ with singular values greater than 1 to converge under Newton-Schulz, we can simply divide by the spectral norm $\Vert A\Vert_2$ before passing it into Newton-Schulz (or divide by the Frobenius norm $\Vert A\Vert_F$, which is an easily computed upper bound).

However, the iterator $f_{\text{classic}}(x)=\frac{3}{2}x-x^3$ is too slow! It can take dozens of iterations to converge; for a singular value of $10^{-5}$ (very common for attention parameters), the number of iterations is roughly $\frac{\ln(10^5)}{\ln(\frac{3}{2})}=28$.  Nobody's got time for that.

We can relax a few constraints to minimize the number of matmuls necessary.
- We can use an iterative method that actually diverges asymptotically; all that matters is at the prespecified number of Newton-Schulz iterations, the approximation to sgn is good.
- Newton-Schulz iteration composes the same polynomial many times.  We can use different polynomials at each iteration and learn the best coefficients for each iteration by gradient descent.
- We don't need to use cubics.  Cubic was nice because the constraints uniquely specific $f_{\text{classic}}$, so the coefficients can be discovered by hand quickly.  But if we commit to learning the polynomial iteration anyway, we can use quintic or septic polynomials.

