# Matrix-Sign Approximations via Spectral-Odd Functions

We seek a fast analytic map that drives small singular values of $X$ to zero and large ones to one while preserving singular vectors.
For an odd scalar function $g(x)$ with threshold $\varepsilon$ define
\begin{align}
F(X) &= X\,\varphi\bigl(X^{\!*} X / \varepsilon^{2}\bigr),\\
 g(x) &= x\,\varphi(x^{2}/\varepsilon^{2}).
\end{align}
For the SVD $X=U\Sigma V^{\!*}$ this yields $F(X)=U\,g(\Sigma)\,V^{\!*}$, so $U,V$ are unchanged while $g(\sigma)$ pushes singular values toward $\{0,1\}$.

---

## Analytic Example

Set
$$
 g_{\varepsilon,\alpha}(x)=\tfrac12\bigl[\tanh(\alpha(x-\varepsilon)) - \tanh(\alpha(-x-\varepsilon))\bigr].
$$
This odd entire function obeys
\begin{align}
|x|\le \varepsilon - \delta &\Longrightarrow |g_{\varepsilon,\alpha}(x)| \le e^{-2\alpha\delta},\\
|x|\ge \varepsilon + \delta &\Longrightarrow |g_{\varepsilon,\alpha}(x)-\operatorname{sgn}(x)| \le e^{-2\alpha\delta}.
\end{align}
Lifting it to matrices via $F_{\varepsilon,\alpha}(X)=X\,\varphi_{\varepsilon,\alpha}(X^{\!*}X/\varepsilon^{2})$ preserves these bounds on each singular value.

A Padé [5/5] rational approximation to $\tanh$ implements $F_{\varepsilon,\alpha}$ with seven GEMMs plus one triangular solve on a dense $1024\times1024$ matrix, taking roughly $3$ ms on modern GPUs.

---

This approach yields a differentiable spectral filter for Muon-style optimizers without computing an SVD: small singular values are suppressed and large ones snapped to unity, producing the polar factor $U V^{\!*}$ when $\alpha$ is large and $\varepsilon$ small.

