# The M in Muon

Muon stands for **M**oment**U**m **O**rthogonalized by **N**ewton-Schulz.

We will now build the final piece of the original Muon optimizer: incorporating momentum.

In its simplest form, momentum reduces the variance of a gradient estimator by taking an exponentially-weighted moving average of a history of gradient estimates.

$$
\begin{aligned}
G_0&=\nabla_{W_0} L\\
G_1&=(1-\gamma)\nabla_{W_1} L + \gamma G_0\\
G_2&=(1-\gamma)\nabla_{W_2} L + \gamma G_1\\
&...\\
G_t&=(1-\gamma)\sum_t \left[\gamma^t \nabla_{W_t} L\right]\\
W_{t+1}&=W_t+\eta G_t
\end{aligned}
$$

Momentum for variance reduction is not a free lunch.  Averaging with stale gradients introduces bias.  We tune the momentum strength $\gamma$ to navigate the bias-variance tradeoff carefully.

Since we apply a nonlinear operation to our gradients, we have an additional choice to make in navigating that tradeoff.  **Do we accumulate momentum inside or outside of the msign?**

$$
\begin{aligned}
&\text{Accumulate inside?}\\
W_{t+1}&\overset{?}{=}W_t+\eta\cdot \text{msign}\left[\sum_t \gamma^t \nabla_{W_t} L\right]\tag{$1-\gamma$ absorbed into tuned $\eta$}\\
\\
&\text{Or accumulate outside?}\\
W_{t+1}&\overset{?}{=}W_t+\eta\cdot \sum_t \gamma^t \text{msign}\left[\nabla_{W_t} L\right]
\end{aligned}
$$

One of these is both theoretically and empirically better than the other.

# Shampoo, the Preconditioner

We have previously motivated Muon (and its equivalents like Shampoo and SOAP) as the unique steepest descent optimizer under the spectral norm.  For this analysis, we will need to take a different perspective on why these work: preconditioning.

Typically, gradients are not well-conditioned, so if we apply naive SGD, most of the updates at each minibatch are dominated by a single direction in parameter space.  This is a shame since each parameter matrix contains multitudes.

We want to **precondition** the gradient matrix so that similar progress is made in all update directions.  For simplicity, we consider the following class of preconditioners:

$$
\boxed{
\begin{aligned}
\textbf{Definition}\quad
& \text{A preconditioner $C\in\mathbb{R}^{n\times n}$ left-multiplies in the update rule}\\
& W_{t+1}=W_t+\eta \cdot CG_t\\
& \text{such that $CG_t$ is roughly orthogonal: $(CG_t)^\top (CG_t)\approx \mathbb{I}_n$}
\end{aligned}}
$$

Preconditioners almost always look like the inverse-square-root of the Fisher Information matrix $\mathbb{E}_{x\sim p_{\text{data}}}{\left[(\nabla_W L(x))^\top (\nabla_W L(x))\right]}$.  We note that msign descent looks a lot like preconditioning under the Fisher Information matrix without the expectation.

$$
\begin{aligned}
\text{For $\nabla_W L=G=USV^T$}\\
(GG^\top)^{-\frac{1}{2}}G&=(USV^TVSU^T)^{-\frac{1}{2}}USV^T\\
&=(US^2U^T)^{-\frac{1}{2}}USV^T\\
&=US^{-1}U^TUSV^T\\
&=US^{-1}SV^T\\
&=UV^T\\
&=\boxed{\text{msign}(G)}
\end{aligned}
$$

# Reality has a Well-Known Bias

We want to analyze which design for Muon is a good estimate for the true preconditioned gradient.  The time has come to reformulate our optimization problem with minibatch noise.

Let $G_t=\nabla_{W_t} L$ be the true gradient at minibatch $g$.  Suppose we have a noisy estimate $\tilde{g}_t=G_t+\epsilon_t$, where $\mathbb{E}[\epsilon_t]=0$ and $\text{Cov}[\epsilon_t]=\Sigma$.

We revisit our choice of momentum implementation:
- do we accumulate first and then take the matrix sign?
- or do we compute the matrix sign of each gradient in the history and then accumulate?
$$
\begin{aligned}
\text{MsignThenAcc}&=\text{msign}\left[\sum_t \gamma^t \tilde{g}_t \right]\\
\text{AccThenMsign}&=\sum_t \gamma^t \text{msign}[\tilde{g}_t]\\
\text{True Preconditioned Update}&=\boxed{\left(g_tg_t^\top\right)^{-\frac{1}{2}}g_t}\\
\end{aligned}
$$

Foundation models train with extremely large batch sizes, so the only thing that matters is the asymptotic order of the bias and variance of MsignThenAcc and AccThenMsign as estimators of the true preconditioned update.