# Muon is a steepest descent optimizer.

The gradient is referred to as the direction of steepest descent, but this masks a silent assumption. For a vector $w\in\mathbb{R}^m$, function $L:\mathbb{R}^m\rightarrow\mathbb{R}$, and arbitrary norm $\Vert \cdot \Vert_\_ $ we define the direction of steepest descent $\Delta w$ at $w$ to be:

$$\argmax_{\Vert \Delta w\Vert_\_=1} \langle \Delta w, -\nabla_w L\rangle$$

If we constrain $\Delta w$ to the L2 hypersphere, we do recover that the unit-perturbation of steepest descent is proportional to the gradient:

$$\argmax_{\Vert \Delta w\Vert_2=1} \langle \Delta w, -\nabla_w L\rangle=\frac{-\nabla_w L}{\Vert \nabla_w L\Vert_2}$$

However, this does not hold true of other norms.  If we instead constrain $\Delta w$ to the L-$\infty$ hypercube, we get:

$$\argmax_{\Vert \Delta w\Vert_\infty=1} \langle \Delta w, -\nabla_w L\rangle=-\text{sgn} \left(\nabla_w L\right)$$

aka the element-wise sign of the vector; this corresponds to the nearest corner on the hypercube where each vertex coordinate is $\pm 1$.

Different choices of norm surfaces lead to different steepest descent update rules.

# The norm choice for LLM optimization

It is known that the kernel regime has poor regularization.  We instead want feature learning, where for every layer $\ell$ with weights $W_\ell\in \mathbb{R}^{n\times m}$, both activations $h_\ell$ and gradients $\nabla_{h_\ell} L$ are of order $\Theta(\sqrt{n})$.

As per Condition 1 from Yang, Simon, and Bernstein's "A Spectral Condition for Feature Learning", we get feature learning regime ($\mu P$) when the both the weights $W_\ell$ and the updates$\Vert\Delta W_\ell\Vert_*$ have spectral norm $\Theta\left(\frac{n}{m}\right)$.

# The steepest descent under the spectral norm.

Given weight matrix $W\in \mathbb{R}^{n\times m}$, inner product $\langle A, B\rangle=\text{tr}(A^TB)$ for $A,B\in\mathbb{R}^{n\times m}$, spectral norm $\Vert\cdot\Vert_2$, and loss function $L$, we write shorthand $G=\nabla_W L$ and specify the SVD $G=\sum_{i=1}^r \sigma_i u_i v_i^T$.

We can show that the inner product achieved by the direction of steepest descent is bounded by the nuclear norm of the perturbation.

$$
\begin{aligned}
\max_{\Vert\Delta W\Vert_2=1} \langle \Delta W, G\rangle&=\max_{\Vert\Delta W\Vert_2=1} \text{tr}\left[\left(\sum_{i=1}^r \sigma_i u_iv_i^T\right)^T \Delta W\right]\\
&=\max_{\Vert\Delta W\Vert_2=1} \sum_{i=1}^r \sigma_i \text{tr}\left(v_i u_i^T \Delta W\right)\tag{Distribute + linearity of trace}\\
&=\max_{\Vert\Delta W\Vert_2=1} \sum_{i=1}^r \sigma_i u_i^T \Delta W v_i\\
&\leq \max_{\Vert\Delta W\Vert_2=1} \sum_{i=1}^r \sigma_i\\
&=\Vert G\Vert_*
\end{aligned}
$$

It's trivial to show that this equality is achieved by $\Delta W=UV^T$.  Therefore, the direction of steepest descent is the polar factor of the gradient: $\argmax_{\Vert\Delta W\Vert_2=1} \langle \Delta W, G\rangle=UV^T$.  We notate this as $\text{msign}(G)$.

Then the steepest descent optimizer is uniquely:
$$W_{t+1}=W_t-\eta\cdot \text{msign}(\nabla_{W_t} L)$$