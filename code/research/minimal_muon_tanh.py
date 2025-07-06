import sys
with open(sys.argv[0]) as f:
    code = f.read()

from code.research.utils import *
import math

# -----------------------------------------------------------------------------
# Muon optimizer with tanh-based orthogonalization


def matrix_tanh(A: torch.Tensor, *, tol: float = 0.5):
    """
    Back-stable tanh(A) for a stack of square matrices [..., n, n].
    Works with torch.compile; no grads recorded.
    """
    dtype, device = A.dtype, A.device
    n = A.shape[-1]
    I = torch.eye(n, dtype=dtype, device=device)

    # 1. scaling (batch-aware)
    one_norm = torch.linalg.matrix_norm(A, ord=1)          # [...,]
    s = int(torch.ceil(torch.log2(torch.clamp(one_norm / tol, min=1.0))).max().item())
    B = A / (2.0 ** s)

    # 2. core evaluation via expm
    E = torch.matrix_exp(2.0 * B)
    Y = torch.linalg.solve(E + I, E - I)                   # (E-I)(E+I)^{-1}

    # 3. unsquaring
    for _ in range(s):
        Y2 = Y @ Y
        Y  = torch.linalg.solve(I + Y2, 2.0 * Y)

    return Y

def tanh_residual(A, Y):
    E2 = torch.matrix_exp(2.0 * A)
    delta = (E2 + torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)) @ Y - (E2 - torch.eye(A.shape[-1], dtype=A.dtype, device=A.device))
    return torch.linalg.matrix_norm(delta) / torch.linalg.matrix_norm(E2 - torch.eye(A.shape[-1], dtype=A.dtype, device=A.device))

def orth_residual(Z):
    """‖Zᵀ Z − I‖_F"""
    n = Z.shape[-1]
    I = torch.eye(n, dtype=Z.dtype, device=Z.device)
    return torch.linalg.matrix_norm(Z.transpose(-2, -1) @ Z - I)

def skew_residual(G, Z):
    """‖Gᵀ Z − Zᵀ G‖_F"""
    return torch.linalg.matrix_norm(
        G.transpose(-2, -1) @ Z - Z.transpose(-2, -1) @ G
    )


def zeropower_via_tanh_square(G, alpha=128.0):
    tanh_alpha = torch.tanh(torch.tensor(alpha, dtype=G.dtype, device=G.device))
    tanh_alpha_G = matrix_tanh(alpha * G)
    return (tanh_alpha_G / tanh_alpha).to(torch.bfloat16)


# def zeropower_via_tanh_square(G: Tensor, alpha: float = 10_000.0, eps: float = 1e-7) -> Tensor:
#     """
#     Computes the zeroth power / orthogonalization of a square matrix G using tanh approximation.
#     Uses the approximation sign(x) = tanh(alpha * x) / tanh(alpha).
#     """
#     assert G.ndim >= 2
#     assert G.size(-2) == G.size(-1), "Matrix must be square"
    
#     # Use .mT (matrix transpose) like the original Newton-Schulz implementation
#     Gram = G.mT @ G                                 # one GEMM

#     # 2. Symmetric eigendecomposition
#     lam, V = torch.linalg.eigh(Gram.to(torch.float64))

#     G = G.to(torch.bfloat16)
#     lam = lam.to(torch.bfloat16)
#     V = V.to(torch.bfloat16)

#     # 3. Scalar map  φ(λ) = tanh(alpha√λ)/(√λ tanh alpha)
#     sqrtlam = torch.sqrt(lam.clamp_min(eps))
#     phi = torch.tanh(alpha * sqrtlam) / (sqrtlam * math.tanh(alpha))
#     phi = torch.where(lam < eps, torch.full_like(phi, alpha), phi)

#     # 4. Assemble  V diag(φ) Vᵀ  without an extra GEMM
#     Vphi = V * phi.unsqueeze(-1)    # scale columns (cheap, n² ops) - explicit broadcasting
#     Y    = G @ Vphi                 # GEMM 1
#     F    = Y @ V.mT                 # GEMM 2

#     return F.to(dtype=G.dtype)

@torch.compile
def zeropower_via_tanh(G: Tensor, alpha: float = 10_000.0) -> Tensor:
    """
    Computes the zeroth power / orthogonalization of G using tanh approximation.
    Handles 3 specific matrix shapes: 1024x1024, 1024x4096, and 4096x1024.
    """
    assert G.ndim >= 2
    m, n = G.size(-2), G.size(-1)
    
    # Case 1: Square matrix (1024x1024)
    if m == n:
        return zeropower_via_tanh_square(G, alpha)
    
    # Case 2: Wide matrix (1024x4096) - split into 4 blocks along columns
    elif m == 1024 and n == 4096:
        # Split into 4 blocks of 1024x1024
        blocks = G.view(*G.shape[:-1], 4, 1024)  # (..., 1024, 4, 1024)
        block0 = zeropower_via_tanh_square(blocks[..., 0, :], alpha)
        block1 = zeropower_via_tanh_square(blocks[..., 1, :], alpha)
        block2 = zeropower_via_tanh_square(blocks[..., 2, :], alpha)
        block3 = zeropower_via_tanh_square(blocks[..., 3, :], alpha)
        return torch.stack([block0, block1, block2, block3], dim=-2).view(*G.shape[:-1], 4096)
    
    # Case 3: Tall matrix (4096x1024) - split into 4 blocks along rows
    elif m == 4096 and n == 1024:
        # Split into 4 blocks of 1024x1024
        blocks = G.view(*G.shape[:-2], 4, 1024, 1024)  # (..., 4, 1024, 1024)
        block0 = zeropower_via_tanh_square(blocks[..., 0, :, :], alpha)
        block1 = zeropower_via_tanh_square(blocks[..., 1, :, :], alpha)
        block2 = zeropower_via_tanh_square(blocks[..., 2, :, :], alpha)
        block3 = zeropower_via_tanh_square(blocks[..., 3, :, :], alpha)
        return torch.stack([block0, block1, block2, block3], dim=-3).view(*G.shape[:-2], 4096, 1024)
    
    else:
        raise ValueError(f"Unsupported matrix shape: {m}x{n}. Only 1024x1024, 1024x4096, and 4096x1024 are supported.")

# @torch.compile
def update_tanh(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_tanh(momentum * momentum_buffer + (1 - momentum) * grad)

    # checks for square matrices
    if grad.shape[-2] == grad.shape[-1]:
        grad = grad.to(torch.float32)
        v = v.to(torch.float32)

        # tanh_res = tanh_residual(grad, v)
        orth_res = orth_residual(v)
        skew_res = skew_residual(grad, v)

        # assert tanh_res < 1e-10
        assert orth_res < 1e-10
        assert skew_res < 1e-10
        print0(f"orth_res {orth_res} skew_res {skew_res}")
    else:
        # just do the check for the first 1024x1024 block
        grad = grad.to(torch.float32)
        v = v.to(torch.float32)
        grad = grad[..., :1024, :1024]
        v = v[..., :1024, :1024]
        # tanh_res = tanh_residual(grad, v)
        orth_res = orth_residual(v)
        skew_res = skew_residual(grad, v)
        # assert tanh_res < 1e-10, tanh_res
        assert orth_res < 1e-10, orth_res
        assert skew_res < 1e-10, skew_res
        print0(f"orth_res {orth_res} skew_res {skew_res}")

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))

class MuonTanh(torch.optim.Optimizer):
    """
    MuonTanh - MomentUm Orthogonalized by tanh approximation
    
    Identical to Muon except it uses tanh-based orthogonalization instead of Newton-Schulz.
    Uses the approximation sign(x) = tanh(alpha * x) / tanh(alpha).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    update_tanh(
                        p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                        p.grad, momentum,
                        eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                        eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                    )
                futures.append(dist.all_gather(params_pad[base_i:base_i + self.world_size], params_pad[base_i + self.rank], async_op=True).get_future())
        torch.futures.collect_all(futures).wait()

# Modify create_model_and_optimizers to use MuonTanh
def create_model_and_optimizers_tanh(args, rank, world_size):
    """Create model and optimizers using MuonTanh instead of Muon."""
    model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                           max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
    scalar_params = [model.scalars]
    head_params: list[nn.Parameter] = [model.lm_head_w]
    hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True)

    # Create optimizers using MuonTanh instead of Muon
    adam_param_groups = [
        dict(params=head_params, lr=1/320), 
        dict(params=embed_params, lr=0.3), 
        dict(params=scalar_params, lr=0.015)
    ]
    optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
    optimizer2 = MuonTanh(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2]

    opt2params = {opt: opt_params(opt) for opt in optimizers}
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return model, optimizers, opt2params

# -----------------------------------------------------------------------------
# Main training loop (identical to minimal_medium.py except using MuonTanh)

run_id, rank, world_size, device, master_process = setup_distributed_training()
print0, run_id_full, logfile = setup_logging(run_id, master_process)
log_system_info(print0, code)

args = Hyperparameters()
model, optimizers, opt2params = create_model_and_optimizers_tanh(args, rank, world_size)
model = torch.compile(model, dynamic=False)
warmup_kernels(model, optimizers, args)

torch.cuda.reset_peak_memory_stats()
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
dist.barrier()
t0 = time.perf_counter()

for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step, args.num_iterations))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{args.num_iterations} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        dist.barrier()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id_full}", exist_ok=True)
            torch.save(log, f"logs/{run_id_full}/state_step{step:06d}.pt")
        break

    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step, args.num_iterations)).backward()
    opt2futures = {
        opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
        for opt, params in opt2params.items()
    }
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step, args.num_iterations, args.cooldown_frac)
    if len(optimizers) > 1:  # Handle MuonTanh momentum warmup
        for group in optimizers[1].param_groups:
            frac = min(step / 300, 1)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt in optimizers:
        torch.futures.collect_all(opt2futures[opt]).wait()
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()