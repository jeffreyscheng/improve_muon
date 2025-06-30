import sys
with open(sys.argv[0]) as f:
    code = f.read()

from code.research.utils import *
import math

# -----------------------------------------------------------------------------
# Muon optimizer with tanh-based orthogonalization

def zeropower_via_tanh(G: Tensor, alpha: float = 10_000.0, eps: float = 1e-7) -> Tensor:
    """
    Computes the zeroth power / orthogonalization of G using tanh approximation.
    Uses the approximation sign(x) = tanh(alpha * x) / tanh(alpha).
    Implementation based on whiten_grad function.
    
    Handles rectangular matrices by splitting them into square blocks.
    """
    assert G.ndim >= 2
    
    # Check if matrix is square
    if G.size(-2) == G.size(-1):
        # Square matrix - original implementation
        eye = torch.eye(G.size(-1), device=G.device, dtype=G.dtype)
        E = torch.linalg.matrix_exp(2 * alpha * G)
        Y = torch.linalg.solve(E + eye, E - eye)
        return Y / math.tanh(alpha + eps)
    else:
        # Rectangular matrix - split into square blocks
        m, n = G.size(-2), G.size(-1)
        
        # Determine how to split: larger dimension must be divisible by smaller
        if m > n:
            assert m % n == 0, f"Matrix dimension {m} must be divisible by {n} for rectangular matrix handling"
            num_blocks = m // n
            # Split along first dimension: [m, n] -> [num_blocks, n, n]
            blocks = G.view(*G.shape[:-2], num_blocks, n, n)
            orthogonalized_blocks = []
            for i in range(num_blocks):
                block = blocks[..., i, :, :]
                eye = torch.eye(n, device=G.device, dtype=G.dtype)
                E = torch.linalg.matrix_exp(2 * alpha * block)
                Y = torch.linalg.solve(E + eye, E - eye)
                orthogonalized_blocks.append(Y / math.tanh(alpha + eps))
            return torch.stack(orthogonalized_blocks, dim=-3).view(*G.shape[:-2], m, n)
        else:
            assert n % m == 0, f"Matrix dimension {n} must be divisible by {m} for rectangular matrix handling"
            num_blocks = n // m
            # Split along second dimension: [m, n] -> [m, num_blocks, m]
            blocks = G.view(*G.shape[:-1], num_blocks, m).transpose(-2, -1)
            orthogonalized_blocks = []
            for i in range(num_blocks):
                block = blocks[..., i, :, :]
                eye = torch.eye(m, device=G.device, dtype=G.dtype)
                E = torch.linalg.matrix_exp(2 * alpha * block)
                Y = torch.linalg.solve(E + eye, E - eye)
                orthogonalized_blocks.append(Y / math.tanh(alpha + eps))
            result = torch.stack(orthogonalized_blocks, dim=-3).transpose(-2, -1)
            return result.view(*G.shape[:-2], m, n)

@torch.compile
def update_tanh(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_tanh(momentum * momentum_buffer + (1 - momentum) * grad)

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