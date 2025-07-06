# =============================================================================
# ZEROPOWER BACKEND SELECTION - CAN BE OVERRIDDEN BY CLI
# =============================================================================
DEFAULT_ZEROPOWER_METHOD = "newton_schulz"  # Options: "newton_schulz", "svd_polar", "classic_newton_schulz", "tanh_matrix"
DEFAULT_ZEROPOWER_HYPERPARAMS = {}

import sys
import argparse
import json
with open(sys.argv[0]) as f:
    code = f.read()

import os
import uuid
import time
import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import itertools

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention
torch._inductor.config.coordinate_descent_tuning = True
# torch._dynamo.config.compiled_autograd = True  # Disabled due to FlexAttention incompatibility

# =============================================================================
# ZEROPOWER BACKEND IMPLEMENTATIONS
# =============================================================================

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_svd_polar(G: Tensor) -> Tensor:
    """
    SVD-based polar decomposition for orthogonalization.
    """
    try:
        # Handle non-square matrices like Newton-Schulz does
        was_transposed = False
        X = G.float()
        if G.size(-2) > G.size(-1):
            X = X.mT
            was_transposed = True
        
        # Use modern linalg.svd instead of deprecated torch.svd
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # Note: torch.linalg.svd returns Vh (V hermitian), not V
        result = U @ Vh
        
        if was_transposed:
            result = result.mT
            
        return result.to(G.dtype)
    except Exception:
        # Fallback to Newton-Schulz if SVD fails
        return zeropower_via_newtonschulz5(G)

def zeropower_via_classic_newton_schulz(G: Tensor, num_iters: int = 15) -> Tensor:
    """
    Classic Newton-Schulz iteration using f(x) = 1.5x - 0.5x^3.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Classic Newton-Schulz iteration: f(x) = 1.5x - 0.5x^3
    for _ in range(num_iters):
        X = 1.5 * X - 0.5 * X @ X @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_tanh_matrix(G: Tensor, alpha: float = 10.0) -> Tensor:
    """
    Matrix tanh approximation (simplified version to avoid hangs).
    Falls back to Newton-Schulz for safety.
    """
    # For now, fallback to Newton-Schulz to avoid the hanging issues we discovered
    # This can be improved with a proper safe matrix tanh implementation
    return zeropower_via_newtonschulz5(G)

# =============================================================================
# ZEROPOWER BACKEND REGISTRY AND SELECTOR
# =============================================================================

ZEROPOWER_BACKENDS = {
    "newton_schulz": zeropower_via_newtonschulz5,
    "svd_polar": zeropower_via_svd_polar,
    "classic_newton_schulz": zeropower_via_classic_newton_schulz,
    "tanh_matrix": zeropower_via_tanh_matrix,
}

def get_zeropower_function(method: str, hyperparams: dict):
    """Get the zeropower function with hyperparameters applied."""
    if method not in ZEROPOWER_BACKENDS:
        available = ", ".join(ZEROPOWER_BACKENDS.keys())
        raise ValueError(f"Unknown zeropower method '{method}'. Available: {available}")
    
    base_func = ZEROPOWER_BACKENDS[method]
    
    # Create a wrapper that applies hyperparameters
    def zeropower_with_hyperparams(G: Tensor) -> Tensor:
        return base_func(G, **hyperparams)
    
    return zeropower_with_hyperparams

# Global variables to be set by CLI parsing
ZEROPOWER_METHOD = DEFAULT_ZEROPOWER_METHOD
ZEROPOWER_HYPERPARAMS = DEFAULT_ZEROPOWER_HYPERPARAMS
zeropower_func = None  # Will be set after CLI parsing

# =============================================================================
# CONFIGURABLE MUON OPTIMIZER
# =============================================================================

def make_update_function(zeropower_function):
    """Create an update function with a specific zeropower function."""
    @torch.compile
    def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
        assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
        grad = grad.float()
        momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
        v = zeropower_function(momentum * momentum_buffer + (1 - momentum) * grad)

        acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
        acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
        acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
        mantissa.copy_(acc_m_u32.to(torch.uint16))
    return update

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by configurable zeropower method.
    
    This version accepts a zeropower function in the constructor, allowing
    different optimizers to use different orthogonalization methods.
    """
    def __init__(self, params, zeropower_function, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.update_func = make_update_function(zeropower_function)
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
                    self.update_func(
                        p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                        p.grad, momentum,
                        eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                        eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                    )
                futures.append(dist.all_gather(params_pad[base_i:base_i + self.world_size], params_pad[base_i + self.rank], async_op=True).get_future())
        torch.futures.collect_all(futures).wait()

# =============================================================================
# MODEL COMPONENTS (copied from utils.py)
# =============================================================================

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        self.qkvo_w = nn.Parameter(init_linear(torch.empty(4, hdim, dim)).bfloat16())
        self.qkvo_w.detach()[3].zero_()
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask, lambdas: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        v = norm(v)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            v = lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w[3])
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.proj_w = nn.Parameter(torch.zeros(dim, hdim).bfloat16())
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor):
        x = F.linear(x, self.fc_w)
        x = F.relu(x).square()
        x = F.linear(x, self.proj_w)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask, lambdas: Tensor, sa_lambdas: Tensor):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, block_mask, sa_lambdas)
        x = x + self.mlp(norm(x))
        return x

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head_w = nn.Parameter(torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim))
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None])

        skip_connections = []
        skip_map = {9: 6, 10: 4, 11: 2}
        skip_weights = self.scalars[:len(self.blocks)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        for i in range(len(self.blocks)):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
            skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1), self.lm_head_w.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        loss = 0
        for i in range(4):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(4)[i], self.lm_head_w.bfloat16()).float()
            loss += F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq.chunk(4)[i]) / 4
        return loss

# =============================================================================
# DATA LOADER
# =============================================================================

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@dataclass
class Hyperparameters:
    train_files = "data/fineweb10B/fineweb_train_*.bin"
    val_files = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens = 10485760
    train_seq_len = 64*1024
    val_seq_len = 4*64*1024
    num_iterations = 5960
    cooldown_frac = 0.7
    vocab_size = 50257
    val_loss_every = 125
    save_checkpoint = False

def nvidia_smi():
    import subprocess
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

def get_lr(step: int, num_iterations: int, cooldown_frac: float):
    x = step / num_iterations
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac

@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

def get_window_size_blocks(step: int, num_iterations: int):
    x = step / num_iterations
    assert 0 <= x <= 1
    factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x
    window_size = next_multiple_of_n(3456 * factor, n=128)
    return get_window_size_blocks_helper(window_size)

def setup_distributed_training():
    run_id = int(os.environ.get("RUN_ID", 0))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = (rank == 0)
    return run_id, rank, world_size, device, master_process

def setup_logging(run_id, master_process):
    if master_process:
        run_id_full = f"{run_id:03d}_{uuid.uuid4()}"
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id_full}.txt"
        print(logfile)
    else:
        run_id_full = None
        logfile = None
    
    def print0(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)
    
    return print0, run_id_full, logfile

def log_system_info(print0, code, mlp_method, mlp_hyperparams, attn_method, attn_hyperparams):
    print0(code)
    print0("="*100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0(f"ZEROPOWER METHOD: {ZEROPOWER_METHOD} (hyperparams: {ZEROPOWER_HYPERPARAMS})")
    print0(f"MLP METHOD: {mlp_method} (hyperparams: {mlp_hyperparams})")
    print0(f"ATTENTION METHOD: {attn_method} (hyperparams: {attn_hyperparams})")
    print0(nvidia_smi())
    print0("="*100)

def create_model_and_optimizers(args, rank, world_size, zeropower_func_mlp, zeropower_func_attn):
    model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                           max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # Separate MLP and attention parameters
    mlp_matrix_params = []
    attn_matrix_params = []
    
    for block in model.blocks:
        # MLP parameters
        mlp_matrix_params.extend([p for p in block.mlp.parameters() if p.ndim >= 2])
        # Attention parameters
        if block.attn is not None:
            attn_matrix_params.extend([p for p in block.attn.parameters() if p.ndim >= 2])
    
    # Sort by size for efficiency
    mlp_matrix_params = sorted(mlp_matrix_params, key=lambda x: x.size(), reverse=True)
    attn_matrix_params = sorted(attn_matrix_params, key=lambda x: x.size(), reverse=True)

    embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
    scalar_params = [model.scalars]
    head_params: list[nn.Parameter] = [model.lm_head_w]

    adam_param_groups = [
        dict(params=head_params, lr=1/320), 
        dict(params=embed_params, lr=0.3), 
        dict(params=scalar_params, lr=0.015)
    ]
    optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
    optimizer2 = Muon(mlp_matrix_params, zeropower_func_mlp, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
    optimizer3 = Muon(attn_matrix_params, zeropower_func_attn, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2, optimizer3]

    opt2params = {opt: [p for group in opt.param_groups for p in group["params"]] for opt in optimizers}
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return model, optimizers, opt2params

def warmup_kernels(model, optimizers, args):
    warmup_steps = 10
    initial_state = copy.deepcopy(dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers]))
    for _ in range(warmup_steps):
        inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
        model(inputs.to(torch.int32), targets, get_window_size_blocks(0, args.num_iterations)).backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state

# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments for zeropower backend selection."""
    parser = argparse.ArgumentParser(description='Zeropower Testing with Flexible Backends')
    parser.add_argument('--zeropower-method', default=DEFAULT_ZEROPOWER_METHOD, 
                        choices=list(ZEROPOWER_BACKENDS.keys()),
                        help='Zeropower backend method')
    parser.add_argument('--zeropower-hyperparams', default='{}', type=str,
                        help='JSON string of hyperparameters for zeropower method')
    parser.add_argument('--mlp-method', default=None,
                        choices=list(ZEROPOWER_BACKENDS.keys()),
                        help='Zeropower method for MLP layers (defaults to main method)')
    parser.add_argument('--mlp-hyperparams', default='{}', type=str,
                        help='JSON string of hyperparameters for MLP zeropower method')
    parser.add_argument('--attn-method', default=None,
                        choices=list(ZEROPOWER_BACKENDS.keys()),
                        help='Zeropower method for attention layers (defaults to main method)')
    parser.add_argument('--attn-hyperparams', default='{}', type=str,
                        help='JSON string of hyperparameters for attention zeropower method')
    
    args = parser.parse_args()
    
    # Parse hyperparameters
    try:
        main_hyperparams = json.loads(args.zeropower_hyperparams)
        mlp_hyperparams = json.loads(args.mlp_hyperparams) 
        attn_hyperparams = json.loads(args.attn_hyperparams)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in hyperparameters: {e}")
    
    # Set defaults for MLP and attention if not specified
    mlp_method = args.mlp_method or args.zeropower_method
    attn_method = args.attn_method or args.zeropower_method
    
    # Use main hyperparams as base, then override with layer-specific ones
    mlp_hyperparams = {**main_hyperparams, **mlp_hyperparams}
    attn_hyperparams = {**main_hyperparams, **attn_hyperparams}
    
    return args.zeropower_method, main_hyperparams, mlp_method, mlp_hyperparams, attn_method, attn_hyperparams

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

# Parse CLI arguments
ZEROPOWER_METHOD, ZEROPOWER_HYPERPARAMS, mlp_method, mlp_hyperparams, attn_method, attn_hyperparams = parse_args()

# Create zeropower functions
zeropower_func = get_zeropower_function(ZEROPOWER_METHOD, ZEROPOWER_HYPERPARAMS)
zeropower_func_mlp = get_zeropower_function(mlp_method, mlp_hyperparams)
zeropower_func_attn = get_zeropower_function(attn_method, attn_hyperparams)

# Setup
run_id, rank, world_size, device, master_process = setup_distributed_training()
print0, run_id_full, logfile = setup_logging(run_id, master_process)
log_system_info(print0, code, mlp_method, mlp_hyperparams, attn_method, attn_hyperparams)

args = Hyperparameters()
model, optimizers, opt2params = create_model_and_optimizers(args, rank, world_size, zeropower_func_mlp, zeropower_func_attn)
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
    # Handle Muon momentum warmup for MLP and attention optimizers
    for opt_idx in [1, 2]:  # optimizer1 is AdamW, optimizer2 is MLP Muon, optimizer3 is Attention Muon
        if opt_idx < len(optimizers):
            for group in optimizers[opt_idx].param_groups:
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