import os
import sys
import uuid
import time
import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import itertools

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
if torch.cuda.is_available():
    torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
torch._inductor.config.coordinate_descent_tuning = True # we allow this flag for medium track
# torch._dynamo.config.compiled_autograd = True  # Disabled due to FlexAttention incompatibility

# Apply trace_structured patch at module level to avoid metadata_fn errors
from torch._logging._internal import trace_structured
import torch._inductor.codecache
import torch._inductor.graph
_original_trace_structured = trace_structured

# Global print function for the patch - will be set by setup_logging
_global_print0 = lambda s, console=False: None

def _patched_trace_structured(*args, **kwargs):
    # Support both (name, metadata_fn, **kwargs) and (name, **kwargs) forms
    name = args[0] if args else kwargs.get("name")
    metadata_fn = kwargs.get("metadata_fn")
    if metadata_fn is None and len(args) >= 2 and callable(args[1]):
        metadata_fn = args[1]
    try:
        if name == "inductor_output_code" and callable(metadata_fn):
            meta = metadata_fn() or {}
            _global_print0(f"inductor_output_code: {meta.get('filename', 'Unknown')}")
    except Exception:
        pass
    return _original_trace_structured(*args, **kwargs)

torch._inductor.codecache.trace_structured = _patched_trace_structured
torch._inductor.graph.trace_structured = _patched_trace_structured


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# Hyperparameters

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 64*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 5960 # number of iterations to run
    max_minibatches = None # maximum number of minibatches to process (None for no limit)
    cooldown_frac = 0.7 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False

# -----------------------------------------------------------------------------
# Utility functions

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

def get_lr(step: int, num_iterations: int, cooldown_frac: float):
    x = step / num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac

@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

def get_window_size_blocks(step: int, num_iterations: int):
    x = step / num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x # cubic schedule by @jadenj3o
    window_size = next_multiple_of_n(3456 * factor, n=128)
    return get_window_size_blocks_helper(window_size)

def opt_params(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [p for group in opt.param_groups for p in group["params"]]

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

# Common setup functions
def setup_distributed_training():
    """Setup distributed training environment and return key variables."""
    run_id = int(os.environ.get("RUN_ID", 0))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8 # this code is designed for 8xH100
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = (rank == 0)
    
    # Configure inductor for distributed-safe compilation
    import torch._inductor.config as config
    config.coordinate_descent_tuning = True  # Keep existing setting
    config.triton.unique_kernel_names = True  # Prevents name collisions
    config.compile_threads = 1  # Prevents race conditions
    config.worker_start_method = "spawn"  # Clean process isolation
    
    return run_id, rank, world_size, device, master_process

def safe_torch_compile(model, **compile_kwargs):
    """Standard torch.compile wrapper with an opt-out via env.

    Set TORCH_COMPILE_DISABLE=1 (or TORCHINDUCTOR_DISABLE/AOT_INDUCTOR_DISABLE) to skip compilation,
    useful on clusters where Triton/PTX versions are incompatible.
    """
    if os.environ.get("TORCH_COMPILE_DISABLE") == "1" or \
       os.environ.get("TORCHINDUCTOR_DISABLE") == "1" or \
       os.environ.get("AOT_INDUCTOR_DISABLE") == "1":
        return model
    return torch.compile(model, **compile_kwargs)

def setup_logging(run_id, master_process):
    """Setup logging infrastructure and return logging function and paths."""
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
    
    # Set the global print function for the trace_structured patch
    global _global_print0
    _global_print0 = print0
    
    return print0, run_id_full, logfile

def log_system_info(print0, code):
    """Log system information and code."""
    print0(code)
    print0("="*100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0(nvidia_smi())
    print0("="*100)

def warmup_kernels(model, optimizers, args):
    """Warmup training kernels and restore initial state."""
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

# -----------------------------------------------------------------------------
# New API Functions

def get_param_groups(model):
    """Extract parameter groups from model."""
    return {
        "hidden": sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True),
        "embed": [*model.embed.parameters(), *model.value_embeds.parameters()],
        "scalar": [model.scalars],
        "head": [model.lm_head_w]
    }

def create_gpt_with_muon(args, zeropower_fn):
    """Factory function that sets up everything for training."""
    from .architecture import GPT
    from .muon import Muon
    from .zeropower import make_update_function
    
    # Setup distributed training
    run_id, rank, world_size, device, master_process = setup_distributed_training()
    print0, run_id_full, logfile = setup_logging(run_id, master_process)
    
    # Read and log code
    with open(sys.argv[0]) as f:
        code = f.read()
    log_system_info(print0, code)
    
    # Create model
    model = GPT(args.vocab_size, 16, 8, 1024, max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    
    # Get parameter groups
    param_groups = get_param_groups(model)
    
    # Sanity check parameter coverage
    params_collections = [param_groups["hidden"], param_groups["embed"], param_groups["scalar"], param_groups["head"]]
    optimized_parameters_set = {p for params in params_collections for p in params}
    assert optimized_parameters_set == {*model.parameters()}
    assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)
    
    # Create optimizers
    update_fn = make_update_function(zeropower_fn)
    
    adam_param_groups = [
        dict(params=param_groups["head"], lr=1/320),
        dict(params=param_groups["embed"], lr=0.3),
        dict(params=param_groups["scalar"], lr=0.015)
    ]
    optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
    optimizer2 = Muon(param_groups["hidden"], update_fn, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2]
    
    # Set initial learning rates
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    # Compile model
    model = torch.compile(model, dynamic=False)
    
    # Warmup kernels
    warmup_kernels(model, optimizers, args)
    
    # Create train loader
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
    
    # Store global state for other functions
    global _global_print0, _global_run_id_full, _global_rank, _global_world_size, _global_master_process, _global_training_time_ms, _global_t0, _global_code
    _global_print0 = print0
    _global_run_id_full = run_id_full
    _global_rank = rank
    _global_world_size = world_size
    _global_master_process = master_process
    _global_training_time_ms = 0
    _global_t0 = time.perf_counter()  # Start the clock
    _global_code = code
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()  # Sync before starting timing
    
    return model, optimizers, train_loader

def create_gpt_with_optimizer(args,
                              build_hidden_optimizer_fn,
                              *,
                              hidden_lr: float = 0.025,
                              hidden_weight_decay: float = 0.01,
                              hidden_momentum: float = 0.95):
    """Generalized assembly: build model and optimizers with an injected hidden optimizer builder.

    Returns: model, (adamw, hidden_opt)
    Train loader should be created via create_train_loader(args).
    """
    # Setup distributed training
    run_id, rank, world_size, device, master_process = setup_distributed_training()
    print0, run_id_full, logfile = setup_logging(run_id, master_process)

    # Read and log code
    with open(sys.argv[0]) as f:
        code = f.read()
    log_system_info(print0, code)

    # Create model
    from .architecture import GPT
    model = GPT(args.vocab_size, 16, 8, 1024, max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # Get parameter groups
    param_groups = get_param_groups(model)

    # Coverage check
    params_collections = [param_groups["hidden"], param_groups["embed"], param_groups["scalar"], param_groups["head"]]
    optimized_parameters_set = {p for params in params_collections for p in params}
    assert optimized_parameters_set == {*model.parameters()}
    assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

    # AdamW for non-hidden
    adam_param_groups = [
        dict(params=param_groups["head"], lr=1/320),
        dict(params=param_groups["embed"], lr=0.3),
        dict(params=param_groups["scalar"], lr=0.015)
    ]
    optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)

    # Hidden optimizer via builder fn
    param_to_name = {p: n for n, p in model.named_parameters()}
    optimizer2 = build_hidden_optimizer_fn(
        param_groups["hidden"],
        model=model,
        param_to_name=param_to_name,
        device=device,
        rank=rank,
        world_size=world_size,
        lr=hidden_lr,
        weight_decay=hidden_weight_decay,
        momentum=hidden_momentum,
    )
    optimizers = [optimizer1, optimizer2]

    # Initial learning rates
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Compile model and warmup
    model = torch.compile(model, dynamic=False)
    warmup_kernels(model, optimizers, args)

    # Stash globals for train_loader helper
    global _global_print0, _global_run_id_full, _global_rank, _global_world_size, _global_master_process, _global_training_time_ms, _global_t0, _global_code
    _global_print0 = print0
    _global_run_id_full = run_id_full
    _global_rank = rank
    _global_world_size = world_size
    _global_master_process = master_process
    _global_training_time_ms = 0
    _global_t0 = time.perf_counter()
    _global_code = code

    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    return model, optimizers

def create_train_loader(args):
    """Return a distributed train data generator using stored globals."""
    assert _global_rank is not None and _global_world_size is not None
    return distributed_data_generator(args.train_files, _global_world_size * args.train_seq_len, _global_rank, _global_world_size)

def should_terminate(step, args):
    """Determine if training should terminate at given step."""
    if args.max_minibatches is not None and step >= args.max_minibatches:
        return True
    return step >= args.num_iterations

def should_validate(step, args):
    """Determine if validation should occur at given step."""
    last_step = should_terminate(step, args)
    return last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

def is_logging_step_dummy_every_100(step):
    """Determine if logging should occur at given step (dummy implementation - every 100 steps)."""
    return step % 100 == 0

def validate_and_log(model, step, args, optimizers=None):
    """Run validation and log results."""
    global _global_print0, _global_rank, _global_world_size, _global_master_process, _global_training_time_ms, _global_t0, _global_run_id_full, _global_code
    
    # Stop the clock and update training time
    dist.barrier()
    _global_training_time_ms += 1000 * (time.perf_counter() - _global_t0)
    
    model.eval()
    val_batch_size = _global_world_size * args.val_seq_len
    assert args.val_tokens % val_batch_size == 0
    val_steps = args.val_tokens // val_batch_size
    val_loader = distributed_data_generator(args.val_files, val_batch_size, _global_rank, _global_world_size)
    val_loss = 0
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets = next(val_loader)
            val_loss += model(inputs, targets, get_window_size_blocks(step, args.num_iterations))
    val_loss /= val_steps
    del val_loader
    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    
    # Log with training time like original
    max_steps = args.max_minibatches if args.max_minibatches is not None else args.num_iterations
    effective_max_steps = min(max_steps, args.num_iterations)
    _global_print0(f"step:{step}/{effective_max_steps} val_loss:{val_loss:.6f} train_time:{_global_training_time_ms:.0f}ms step_avg:{_global_training_time_ms/max(step, 1):.2f}ms", console=True)
    model.train()
    
    # Handle final checkpoint saving
    if should_terminate(step, args):
        if _global_master_process and args.save_checkpoint and optimizers is not None:
            log = dict(step=step, code=_global_code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{_global_run_id_full}", exist_ok=True)
            torch.save(log, f"logs/{_global_run_id_full}/state_step{step:06d}.pt")
    
    # Start the clock again
    dist.barrier()
    _global_t0 = time.perf_counter()

def train_step(model, inputs, targets, step, args):
    """Execute forward pass and return loss."""
    loss = model(inputs, targets, get_window_size_blocks(step, args.num_iterations))
    loss.backward()
    
    # Log training step like original (with step+1)
    global _global_print0, _global_training_time_ms, _global_t0
    approx_training_time_ms = _global_training_time_ms + 1000 * (time.perf_counter() - _global_t0)
    max_steps = args.max_minibatches if args.max_minibatches is not None else args.num_iterations
    train_steps = min(max_steps, args.num_iterations)  # Match original variable name
    _global_print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
    
    return loss

def optimize_step(model, optimizers, step, args):
    """Distributed gradient synchronization and optimizer stepping."""
    global _global_rank, _global_world_size
    
    # Create parameter mapping
    opt2params = {opt: opt_params(opt) for opt in optimizers}
    # Identify Muon optimizer and its params (Muon is second as per create_gpt_with_muon)
    muon_opt = optimizers[1] if len(optimizers) > 1 else None
    muon_params = opt2params.get(muon_opt, [])

    # 0) Compute local normalized squared norms for hidden grads (A numerator) before any grad averaging
    #    A = E_i[(1/N)||G_i||_F^2] is obtained via AVG all_reduce over ranks of these local values
    if muon_opt is not None and len(muon_params) > 0:
        a_local = torch.zeros(len(muon_params), device=torch.device("cuda"), dtype=torch.float32)
        # For packed attention params, also track per-slice A
        a_slices_map: dict[Tensor, torch.Tensor] = {}
        for idx, p in enumerate(muon_params):
            if p.grad is not None:
                g = p.grad.float()
                if g.ndim == 3 and g.size(0) == 4:
                    # Per-slice mean square
                    H, W = g.size(-2), g.size(-1)
                    a_slices = torch.zeros(4, device=g.device, dtype=torch.float32)
                    for i in range(4):
                        gi = g[i]
                        a_slices[i] = (gi.mul(gi).sum() / (H * W))
                    a_slices_map[p] = a_slices
                    # Also aggregate full tensor value for legacy scalar path
                    a_local[idx] = (g.mul(g).sum() / p.numel())
                else:
                    a_local[idx] = (g.mul(g).sum() / p.numel())
            else:
                a_local[idx] = 0.0
        # Average across ranks to get A
        dist.all_reduce(a_local, op=dist.ReduceOp.AVG)
        # Average per-slice across ranks
        for p, vec in a_slices_map.items():
            dist.all_reduce(vec, op=dist.ReduceOp.AVG)
    else:
        a_local = None
        a_slices_map = {}

    # 1. Create async all_reduce futures for each optimizer's parameters
    opt2futures = {
        opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
        for opt, params in opt2params.items()
    }
    
    # 2. Update learning rates using get_lr(step) schedule
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step, args.num_iterations, args.cooldown_frac)
    
    # 3. Apply Muon momentum warmup for optimizer2
    if len(optimizers) > 1:  # Muon is the second optimizer
        for group in optimizers[1].param_groups:
            frac = min(step / 300, 1)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    
    # 4. Wait for gradient sync
    for opt in optimizers:
        torch.futures.collect_all(opt2futures[opt]).wait()

    # 5. With averaged grads, compute B for Muon and derive sigma
    #    B = (1/N)||Ḡ||_F^2, and under the simple i.i.d. noise model
    #    A - B ≈ (1 - 1/W) * sigma^2  ⇒  sigma^2 ≈ (A - B) / (1 - 1/W)
    if muon_opt is not None and len(muon_params) > 0 and a_local is not None:
        W = dist.get_world_size() if dist.is_initialized() else 1
        b_vec = torch.zeros_like(a_local)
        # For packed attention, track per-slice B
        b_slices_map: dict[Tensor, torch.Tensor] = {}
        for idx, p in enumerate(muon_params):
            if p.grad is not None:
                gbar = p.grad.float()
                if gbar.ndim == 3 and gbar.size(0) == 4:
                    H, W = gbar.size(-2), gbar.size(-1)
                    b_slices = torch.zeros(4, device=gbar.device, dtype=torch.float32)
                    for i in range(4):
                        gi = gbar[i]
                        b_slices[i] = (gi.mul(gi).sum() / (H * W))
                    b_slices_map[p] = b_slices
                    b_vec[idx] = (gbar.mul(gbar).sum() / p.numel())
                else:
                    b_vec[idx] = (gbar.mul(gbar).sum() / p.numel())
            else:
                b_vec[idx] = 0.0
        # Denominator (1 - 1/W); guard W=1 edge case
        denom = 1.0 - (1.0 / float(max(W, 1)))
        if denom <= 0.0:
            sigma2 = torch.zeros_like(a_local)
        else:
            sigma2 = torch.clamp_min(a_local - b_vec, 0.0) / denom
        sigma_vec = torch.sqrt(sigma2)
        # Compute per-slice sigma2 for packed attention params
        sigma2_slices_map: dict[Tensor, torch.Tensor] = {}
        for p in a_slices_map.keys():
            a_s = a_slices_map[p]
            b_s = b_slices_map.get(p, torch.zeros_like(a_s))
            if denom <= 0.0:
                sigma2_slices = torch.zeros_like(a_s)
            else:
                sigma2_slices = torch.clamp_min(a_s - b_s, 0.0) / denom
            sigma2_slices_map[p] = sigma2_slices
    else:
        sigma_vec = None
        sigma2_slices_map = {}

    # 6. Step optimizers
    for opt in optimizers:
        opt.step()

    # 7. Feed noise sigma^2 to hidden optimizer if it expects it (duck-typed)
    if muon_opt is not None and len(muon_params) > 0 and sigma_vec is not None:
        if hasattr(muon_opt, 'expects_noise_sigma2') and getattr(muon_opt, 'expects_noise_sigma2'):
            # Build a mixed list: scalar variances for normal params; 4-element lists for packed attention
            values = []
            for idx, p in enumerate(muon_params):
                if p in sigma2_slices_map:
                    vals = sigma2_slices_map[p].detach().cpu().tolist()
                    values.append([float(x) for x in vals])
                else:
                    values.append(float((sigma_vec[idx] * sigma_vec[idx]).item()))
            muon_opt.feed_noise_sigma2(muon_params, values)

    # 8. Zero gradients
    # Zero gradients
    model.zero_grad(set_to_none=True)

def run_loggers(loggers, model, optimizers, step):
    """Iterate through loggers and call each one."""
    global _global_run_id_full, _global_master_process
    
    if _global_master_process:
        other_state = {"run_name": _global_run_id_full, "step": step}
        for logger_fn in loggers:
            logger_fn(model, optimizers[0], other_state)  # Pass first optimizer like original

# Global variables for state sharing
_global_print0 = None
_global_run_id_full = None
_global_rank = None
_global_world_size = None
_global_master_process = None
_global_training_time_ms = 0
_global_t0 = None
_global_code = None
