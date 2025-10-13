import sys
with open(sys.argv[0]) as f:
    code = f.read()
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor
import torch.distributed as dist

layer_type_to_kappa = {
    'Attention Q': 2.01e5,
    'Attention K': 1.61e5,
    'Attention V': 4.44e4,
    'Attention O': 1.37e5,
    'MLP Input': 4.27e5,
    'MLP Output': 3.83e5
}

def spectral_echo_via_svd(Ghat: Tensor, kappa: Tensor, sigma2: Tensor):
    """
    1. computes Ghat = Uhat Shat Vhat^T via SVD.
    2. creates spectral_echo by applying f(s) = 1 / (1 + kappa * sigma^2 / s^2) elementwise to Shat.
    3. returns Uhat Shat Vhat^T

    TODO: implement
    """
    assert Ghat.ndim >= 2
    was_transposed = False
    X = Ghat.float()
    if X.size(-2) > X.size(-1):
        X = X.mT
        was_transposed = True

    # Handle packed QKVO: shape [4, H, W]
    if X.ndim == 3 and X.size(0) == 4:
        slices = []
        # Expect kappa as scalar (broadcast) or vector[4]; sigma2 as scalar or vector[4]
        if kappa.numel() == 1:
            kappa_vec = kappa.expand(4)
        else:
            kappa_vec = kappa.flatten()[:4]
        if sigma2.numel() == 1:
            sigma2_vec = sigma2.expand(4)
        else:
            sigma2_vec = sigma2.flatten()[:4]
        for i, sl in enumerate(X):
            U, S, Vh = torch.linalg.svd(sl, full_matrices=False)
            S2 = torch.clamp(S * S, min=1e-12)
            w = 1.0 / (1.0 + (kappa_vec[i] * sigma2_vec[i]) / S2)
            Y = U @ torch.diag_embed(w) @ Vh
            slices.append(Y)
        Y = torch.stack(slices, dim=0)
    else:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S2 = torch.clamp(S * S, min=1e-12)
        # Ensure scalar tensors for broadcasting
        if kappa.numel() != 1:
            kappa = kappa.flatten()[0:1]
        if sigma2.numel() != 1:
            sigma2 = sigma2.flatten()[0:1]
        w = 1.0 / (1.0 + (kappa * sigma2) / S2)
        Y = U @ torch.diag_embed(w) @ Vh

    if was_transposed:
        Y = Y.mT
    return Y.to(Ghat.dtype)

zeropower_function = spectral_echo_via_svd

@torch.compile
def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor, kappa: Tensor, sigma2: Tensor):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_function(momentum * momentum_buffer + (1 - momentum) * grad, kappa, sigma2)

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))

class SpectralEcho(torch.optim.Optimizer):
    """
    TODO
    """
    def __init__(self, params, update_fn, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1, *, param_to_name: dict[Tensor, str]):
        self.rank = rank
        self.world_size = world_size
        self.update_func = update_fn
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])
        # Keep a reference to names for logging
        self.param_to_name: dict[Tensor, str] = dict(param_to_name)
        # Build param -> kind mapping (strict)
        # kind in { 'packed_attention', 'MLP Input', 'MLP Output' }
        self.param_kind: dict[Tensor, str] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p not in param_to_name:
                    raise RuntimeError("Missing parameter name mapping for SpectralEcho")
                name = param_to_name[p]
                kind = _classify_param_kind(name, p)
                if kind is None:
                    raise RuntimeError(f"Unrecognized parameter for SpectralEcho: {name}")
                self.param_kind[p] = kind

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
                    if "mantissa" not in state:
                        state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    kind = self.param_kind[p]
                    # Build per-kind kappa and sigma2 tensors, then update once
                    if kind == 'MLP Input' or kind == 'MLP Output':
                        kappa_t = torch.tensor([layer_type_to_kappa[kind]], dtype=torch.float32, device=p.device)
                        sigma2_val = float(state["noise_sigma2"])  # must be provided
                        # Log received sigma2 value and zero cases (rank 0 only)
                        if self.rank == 0:
                            name = self.param_to_name[p]
                            print(f"SpectralEcho: sigma2 for {name} ({kind}) = {sigma2_val:.6e}")
                            if sigma2_val == 0.0:
                                print(f"SpectralEcho: received zero noise sigma^2 for {name} ({kind})")
                        sigma2_t = torch.tensor([sigma2_val], dtype=torch.float32, device=p.device)
                    elif kind == 'packed_attention':
                        kappas = [layer_type_to_kappa['Attention Q'], layer_type_to_kappa['Attention K'], layer_type_to_kappa['Attention V'], layer_type_to_kappa['Attention O']]
                        kappa_t = torch.tensor(kappas, dtype=torch.float32, device=p.device)
                        s2 = state['noise_sigma2']  # must be provided (sequence of 4)
                        if not isinstance(s2, (list, tuple)):
                            raise RuntimeError("packed_attention expects noise_sigma2 as a 4-element list/tuple")
                        s2_slices = [float(x) for x in s2]
                        if len(s2_slices) != 4:
                            raise RuntimeError("noise_sigma2_slices must have length 4 for packed attention")
                        if self.rank == 0:
                            name = self.param_to_name[p]
                            print(f"SpectralEcho: sigma2 for {name} (packed_attention) = {[float(x) for x in s2_slices]}")
                            zero_idx = [i for i, x in enumerate(s2_slices) if float(x) == 0.0]
                            if zero_idx:
                                print(f"SpectralEcho: received zero noise sigma^2 for {name} (packed_attention) slices {zero_idx}")
                        sigma2_t = torch.tensor(s2_slices, dtype=torch.float32, device=p.device)
                    else:
                        raise RuntimeError(f"Unexpected param kind: {kind}")
                    self.update_func(
                        p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                        p.grad, momentum,
                        eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                        eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                        kappa=kappa_t,
                        sigma2=sigma2_t,
                    )
                futures.append(dist.all_gather(params_pad[base_i:base_i + self.world_size], params_pad[base_i + self.rank], async_op=True).get_future())
        torch.futures.collect_all(futures).wait()

def _classify_param_kind(name: str, tensor: Tensor) -> str | None:
    # Strict classification based on naming and tensor shape
    if name.startswith("_orig_mod."):
        name = name[10:]
    if not name.startswith("blocks."):
        return None
    if "attn.qkvo_w" in name:
        # Expect packed attention [4, H, W]
        if tensor.ndim == 3 and tensor.size(0) == 4:
            return "packed_attention"
        # Future: if model splits Q/K/V/O into separate params, we would map by index; not supported yet
        return None
    if "mlp.fc_w" in name:
        return "MLP Input"
    if "mlp.proj_w" in name:
        return "MLP Output"
    return None
