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
    'Attention Q': 0.0,
    'Attention K': 0.0,
    'Attention V': 0.0,
    'Attention O': 0.0,
    'MLP Input': 0.0,
    'MLP Output': 0.0
}

def spectral_echo_via_svd(Ghat: Tensor, kappa: float | Tensor, sigma2: float | Tensor):
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
        # Map slice index to layer type kappa
        slice_types = ['Attention Q', 'Attention K', 'Attention V', 'Attention O']
        sigma_slices = None
        if isinstance(sigma2, torch.Tensor) and sigma2.numel() == 4:
            sigma_slices = sigma2.flatten().tolist()
        for i, sl in enumerate(X):
            U, S, Vh = torch.linalg.svd(sl, full_matrices=False)
            S2 = torch.clamp(S * S, min=1e-12)
            k = float(layer_type_to_kappa.get(slice_types[i], float(kappa) if not isinstance(kappa, torch.Tensor) else float(kappa.item())))
            s2 = float(sigma_slices[i]) if sigma_slices is not None else (float(sigma2) if not isinstance(sigma2, torch.Tensor) else float(sigma2.item()))
            w = 1.0 / (1.0 + (k * s2) / S2)
            Y = U @ torch.diag_embed(w) @ Vh
            slices.append(Y)
        Y = torch.stack(slices, dim=0)
    else:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S2 = torch.clamp(S * S, min=1e-12)
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
    v = zeropower_function(momentum * momentum_buffer + (1 - momentum) * grad, float(kappa.item()), float(sigma2.item()))

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
        # Build param -> kind mapping (strict)
        # kind in { 'packed_attention', 'MLP Input', 'MLP Output' }
        self.param_kind: dict[Tensor, str] = {}
        # Advertise noise sigma2 consumption (duck typing for training_core)
        self.expects_noise_sigma2 = True
        def _feed_sigma2(ps, s2):
            # s2 is a sequence aligned to ps; each element can be a float or 4‑length list for packed attention
            if hasattr(s2, 'detach'):
                vals = s2.detach().cpu().tolist()
            else:
                vals = list(s2)
            if len(vals) != len(ps):
                raise RuntimeError("feed_noise_sigma2: length mismatch")
            for p, v in zip(ps, vals):
                if isinstance(v, (list, tuple)):
                    if len(v) != 4:
                        raise RuntimeError("noise_sigma2_slices must have length 4 for packed attention")
                    self.state[p]['noise_sigma2_slices'] = [float(x) for x in v]
                else:
                    self.state[p]['noise_sigma2'] = float(v)
        self.feed_noise_sigma2 = _feed_sigma2
        for group in self.param_groups:
            for p in group["params"]:
                name = param_to_name.get(p)
                if name is None:
                    raise RuntimeError("Missing parameter name mapping for SpectralEcho")
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
                    if kind == 'MLP Input' or kind == 'MLP Output':
                        kappa_val = float(layer_type_to_kappa[kind])
                        sigma2_val = float(state.get("noise_sigma2", 0.0))
                        kappa_t = torch._as_tensor_fullprec(kappa_val)
                        sigma2_t = torch._as_tensor_fullprec(sigma2_val)
                        self.update_func(
                            p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                            p.grad, momentum,
                            eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                            eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                            kappa=kappa_t,
                            sigma2=sigma2_t,
                        )
                        continue
                    elif kind == 'packed_attention':
                        # Per-slice kappas handled inside spectral_echo_via_svd; sigma2 is a 4‑vector if available
                        s2_slices = state.get('noise_sigma2_slices', None)
                        if s2_slices is not None:
                            sigma2_t = torch.tensor(s2_slices, dtype=torch.float32, device=p.device)
                        else:
                            sigma2_t = torch.zeros(4, dtype=torch.float32, device=p.device)
                        kappa_t = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                        self.update_func(
                            p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                            p.grad, momentum,
                            eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                            eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                            kappa=kappa_t,
                            sigma2=sigma2_t,
                        )
                        continue
                    else:
                        raise RuntimeError(f"Unexpected param kind: {kind}")
                    # (handled in branches above)
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
