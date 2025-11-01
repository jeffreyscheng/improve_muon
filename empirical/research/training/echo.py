# empirical/research/training/echo.py
import torch
import torch.distributed as dist
from torch.optim import Optimizer

from empirical.research.analysis.core_math import (
    get_aligned_svds,
    solve_for_spectral_echo_using_reverb,
)

class SpectralEcho(Optimizer):
    """
    Noise-aware spectral update via reverb OLS echoes.
    Interface matches Muon(params, lr, weight_decay, momentum, rank, world_size).
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.9, rank=0, world_size=1):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._m = {}
        self._local = {}

        # capture local (pre-allreduce) grads per param
        for g in self.param_groups:
            for p in g["params"]:
                if p.ndim >= 2:
                    p.register_hook(lambda grad, p=p: self._local.__setitem__(p, grad.detach().float().clone()))

    @torch.no_grad()
    def _gather(self, G_local: torch.Tensor) -> torch.Tensor:
        bufs = [torch.empty_like(G_local) for _ in range(self.world_size)]
        dist.all_gather(bufs, G_local.contiguous())
        return torch.stack(bufs, dim=0)  # [R,H,W]

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()

        for group in self.param_groups:
            lr = group["lr"]; wd = group["weight_decay"]; mu = group["momentum"]

            for p in group["params"]:
                if p.ndim < 2 or p.grad is None:
                    continue

                # decoupled WD
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # momentum on mean gradient (post-allreduce)
                gbar = p.grad.detach().float()
                m = self._m.get(p)
                if m is None:
                    m = torch.zeros_like(gbar, dtype=torch.float32)
                    self._m[p] = m
                m.mul_(mu).add_(gbar, alpha=(1.0 - mu))

                # gather replicas of local grads and align
                G_repl = self._gather(self._local[p])  # [R,H,W]
                U_aligned, _, V_aligned = get_aligned_svds(G_repl)  # U:(R,H,D), V:(R,W,D)
                # echoes per-replica, per-direction
                echoes = solve_for_spectral_echo_using_reverb(
                    left_bases_U=U_aligned,            # (R,H,D)
                    right_bases_V=V_aligned,           # (R,W,D)
                )                                      # (R,D)

                # aggregate echoes across replicas
                zeta = echoes.median(dim=0).values     # (D,)

                # spectral update in momentum-mean basis
                U, S, Vh = torch.linalg.svd(m, full_matrices=False)
                V = Vh.transpose(-2, -1)
                r = zeta.numel()
                U_r = U[:, :r]
                V_r = V[:, :r]
                # U diag(zeta) V^T  (broadcast scaling of columns in U)
                update = (U_r * zeta.unsqueeze(0)) @ V_r.transpose(0, 1)

                p.add_(update, alpha=-lr)

        return None
