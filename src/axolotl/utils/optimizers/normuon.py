"""NorMuon optimizer: neuron-wise normalized Muon (https://arxiv.org/abs/2510.05491).

Keeps Muon's Newton-Schulz orthogonalization, then normalizes each row of the update by a
per-neuron second moment and rescales to a fixed RMS magnitude. NS iteration adapted from
axolotl.contribs.mit.muon (https://github.com/KellerJordan/Muon).
"""

import math

import torch

from axolotl.integrations.base import BaseOptimizerFactory


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """Newton-Schulz quintic iteration approximating the orthogonalization of G."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)  # spectral norm <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class NorMuon(torch.optim.Optimizer):
    """Neuron-wise normalized Muon; {0,1}-D / embed / lm_head params fall back to AdamW."""

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        normuon_beta2=0.95,
        normuon_eps=1e-8,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            normuon_beta2=normuon_beta2,
            normuon_eps=normuon_eps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        muon_params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        super().__init__(muon_params + adamw_params, defaults)
        for p in muon_params:
            assert p.ndim >= 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # NorMuon branch
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            beta2 = group["normuon_beta2"]
            eps = group["normuon_eps"]
            for p in [p for p in group["params"] if self.state[p]["use_muon"]]:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["neuron_moment2"] = g.new_zeros(g.size(0), 1)  # per-row 2nd moment
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                o = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(g.dtype)

                v = state["neuron_moment2"]
                v.mul_(beta2).add_(o.square().mean(dim=1, keepdim=True), alpha=1 - beta2)
                o = o / (v.sqrt() + eps)

                rows, cols = o.shape
                adjusted_lr = 0.2 * lr * math.sqrt(rows * cols) / (o.norm() + 1e-7)  # fixed-RMS scale
                p.data.mul_(1 - lr * wd)
                p.data.add_(o, alpha=-adjusted_lr)

            # AdamW fallback branch
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            wd = group["wd"]
            for p in [p for p in group["params"] if not self.state[p]["use_muon"]]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1, buf2 = state["moment1"], state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())

                scale = (1 - beta1**step) / (1 - beta2**step) ** 0.5
                p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr / scale)

        return loss


class NorMuonOptimizerFactory(BaseOptimizerFactory):
    def __call__(self, opt_model, training_args, **optimizer_kwargs) -> "NorMuon":
        lr = optimizer_kwargs.pop("lr")
        wd = optimizer_kwargs.pop("weight_decay")
        adamw_betas = optimizer_kwargs.pop("betas", (0.95, 0.95))
        adamw_eps = optimizer_kwargs.pop("eps", 1.0e-8)

        muon_params = []
        adamw_params = []
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim < 2
                or name.endswith("modules_to_save.default.weight")
                or any(embed_name in name for embed_name in ["embed_tokens", "lm_head"])
            ):
                adamw_params.append(param)
            else:
                muon_params.append(param)

        return NorMuon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            **optimizer_kwargs,
        )
