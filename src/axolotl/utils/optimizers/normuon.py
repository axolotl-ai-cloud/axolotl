"""
NorMuon optimizer

Neuron-wise normalized Muon, from "NorMuon: Making Muon more efficient and
scalable" (https://arxiv.org/abs/2510.05491).

NorMuon keeps Muon's Newton-Schulz orthogonalization but additionally maintains a
per-neuron (row-wise) second moment of the orthogonalized update and normalizes
each row by it, then rescales the whole update to a fixed RMS magnitude. This
counteracts the highly non-uniform neuron norms produced by orthogonalization.

Muon orthogonalization adapted from the existing Muon contrib
(axolotl.contribs.mit.muon), originally from https://github.com/KellerJordan/Muon.
"""

import math

import torch

from axolotl.integrations.base import BaseOptimizerFactory


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class NorMuon(torch.optim.Optimizer):
    """
    NorMuon - neuron-wise Normalized Muon

    Runs SGD-momentum, orthogonalizes each 2D parameter's update with a Newton-Schulz iteration
    (as in Muon), then normalizes the orthogonalized update per output neuron (row) using an
    Adam-style second moment, and rescales it to a fixed RMS magnitude before applying it.

    Arguments:
        muon_params: The parameters to be optimized by NorMuon.
        lr: The learning rate.
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        normuon_beta2: Decay for the per-neuron second moment. (0.95 is a good default)
        normuon_eps: Epsilon for the per-neuron normalization.
        adamw_params: Parameters to be optimized by the AdamW backup. Any {0, 1}-D params or params
            detected as embed / lm_head are optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
    """

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

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use NorMuon, and those for which we will not
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #         NorMuon          #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            beta2 = group["normuon_beta2"]
            eps = group["normuon_eps"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    # per-neuron (row) second moment of the orthogonalized update
                    state["neuron_moment2"] = g.new_zeros(g.size(0), 1)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # orthogonalize the (momentum) update
                o = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(g.dtype)

                # per-neuron second moment and row-wise normalization
                v = state["neuron_moment2"]
                v.mul_(beta2).add_(
                    o.square().mean(dim=1, keepdim=True), alpha=1 - beta2
                )
                o = o / (v.sqrt() + eps)

                # rescale to a fixed RMS magnitude (matches Adam's update norm)
                rows, cols = o.shape
                adjusted_lr = 0.2 * lr * math.sqrt(rows * cols) / (o.norm() + 1e-7)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(o, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
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
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
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
            if not param.requires_grad or param.ndim < 2:
                continue
            if name.endswith("modules_to_save.default.weight") or any(
                embed_name in name for embed_name in ["embed_tokens", "lm_head"]
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
