# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SinkGD: stateless gradient multi-normalization optimizer.

Implements the SinkGD optimizer from "Gradient Multi-Normalization for Stateless
and Scalable LLM Training" (Scetbon et al., 2025, https://arxiv.org/abs/2502.06742).

Linear weight matrices are updated with the stateless SR-Sinkhorn procedure
(Algorithm 3/4): the raw gradient is alternately row- and column-normalized for a
few iterations, then applied as the update. No optimizer state is stored for these
parameters. Embeddings, the output head, and 1D parameters (norms, biases) fall
back to AdamW, reusing torchao's low-bit optimizer base so that this state is kept
in 8-bit and the overall footprint stays close to plain SGD.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torchao.optim.adam import _AdamBase, single_param_adam
from torchao.optim.quant_utils import _fp32_to_bf16_sr
from torchao.optim.subclass_8bit import OptimState8bit

from axolotl.integrations.base import BaseOptimizerFactory
from axolotl.utils.optimizers.sinkgd_triton import fused_available, fused_sinkgd_step


def sr_sinkhorn(grad: Tensor, iters: int, eps: float) -> Tensor:
    """SR-Sinkhorn (Algorithm 3): alternate row/column L2 normalization.

    Each iteration rescales rows to L2 norm sqrt(n) then columns to L2 norm
    sqrt(m), driving the gradient towards a doubly-balanced fixed point.

    Operates on the last two dims, so any leading dims (e.g. the expert axis of
    a fused MoE weight ``[num_experts, in, out]``) are treated as an independent
    batch and each matrix is normalized separately.
    """
    m, n = grad.shape[-2], grad.shape[-1]
    sqrt_n = n**0.5
    sqrt_m = m**0.5
    x = grad
    for _ in range(iters):
        x = x * (sqrt_n / x.norm(dim=-1, keepdim=True).clamp_min(eps))
        x = x * (sqrt_m / x.norm(dim=-2, keepdim=True).clamp_min(eps))
    return x


def single_param_sinkgd(
    p: Tensor,
    grad: Tensor,
    lr: Tensor,
    weight_decay: float,
    iters: int,
    eps: float,
    stochastic_round: bool,
) -> None:
    """Fused stateless SinkGD update for one weight; meant to be torch.compiled.

    Normalization runs over the last two dims (leading dims, e.g. the expert axis
    of a fused MoE weight, are batched and normalized per-matrix). The iterate
    stays in the gradient dtype to halve memory traffic while norm reductions
    accumulate in fp32. `iters`/`eps`/`weight_decay` are constants so the loop
    unrolls, and `lr` is a tensor input so a changing schedule never recompiles.
    """
    m, n = grad.shape[-2], grad.shape[-1]
    sqrt_n = n**0.5
    sqrt_m = m**0.5
    x = grad
    for _ in range(iters):
        rn = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
        x = x * (sqrt_n / rn.clamp_min(eps)).to(x.dtype)
        cn = torch.linalg.vector_norm(x, dim=-2, keepdim=True, dtype=torch.float32)
        x = x * (sqrt_m / cn.clamp_min(eps)).to(x.dtype)

    p_f32 = p.float()
    if weight_decay != 0.0:
        p_f32 = p_f32 * (1 - lr * weight_decay)
    p_f32 = p_f32 - lr * x.float()

    if stochastic_round:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32)


def single_param_sinkgd_specnorm(
    p: Tensor,
    grad: Tensor,
    u: Tensor,
    lr: Tensor,
    weight_decay: float,
    iters: int,
    eps: float,
    target: float,
    sn_iters: int,
    stochastic_round: bool,
) -> None:
    """SinkGD update (Feature A) with an operator-norm rescale of the update.

    After SR-Sinkhorn produces the update ``U``, its spectral norm ``sigma = ||U||_2`` is
    estimated by ``sn_iters`` of power iteration warm-started from ``u`` (the persisted
    right-singular-vector estimate, the only added state, O(n) per matrix), and ``U`` is
    globally rescaled to ``target = sqrt(d_out/d_in)`` operator norm. ``u`` is updated
    in place across steps. Batched over any leading (expert) dims. Compile-clean: the
    power iteration is matmuls + norms, no graph break.

    This is Option 2 of the handoff INTERACTION WARNING — the spectral rescale replaces
    SinkGD's implicit ``sqrt(nm)`` Frobenius magnitude, so the effective step size shifts
    vs the flag-off path (expected; measured/reported in Phase 2, tuned via lr/alpha).
    """
    m, n = grad.shape[-2], grad.shape[-1]
    sqrt_n = n**0.5
    sqrt_m = m**0.5
    x = grad
    for _ in range(max(iters - 1, 0)):
        rn = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
        x = x * (sqrt_n / rn.clamp_min(eps)).to(x.dtype)
        cn = torch.linalg.vector_norm(x, dim=-2, keepdim=True, dtype=torch.float32)
        x = x * (sqrt_m / cn.clamp_min(eps)).to(x.dtype)
    # The FINAL column scale is a diagonal on the right: (X·D)u = X(D⊙u) and vᵀ(X·D) =
    # (vᵀX)⊙D, so it folds into the O(n) power-iteration vectors and the apply — the scaled
    # matrix is never materialized (saves one full read+write per step vs the base loop).
    if iters > 0:
        rn = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
        x = x * (sqrt_n / rn.clamp_min(eps)).to(x.dtype)
        cn = torch.linalg.vector_norm(x, dim=-2, keepdim=True, dtype=torch.float32)
        d = sqrt_m / cn.clamp_min(eps)  # fp32 [*lead, 1, n]
    else:
        d = torch.ones_like(x[..., :1, :], dtype=torch.float32)
    dv = d.squeeze(-2)  # [*lead, n]

    # Power iteration in the update dtype: matmuls hit tensor cores with fp32 accumulation, so
    # the full matrix is never cast to fp32 (only the O(m)/O(n) vectors are). `v @ X` avoids
    # materializing Xᵀ.
    uu = u
    sigma: Tensor
    for _ in range(sn_iters):
        v = torch.matmul(x, (dv * uu).to(x.dtype).unsqueeze(-1)).squeeze(-1).float()
        v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
        uu = torch.matmul(v.to(x.dtype).unsqueeze(-2), x).squeeze(-2).float() * dv
        sigma = torch.linalg.vector_norm(uu, dim=-1, keepdim=True).clamp_min(eps)
        uu = uu / sigma
    u.copy_(uu)
    # per-column scale: operator-norm rescale x held-out final column scale, fp32 [*lead,1,n]
    scale = (target / sigma.reshape(*sigma.shape[:-1], 1, 1)) * d

    p_f32 = p.float()
    if weight_decay != 0.0:
        p_f32 = p_f32 * (1 - lr * weight_decay)
    p_f32 = p_f32 - (lr * scale) * x.float()

    if stochastic_round:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32)


def single_param_sinkgd_md(
    p: Tensor,
    grad: Tensor,
    u: Tensor,
    lr: Tensor,
    iters: int,
    eps: float,
    target_norm: Tensor,
    sn_iters: int,
    stochastic_round: bool,
) -> None:
    """MD-sphere update (the A+B "unit" variant): SR-Sinkhorn -> spectral-normalize the
    direction to unit operator norm scaled to the sphere radius -> step -> project the weight
    back onto the Frobenius sphere ``||W||_F = target_norm``.

    ``target_norm`` is the per-matrix sphere radius (anchored at the enable-time ``||W||_F``,
    so the weight magnitude is preserved). Gains are dropped (Phase 3: SinkGD already balances
    per-row/col, so the MD gains are redundant). Batched over leading (expert) dims; the sphere
    projection and Frobenius are per-matrix.
    """
    m, n = grad.shape[-2], grad.shape[-1]
    sqrt_n = n**0.5
    sqrt_m = m**0.5
    x = grad
    for _ in range(max(iters - 1, 0)):
        rn = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
        x = x * (sqrt_n / rn.clamp_min(eps)).to(x.dtype)
        cn = torch.linalg.vector_norm(x, dim=-2, keepdim=True, dtype=torch.float32)
        x = x * (sqrt_m / cn.clamp_min(eps)).to(x.dtype)
    # final column scale held out as a diagonal (see single_param_sinkgd_specnorm)
    if iters > 0:
        rn = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
        x = x * (sqrt_n / rn.clamp_min(eps)).to(x.dtype)
        cn = torch.linalg.vector_norm(x, dim=-2, keepdim=True, dtype=torch.float32)
        d = sqrt_m / cn.clamp_min(eps)  # fp32 [*lead, 1, n]
    else:
        d = torch.ones_like(x[..., :1, :], dtype=torch.float32)
    dv = d.squeeze(-2)

    # bf16 tensor-core power iteration with the column scale folded into the vectors
    uu = u
    sigma: Tensor
    for _ in range(sn_iters):
        v = torch.matmul(x, (dv * uu).to(x.dtype).unsqueeze(-1)).squeeze(-1).float()
        v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
        uu = torch.matmul(v.to(x.dtype).unsqueeze(-2), x).squeeze(-2).float() * dv
        sigma = torch.linalg.vector_norm(uu, dim=-1, keepdim=True).clamp_min(eps)
        uu = uu / sigma
    u.copy_(uu)
    tn = target_norm.reshape(*target_norm.shape, 1, 1)
    a = lr * tn / sigma.reshape(*sigma.shape[:-1], 1, 1)  # step coefficient [*lead,1,1]

    # Analytic sphere projection: ||W - aU||_F^2 = ||W||^2 - 2a<W,U> + a^2||U||^2, so three
    # fused reductions over (p, x) replace materializing the fp32 intermediate, reducing over
    # it, and re-reading it to rescale — the projection collapses into one fused write pass.
    p_f32 = p.float()
    xd = x.float() * d
    s0 = (p_f32 * p_f32).sum(dim=(-2, -1), keepdim=True)
    s1 = (p_f32 * xd).sum(dim=(-2, -1), keepdim=True)
    s2 = (xd * xd).sum(dim=(-2, -1), keepdim=True)
    fro = (s0 - 2 * a * s1 + a * a * s2).clamp_min(eps).sqrt()
    p_new = (p_f32 - a * xd) * (tn / fro)

    if stochastic_round:
        p.copy_(_fp32_to_bf16_sr(p_new))
    else:
        p.copy_(p_new)


class SinkGD(_AdamBase):
    """SinkGD optimizer with an 8-bit AdamW fallback for non-matrix parameters.

    Parameter groups flagged with ``use_sinkgd=True`` get the stateless SR-Sinkhorn
    update for their >=2D tensors (3D fused-MoE-expert weights are normalized
    per-expert); every other parameter is optimized with the quantized 8-bit AdamW
    step inherited from torchao's ``_AdamBase``.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        *,
        sinkhorn_iters=5,
        sinkgd_lr_scale=0.05,
        sinkgd_eps=1e-8,
        sinkgd_spectral_norm=False,
        sinkgd_spectral_norm_iters=1,
        sinkgd_spectral_target="unit",
        sinkgd_base_width=None,
        sinkgd_lr_width_exponent=1.0,
        sinkgd_fused_kernel=False,
        sinkgd_fused_min_numel=1 << 25,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad=False,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkgd_lr_scale = sinkgd_lr_scale
        self.sinkgd_eps = sinkgd_eps
        self.sinkgd_spectral_norm = sinkgd_spectral_norm
        self.sinkgd_spectral_norm_iters = sinkgd_spectral_norm_iters
        self.sinkgd_spectral_target = sinkgd_spectral_target
        self.sinkgd_base_width = sinkgd_base_width
        self.sinkgd_lr_width_exponent = sinkgd_lr_width_exponent
        self.sinkgd_fused_kernel = sinkgd_fused_kernel
        self.sinkgd_fused_min_numel = sinkgd_fused_min_numel
        self._compiled_sinkgd = torch.compile(
            single_param_sinkgd, fullgraph=True, dynamic=False
        )
        self._compiled_sinkgd_sn = torch.compile(
            single_param_sinkgd_specnorm, fullgraph=True, dynamic=False
        )
        self._compiled_adam = torch.compile(
            single_param_adam, fullgraph=True, dynamic=False
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState8bit.zeros(
            p.shape, signed, block_size, p.device, dtype=p.dtype
        )

    def _specnorm_u(self, p: Tensor, n_local: int, lead) -> Tensor:
        """Persisted right-singular-vector estimate for power iteration (warm-start).

        ``n_local`` is the (local, for the dist path) length of the vector; ``lead`` the
        leading/expert dims. The only optimizer state on a SinkGD-routed matrix.
        """
        state = self.state[p]
        u = state.get("specnorm_u")
        if u is None:
            u = torch.randn(*lead, n_local, device=p.device, dtype=torch.float32)
            u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1e-12)
            state["specnorm_u"] = u
        return u

    def _alpha_eff(self, p: Tensor) -> float:
        """Per-layer update scale. Width-aware when ``sinkgd_base_width`` is set:
        ``alpha_eff = sinkgd_lr_scale * (base_width / d_in) ** exponent`` (the Phase-1
        ``eta ∝ 1/d_in`` rule). Unset -> plain scalar (old behavior, backward compatible).
        ``d_in`` is the logical input dim ``p.shape[-1]`` (the shared input of a fused
        QKV / gate-up matrix), which for a DTensor is already the global dim."""
        if self.sinkgd_base_width is None:
            return self.sinkgd_lr_scale
        d_in = p.shape[-1]
        return (
            self.sinkgd_lr_scale
            * (self.sinkgd_base_width / d_in) ** self.sinkgd_lr_width_exponent
        )

    def _fused_ok(self, p: Tensor, epilogue: bool = False) -> bool:
        # Stochastic rounding is not implemented in the fused kernels -> compiled fallback.
        # The spec/md epilogue adds ~20 small eager ops per param; without a process group
        # to amortize against, small matrices lose to the compiled path (H100: 0.6-0.7x at
        # d=2048) -> gate single-device epilogue modes on tensor size.
        if epilogue and p.numel() < self.sinkgd_fused_min_numel:
            return False
        return (
            self.sinkgd_fused_kernel
            and fused_available()
            and p.is_cuda
            # kernels index p with dense strides and update it in place
            and p.is_contiguous()
            and not (self.bf16_stochastic_round and p.dtype is torch.bfloat16)
        )

    def _sinkgd_update(self, p: Tensor, grad: Tensor, group: dict, lr: Tensor) -> None:
        alpha = self._alpha_eff(p)
        if self._fused_ok(p, epilogue=self.sinkgd_spectral_norm):
            if self.sinkgd_spectral_norm:
                u = self._specnorm_u(p, p.shape[-1], p.shape[:-2])
                fused_sinkgd_step(
                    p.detach(),
                    grad,
                    float(lr) * alpha,
                    group["weight_decay"],
                    self.sinkhorn_iters,
                    self.sinkgd_eps,
                    mode="spec",
                    u=u,
                    spectral_target=self.sinkgd_spectral_target,
                    sn_iters=self.sinkgd_spectral_norm_iters,
                )
            else:
                fused_sinkgd_step(
                    p.detach(),
                    grad,
                    float(lr) * alpha,
                    group["weight_decay"],
                    self.sinkhorn_iters,
                    self.sinkgd_eps,
                    mode="base",
                )
            return
        if not self.sinkgd_spectral_norm:
            self._compiled_sinkgd(
                p.detach(),
                grad,
                lr * alpha,
                group["weight_decay"],
                self.sinkhorn_iters,
                self.sinkgd_eps,
                self.bf16_stochastic_round and p.dtype is torch.bfloat16,
            )
            return
        m, n = p.shape[-2], p.shape[-1]
        # "unit": pin operator norm to a width-independent constant so spectral norm is a
        # pure conditioning stabilizer and 1/d_in owns width (Adam-class). "muon":
        # sqrt(d_out/d_in) makes spectral own width (double-counts with 1/d_in — see Phase 2).
        target = 1.0 if self.sinkgd_spectral_target == "unit" else (m / n) ** 0.5
        u = self._specnorm_u(p, n, p.shape[:-2])
        self._compiled_sinkgd_sn(
            p.detach(),
            grad,
            u,
            lr * alpha,
            group["weight_decay"],
            self.sinkhorn_iters,
            self.sinkgd_eps,
            target,
            self.sinkgd_spectral_norm_iters,
            self.bf16_stochastic_round and p.dtype is torch.bfloat16,
        )

    def _adam_fallback(self, p: Tensor, grad: Tensor, group: dict, lr: Tensor) -> None:
        # Adam is elementwise, so run on the local shard: no cross-rank comm, and it
        # sidesteps the OptimState8bit + DTensor + compile dispatch gap under FSDP2.
        # State is keyed by the original param (to_local returns a fresh object).
        p_local = p.to_local() if isinstance(p, DTensor) else p
        grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
        state = self.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = self._new_buffer(p_local, True)
            state["exp_avg_sq"] = self._new_buffer(p_local, False)
        state["step"] += 1
        self._compiled_adam(
            p_local.detach(),
            grad_local,
            state["step"],
            state["exp_avg"],
            state["exp_avg_sq"],
            None,
            lr,
            group["betas"][0],
            group["betas"][1],
            group["weight_decay"],
            group["eps"],
            self.is_adamw,
            self.bf16_stochastic_round and p_local.dtype is torch.bfloat16,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                use_sinkgd = group.get("use_sinkgd", False)
                # CPU scalar lr avoids recompiling the compiled steps on lr changes
                lr = group["lr"]
                if not isinstance(lr, Tensor):
                    lr = torch.tensor(lr, dtype=torch.float32)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Sparse gradient is not supported")

                    if use_sinkgd and p.ndim >= 2:
                        self._sinkgd_update(p, grad, group, lr)
                    else:
                        self._adam_fallback(p, grad, group, lr)

        return loss


class SinkGDMD(SinkGD):
    """Experimental A+B variant: SinkGD direction on a Frobenius weight-sphere (MD, "unit").

    Each SinkGD-routed 2D weight is kept on a fixed-Frobenius sphere (``||W||_F`` anchored at
    enable time); the SR-Sinkhorn update is spectral-normalized to unit operator norm, applied,
    and the weight is reprojected onto the sphere. No gains (redundant with SinkGD's balancing).
    In width sweeps this bounds the deep-layer activation blow-up most tightly, but its optimal
    LR is width-dependent (``lr_opt ~ d_model**-0.6``) and its optimum is narrow, so it is
    OFF by default and gated behind ``sinkgd_md_sphere: true``. Non-matrix params still use the
    8-bit AdamW fallback.
    """

    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._compiled_md = torch.compile(
            single_param_sinkgd_md, fullgraph=True, dynamic=False
        )

    def _md_target_norm(self, p: Tensor, state: dict) -> Tensor:
        tn = state.get("md_target_norm")
        if tn is None:
            tn = torch.linalg.vector_norm(p.detach().float(), dim=(-2, -1))
            state["md_target_norm"] = tn
        return tn

    def _sinkgd_update(self, p: Tensor, grad: Tensor, group: dict, lr: Tensor) -> None:
        alpha = self._alpha_eff(p)
        state = self.state[p]
        tn = self._md_target_norm(p, state)
        u = self._specnorm_u(p, p.shape[-1], p.shape[:-2])
        if self._fused_ok(p, epilogue=True):
            fused_sinkgd_step(
                p.detach(),
                grad,
                float(lr) * alpha,
                0.0,
                self.sinkhorn_iters,
                self.sinkgd_eps,
                mode="md",
                u=u,
                target_norm=tn,
                sn_iters=self.sinkgd_spectral_norm_iters,
            )
            return
        self._compiled_md(
            p.detach(),
            grad,
            u,
            lr * alpha,
            self.sinkhorn_iters,
            self.sinkgd_eps,
            tn,
            self.sinkgd_spectral_norm_iters,
            self.bf16_stochastic_round and p.dtype is torch.bfloat16,
        )


def _sinkgd_param_groups(opt_model, weight_decay):
    """Split params: 2D/3D weight matrices -> SR-Sinkhorn; everything else -> AdamW.

    Routing is by tensor rank, not module type, so fused MoE experts (transformers
    v5 stores them as 3D ``[num_experts, in, out]`` parameters rather than
    ``nn.Linear`` modules) are picked up and normalized per-expert. The AdamW
    fallback group (embeddings, head, norms, biases) uses no weight decay.
    """
    sinkgd_params = []
    adamw_params = []
    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim < 2
            or name.endswith("modules_to_save.default.weight")
            or any(
                embed in name
                for embed in ["embed_tokens", "lm_head", "wte", "word_embeddings"]
            )
        ):
            adamw_params.append(param)
        else:
            sinkgd_params.append(param)

    param_groups = []
    if sinkgd_params:
        param_groups.append(
            {"params": sinkgd_params, "use_sinkgd": True, "weight_decay": weight_decay}
        )
    if adamw_params:
        param_groups.append(
            {"params": adamw_params, "use_sinkgd": False, "weight_decay": 0.0}
        )
    return param_groups


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _pop_sinkgd_extra_kwargs(optimizer_kwargs: dict) -> dict:
    """Type-cast the optional Feature-A ``optim_args`` (they may arrive as strings from the
    ``key=value`` form) and validate the width-transfer mutual exclusion. Returns kwargs to
    forward to the ``SinkGD`` constructor (only the keys that were provided)."""
    out = dict(optimizer_kwargs)
    if "sinkgd_spectral_norm" in out:
        out["sinkgd_spectral_norm"] = _as_bool(out["sinkgd_spectral_norm"])
    if "sinkgd_spectral_norm_iters" in out:
        out["sinkgd_spectral_norm_iters"] = int(out["sinkgd_spectral_norm_iters"])
    if out.get("sinkgd_spectral_target") not in (None, "unit", "muon"):
        raise ValueError("sinkgd_spectral_target must be 'unit' or 'muon'")
    if out.get("sinkgd_base_width") is not None:
        out["sinkgd_base_width"] = int(out["sinkgd_base_width"])
    if "sinkgd_lr_width_exponent" in out:
        out["sinkgd_lr_width_exponent"] = float(out["sinkgd_lr_width_exponent"])
    if "sinkgd_fused_kernel" in out:
        out["sinkgd_fused_kernel"] = _as_bool(out["sinkgd_fused_kernel"])
    if "sinkgd_fused_min_numel" in out:
        out["sinkgd_fused_min_numel"] = int(out["sinkgd_fused_min_numel"])
    # Mutual exclusion: 1/d_in (base_width) and the Muon spectral target are two width
    # corrections; stacking them double-counts (Phase 2). Let the sphere/spectral own width.
    if (
        out.get("sinkgd_base_width") is not None
        and _as_bool(out.get("sinkgd_spectral_norm", False))
        and out.get("sinkgd_spectral_target", "unit") == "muon"
    ):
        raise ValueError(
            "sinkgd_base_width (1/d_in width scaling) and sinkgd_spectral_target='muon' both "
            "correct for width and double-count; set base_width for the plain/spectral-unit "
            "path, or use spectral_target='muon' with base_width unset."
        )
    return out


class SinkGDOptimizerFactory(BaseOptimizerFactory):
    """Builds a :class:`SinkGD` optimizer, routing weight matrices to SR-Sinkhorn."""

    def __call__(self, opt_model, training_args=None, **optimizer_kwargs) -> "SinkGD":
        lr = optimizer_kwargs.pop("lr")
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)
        betas = optimizer_kwargs.pop("betas", (0.9, 0.999))
        eps = optimizer_kwargs.pop("eps", 1e-8)
        sinkhorn_iters = int(optimizer_kwargs.pop("sinkhorn_iters", 5))
        sinkgd_lr_scale = float(optimizer_kwargs.pop("sinkgd_lr_scale", 0.05))
        optimizer_kwargs.pop(
            "device_mesh", None
        )  # ignored by the single-device variant
        md_sphere = _as_bool(optimizer_kwargs.pop("sinkgd_md_sphere", False))
        cls = SinkGDMD if md_sphere else SinkGD

        return cls(
            _sinkgd_param_groups(opt_model, weight_decay),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sinkhorn_iters=sinkhorn_iters,
            sinkgd_lr_scale=sinkgd_lr_scale,
            **_pop_sinkgd_extra_kwargs(optimizer_kwargs),
        )


def _matrix_shard_dim(p: Tensor):
    """Return the negative matrix dim (-2 or -1) sharded across >1 ranks, else None.

    Leading-dim sharding (e.g. the expert axis of a fused MoE weight) and replicated
    tensors return None — their row/column norms are then fully shard-local.
    """
    if not isinstance(p, DTensor):
        return None
    matrix_dims = {p.ndim - 1, p.ndim - 2}
    for i, placement in enumerate(p.placements):
        if (
            placement.is_shard()
            and placement.dim in matrix_dims
            and p.device_mesh.size(i) > 1
        ):
            return placement.dim - p.ndim
    return None


# Compiled local pieces of one SR-Sinkhorn iteration; the only thing kept eager and
# uncompiled between them is the tiny norm-vector all-reduce (a collective would graph
# break fullgraph anyway). This recovers the fusion the single-device path enjoys.
def _row_scale_col_partial(x, sqrt_n, eps):
    rn = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
    x = x * (sqrt_n / rn.clamp_min(eps)).to(x.dtype)
    return x, (x.float() ** 2).sum(dim=-2, keepdim=True)


def _apply_col_scale(x, csq, sqrt_m, eps):
    return x * (sqrt_m / csq.sqrt().clamp_min(eps)).to(x.dtype)


def _row_partial(x):
    return (x.float() ** 2).sum(dim=-1, keepdim=True)


def _row_scale_then_col_scale(x, rsq, sqrt_n, sqrt_m, eps):
    # row norm is the reduced rsq (cols sharded); column norm is then shard-local.
    x = x * (sqrt_n / rsq.sqrt().clamp_min(eps)).to(x.dtype)
    cn = torch.linalg.vector_norm(x, dim=-2, keepdim=True, dtype=torch.float32)
    return x * (sqrt_m / cn.clamp_min(eps)).to(x.dtype)


def _sinkgd_apply_update(p, x, lr, weight_decay, stochastic_round):
    p_f32 = p.float()
    if weight_decay != 0.0:
        p_f32 = p_f32 * (1 - lr * weight_decay)
    p_f32 = p_f32 - lr * x.float()
    if stochastic_round:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32)


# One power-iteration step on the Gram matrix, returning the LOCAL partial of the full
# product (summed across the shard group by an eager all-reduce). Iterating the Gram matrix
# (rather than U, Uᵀ separately) needs only a single vector all-reduce per step, and the
# implicit v-normalization cancels. Batched over leading (expert) dims.
def _sinkgd_apply_update_scaled(p, x, scale, lr, weight_decay, stochastic_round):
    # apply with a per-matrix fp32 scale folded into the update multiply (the spectral
    # rescale is never materialized as a separate full-matrix pass).
    p_f32 = p.float()
    if weight_decay != 0.0:
        p_f32 = p_f32 * (1 - lr * weight_decay)
    p_f32 = p_f32 - (lr * scale) * x.float()
    if stochastic_round:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32)


def _md_local_partials(p_local, x):
    # local partial sums for the analytic sphere norm: ||W||^2, <W,U>, ||U||^2 per matrix,
    # packed into one tensor so a single all-reduce covers all three.
    pf = p_local.float()
    xf = x.float()
    s0 = (pf * pf).sum(dim=(-2, -1))
    s1 = (pf * xf).sum(dim=(-2, -1))
    s2 = (xf * xf).sum(dim=(-2, -1))
    return torch.stack([s0, s1, s2])


def _md_apply_analytic(p_local, x, sums, a, tn, eps, stochastic_round):
    # analytic projection (see single_param_sinkgd_md): fro from the reduced partials, then
    # one fused step + rescale pass; no fp32 intermediate.
    shape = (*sums.shape[1:], 1, 1)
    s0 = sums[0].reshape(shape)
    s1 = sums[1].reshape(shape)
    s2 = sums[2].reshape(shape)
    fro = (s0 - 2 * a * s1 + a * a * s2).clamp_min(eps).sqrt()
    tn_b = tn.reshape(*tn.shape, 1, 1)
    pf = (p_local.float() - a * x.float()) * (tn_b / fro)
    if stochastic_round:
        p_local.copy_(_fp32_to_bf16_sr(pf))
    else:
        p_local.copy_(pf)


def _specnorm_gram_rows(x, u):
    # rows sharded: local part of (Xᵀ X) u = X_localᵀ (X_local u). matmuls in x's dtype
    # (tensor cores, fp32 accumulation), transpose-free via w @ X; fp32 partial for the reduce.
    w = torch.matmul(x, u.unsqueeze(-1)).squeeze(-1)
    return torch.matmul(w.unsqueeze(-2), x).squeeze(-2).float()


def _specnorm_gram_cols(x, v):
    # cols sharded: local part of (X Xᵀ) v = X_local (X_localᵀ v). x[...,M,N_local], v[...,M].
    t = torch.matmul(v.unsqueeze(-2), x).squeeze(-2)
    return torch.matmul(x, t.unsqueeze(-1)).squeeze(-1).float()


class DistSinkGD(SinkGD):
    """Distributed SinkGD for FSDP2/TP-sharded weights.

    The SR-Sinkhorn matrices are updated on their local shards, all-reducing only the
    ``[N]`` (or ``[M]``) norm vector over the shard process group — never the matrix
    itself, so optimizer communication is orders of magnitude lighter than gathering
    the full parameter. Local work is torch.compiled (the all-reduce is the only eager
    break); the 8-bit AdamW fallback is inherited unchanged.
    """

    def __init__(self, params, *, process_group=None, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._process_group = process_group
        c = lambda f: torch.compile(f, fullgraph=True, dynamic=False)  # noqa: E731
        self._c_row_scale_col_partial = c(_row_scale_col_partial)
        self._c_apply_col_scale = c(_apply_col_scale)
        self._c_row_partial = c(_row_partial)
        self._c_row_scale_then_col_scale = c(_row_scale_then_col_scale)
        self._c_apply_update = c(_sinkgd_apply_update)
        self._c_specnorm_gram_rows = c(_specnorm_gram_rows)
        self._c_specnorm_gram_cols = c(_specnorm_gram_cols)
        self._c_apply_update_scaled = c(_sinkgd_apply_update_scaled)

    def _dist_specnorm_vec(self, p, device, lead, vec_len) -> Tensor:
        """Replicated warm-start vector for the sharded power iteration, living on the
        UNSHARDED matrix axis (length ``vec_len``). Deterministic init (fixed-seed philox)
        so every rank starts from the same vector without a broadcast."""
        st = self.state[p]
        u = st.get("specnorm_u")
        if u is None or u.shape[-1] != vec_len:
            g = torch.Generator(device=device).manual_seed(0)
            u = torch.randn(
                *lead, vec_len, generator=g, device=device, dtype=torch.float32
            )
            u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1e-12)
            st["specnorm_u"] = u
        return u

    def _dist_spectral_scale(
        self, x, p, device, shard_dim, global_M, global_N, op_target=None
    ):
        """Estimate ``||U||_2`` on the matrix-dim-sharded update via power iteration on the
        Gram matrix — one ``[N]`` (rows sharded) or ``[M]`` (cols sharded) vector all-reduce
        per iter, the same shape and ``dp_shard`` group as the Sinkhorn norm reduce, matrix
        never gathered — and return the fp32 ``[*lead,1,1]`` rescale factor (folded into the
        apply by the caller; the rescaled matrix is never materialized).

        ``op_target`` overrides the target with a per-matrix tensor (the MD sphere radius);
        otherwise the class ``unit``/``muon`` scalar is used."""
        lead = x.shape[:-2]
        eps = self.sinkgd_eps
        vec_len = global_N if shard_dim == -2 else global_M
        u = self._dist_specnorm_vec(p, device, lead, vec_len)
        nrm: Tensor
        for _ in range(self.sinkgd_spectral_norm_iters):
            ud = u.to(x.dtype)
            if shard_dim == -2:
                up = self._c_specnorm_gram_rows(x, ud)
            else:
                up = self._c_specnorm_gram_cols(x, ud)
            dist.all_reduce(up, group=self._process_group)
            nrm = torch.linalg.vector_norm(up, dim=-1, keepdim=True).clamp_min(eps)
            u = up / nrm
        self.state[p]["specnorm_u"] = u
        sigma = nrm.squeeze(-1).sqrt()  # nrm = ||(XᵀX)u|| -> sigma^2 at convergence
        sig_b = sigma.reshape(*sigma.shape, 1, 1)
        if op_target is not None:
            target = op_target.reshape(*op_target.shape, 1, 1)
        else:
            target = (
                1.0
                if self.sinkgd_spectral_target == "unit"
                else (global_M / global_N) ** 0.5
            )
        return target / sig_b

    def _dist_sinkgd_step(self, p, grad, group, lr):
        shard_dim = _matrix_shard_dim(p)
        sr = self.bf16_stochastic_round and p.dtype is torch.bfloat16
        global_M, global_N = p.shape[-2], p.shape[-1]  # DTensor -> global dims
        p_local = p.to_local() if isinstance(p, DTensor) else p
        x = grad.to_local() if isinstance(grad, DTensor) else grad

        # fused Triton path: replicated/expert-sharded runs locally; rows-sharded (the
        # FSDP2 dim-0 layout) all-reduces the same vectors as the compiled path. A
        # cols-sharded matrix dim falls through to the compiled pipeline below.
        if self._fused_ok(
            p_local, epilogue=self.sinkgd_spectral_norm and shard_dim is None
        ) and shard_dim in (None, -2):
            lr_f = float(lr) * self._alpha_eff(p)
            grp = self._process_group if shard_dim == -2 else None
            m_glob = global_M if shard_dim == -2 else None
            if self.sinkgd_spectral_norm:
                if shard_dim == -2:
                    u = self._dist_specnorm_vec(
                        p, p_local.device, p_local.shape[:-2], global_N
                    )
                else:
                    u = self._specnorm_u(p, global_N, p_local.shape[:-2])
                fused_sinkgd_step(
                    p_local,
                    x,
                    lr_f,
                    group["weight_decay"],
                    self.sinkhorn_iters,
                    self.sinkgd_eps,
                    mode="spec",
                    u=u,
                    spectral_target=self.sinkgd_spectral_target,
                    sn_iters=self.sinkgd_spectral_norm_iters,
                    m_global=m_glob,
                    process_group=grp,
                )
            else:
                fused_sinkgd_step(
                    p_local,
                    x,
                    lr_f,
                    group["weight_decay"],
                    self.sinkhorn_iters,
                    self.sinkgd_eps,
                    mode="base",
                    m_global=m_glob,
                    process_group=grp,
                )
            return
        lr = lr * self._alpha_eff(p)  # width-aware; p.shape[-1] is the global d_in

        if shard_dim is None:
            # matrix dims fully local (replicated / expert-sharded) -> reuse the
            # fully-compiled single-device path (incl. spectral norm), no communication.
            if self.sinkgd_spectral_norm:
                m, n = global_M, global_N
                target = (
                    1.0 if self.sinkgd_spectral_target == "unit" else (m / n) ** 0.5
                )
                u = self._specnorm_u(p, n, p_local.shape[:-2])
                self._compiled_sinkgd_sn(
                    p_local,
                    x,
                    u,
                    lr,
                    group["weight_decay"],
                    self.sinkhorn_iters,
                    self.sinkgd_eps,
                    target,
                    self.sinkgd_spectral_norm_iters,
                    sr,
                )
            else:
                self._compiled_sinkgd(
                    p_local,
                    x,
                    lr,
                    group["weight_decay"],
                    self.sinkhorn_iters,
                    self.sinkgd_eps,
                    sr,
                )
            return

        sqrt_n, sqrt_m = global_N**0.5, global_M**0.5
        eps = self.sinkgd_eps
        for _ in range(self.sinkhorn_iters):
            if shard_dim == -2:  # rows sharded -> column norm needs a reduction
                x, csq = self._c_row_scale_col_partial(x, sqrt_n, eps)
                dist.all_reduce(csq, group=self._process_group)
                x = self._c_apply_col_scale(x, csq, sqrt_m, eps)
            else:  # columns sharded (-1) -> row norm needs a reduction (row first)
                rsq = self._c_row_partial(x)
                dist.all_reduce(rsq, group=self._process_group)
                x = self._c_row_scale_then_col_scale(x, rsq, sqrt_n, sqrt_m, eps)
        if self.sinkgd_spectral_norm:
            # one extra small vector all-reduce (size d_in/d_out) per power-iteration step,
            # mirroring the Sinkhorn norm reduce above; the matrix is never gathered and the
            # rescale folds into the apply.
            scale = self._dist_spectral_scale(
                x, p, p_local.device, shard_dim, global_M, global_N
            )
            self._c_apply_update_scaled(
                p_local, x, scale, lr, group["weight_decay"], sr
            )
        else:
            self._c_apply_update(p_local, x, lr, group["weight_decay"], sr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                use_sinkgd = group.get("use_sinkgd", False)
                lr = group["lr"]
                if not isinstance(lr, Tensor):
                    lr = torch.tensor(lr, dtype=torch.float32)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("Sparse gradient is not supported")
                    if use_sinkgd and p.ndim >= 2:
                        self._dist_sinkgd_step(p, p.grad, group, lr)
                    else:
                        self._adam_fallback(p, p.grad, group, lr)

        return loss


class DistSinkGDMD(DistSinkGD):
    """Distributed :class:`SinkGDMD` (A+B "unit") for FSDP2/TP-sharded weights.

    On a matrix-dim-sharded weight the added communication over the ``dp_shard`` group is: the
    sharded power iteration's vector all-reduce (spectral norm) plus one per-matrix scalar
    all-reduce for the global Frobenius norm (sphere projection). The matrix is never gathered;
    the sphere radius ``target_norm`` and the power-iteration vector round-trip through the
    checkpoint. Replicated / expert-sharded weights reuse the fully-local single-device MD path.
    """

    def __init__(self, params, *, process_group=None, **kwargs) -> None:
        super().__init__(params, process_group=process_group, **kwargs)
        c = lambda f: torch.compile(f, fullgraph=True, dynamic=False)  # noqa: E731
        self._compiled_md = c(single_param_sinkgd_md)
        self._c_md_local_partials = c(_md_local_partials)
        self._c_md_apply_analytic = c(_md_apply_analytic)

    def _dist_md_target(self, p, p_local, shard_dim) -> Tensor:
        state = self.state[p]
        tn = state.get("md_target_norm")
        if tn is None:
            sq = (p_local.float() ** 2).sum(dim=(-2, -1))  # [*lead] local partial
            if shard_dim is not None:
                dist.all_reduce(sq, group=self._process_group)
            tn = sq.sqrt()
            state["md_target_norm"] = tn
        return tn

    def _dist_sinkgd_step(self, p, grad, group, lr):
        shard_dim = _matrix_shard_dim(p)
        sr = self.bf16_stochastic_round and p.dtype is torch.bfloat16
        global_M, global_N = p.shape[-2], p.shape[-1]
        p_local = p.to_local() if isinstance(p, DTensor) else p
        x = grad.to_local() if isinstance(grad, DTensor) else grad
        tn = self._dist_md_target(p, p_local, shard_dim)

        if self._fused_ok(p_local, epilogue=shard_dim is None) and shard_dim in (
            None,
            -2,
        ):
            if shard_dim == -2:
                u = self._dist_specnorm_vec(
                    p, p_local.device, p_local.shape[:-2], global_N
                )
            else:
                u = self._specnorm_u(p, global_N, p_local.shape[:-2])
            fused_sinkgd_step(
                p_local,
                x,
                float(lr) * self._alpha_eff(p),
                0.0,
                self.sinkhorn_iters,
                self.sinkgd_eps,
                mode="md",
                u=u,
                target_norm=tn,
                sn_iters=self.sinkgd_spectral_norm_iters,
                m_global=global_M if shard_dim == -2 else None,
                process_group=self._process_group if shard_dim == -2 else None,
            )
            return
        lr = lr * self._alpha_eff(p)

        if shard_dim is None:
            # full matrix local -> reuse the fully-compiled single-device MD step.
            u = self._specnorm_u(p, global_N, p_local.shape[:-2])
            self._compiled_md(
                p_local,
                x,
                u,
                lr,
                self.sinkhorn_iters,
                self.sinkgd_eps,
                tn,
                self.sinkgd_spectral_norm_iters,
                sr,
            )
            return

        sqrt_n, sqrt_m = global_N**0.5, global_M**0.5
        eps = self.sinkgd_eps
        for _ in range(self.sinkhorn_iters):
            if shard_dim == -2:
                x, csq = self._c_row_scale_col_partial(x, sqrt_n, eps)
                dist.all_reduce(csq, group=self._process_group)
                x = self._c_apply_col_scale(x, csq, sqrt_m, eps)
            else:
                rsq = self._c_row_partial(x)
                dist.all_reduce(rsq, group=self._process_group)
                x = self._c_row_scale_then_col_scale(x, rsq, sqrt_n, sqrt_m, eps)
        # spectral scale for operator norm == sphere radius tn (folded into the apply)
        scale = self._dist_spectral_scale(
            x, p, p_local.device, shard_dim, global_M, global_N, op_target=tn
        )
        a = lr.to(scale.device) * scale  # step coefficient [*lead,1,1]
        # analytic sphere projection: one packed all-reduce of the three Frobenius partials
        # replaces materializing the fp32 W' shard (see single_param_sinkgd_md).
        sums = self._c_md_local_partials(p_local, x)
        dist.all_reduce(sums, group=self._process_group)
        self._c_md_apply_analytic(p_local, x, sums, a, tn, eps, sr)


class DistSinkGDOptimizerFactory(BaseOptimizerFactory):
    """Builds a :class:`DistSinkGD` for sharded training; pulls the dp_shard group."""

    def __call__(
        self, opt_model, training_args=None, **optimizer_kwargs
    ) -> "DistSinkGD":
        lr = optimizer_kwargs.pop("lr")
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)
        betas = optimizer_kwargs.pop("betas", (0.9, 0.999))
        eps = optimizer_kwargs.pop("eps", 1e-8)
        sinkhorn_iters = int(optimizer_kwargs.pop("sinkhorn_iters", 5))
        sinkgd_lr_scale = float(optimizer_kwargs.pop("sinkgd_lr_scale", 0.05))

        device_mesh = optimizer_kwargs.pop("device_mesh", None)
        process_group = None
        if isinstance(device_mesh, DeviceMesh):
            if "dp_shard" in (device_mesh.mesh_dim_names or ()):
                process_group = device_mesh["dp_shard"].get_group()
            elif device_mesh.ndim == 1:
                process_group = device_mesh.get_group()

        md_sphere = _as_bool(optimizer_kwargs.pop("sinkgd_md_sphere", False))
        cls = DistSinkGDMD if md_sphere else DistSinkGD

        return cls(
            _sinkgd_param_groups(opt_model, weight_decay),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sinkhorn_iters=sinkhorn_iters,
            sinkgd_lr_scale=sinkgd_lr_scale,
            process_group=process_group,
            **_pop_sinkgd_extra_kwargs(optimizer_kwargs),
        )
