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
        self._compiled_sinkgd = torch.compile(
            single_param_sinkgd, fullgraph=True, dynamic=False
        )
        self._compiled_adam = torch.compile(
            single_param_adam, fullgraph=True, dynamic=False
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState8bit.zeros(
            p.shape, signed, block_size, p.device, dtype=p.dtype
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
                        self._compiled_sinkgd(
                            p.detach(),
                            grad,
                            lr * self.sinkgd_lr_scale,
                            group["weight_decay"],
                            self.sinkhorn_iters,
                            self.sinkgd_eps,
                            self.bf16_stochastic_round and p.dtype is torch.bfloat16,
                        )
                    else:
                        self._adam_fallback(p, grad, group, lr)

        return loss


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

        return SinkGD(
            _sinkgd_param_groups(opt_model, weight_decay),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sinkhorn_iters=sinkhorn_iters,
            sinkgd_lr_scale=sinkgd_lr_scale,
            **optimizer_kwargs,
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

    def _dist_sinkgd_step(self, p, grad, group, lr):
        shard_dim = _matrix_shard_dim(p)
        sr = self.bf16_stochastic_round and p.dtype is torch.bfloat16
        global_M, global_N = p.shape[-2], p.shape[-1]  # DTensor -> global dims
        p_local = p.to_local() if isinstance(p, DTensor) else p
        x = grad.to_local() if isinstance(grad, DTensor) else grad
        lr = lr * self.sinkgd_lr_scale

        if shard_dim is None:
            # matrix dims fully local (replicated / expert-sharded) -> reuse the
            # fully-compiled single-device path, no communication.
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

        return DistSinkGD(
            _sinkgd_param_groups(opt_model, weight_decay),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sinkhorn_iters=sinkhorn_iters,
            sinkgd_lr_scale=sinkgd_lr_scale,
            process_group=process_group,
            **optimizer_kwargs,
        )
