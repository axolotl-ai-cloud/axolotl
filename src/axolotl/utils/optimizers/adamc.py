"""
AdamC: Adam with Corrected Weight Decay

Aaron Defazio (2025), "Why Gradients Rapidly Increase Near the End of Training"
https://arxiv.org/abs/2506.02285
"""

import math
from collections.abc import Callable, Iterable
from typing import cast

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["AdamC"]


class AdamC(Optimizer):
    """
    AdamW with the weight decay corrected for the learning rate schedule.

    The decay applied at step t is ``(lr_t^2 / max_lr) * weight_decay`` instead of
    AdamW's ``lr_t * weight_decay``, which keeps the steady-state
    gradient-to-weight-norm ratio independent of the schedule.

    ``max_lr`` defaults to each group's initial ``lr``, the peak for warmup/decay
    schedules.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        max_lr: float | None = None,
        foreach: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if max_lr is not None and not 0.0 < max_lr:
            raise ValueError(f"Invalid max_lr value: {max_lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_lr": max_lr,
            "foreach": foreach,
        }
        super().__init__(params, defaults)

        # captured before any scheduler touches `lr`, so this is the peak of the schedule
        for group in self.param_groups:
            if group["max_lr"] is None:
                group["max_lr"] = group["lr"]

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", True)
            group.setdefault("max_lr", group["lr"])

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamC does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                params.append(param)
                grads.append(param.grad)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

            if not params:
                continue

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            max_lr = group["max_lr"] or lr
            decay = group["weight_decay"] * lr * lr / max_lr if max_lr else 0.0

            impl = _multi_tensor_adamc if group["foreach"] else _single_tensor_adamc
            impl(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                eps=group["eps"],
                decay=decay,
            )

        return loss


def _single_tensor_adamc(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
    decay: float,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        state_steps[i] += 1
        step = state_steps[i].item()

        if decay:
            param.mul_(1 - decay)

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2_sqrt = math.sqrt(1 - beta2**step)

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


def _multi_tensor_adamc(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
    decay: float,
):
    tensor_lists: list[list[Tensor | None]] = [
        cast(list[Tensor | None], tensors)
        for tensors in (params, grads, exp_avgs, exp_avg_sqs, state_steps)
    ]
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(tensor_lists)
    for grouped, _ in grouped_tensors.values():
        (
            dev_params,
            dev_grads,
            dev_exp_avgs,
            dev_exp_avg_sqs,
            dev_state_steps,
        ) = cast(list[list[Tensor]], grouped)
        torch._foreach_add_(dev_state_steps, 1)
        steps = [step.item() for step in dev_state_steps]

        if decay:
            torch._foreach_mul_(dev_params, 1 - decay)

        torch._foreach_lerp_(dev_exp_avgs, dev_grads, 1 - beta1)
        torch._foreach_mul_(dev_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(dev_exp_avg_sqs, dev_grads, dev_grads, value=1 - beta2)

        denom = torch._foreach_sqrt(dev_exp_avg_sqs)
        torch._foreach_div_(denom, [math.sqrt(1 - beta2**step) for step in steps])
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(
            dev_params,
            dev_exp_avgs,
            denom,
            [-lr / (1 - beta1**step) for step in steps],
        )
