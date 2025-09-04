"""
Copied from https://github.com/iShohei220/adopt

ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate (2024)
Taniguchi, Shohei and Harada, Keno and Minegishi, Gouki and Oshima, Yuta and Jeong, Seong Cheol and Nagahara, Go and Iiyama, Tomoshi and Suzuki, Masahiro and Iwasawa, Yusuke and Matsuo, Yutaka
"""

# mypy: ignore-errors
# pylint: skip-file
# flake8: noqa
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import Callable, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor
from torch.optim.optimizer import (  # DeviceDict,; _capturable_doc,; _differentiable_doc,; _foreach_doc,; _fused_doc,; _maximize_doc,; _stack_if_compiling,
    DeviceDict,
    Optimizer,
    ParamsT,
    _capturable_doc,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _stack_if_compiling,
    _use_grad_for_differentiable,
    _view_as_real,
)

__all__ = ["ADOPT", "adopt"]


class ADOPT(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        clip_lambda: Optional[Callable[[int], float]] = lambda step: step**0.25,
        weight_decay: float = 0.0,
        decouple: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
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

        self.clip_lambda = clip_lambda

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            # TODO: support fused
            raise RuntimeError("`fused` is not currently supported")

            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Support AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    if group["fused"]:
                        _device_dtype_check_for_fused(p)
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                # Foreach without capturable does not support a tensor lr
                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            adopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                clip_lambda=self.clip_lambda,
                weight_decay=group["weight_decay"],
                decouple=group["decouple"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def _single_tensor_adopt(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[int], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        step = step_t if capturable or differentiable else _get_value(step_t)

        if weight_decay != 0 and not decouple:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            if exp_avg is not None:
                exp_avg = torch.view_as_real(exp_avg)
            if exp_avg_sq is not None:
                exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        if step == 0:
            exp_avg_sq.addcmul_(grad, grad.conj())
            # update step
            step_t += 1
            continue

        if weight_decay != 0 and decouple:
            param.add_(param, alpha=-lr * weight_decay)

        denom = torch.clamp(exp_avg_sq.sqrt(), eps)
        normed_grad = grad.div(denom)
        if clip_lambda is not None:
            clip = clip_lambda(step)
            normed_grad.clamp_(-clip, clip)

        exp_avg.lerp_(normed_grad, 1 - beta1)

        param.add_(exp_avg, alpha=-lr)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        # update step
        step_t += 1


def _multi_tensor_adopt(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[int], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        # Handle complex parameters
        if has_complex:
            _view_as_real(
                device_params, device_grads, device_exp_avgs, device_exp_avg_sqs
            )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0 and not decouple:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        if device_state_steps[0] == 0:
            torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads)

            # Update steps
            # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
            # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
            # wrapped it once now. The alpha is required to assure we go to the right overload.
            if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:
                torch._foreach_add_(
                    device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
                )
            else:
                torch._foreach_add_(device_state_steps, 1)

            continue

        if weight_decay != 0 and decouple:
            torch._foreach_add_(device_params, device_params, alpha=-lr * weight_decay)

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        torch._foreach_maximum_(exp_avg_sq_sqrt, eps)

        normed_grad = torch._foreach_div(device_grads, exp_avg_sq_sqrt)
        if clip_lambda is not None:
            clip = clip_lambda(device_state_steps[0])
            torch._foreach_maximum_(normed_grad, -clip)
            torch._foreach_minimum_(normed_grad, clip)

        torch._foreach_lerp_(device_exp_avgs, normed_grad, 1 - beta1)

        torch._foreach_add_(device_params, device_exp_avgs, alpha=-lr)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, value=1 - beta2
        )

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adopt)
def adopt(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[int], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs ADOPT algorithm computation."""
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    # if fused and not torch.jit.is_scripting():
    #     func = _fused_adopt
    # elif foreach and not torch.jit.is_scripting():
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adopt
    else:
        func = _single_tensor_adopt

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        clip_lambda=clip_lambda,
        weight_decay=weight_decay,
        decouple=decouple,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
