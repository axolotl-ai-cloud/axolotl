from typing import Optional

import torch
from torch import Tensor
from torch.distributed._tensor import DTensor
from torch.optim import Optimizer
from torchao.prototype.low_bit_optim.subclass_4bit import OptimState4bit
from torchao.prototype.low_bit_optim.subclass_8bit import OptimState8bit
from torchao.prototype.low_bit_optim.subclass_fp8 import OptimStateFp8


class _ShampooBase(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-1,
        momentum=0.0,
        weight_decay=0.0,
        eps=1e-4,
        update_freq=1,
        *,
        block_size,
        quantization_bits,
        optimizer_state_class,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if update_freq < 1:
            raise ValueError(f"Invalid update_freq value: {update_freq}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            update_freq=update_freq,
        )
        super().__init__(params, defaults)
        self.block_size = block_size
        self.quantization_bits = quantization_bits
        self.optimizer_state_class = optimizer_state_class

    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = self._new_buffer(grad, True)
                    state["preconds"] = []
                    state["inv_preconds"] = []
                    for dim in grad.size():
                        state["preconds"].append(
                            self.optimizer_state_class.zeros(
                                (dim, dim),
                                signed=False,
                                block_size=self.block_size,
                                device=grad.device,
                            )
                        )
                        state["inv_preconds"].append(
                            torch.zeros((dim, dim), device=grad.device)
                        )

                state["step"] += 1
                beta = group["momentum"]
                weight_decay = group["weight_decay"]
                lr = group["lr"]
                eps = group["eps"]
                update_freq = group["update_freq"]

                # Apply momentum
                if beta > 0:
                    state["momentum_buffer"].mul_(beta).add_(grad, alpha=1 - beta)
                    grad = state["momentum_buffer"]

                # Apply weight decay
                if weight_decay > 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Preconditioning
                order = grad.ndimension()
                original_size = grad.size()
                for dim_id, dim in enumerate(grad.size()):
                    precond = state["preconds"][dim_id]
                    inv_precond = state["inv_preconds"][dim_id]

                    # Reshape grad
                    grad = grad.transpose(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()

                    # Update preconditioner
                    precond_fp32 = precond.dequantize()
                    precond_update = grad @ grad_t
                    precond_fp32.add_(precond_update)

                    # Quantize preconditioner back
                    precond.copy_(precond_fp32)

                    # Update inverse preconditioner
                    if state["step"] % update_freq == 0:
                        inv_precond.copy_(
                            self._compute_inv_precond(precond_fp32, eps, order)
                        )

                    # Precondition grad
                    if dim_id == order - 1:
                        # Last dimension
                        grad = grad_t @ inv_precond
                        grad = grad.view(original_size)
                    else:
                        grad = inv_precond @ grad
                        grad = grad.view(transposed_size)

                # Update parameter
                p.data.add_(grad, alpha=-lr)

        return loss

    def _compute_inv_precond(self, precond: Tensor, eps: float, order: int):
        # Add eps for numerical stability
        precond = precond + torch.eye(precond.size(0), device=precond.device) * eps

        # Compute matrix power
        inv_precond = self._matrix_power(precond, -1.0 / (2 * order))

        return inv_precond

    def _matrix_power(self, matrix: Tensor, power: float) -> Tensor:
        # Compute matrix power using SVD
        u, s, v = torch.svd(matrix)
        s_pow = s.pow(power)
        return u @ torch.diag(s_pow) @ v.t()

    # bring your own function to create zero-filled subclass
    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        raise NotImplementedError

    # follow bitsandbytes, only quantize tensors >= 4096 values
    # also wrap subclass in DTensor when needed
    def _new_buffer(self, p: Tensor, signed: bool):
        if p.numel() >= 4096 and p.numel() % self.block_size == 0:
            if isinstance(p, DTensor):
                out = DTensor.from_local(
                    local_tensor=self._subclass_zeros(
                        p.to_local(), signed, self.block_size
                    ),
                    device_mesh=p.device_mesh,
                    placements=p.placements,
                    run_check=False,
                )
            else:
                out = self._subclass_zeros(p, signed, self.block_size)
        else:
            out = torch.zeros_like(p)
        return out


class Shampoo8bit(_ShampooBase):
    def __init__(
        self,
        params,
        lr=1e-1,
        momentum=0.0,
        weight_decay=0.0,
        eps=1e-4,
        update_freq=1,
        *,
        block_size=256,
    ):
        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            eps,
            update_freq,
            block_size=block_size,
            quantization_bits=8,
            optimizer_state_class=OptimState8bit,
        )


class Shampoo4bit(_ShampooBase):
    def __init__(
        self,
        params,
        lr=1e-1,
        momentum=0.0,
        weight_decay=0.0,
        eps=1e-4,
        update_freq=1,
        *,
        block_size=128,
    ):
        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            eps,
            update_freq,
            block_size=block_size,
            quantization_bits=4,
            optimizer_state_class=OptimState4bit,
        )


class ShampooFp8(_ShampooBase):
    def __init__(
        self,
        params,
        lr=1e-1,
        momentum=0.0,
        weight_decay=0.0,
        eps=1e-4,
        update_freq=1,
        *,
        block_size=256,
    ):
        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            eps,
            update_freq,
            block_size=block_size,
            quantization_bits=8,  # FP8 uses 8 bits
            optimizer_state_class=OptimStateFp8,
        )
