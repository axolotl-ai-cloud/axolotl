from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

try:
    import triton
    import triton.language as tl
except ImportError as e:
    print('triton is not installed, please install by running `pip install triton -U --pre`')
    exit()


def exists(val):
    return val is not None


# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


def clone_inplace_updated_params(nargs):
    nargs['p_ptr'] = nargs['p_ptr'].clone()
    nargs['exp_avg_ptr'] = nargs['exp_avg_ptr'].clone()


# triton cuda kernel

@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE': 128}, num_warps=4, pre_hook=clone_inplace_updated_params),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, pre_hook=clone_inplace_updated_params),
], key=['n_elements'])
@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # offsetted pointers

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # load

    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # stepweight decay

    p = p * (1 - lr * wd)

    # diff between momentum running average and grad

    diff = exp_avg - grad

    # weight update

    update = diff * beta1 + grad

    # torch.sign

    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    p = p + update_sign * can_update

    # decay the momentum running average coefficient

    exp_avg = diff * beta2 + grad

    # store new params and momentum running average coefficient

    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)


def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float
):
    assert all([t.is_cuda for t in (p, grad, exp_avg)])
    n_elements = p.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    update_fn_kernel[grid](
        p,
        grad,
        exp_avg,
        lr,
        wd,
        beta1,
        beta2,
        n_elements
    )


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], \
                self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss
