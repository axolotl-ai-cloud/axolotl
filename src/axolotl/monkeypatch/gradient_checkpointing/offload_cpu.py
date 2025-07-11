"""CPU offloaded checkpointing"""

# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

import contextlib
import inspect

import torch
from packaging import version
from torch.utils.checkpoint import (
    _get_autocast_kwargs,
    _get_device_module,
    _infer_device_type,
    check_backward_validity,
    detach_variable,
    get_device_states,
    set_device_states,
)

# support different pytorch versions
has_device_type = "device_type" in inspect.signature(set_device_states).parameters

torch_version = version.parse(torch.__version__)

if torch_version < version.parse("2.4.0"):
    torch_cuda_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_cuda_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_cuda_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_cuda_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")


class CPU_Offloaded_Gradient_Checkpointer(  # pylint: disable=invalid-name
    torch.autograd.Function
):
    """
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """

    @staticmethod
    @torch_cuda_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    @staticmethod
    @torch_cuda_amp_custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
        hidden_states.requires_grad = True
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (
            None,
            hidden_states.grad,
        ) + (
            None,
        ) * len(ctx.args)


# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
# https://github.com/snowflakedb/ArcticTraining/blob/main/arctic_training/monkey_patches.py
class CheckpointFunctionWithCPUOffload(torch.autograd.Function):
    """
    This is a torch/utils/checkpoint.py CheckpointFunction monkey patch that offloads the first tensor to cpu during forward and back to cuda during backward. This allows significant memory savings when using a very long seqlen. e.g. for llama 8b at 100k it's 24GB saved per gpu: `((100_000*4096)*2*32/2**30)`
    In the case of a very long seqlen 100k+ the copying to/from cpu overhead is not big, because dense quadratic attention compute will dominate.
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        # x = None
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                # cpu-offload
                # we don't want the 2nd tensor - usually it's a shared 4D attn mask which is huge [seq,seq]
                # upstream could accept a list of arg indices to offload
                if i == 0:
                    # print(f"{arg.shape=}")
                    ctx.x_device = arg.device
                    ctx.x_requires_grad = arg.requires_grad
                    t = arg.detach().cpu()
                else:
                    t = arg
                tensor_inputs.append(t)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if (
            not torch.autograd._is_checkpoint_valid()  # pylint: disable=protected-access
        ):
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            if i == 0:
                t = (
                    tensors[i]
                    .to(ctx.x_device)
                    .detach()
                    .requires_grad_(ctx.x_requires_grad)
                )
            else:
                t = tensors[i]
            inputs[idx] = t

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices,
            enabled=ctx.preserve_rng_state,
            device_type=ctx.device_type,
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    if has_device_type:
                        # newer pytorch (as early as 2.7)
                        set_device_states(
                            ctx.fwd_devices,
                            ctx.fwd_device_states,
                            device_type=ctx.device_type,
                        )
                    else:
                        # older pytorch (at least 2.4)
                        set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            detached_inputs = detach_variable(tuple(inputs))

            device_autocast_ctx = (
                torch.amp.autocast(
                    device_type=ctx.device_type, **ctx.device_autocast_kwargs
                )
                if torch.amp.is_autocast_available(ctx.device_type)
                else contextlib.nullcontext()
            )
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):  # pylint: disable=consider-using-enumerate
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True, this checkpoint() is not necessary"
            )
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        return (None, None) + grads
