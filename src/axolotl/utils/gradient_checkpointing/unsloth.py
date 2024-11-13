"""Unsloth checkpointing"""

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
import torch
from packaging import version

torch_version = version.parse(torch.__version__)

if torch_version < version.parse("2.4.0"):
    torch_cuda_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_cuda_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_cuda_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_cuda_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")


class Unsloth_Offloaded_Gradient_Checkpointer(  # pylint: disable=invalid-name
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
