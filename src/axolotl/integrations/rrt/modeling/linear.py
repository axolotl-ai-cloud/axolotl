import math

import torch
import torch.nn.functional as F
from peft.utils import transpose
from torch import nn


class RelaxedRecursiveDoraLinear(nn.Module):
    """
    A single linear layer that is "shared" across multiple loop iterations,
    but each iteration has its own DoRA offsets (A_i, B_i, magnitude_i).

    The constructor expects you to specify:
      - in_features, out_features
      - B: number of loop iterations (i.e., how many times we "unroll")
      - fan_in_fan_out: pass True if your underlying base weight is transposed, etc.

    The forward(...) expects an additional argument "loop_idx" in [0..B-1],
    which picks out the iteration-specific DoRA offsets.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        B: int,
        rank: int,
        alpha: int,
        fan_in_fan_out: bool = False,
        bias: bool = True,
        use_dora: bool = True,
    ):
        super().__init__()
        self.B = B
        self.fan_in_fan_out = fan_in_fan_out

        self.weight_base = nn.Parameter(torch.empty(out_features, in_features))

        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.lora_A_list = nn.ParameterList(
            [nn.Parameter(torch.zeros(rank, in_features)) for _ in range(B)]
        )
        self.lora_B_list = nn.ParameterList(
            [nn.Parameter(torch.zeros(out_features, rank)) for _ in range(B)]
        )
        # rslora
        self.scaling = alpha / math.sqrt(rank)
        self.use_dora = use_dora
        if use_dora:
            self.lora_magnitude_vector_list = nn.ParameterList(
                [nn.Parameter(torch.ones(out_features)) for _ in range(B)]
            )

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def forward(self, x, loop_idx: int):
        """

        :param x: hidden state of shape (batch_size, seq_len, in_features)
        :param loop_idx:
        :return:
        """
        w_base = self.weight_base
        w_base = w_base.to(x.dtype)

        lora_A: torch.Tensor = self.lora_A_list[loop_idx]
        lora_B: torch.Tensor = self.lora_B_list[loop_idx]

        base_out: torch.Tensor = F.linear(x, w_base, self.bias)

        lora_out: torch.Tensor = F.linear(F.linear(x, lora_A), lora_B)

        if self.use_dora:
            x_eye: torch.Tensor = torch.eye(
                lora_A.shape[1], device=lora_A.device, dtype=x.dtype
            )
            tmp = F.linear(x_eye, lora_A)  # [hidden_size, rank]
            w_dora_full: torch.Tensor = F.linear(tmp, lora_B)
            w_dora_full = w_dora_full.t()

            magnitude_vector: torch.Tensor = self.lora_magnitude_vector_list[loop_idx]
            w_dora_norm: torch.Tensor = self.get_weight_norm(
                w_base, w_dora_full.detach(), self.scaling
            )
            w_dora_norm = w_dora_norm.detach()
            scale_factor = (magnitude_vector / w_dora_norm).unsqueeze(
                0
            )  # shape [1, out_features]

            result_dora = (scale_factor - 1) * base_out + scale_factor * lora_out
            return result_dora
        return base_out + lora_out * self.scaling
