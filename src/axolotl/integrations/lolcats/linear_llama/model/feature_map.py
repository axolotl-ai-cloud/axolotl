"""
Learnable linear attention feature map classes and functions
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_feature_map(name: str, mlp: nn.Module, **kwargs):
    """
    Initialize feature map final activation for linear attention
    """
    return FeatureMap(activation_name=name, mlp=mlp, **kwargs)


def init_feature_map_act(name: str, fullspace: bool = True, **kwargs):
    """
    Initialize feature map final activation for linear attention
    """
    if name == "softmax_dim" and fullspace:
        return SoftmaxDim(**kwargs)
    elif name == "softmax_dim" and not fullspace:
        return SoftmaxDimHalfspace(**kwargs)
    elif name == "exp_dim" and fullspace:
        return Exp(**kwargs)
    elif name == "exp_dim" and not fullspace:
        return ExpHalfspace(**kwargs)
    elif name == "pos_elu":
        return PosELU(**kwargs)
    elif name == "relu":
        return ReLU(**kwargs)

    else:
        raise NotImplementedError


def init_learned_kernel(name: str, **kwargs):
    """
    Initialize feature map MLP for linear attention
    """
    if name == "untied_head_einsum":
        return FeatureMapMLP(**kwargs)
    elif name == "untied_head_adapter":
        return FeatureMapAdapter(**kwargs)
    else:
        raise NotImplementedError


class FeatureMap(nn.Module):
    """
    Final 'activation' of feature map. Can probably be combined with
    `FeatureMapMLP` below

    Full feature map is like f(xW + b)
    -> This is the `f` part
    """

    def __init__(
        self,
        activation_name: str,
        head_dim_idx: int = -1,
        eps: float = 1e-12,
        mlp: Optional[nn.Module] = None,
        fullspace: bool = True,
    ):
        super().__init__()
        self.head_dim_idx = head_dim_idx
        self.eps = eps
        self.mlp = mlp if mlp is not None else nn.Identity()
        self.activation = init_feature_map_act(activation_name, fullspace, eps=eps)

    def forward(self, x: torch.Tensor, *mlp_args, **mlp_kwargs):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return self.activation(self.mlp(x, *mlp_args, **mlp_kwargs), x)

    def q_map(self, *args, **kwargs):
        """
        Use for inference in case q and k feature maps differ
        """
        return self.forward(*args, **kwargs)

    def k_map(self, *args, **kwargs):
        """
        Use for inference in case q and k feature maps differ
        """
        return self.forward(*args, **kwargs)


# -----------------------
# Feature map activations
# -----------------------
class FeatureMapAct(nn.Module):
    """
    Base class for feature map activations
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x


class PosELU(FeatureMapAct):
    """
    1 + ELU activation as in https://arxiv.org/abs/2006.16236
    """

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return (1 + F.elu(x)).clamp(min=self.eps)


class ReLU(FeatureMapAct):
    """
    ReLU activation as in https://arxiv.org/abs/2103.13076
    """

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return F.relu(x).clamp(min=self.eps)


class SoftmaxDim(FeatureMapAct):
    """
    Softmax activation as in https://arxiv.org/abs/2402.04347
    """

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return torch.cat(
            [torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)], dim=-1
        ).clamp(min=self.eps)


class SoftmaxDimHalfspace(FeatureMapAct):
    """
    Softmax activation as in https://arxiv.org/abs/2402.04347
    """

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return torch.softmax(x, dim=-1).clamp(min=self.eps)


class Exp(FeatureMapAct):
    """
    Exp activation as in https://arxiv.org/abs/2402.04347
    """

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x_max = torch.amax(x, dim=-1, keepdim=True)
        x_min = torch.amin(x, dim=-1, keepdim=True)
        return torch.cat([torch.exp(x - x_max), torch.exp(-x + x_min)], dim=-1).clamp(
            min=self.eps
        )


class ExpHalfspace(FeatureMapAct):
    """
    Exp activation as in https://arxiv.org/abs/2402.04347
    """

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x_max = torch.amax(x, dim=-1, keepdim=True)
        return torch.exp(x - x_max).clamp(min=self.eps)


# ----------------
# Feature map MLPs
# ----------------


class FeatureMapMLP(nn.Module):
    """
    Learnable MLP in feature map.

    Full feature map is like f(xW + b)
    -> This is the `W` and (optional) `b` part
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,  # input dim
        feature_dim: int,  # output dim
        dtype: torch.dtype,
        device: torch.device,
        skip_connection: bool = False,
        bias: bool = False,
        zero_init: bool = False,
        normal_init: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.device = device
        self.skip_connection = skip_connection
        self.bias = bias
        self.zero_init = zero_init
        self.normal_init = normal_init
        self.init_weights_()

        if self.zero_init:  # Zero-out weights or set as identity post-initialization
            self.zero_init_with_skip_() if self.skip_connection else self.zero_init_()

        if self.normal_init:
            with torch.no_grad():
                nn.init.normal_(self.layer)

        if self.skip_connection:
            assertion_fail = f"If self.skip_connection we need self.head_dim == self.feature_dim but self.head_dim is {self.head_dim} != self.feature_dim is {self.feature_dim}"
            assert self.head_dim == self.feature_dim, assertion_fail

    def init_weights_(self):
        """
        Initialize (W)eights and (b)iases
        """
        self.layer = nn.Parameter(
            torch.zeros(
                (self.num_heads, self.head_dim, self.feature_dim),
                dtype=self.dtype,
                device=self.device,
            )
        )
        nn.init.kaiming_uniform_(self.layer)

        if self.bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    (1, self.num_heads, 1, 1),  # self.feature_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            nn.init.kaiming_uniform_(self.bias)
        else:
            self.bias = 0.0  # hack

    def zero_init_with_skip_(self):
        """
        Initialize weights to zero matrix if skip connection
        """
        with torch.no_grad():
            nn.init.zeros_(self.layer)

    def zero_init_(self):
        """
        Initialize weights to identity matrix if no skip connection
        """
        with torch.no_grad():
            for i in range(self.layer.shape[0]):
                try:
                    nn.init.eye_(self.layer[i])
                except RuntimeError:
                    with torch.no_grad():
                        dtype = self.layer[i].dtype
                        weight = torch.eye(
                            *self.layer[i].shape,
                            requires_grad=self.layer[i].requires_grad,
                            device=self.layer[i].device,
                        )
                        self.layer[i] = weight.to(dtype=dtype)

    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, num_heads, seq_len, head_dim)
        """
        _x = torch.einsum("hdf,bhld->bhlf", self.layer, x) + self.bias
        return x + _x if self.skip_connection else _x


class FeatureMapAdapter(FeatureMapMLP):
    """
    Learnable Feature map with bottleneck adapter
    as in https://arxiv.org/abs/1902.00751

    We don't use but could be fun to try
    """

    def __init__(self, hidden_dim: int, *args, **kwargs):
        kwargs["skip_connection"] = True
        kwargs["bias"] = True
        kwargs["zero_init"] = True
        self.hidden_dim = hidden_dim
        super().__init__(*args, **kwargs)

    def init_weights_(self):
        """
        Initialize (W)eights and (b)iases
        """
        kwargs = {"dtype": self.dtype, "device": self.device}
        self.layer0 = nn.Parameter(
            torch.zeros((self.num_heads, self.head_dim, self.hidden_dim), **kwargs)
        )
        self.layer1 = nn.Parameter(
            torch.zeros((self.num_heads, self.hidden_dim, self.feature_dim), **kwargs)
        )
        nn.init.kaiming_uniform_(self.layer0)
        nn.init.kaiming_uniform_(self.layer1)

        self.bias0 = nn.Parameter(
            torch.zeros((1, self.num_heads, 1, self.hidden_dim), **kwargs)
        )
        self.bias1 = nn.Parameter(
            torch.zeros((1, self.num_heads, 1, self.feature_dim), **kwargs)
        )
        nn.init.kaiming_uniform_(self.bias0)
        nn.init.kaiming_uniform_(self.bias1)

    def zero_init_with_skip_(self):
        with torch.no_grad():
            nn.init.zeros_(self.layer0)
            nn.init.zeros_(self.layer1)
            nn.init.zeros_(self.bias0)
            nn.init.zeros_(self.bias1)

    def zero_init_(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, num_heads, seq_len, head_dim)
        -> Down-project, apply nonlinearity, up-project; add skip connection
        """
        _x = torch.einsum("hde,bhld->bhle", self.layer0, x) + self.bias0
        _x = F.relu(_x)
        _x = torch.einsum("hef,bhle->bhlf", self.layer1, _x) + self.bias1
        return x + _x if self.skip_connection else _x
