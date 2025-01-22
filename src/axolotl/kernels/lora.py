"""
Module for definition of Low-Rank Adaptation (LoRA) Triton kernels.

See "LoRA: Low-Rank Adaptation of Large Language Models"
(https://arxiv.org/abs/2106.09685).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""
# pylint: disable=invalid-name

from typing import Callable

import torch
from bitsandbytes.functional import QuantState

from .geglu import geglu_backward, geglu_forward
from .quantize import dequantize
from .swiglu import swiglu_backward, swiglu_forward
from .utils import torch_amp_custom_bwd, torch_amp_custom_fwd


def quant_state(W):
    return getattr(W, "quant_state", None)


def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    W = base_layer.weight

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, quant_state(W), None, None, None
    pass

    active_adapter = (
        proj.active_adapters[0]
        if hasattr(proj, "active_adapters")
        else proj.active_adapter
    )
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]

    return W, quant_state(W), A, B, s


def matmul_lora(
    X: torch.Tensor,
    W: torch.Tensor,
    W_quant: QuantState,
    A: torch.Tensor,
    B: torch.Tensor,
    s: float,
    out: torch.Tensor | None = None,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Enhanced matmul_lora with distributed support"""
    dtype = X.dtype
    device = X.device
    W = dequantize(W.t(), W_quant)

    if X.dim() == 3:
        batch, seq_len, _ = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    out = torch.matmul(X, W, out=out)
    if W_quant is not None:
        del W

    if A is not None:
        A, B = A.t(), B.t()
        # Split computation across GPUs if in distributed mode
        if process_group is not None:
            # Shard the LoRA matrices across GPUs
            rank = torch.distributed.get_rank(process_group)
            world_size = torch.distributed.get_world_size(process_group)

            # Compute local shard
            shard_size = A.size(1) // world_size
            start_idx = rank * shard_size
            end_idx = start_idx + shard_size if rank < world_size - 1 else A.size(1)

            A_local = A[:, start_idx:end_idx].to(device)
            B_local = B[start_idx:end_idx, :].to(device)

            local_out = (X @ A_local.to(dtype)) @ (s * B_local.to(dtype))

            # All-reduce to get final result
            torch.distributed.all_reduce(
                local_out, op=torch.distributed.ReduceOp.SUM, group=process_group
            )
            out += local_out
        else:
            out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out


class LoRA_MLP(torch.autograd.Function):
    """Optimized LoRA MLP implementation with memory management."""

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        gate_weight: torch.Tensor,
        gate_quant: object | None,
        gate_A: torch.Tensor | None,
        gate_B: torch.Tensor | None,
        gate_scale: float,
        up_weight: torch.Tensor,
        up_quant: object | None,
        up_A: torch.Tensor | None,
        up_B: torch.Tensor | None,
        up_scale: float,
        down_weight: torch.Tensor,
        down_quant: object | None,
        down_A: torch.Tensor | None,
        down_B: torch.Tensor | None,
        down_scale: float,
        activation_fn: Callable,
        activation_fn_backward: Callable,
        inplace: bool | None = True,
        process_group: torch.distributed.ProcessGroup | None = None,
    ) -> torch.Tensor:
        # Compute projections using helper function
        gate = matmul_lora(
            X,
            gate_weight,
            gate_quant,
            gate_A,
            gate_B,
            gate_scale,
            process_group=process_group,
        )
        up = matmul_lora(
            X, up_weight, up_quant, up_A, up_B, up_scale, process_group=process_group
        )

        # Activation
        hidden = activation_fn(gate, up)

        # Down projection
        output = matmul_lora(
            hidden, down_weight, down_quant, down_A, down_B, down_scale
        )

        # Save tensors needed for backward
        ctx.save_for_backward(X, gate, up, gate_A, gate_B, up_A, up_B, down_A, down_B)
        ctx.scales = (gate_scale, up_scale, down_scale)
        ctx.quants = (gate_quant, up_quant, down_quant)
        ctx.weights = (gate_weight, up_weight, down_weight)
        ctx.activation_fn = activation_fn
        ctx.activation_fn_backward = activation_fn_backward
        ctx.inplace = inplace
        ctx.process_group = process_group

        return output

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        (
            X,
            gate,
            up,
            gate_A,
            gate_B,
            up_A,
            up_B,
            down_A,
            down_B,
        ) = ctx.saved_tensors
        gate_scale, up_scale, down_scale = ctx.scales
        gate_quant, up_quant, down_quant = ctx.quants
        gate_weight, up_weight, down_weight = ctx.weights
        process_group = ctx.process_group

        # Transpose all LoRA matrices
        gate_A, gate_B = (
            gate_A.t() if gate_A is not None else None,
            gate_B.t() if gate_B is not None else None,
        )
        up_A, up_B = (
            up_A.t() if up_A is not None else None,
            up_B.t() if up_B is not None else None,
        )
        down_A, down_B = (
            down_A.t() if down_A is not None else None,
            down_B.t() if down_B is not None else None,
        )

        # Reshape inputs
        batch, seq_len, hd = X.shape
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        X = X.view(-1, X.shape[-1])
        gate = gate.view(-1, gate.shape[-1])
        up = up.view(-1, up.shape[-1])
        dtype = X.dtype

        # Down projection with distributed support
        DW = matmul_lora(
            grad_output,
            down_weight.t(),
            down_quant,
            down_B,
            down_A,
            down_scale,
            process_group=process_group,
        )

        # Activation backward
        DW, gate, up = ctx.activation_fn_backward(DW, gate, up)
        h, df, de = DW, gate, up

        # Initialize gradient accumulators
        d_down_A = torch.zeros_like(down_A) if down_A is not None else None
        d_down_B = torch.zeros_like(down_B) if down_B is not None else None
        d_up_A = torch.zeros_like(up_A) if up_A is not None else None
        d_up_B = torch.zeros_like(up_B) if up_B is not None else None
        d_gate_A = torch.zeros_like(gate_A) if gate_A is not None else None
        d_gate_B = torch.zeros_like(gate_B) if gate_B is not None else None

        # Compute LoRA gradients with distributed handling
        if process_group is not None:
            rank = torch.distributed.get_rank(process_group)
            world_size = torch.distributed.get_world_size(process_group)

            # Shard computations across GPUs
            shard_size = h.size(0) // world_size
            start_idx = rank * shard_size
            end_idx = start_idx + shard_size if rank < world_size - 1 else h.size(0)

            # Compute local gradients
            if down_A is not None:
                d_down_A = h[start_idx:end_idx].t() @ (
                    grad_output[start_idx:end_idx] @ down_B.t()
                )
                d_down_B = (down_A.t() @ h[start_idx:end_idx].t()) @ grad_output[
                    start_idx:end_idx
                ]
                d_down_A *= down_scale
                d_down_B *= down_scale

            if up_A is not None:
                d_up_A = X[start_idx:end_idx].t() @ (df[start_idx:end_idx] @ up_B.t())
                d_up_B = (up_A.t() @ X[start_idx:end_idx].t()) @ df[start_idx:end_idx]
                d_up_A *= up_scale
                d_up_B *= up_scale

            if gate_A is not None:
                d_gate_A = X[start_idx:end_idx].t() @ (
                    de[start_idx:end_idx] @ gate_B.t()
                )
                d_gate_B = (gate_A.t() @ X[start_idx:end_idx].t()) @ de[
                    start_idx:end_idx
                ]
                d_gate_A *= gate_scale
                d_gate_B *= gate_scale

            # All-reduce gradients
            grads_to_reduce = [
                grad
                for grad in [d_down_A, d_down_B, d_up_A, d_up_B, d_gate_A, d_gate_B]
                if grad is not None
            ]

            for grad in grads_to_reduce:
                torch.distributed.all_reduce(
                    grad, op=torch.distributed.ReduceOp.SUM, group=process_group
                )
        else:
            # Non-distributed gradient computation
            if down_A is not None:
                d_down_A = h.t() @ (grad_output @ down_B.t())
                d_down_B = (down_A.t() @ h.t()) @ grad_output
                d_down_A *= down_scale
                d_down_B *= down_scale

            if up_A is not None:
                d_up_A = X.t() @ (df @ up_B.t())
                d_up_B = (up_A.t() @ X.t()) @ df
                d_up_A *= up_scale
                d_up_B *= up_scale

            if gate_A is not None:
                d_gate_A = X.t() @ (de @ gate_B.t())
                d_gate_B = (gate_A.t() @ X.t()) @ de
                d_gate_A *= gate_scale
                d_gate_B *= gate_scale

        # Compute input gradients
        dX = torch.zeros_like(X) if ctx.needs_input_grad[0] else None

        if dX is not None:
            # Up projection gradients
            up_weight = dequantize(up_weight.t(), up_quant)
            if ctx.inplace:
                dX = torch.matmul(df, up_weight.t(), out=X)
            else:
                dX = torch.matmul(df, up_weight.t())
            del up_weight

            # Note the .to(dtype) only where mixing LoRA with base weights
            if up_A is not None:
                dX += df @ up_B.to(dtype).t() @ (up_scale * up_A.to(dtype).t())

            # Gate projection gradients
            gate_weight = dequantize(gate_weight.t(), gate_quant)
            dX += de @ gate_weight.t()
            del gate_weight

            if gate_A is not None:
                dX += de @ gate_B.to(dtype).t() @ (gate_scale * gate_A.to(dtype).t())

            # Reshape back
            dX = dX.view(batch, seq_len, hd)

        # Return gradients in correct order matching forward inputs
        return (
            dX,  # X
            None,  # gate_weight
            None,  # gate_quant
            d_gate_A.t() if d_gate_A is not None else None,  # gate_A
            d_gate_B.t() if d_gate_B is not None else None,  # gate_B
            None,  # gate_scale
            None,  # up_weight
            None,  # up_quant
            d_up_A.t() if d_up_A is not None else None,  # up_A
            d_up_B.t() if d_up_B is not None else None,  # up_B
            None,  # up_scale
            None,  # down_weight
            None,  # down_quant
            d_down_A.t() if d_down_A is not None else None,  # down_A
            d_down_B.t() if d_down_B is not None else None,  # down_B
            None,  # down_scale
            None,  # activation_fn
            None,  # activation_fn_backward
            None,  # inplace
        )


def apply_lora_mlp_swiglu(self, X, inplace=True):
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)

    out = LoRA_MLP.apply(
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        swiglu_forward,
        swiglu_backward,
        inplace,
    )

    return out


# TODO: Add approximate GEGLU implementation like unsloth does?
def apply_lora_mlp_geglu(self, X, inplace=True):
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        geglu_forward,
        geglu_backward,
        inplace,
    )

    return out


class LoRAMLPWrapper(torch.nn.Module):
    def __init__(self, config, weights, activation="swiglu"):
        super().__init__()

        self.quant_config = config.get("quant_config")

        # Change to match reference model naming
        self.gate_proj = torch.nn.Linear(
            config["in_features"], config["hidden_features"], bias=False
        )
        self.up_proj = torch.nn.Linear(
            config["in_features"], config["hidden_features"], bias=False
        )
        self.down_proj = torch.nn.Linear(
            config["hidden_features"], config["out_features"], bias=False
        )

        # Base weights
        self.gate_proj.weight.data = weights[0]
        self.up_proj.weight.data = weights[1]
        self.down_proj.weight.data = weights[2]

        # Store weights as parameters
        self.gate_A = torch.nn.Parameter(weights[3])
        self.gate_B = torch.nn.Parameter(weights[4])
        self.up_A = torch.nn.Parameter(weights[5])
        self.up_B = torch.nn.Parameter(weights[6])
        self.down_A = torch.nn.Parameter(weights[7])
        self.down_B = torch.nn.Parameter(weights[8])

        self.gate_weight_q = self.gate_proj.weight.data
        self.up_weight_q = self.up_proj.weight.data
        self.down_weight_q = self.down_proj.weight.data
        self.gate_quant = self.up_quant = self.down_quant = None

        # Store activation function
        activation_fns = {
            "swiglu": (swiglu_forward, swiglu_backward),
            "geglu": (geglu_forward, geglu_backward),
        }
        self.activation_fn, self.backward_activation_fn = activation_fns[activation]

    def forward(self, x):
        return LoRA_MLP.apply(
            x,
            self.gate_weight_q,
            self.gate_quant,
            self.gate_A,
            self.gate_B,
            1.0,  # gate_scale
            self.up_weight_q,
            self.up_quant,
            self.up_A,
            self.up_B,
            1.0,  # up_scale
            self.down_weight_q,
            self.down_quant,
            self.down_A,
            self.down_B,
            1.0,  # down_scale
            self.activation_fn,
            self.backward_activation_fn,
            True,  # inplace
        )


def create_lora_mlp(config, *weights, activation="swiglu"):
    """Create a module wrapper for LoRA MLP."""
    return LoRAMLPWrapper(config, weights, activation=activation)


class LoRA_QKV(torch.autograd.Function):
    """Optimized LoRA QKV implementation with quantization support."""

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X,
        q_weight,
        q_quant,
        q_A,
        q_B,
        q_scale,
        k_weight,
        k_quant,
        k_A,
        k_B,
        k_scale,
        v_weight,
        v_quant,
        v_A,
        v_B,
        v_scale,
        inplace=True,
    ):
        Q = matmul_lora(X, q_weight, q_quant, q_A, q_B, q_scale)
        K = matmul_lora(X, k_weight, k_quant, k_A, k_B, k_scale)
        V = matmul_lora(X, v_weight, v_quant, v_A, v_B, v_scale)

        ctx.save_for_backward(X, q_A, q_B, k_A, k_B, v_A, v_B)
        ctx.scales = (q_scale, k_scale, v_scale)
        ctx.quants = (q_quant, k_quant, v_quant)
        ctx.weights = (q_weight, k_weight, v_weight)
        ctx.inplace = inplace

        return Q, K, V

    @staticmethod
    @torch_amp_custom_fwd
    def backward(ctx, q_grad, k_grad, v_grad):
        X, A_q, B_q, A_k, B_k, A_v, B_v = ctx.saved_tensors
        q_weight, k_weight, v_weight = ctx.weights
        q_quant, k_quant, v_quant = ctx.quants
        q_scale, k_scale, v_scale = ctx.scales
        dtype = X.dtype

        # Reshape gradients
        batch, seq_len = X.shape[:2]
        q_grad = q_grad.view(-1, q_grad.shape[-1])
        k_grad = k_grad.reshape(-1, k_grad.shape[-1])
        v_grad = v_grad.view(-1, v_grad.shape[-1])
        X = X.view(-1, X.shape[-1])

        # Pre-transpose X once
        X_t = X.t()

        # Pre-compute scaled LoRA weights to avoid repeated dtype conversions
        A_q_scaled = (q_scale * A_q).to(dtype)
        B_q_scaled = B_q.to(dtype)
        A_k_scaled = (k_scale * A_k).to(dtype)
        B_k_scaled = B_k.to(dtype)
        A_v_scaled = (v_scale * A_v).to(dtype)
        B_v_scaled = B_v.to(dtype)

        # Compute all LoRA gradients using efficient matrix ops
        # Reuse buffers where possible
        d_A_q = torch.mm(X_t, torch.mm(q_grad, B_q_scaled))
        d_B_q = torch.mm(torch.mm(A_q_scaled, X_t), q_grad)

        d_A_k = torch.mm(X_t, torch.mm(k_grad, B_k_scaled))
        d_B_k = torch.mm(torch.mm(A_k_scaled, X_t), k_grad)

        d_A_v = torch.mm(X_t, torch.mm(v_grad, B_v_scaled))
        d_B_v = torch.mm(torch.mm(A_v_scaled, X_t), v_grad)

        # Compute input gradient, reusing X memory if possible
        out_buffer = X if ctx.inplace else None

        # Q path
        q_weight_t = dequantize(q_weight, q_quant)
        grad_X = torch.mm(q_grad, q_weight_t, out=out_buffer)
        del q_weight
        del q_weight_t
        grad_X.addmm_(q_grad, torch.mm(B_q_scaled, A_q_scaled))

        # K path
        k_weight_t = dequantize(k_weight, k_quant)
        grad_X.addmm_(k_grad, k_weight_t)
        del k_weight
        del k_weight_t
        grad_X.addmm_(k_grad, torch.mm(B_k_scaled, A_k_scaled))

        # V path
        v_weight_t = dequantize(v_weight, v_quant)
        grad_X.addmm_(v_grad, v_weight_t)
        del v_weight
        del v_weight_t
        grad_X.addmm_(v_grad, torch.mm(B_v_scaled, A_v_scaled))

        return (
            grad_X.view(batch, seq_len, -1),
            None,
            None,
            d_A_q.t(),
            d_B_q.t(),
            None,
            None,
            None,
            d_A_k.t(),
            d_B_k.t(),
            None,
            None,
            None,
            d_A_v.t(),
            d_B_v.t(),
            None,
            None,
        )


def apply_lora_qkv(self, X, inplace=True):
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = LoRA_QKV.apply(
        X,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace,
    )

    return Q, K, V


class LoRAQKVWrapper(torch.nn.Module):
    def __init__(self, config, weights, activation="swiglu"):
        super().__init__()

        self.quant_config = config.get("quant_config")

        # Change to match reference model naming
        self.q_proj = torch.nn.Linear(
            config["in_features"], config["hidden_features"], bias=False
        )
        self.k_proj = torch.nn.Linear(
            config["in_features"], config["hidden_features"], bias=False
        )
        self.v_proj = torch.nn.Linear(
            config["hidden_features"], config["out_features"], bias=False
        )

        # Base weights
        self.q_proj.weight.data = weights[0]
        self.k_proj.weight.data = weights[1]
        self.v_proj.weight.data = weights[2]

        # Store weights as parameters
        self.q_A = torch.nn.Parameter(weights[3])
        self.q_B = torch.nn.Parameter(weights[4])
        self.k_A = torch.nn.Parameter(weights[5])
        self.k_B = torch.nn.Parameter(weights[6])
        self.v_A = torch.nn.Parameter(weights[7])
        self.v_B = torch.nn.Parameter(weights[8])

        self.q_weight = self.q_proj.weight.data
        self.k_weight = self.k_proj.weight.data
        self.v_weight = self.v_proj.weight.data
        self.q_quant = self.k_quant = self.v_quant = None

        # Store activation function
        activation_fns = {
            "swiglu": (swiglu_forward, swiglu_backward),
            "geglu": (geglu_forward, geglu_backward),
        }
        self.activation_fn, self.backward_activation_fn = activation_fns[activation]

    def forward(self, x):
        return LoRA_QKV.apply(
            x,
            self.q_weight,
            self.q_quant,
            self.q_A,
            self.q_B,
            1.0,  # query scale
            self.k_weight,
            self.k_quant,
            self.k_A,
            self.k_B,
            1.0,  # key scale
            self.v_weight,
            self.v_quant,
            self.v_A,
            self.v_B,
            1.0,  # value scale
            True,  # inplace
        )


def create_lora_qkv(config, *weights, activation="swiglu"):
    """Create a module wrapper for LoRA QKV."""
    return LoRAQKVWrapper(config, weights, activation=activation)


class LoRA_W(torch.autograd.Function):
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X: torch.Tensor, W, W_quant, A, B, S):
        XW = matmul_lora(X, W, W_quant, A, B, S)
        ctx.custom_saved_tensors = (
            W,
            W_quant,
            S,
        )
        ctx.save_for_backward(A, B, X)

        return XW

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY: torch.Tensor):
        W, W_quant, S = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        A, B = A.t(), B.t()

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        dtype = X.dtype

        # Weight projection
        d_A = X.t() @ (dY @ B.t())
        d_B = (A.t() @ X.t()) @ dY
        d_A *= S
        d_B *= S

        # Get derivative for dX
        W = dequantize(W.t(), W_quant)
        dX = dY @ W.t()
        del W
        dX += dY @ B.to(dtype).t() @ (S * A.to(dtype).t())

        # W, W_quant, A, B, S
        return dX.view(batch, seq_len, hd), None, None, d_A.t(), d_B.t(), None


def apply_lora_o(self, X):
    OW, OW_quant, OA, OB, OS = get_lora_parameters(self.o_proj)
    output = LoRA_W.apply(X, OW, OW_quant, OA, OB, OS)

    return output
