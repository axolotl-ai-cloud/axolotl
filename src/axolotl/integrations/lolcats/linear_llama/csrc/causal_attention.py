#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import torch

try:
    from causal_attention_cuda import causal_dot_backward as causal_dot_backward_cuda
    from causal_attention_cuda import causal_dot_product as causal_dot_product_cuda
except ImportError as e:
    print(e)
    causal_dot_product_cuda = causal_dot_backward_cuda = None


class CausalDotProduct(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""

    dot = {
        # "cpu": causal_dot_product_cpu,
        "cuda": causal_dot_product_cuda
    }
    dot_backward = {
        # "cpu": causal_dot_backward_cpu,
        "cuda": causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, V):
        # Save the inputs for the gradient computation
        ctx.save_for_backward(Q, K, V)

        # Create the output tensor
        device = Q.device
        N, H, L, _ = Q.shape
        _, _, _, M = V.shape
        product = torch.zeros((N, H, L, M), dtype=Q.dtype, device=device)

        # Actually perform the dot product
        CausalDotProduct.dot[device.type](Q.data, K.data, V.data, product)
        # breakpoint()
        # CausalDotProduct.dot[device.type](Q.data, K.data, V.data, product)

        return product

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        Q, K, V = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # Actually compute the gradients
        CausalDotProduct.dot_backward[Q.device.type](
            Q.data, K.data, V.data, grad_out, grad_Q, grad_K, grad_V
        )

        return grad_Q, grad_K, grad_V


# Alias the autograd functions to python style snake case naming
causal_dot_product = CausalDotProduct.apply
