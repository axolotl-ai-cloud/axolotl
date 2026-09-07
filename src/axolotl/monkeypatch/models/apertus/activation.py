"""Monkeypatch for Apertus to dtype mismatch in XIELU act"""

from torch import Tensor


def patch_apertus_xielu_activation():
    try:
        from transformers.activations import XIELUActivation
    except ImportError as err:
        raise ImportError(
            "Cannot import XIELUActivation. "
            "Please make sure to update your transformers version >= 4.56.1."
        ) from err

    from transformers.activations import logger

    # Store the original method
    old_fn = XIELUActivation._xielu_cuda

    def _xielu_cuda_fixed(self, x: Tensor) -> Tensor:
        """Firewall function to prevent torch.compile from seeing .item() calls"""
        original_shape = x.shape
        # CUDA kernel expects 3D tensors, reshape if needed
        while x.dim() < 3:
            x = x.unsqueeze(0)
        if x.dim() > 3:
            x = x.view(-1, 1, x.size(-1))
        if original_shape != x.shape:
            logger.warning_once(
                "Warning: xIELU input tensor expects 3 dimensions but got (shape: %s). Reshaping to (shape: %s).",
                original_shape,
                x.shape,
            )
        result = self._xielu_cuda_obj.forward(
            x,
            self.alpha_p.to(x.dtype),
            self.alpha_n.to(x.dtype),
            # Temporary until xIELU CUDA fully implemented -> self.{beta,eps}.item()
            self._beta_scalar,
            self._eps_scalar,
            self.with_vector_loads,
        )
        return result.view(original_shape)

    # Apply the patch
    XIELUActivation._xielu_cuda = _xielu_cuda_fixed

    def unpatch():
        """Restore the original method"""
        XIELUActivation._xielu_cuda = old_fn

    return unpatch
