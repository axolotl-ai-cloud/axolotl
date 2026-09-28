"""Ternary (BitNet b1.58 style) fake quantization for QAT."""

import torch
from torch import nn
from torch.nn import functional as F

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

EPS = 1e-5


def ternarize(weight: torch.Tensor) -> torch.Tensor:
    """Round to {-1, 0, 1} scaled by the per-row (per output channel) absmean."""
    scale = weight.abs().mean(dim=-1, keepdim=True).clamp_min(EPS)
    return (weight / scale).round().clamp_(-1, 1) * scale


def quantize_activation(x: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 quantization with a per-token absmax scale."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / 127
    return (x / scale).round().clamp_(-128, 127) * scale


class TernaryFakeQuantizer(nn.Module):
    """Applies ``quantize_fn`` with a straight-through gradient estimator."""

    def __init__(self, quantize_fn):
        super().__init__()
        self.quantize_fn = quantize_fn
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return x + (self.quantize_fn(x) - x).detach()


class TernaryFakeQuantizedLinear(nn.Linear):
    """``nn.Linear`` with ternary weights and optional int8 activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_fake_quantizer is not None:
            x = self.activation_fake_quantizer(x)
        return F.linear(x, self.weight_fake_quantizer(self.weight), self.bias)


def _int8_embedding_qat_config():
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.fake_quantize_config import IntxFakeQuantizeConfig

    return QATConfig(
        weight_config=IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_channel"
        )
    )


def has_tied_output_embedding(model) -> bool:
    """Whether the LM head shares its weight with the input embedding table."""
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        return True

    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if get_input is None or get_output is None:
        return False
    try:
        input_embedding, output_embedding = get_input(), get_output()
    except (AttributeError, NotImplementedError):
        return False
    if input_embedding is None or output_embedding is None:
        return False

    input_weight, output_weight = input_embedding.weight, output_embedding.weight
    # data_ptr is 0 for every meta tensor, which would read as tied for all of them
    if input_weight.is_meta or output_weight.is_meta:
        return False
    return input_weight.data_ptr() == output_weight.data_ptr()


def prepare_model_for_ternary_qat(
    model,
    quantize_activations: bool = True,
    quantize_embedding: bool = False,
):
    """Swap every ``nn.Linear`` outside the LM head for a ternary fake quantized one.

    Norms and the LM head stay in high precision; embeddings, when quantized, use
    int8 per-row rather than ternary.
    """
    for name, module in model.named_modules():
        # exact type check so the swap back on convert cannot downgrade a subclass
        if type(module) is nn.Linear and not name.endswith("lm_head"):  # pylint: disable=unidiomatic-typecheck
            module.__class__ = TernaryFakeQuantizedLinear
            module.weight_fake_quantizer = TernaryFakeQuantizer(ternarize)
            module.activation_fake_quantizer = (
                TernaryFakeQuantizer(quantize_activation)
                if quantize_activations
                else None
            )

    if quantize_embedding:
        # tied weights would carry the baked int8 table into the LM head on convert
        if has_tied_output_embedding(model):
            LOG.warning(
                "Skipping quantize_embedding: this model ties the LM head to the input "
                "embeddings, so quantizing the table would also quantize the LM head, "
                "which ternary QAT keeps in high precision. Untie the weights to "
                "quantize the embedding table."
            )
            return

        from torchao.quantization import quantize_

        quantize_(
            model,
            _int8_embedding_qat_config(),
            filter_fn=lambda m, _: isinstance(m, nn.Embedding),
        )


@torch.no_grad()
def convert_ternary_model(model) -> bool:
    """Bake the ternary values into the weights and restore plain ``nn.Linear``.

    Returns whether the model was a ternary one.
    """
    converted = False
    for module in model.modules():
        if isinstance(module, TernaryFakeQuantizedLinear):
            module.weight.copy_(ternarize(module.weight))
            module.__class__ = nn.Linear
            del module.weight_fake_quantizer
            del module.activation_fake_quantizer
            converted = True
    return converted
