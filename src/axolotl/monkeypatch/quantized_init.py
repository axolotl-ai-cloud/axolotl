"""Stop Transformers re-initializing weights that arrive already quantized."""

import functools
import importlib


def _optional_types(*specs: tuple[str, str]) -> tuple[type, ...]:
    """The named types that import here; torchao and bitsandbytes are absent on macOS/aarch64."""
    found: tuple[type, ...] = ()
    for module_name, attribute in specs:
        try:
            found += (getattr(importlib.import_module(module_name), attribute),)
        except (ImportError, AttributeError):
            pass
    return found


def patch_transformers_skip_quantized_init():
    """Stop ``from_pretrained`` from re-initializing already-quantized weights.

    transformers re-runs ``_init_weights`` on every module during loading; the
    generic implementation does ``init.normal_(module.weight.float(), ...)``.
    Re-initializing an already-loaded quantized weight is never correct, so we skip
    those modules entirely. Two shapes reach this, both because the weights are
    quantized but no HF quantizer is registered to claim them:

    - a torchao tensor subclass (e.g. ``MXTensor``) as the parameter itself, where
      ``.float()`` returns a new tensor that drops the ``_is_hf_initialized`` flag and
      does not implement ``normal_``, so an MX checkpoint raises NotImplementedError;
    - a quantized-weight parametrization (staged NF4, or the bitsandbytes expert
      parametrizations from ``quantize_moe_experts``), where the packed weight hides
      behind ``module.parametrizations`` and reads as a missing key, so the module is
      re-initialized silently: one full dequantization plus a ``normal_`` draw each,
      before the first step.

    Other parametrizations (weight norm in the audio models, for instance) are left
    to Transformers, since their weights are not quantized.
    """
    from transformers import PreTrainedModel

    from axolotl.utils.nf4 import BnbNF4Parametrization, TorchaoNF4Parametrization

    torchao_tensors = _optional_types(("torchao.utils", "TorchAOBaseTensor"))
    quantized_parametrizations: tuple[type, ...] = (
        BnbNF4Parametrization,
        TorchaoNF4Parametrization,
    ) + _optional_types(
        ("bitsandbytes.nn.parametrize", "Bnb4bitParametrization"),
        ("axolotl.monkeypatch.moe_quant", "Bnb8bitParametrization"),
    )

    if getattr(PreTrainedModel._initialize_weights, "_axolotl_torchao_patched", False):
        return

    original = PreTrainedModel._initialize_weights

    def holds_quantized_weight(module):
        if any(
            isinstance(param, torchao_tensors)
            for param in module.parameters(recurse=False)
        ):
            return True
        return any(
            isinstance(transform, quantized_parametrizations)
            for chain in (getattr(module, "parametrizations", None) or {}).values()
            for transform in chain
        )

    @functools.wraps(original)
    def _initialize_weights(self, module, *args, **kwargs):
        if holds_quantized_weight(module):
            module._is_hf_initialized = True
            return None
        return original(self, module, *args, **kwargs)

    _initialize_weights._axolotl_torchao_patched = True
    PreTrainedModel._initialize_weights = _initialize_weights
