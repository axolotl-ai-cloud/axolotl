"""Patch PEFT's AuxiliaryTrainingWrapper / ModulesToSaveWrapper so kwargs-only
forward calls work (e.g. Gemma 4 vision_tower / embed_vision in lora_modules_to_save)."""

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCHED_ATTR = "_axolotl_modules_to_save_kwargs_patched"


def _patched_forward(self, *args, **kwargs):
    # _check_forward_args only validates len(x) vs len(adapter_names); skip when no positional x.
    if args:
        self._check_forward_args(*args, **kwargs)

    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters or any(
        adapter not in self._adapters for adapter in self.active_adapters
    ):
        return self._forward_wrapped_passthrough(*args, **kwargs)

    if adapter_names is None:
        return self._forward_wrapped(*args, **kwargs)
    # Mixed-batch path needs positional input for sub-batch indexing; leave unchanged.
    return self._mixed_batch_forward(*args, adapter_names=adapter_names, **kwargs)


def _patched_forward_wrapped(self, *args, **kwargs):
    if not self.active_adapters:
        return self._forward_wrapped_passthrough(*args, **kwargs)
    return self.modules_to_save[self.active_adapters[0]](*args, **kwargs)


def _patched_forward_wrapped_passthrough(self, *args, **kwargs):
    return self.original_module(*args, **kwargs)


def patch_peft_modules_to_save_kwargs() -> None:
    """Apply the kwargs-compatible forward patch to PEFT. Idempotent."""
    from peft.utils.other import AuxiliaryTrainingWrapper, ModulesToSaveWrapper

    if getattr(AuxiliaryTrainingWrapper, _PATCHED_ATTR, False):
        return

    AuxiliaryTrainingWrapper.forward = _patched_forward
    ModulesToSaveWrapper._forward_wrapped = _patched_forward_wrapped
    ModulesToSaveWrapper._forward_wrapped_passthrough = (
        _patched_forward_wrapped_passthrough
    )

    setattr(AuxiliaryTrainingWrapper, _PATCHED_ATTR, True)
    LOG.debug(
        "Patched peft.AuxiliaryTrainingWrapper / ModulesToSaveWrapper to accept "
        "kwargs-only forward calls (enables full-FT of modules called with "
        "keyword args, e.g. Gemma 4 vision_tower/embed_vision)"
    )
