"""Exact LoRA merge -> unmerge round trips for vLLM weight syncs.

PEFT merges in place with ``W += delta`` and unmerges with ``W -= delta.to(W.dtype)``.
In BF16/FP16 the two roundings do not always cancel, so a round trip changes some
base weights. TRL's vLLM weight sync merges and unmerges the adapter at every sync,
so the frozen base weights the trainer uses drift further with each sync.

``temporarily_merge_adapter`` (being added to PEFT) merges for the duration of a
context; inside it, vanilla LoRA layers record the entries the unmerge will not
restore and write them back. This shim provides it for PEFT versions without it,
covering LoRA ``Linear`` and ``Embedding`` layers. Plain ``merge_adapter`` /
``unmerge_adapter``, ``merge_and_unload``, FP32 weights and LoRA variants keep
PEFT's behaviour and record nothing. Delete the shim once the pinned PEFT ships
``temporarily_merge_adapter``.
"""

from contextlib import ExitStack, contextmanager
from functools import wraps

import torch

_CHUNK_ELEMENTS = 1 << 24


def _index_dtype(numel: int) -> torch.dtype:
    return torch.int32 if numel < 2**31 else torch.int64


def _recordable(weight: torch.Tensor) -> bool:
    return (
        type(weight) is torch.Tensor
        and weight.dtype in (torch.float16, torch.bfloat16)
        and weight.is_contiguous()
        and weight.dim() > 0
    )


def _offloaded(module) -> bool:
    # accelerate offload hook: PEFT's onload_layer moves the base weight to the
    # execution device only for the duration of merge() / unmerge()
    hook = getattr(module.get_base_layer(), "_hf_hook", None)
    return bool(getattr(hook, "offload", False))


def unmerge_residual(
    weight: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Entries of ``weight`` that ``(weight += delta) -= delta.to(weight.dtype)`` does not
    restore, as ``(flat indices, original values)``. Built in row chunks; ``weight`` is
    not modified."""
    rows = weight.shape[0]
    weight_2d = weight.view(rows, -1)
    cols = weight_2d.shape[1]
    chunk = max(1, _CHUNK_ELEMENTS // max(1, cols))
    index_dtype = _index_dtype(weight.numel())
    indices, values = [], []
    for start in range(0, rows, chunk):
        before = weight_2d[start : start + chunk]
        delta_chunk = delta[start : start + chunk].reshape(before.shape)
        merged = before.clone()
        merged += delta_chunk
        wrong = (merged - delta_chunk.to(weight.dtype)) != before
        wrong = wrong.view(-1).nonzero().squeeze(-1)
        if wrong.numel():
            indices.append((wrong + start * cols).to(index_dtype))
            values.append(before.reshape(-1)[wrong])
    if not indices:
        return None
    return torch.cat(indices), torch.cat(values)


def _patch_layer_class(cls, casts_delta: bool):
    original_merge = cls.merge
    original_unmerge = cls.unmerge

    @wraps(original_merge)
    def merge(self, safe_merge=False, adapter_names=None):
        if not self.__dict__.get("_exact_unmerge", False):
            return original_merge(self, safe_merge, adapter_names)
        residuals = self.__dict__.setdefault("_unmerge_residuals", {})
        get_delta_weight = self.get_delta_weight

        def recording_get_delta_weight(adapter):
            # PEFT's merge adds this delta to the base weight right after this call
            delta = get_delta_weight(adapter)
            residuals.pop(adapter, None)
            weight = self.get_base_layer().weight.data
            if adapter not in self.lora_variant and _recordable(weight):
                added = delta.to(weight.dtype) if casts_delta or safe_merge else delta
                residual = unmerge_residual(weight, added)
                if residual is not None:
                    if _offloaded(self):
                        residual = (residual[0].cpu(), residual[1].cpu())
                    residuals[adapter] = residual
            return delta

        self.get_delta_weight = recording_get_delta_weight
        try:
            return original_merge(self, safe_merge, adapter_names)
        finally:
            del self.get_delta_weight

    @wraps(original_unmerge)
    def unmerge(self):
        residuals = self.__dict__.get("_unmerge_residuals")
        if not residuals or not self.merged:
            return original_unmerge(self)
        while self.merged_adapters:
            adapter = self.merged_adapters[-1]
            remaining = self.merged_adapters[:-1]
            self.merged_adapters = [adapter]
            original_unmerge(self)
            self.merged_adapters = remaining
            residual = residuals.pop(adapter, None)
            weight = self.get_base_layer().weight.data
            if residual is not None and weight.is_contiguous():
                indices, values = residual
                weight.view(-1)[indices.to(weight.device)] = values.to(weight.device)

    cls.merge = merge
    cls.unmerge = unmerge


@contextmanager
def _temporarily_merge_adapter(self, adapter_names=None, safe_merge=False):
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [m for m in self.model.modules() if isinstance(m, BaseTunerLayer)]
    if any(layer.merged for layer in layers):
        raise ValueError(
            "temporarily_merge_adapter needs unmerged adapters; call unmerge_adapter first."
        )
    for layer in layers:
        layer._exact_unmerge = True
    try:
        self.merge_adapter(adapter_names=adapter_names, safe_merge=safe_merge)
        yield
    finally:
        try:
            self.unmerge_adapter()
        finally:
            for layer in layers:
                layer.__dict__.pop("_exact_unmerge", None)
                layer.__dict__.pop("_unmerge_residuals", None)


_temporarily_merge_adapter._axolotl_shim = True


def patch_peft_lora_exact_unmerge():
    """Provide ``temporarily_merge_adapter`` unless the installed PEFT already has it."""
    from peft.tuners.lora.layer import Embedding, Linear
    from peft.tuners.tuners_utils import BaseTuner

    if hasattr(BaseTuner, "temporarily_merge_adapter"):
        return
    _patch_layer_class(Linear, casts_delta=False)
    _patch_layer_class(Embedding, casts_delta=True)
    BaseTuner.temporarily_merge_adapter = _temporarily_merge_adapter


@contextmanager
def exact_adapter_merges(model):
    """Route ``model.merge_adapter()`` / ``model.unmerge_adapter()`` calls made inside
    this context, e.g. by TRL's vLLM weight sync, through ``temporarily_merge_adapter``."""
    stack = ExitStack()

    def merge_adapter(adapter_names=None, safe_merge=False):
        stack.enter_context(
            model.temporarily_merge_adapter(
                adapter_names=adapter_names, safe_merge=safe_merge
            )
        )

    model.merge_adapter = merge_adapter
    model.unmerge_adapter = stack.close
    try:
        with stack:
            yield
    finally:
        del model.merge_adapter
        del model.unmerge_adapter


def exact_unmerge_vllm_sync(vllm_generation) -> None:
    """Make the merged-weight syncs of a TRL ``VLLMGeneration`` leave the PEFT base
    weights unchanged."""
    from accelerate.utils import is_peft_model

    model = getattr(vllm_generation, "model", None)
    if model is None or not is_peft_model(model):
        return
    patch_peft_lora_exact_unmerge()
    sync_weights = vllm_generation.sync_weights

    @wraps(sync_weights)
    def exact_sync_weights(*args, **kwargs):
        with exact_adapter_merges(model):
            return sync_weights(*args, **kwargs)

    vllm_generation.sync_weights = exact_sync_weights
