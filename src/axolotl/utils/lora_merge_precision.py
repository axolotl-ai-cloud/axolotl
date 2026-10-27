"""How much of a LoRA update survives rounding the merged weight to the served dtype.

Merged serving (vLLM merged weight sync, ``merge-lora``) computes
``cast(W + scaling * B @ A)``. At RL learning rates most entries of the LoRA
delta are below half an ulp of the base weight in BF16, so the cast drops most
of the update the trainer optimized against the unrounded ``W + scaling * B @ A``.
"""

from dataclasses import dataclass

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

LOW_RETENTION_THRESHOLD = 0.5
_CHUNK_ELEMENTS = 1 << 24
_HALF_DTYPES = (torch.bfloat16, torch.float16)
_FLOAT_DTYPES = (torch.float32, *_HALF_DTYPES)


def lora_delta(
    lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float
) -> torch.Tensor:
    """FP32 ``scaling * B @ A``, with the matmul in the adapter dtype as PEFT's merge computes it."""
    dtype = torch.promote_types(lora_a.dtype, lora_b.dtype)
    with torch.autocast(device_type=lora_b.device.type, enabled=False):
        return ((lora_b.to(dtype) @ lora_a.to(dtype)) * scaling).float()


def merged_lora_weight(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``cast(W + scaling * B @ A)`` rounded once, bitwise equal to PEFT's in-place merge
    into a ``dtype`` weight."""
    with torch.autocast(device_type=weight.device.type, enabled=False):
        return (weight.float() + lora_delta(lora_a, lora_b, scaling)).to(dtype)


def served_dtype_for(weight_dtype: torch.dtype) -> torch.dtype:
    """vLLM serves half-precision checkpoints in their own dtype and FP32 ones in BF16."""
    return weight_dtype if weight_dtype in _HALF_DTYPES else torch.bfloat16


@dataclass
class MergeRetention:
    """Projection of the served weight change onto the intended LoRA delta.

    ``retained = <served(W + D) - served(W), D> / |D|^2``: 1.0 when the merge keeps the
    whole update, ~0 when it rounds back to the base weight.
    """

    dot: float = 0.0
    delta_sq: float = 0.0
    changed: int = 0
    numel: int = 0
    modules: int = 0
    skipped: int = 0

    def update(self, delta: torch.Tensor, served_change: torch.Tensor) -> None:
        delta = delta.float()
        served_change = served_change.float()
        self.dot += torch.sum(delta * served_change).item()
        self.delta_sq += torch.sum(delta * delta).item()
        self.changed += int(torch.count_nonzero(served_change).item())
        self.numel += delta.numel()

    @property
    def retained(self) -> float | None:
        return self.dot / self.delta_sq if self.delta_sq > 0 else None

    @property
    def changed_fraction(self) -> float | None:
        return self.changed / self.numel if self.numel else None

    def describe(self) -> str:
        return (
            f"{100 * (self.retained or 0.0):.1f}% of the LoRA delta survives "
            f"({100 * (self.changed_fraction or 0.0):.1f}% of weight entries change) "
            f"across {self.modules} merged LoRA linears"
        )


def accumulate_lora_linear(
    stats: MergeRetention,
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    dtype: torch.dtype,
) -> None:
    """Add one LoRA linear's merge rounding to ``stats``, ``_CHUNK_ELEMENTS`` at a time."""
    rows = max(1, _CHUNK_ELEMENTS // max(1, weight.shape[1]))
    with torch.no_grad():
        for start in range(0, weight.shape[0], rows):
            base = weight[start : start + rows].float()
            delta = lora_delta(lora_a, lora_b[start : start + rows], scaling)
            served_change = (base + delta).to(dtype).float() - base.to(dtype).float()
            stats.update(delta, served_change)
    stats.modules += 1


def _is_dense_float(weight: torch.Tensor) -> bool:
    # Tensor subclasses (DTensor, Params4bit, NVFP4/FP8 tensors) and ZeRO-3
    # placeholders do not hold the full local weight.
    return (
        type(weight) in (torch.Tensor, torch.nn.Parameter)
        and weight.dtype in _FLOAT_DTYPES
        and weight.numel() > 0
        and not hasattr(weight, "ds_id")
    )


def measure_lora_merge_retention(
    model: torch.nn.Module, dtype: torch.dtype | None = None
) -> MergeRetention:
    """Measure the merge rounding of every active, unmerged PEFT LoRA linear in ``model``.

    ``dtype`` is the served weight dtype; by default the base weight's own half dtype,
    or BF16 for FP32 bases. Modules already trained merge-aware are left out: the
    trainer optimizes their rounded weights directly.
    """
    from peft.tuners.lora.layer import Linear as LoraLinear

    stats = MergeRetention()
    for module in model.modules():
        if not isinstance(module, LoraLinear) or module.merged:
            continue
        if module.disable_adapters or module.fan_in_fan_out:
            continue
        if getattr(module, "_axolotl_lora_merge_aware_dtype", None) is not None:
            continue
        adapters = [a for a in module.active_adapters if a in module.lora_A]
        if len(adapters) != 1 or adapters[0] in module.lora_variant:
            stats.skipped += 1
            continue
        weight = module.get_base_layer().weight
        if not _is_dense_float(weight):
            stats.skipped += 1
            continue
        adapter = adapters[0]
        accumulate_lora_linear(
            stats,
            weight,
            module.lora_A[adapter].weight,
            module.lora_B[adapter].weight,
            module.scaling[adapter],
            dtype or served_dtype_for(weight.dtype),
        )
    return stats


VLLM_SYNC_REMEDY = (
    "Serve the adapter unmerged (trl.vllm_lora_sync: true with the async GRPO trainer "
    "and a vLLM server), or train against the merged weights with lora_merge_aware: bf16."
)


def log_merge_retention(
    stats: MergeRetention, context: str, remedy: str, warn: bool = True
) -> None:
    """Log ``stats``; with ``warn``, a retained share below ``LOW_RETENTION_THRESHOLD``
    is a warning that suggests ``remedy``."""
    retained = stats.retained
    if retained is None:
        return
    if warn and retained < LOW_RETENTION_THRESHOLD:
        LOG.warning(
            "LoRA MERGE ROUNDING: %s: %s. The merged weights lose most of the trained "
            "update, so they behave differently from the model the trainer optimized. %s",
            context,
            stats.describe(),
            remedy,
        )
    else:
        LOG.info("LoRA merge rounding: %s: %s", context, stats.describe())
