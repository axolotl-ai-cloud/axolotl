"""SubLN: an extra RMSNorm in front of the `o_proj` / `down_proj` inputs.

Both of those projections consume a tensor no norm has touched — attention output and
the SwiGLU product — so their per-token dynamic range is the widest in the block and
they take the most damage from int8 activations and ternary weights. Normalizing that
input is the one architectural change the BitNet-distillation recipe keeps.

The norm lives *inside* `TernaryLinear` (attribute `modules.SUBLN_ATTR`), not as a
sibling module, for four reasons: the swap is what knows which linears qualify, the
norm has to run before the shared activation-quant memo, the parameter serializes as
`<linear>.sub_norm.weight` so nothing outside the swap has to rewrite the graph, and
the `_norm` in that name is what keeps the gain out of the trainer's decay group.

Insertion is *not* function-preserving, and no initialization makes it so:

- at unit gain the norm still maps `x → x / rms(x)`, so the projection's output picks
  up a per-token scalar gain of `1 / rms(x)`. Even at λ=0, with the ternary quantizer
  switched off, a converted block already computes something else,
- the recipe accepts that because the conversion is destructive by construction
  (ternary weights, int8 activations) and the heal is what re-fits the block. What
  SubLN buys is a bounded, unit-variance input for the two projections the block's
  own norms never reach — the precondition per-token int8 quantization and ternary
  weight rounding both want,
- so `ternary.subln` belongs with a healing budget. Turning it on for a short QAT run
  spends accuracy the run has no time to earn back.

Persistence, and the sharp edge that comes with it:

- the gain is a plain FP parameter trained with everything else; `_post_training`
  bakes only the ternary weight, so it is saved as-is into the bf16 master,
- the master therefore carries tensors the stock architecture has no home for.
  `from_pretrained` does *not* fail on them: transformers lists them as UNEXPECTED in
  its load report and drops them, so the reload is quietly a different model,
- the conversion therefore stamps `SUBLN_CONFIG_KEY` on the model config (it survives
  `save_pretrained` and comes back as an attribute) and records `manifest.subln`.
  `assert_subln_reloadable` refuses a marked checkpoint whenever `ternary.subln` is
  off, and `insert_subln` restores the dropped gains from the checkpoint's shards
  once it has rebuilt the modules,
- `assert_subln_exportable` refuses the deployment formats that have no slot for the
  extra tensor, which the config schema also rejects up front.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .args import resolve_ternary_config
from .modules import SUBLN_ATTR
from .swap import SwapManifest, module_family

LOG = get_logger(__name__)

# module families whose input is not normalized by the block's own norms
SUBLN_FAMILIES: tuple[str, ...] = ("o_proj", "down_proj")

SUBLN_EPS: float = 1e-6

# marks a saved checkpoint as one whose state dict carries sub-norm gains
SUBLN_CONFIG_KEY: str = "ternary_subln"

# set on the live model once its norms exist, so a second insert does not read the
# marker this one stamped and try to restore gains that are already in memory
_INSERTED_ATTR: str = "_ternary_subln_inserted"

# export formats that cannot carry the extra per-module gain, and why
SUBLN_UNSUPPORTED_FORMATS: dict[str, str] = {
    "gguf_tq1_0": "the GGUF ternary archs have no tensor slot for it",
    "gguf_tq2_0": "the GGUF ternary archs have no tensor slot for it",
    "i2_s": "the bitnet.cpp arch has no tensor slot for it",
    "hf_bitnet": (
        "BitNetForCausalLM keeps its sub-norms as `attn_sub_norm` / `ffn_sub_norm` on "
        f"the attention and MLP blocks, and the packer does not map `<linear>."
        f"{SUBLN_ATTR}.weight` onto those names yet"
    ),
    "onebitllms_bf16": (
        "`replace_linear_with_bitnet_linear` swaps stock `nn.Linear` modules, so the "
        "sub-norm the swap inserted has nowhere to live in the finetuning model"
    ),
}


class TernarySubNorm(nn.Module):
    """RMSNorm over the input features of the linear it is attached to.

    Attributes:
        weight: Per-input-feature gain of shape `(num_features,)`, starting at ones.
    """

    weight: nn.Parameter

    def __init__(
        self,
        num_features: int,
        eps: float = SUBLN_EPS,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Build a unit-gain RMSNorm.

        Unit gain is the neutral starting point, not an identity: the norm still
        rescales every token to unit RMS. See the module docstring.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return `x / rms(x) * weight`, reduced over the last dimension in fp32."""
        input_dtype = x.dtype
        hidden = x.float()
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * hidden.to(input_dtype)

    def extra_repr(self) -> str:
        """Return the feature count and epsilon for `repr(model)`."""
        return f"{self.num_features}, eps={self.eps:g}"


def insert_subln(
    model: PreTrainedModel, cfg: DictDefault, target_names: Sequence[str]
) -> list[str]:
    """Attach a `TernarySubNorm` to every qualifying Linear about to be ternarized.

    Runs inside `swap.convert_model` *before* the modules are replaced: the norm is
    set on the source `nn.Linear` as `SUBLN_ATTR`, and `TernaryLinear.from_linear`
    adopts it as its own child. Idempotent, so a repeated call does not stack norms.

    When `model` was loaded from a checkpoint that already carries sub-norms (its
    config holds `SUBLN_CONFIG_KEY`), the gains transformers dropped as unexpected
    keys are read back out of that checkpoint's shards here — this is the only point
    at which a subln master becomes the model it was saved as.

    Args:
        model: The model being converted, still holding `nn.Linear` targets.
        cfg: The axolotl config; `ternary.*` decides which families qualify.
        target_names: Names of the Linears the swap is about to replace, in
            `named_modules()` order.

    Returns:
        The qualified names of the inserted norms (`<linear name>.sub_norm`), in the
        order they were inserted, for `manifest.subln_modules`.

    Raises:
        ValueError: If `ternary.subln` is off, a name in `target_names` is not a
            Linear on `model`, a `SUBLN_FAMILIES` member matches nothing, or a marked
            checkpoint's saved gains cannot be restored.
    """
    ternary_cfg = resolve_ternary_config(cfg)
    if not ternary_cfg.subln:
        raise ValueError(
            "insert_subln runs only under ternary.subln: true; otherwise neither the "
            "manifest nor the model config records the norms and the master it saves "
            "cannot be reloaded"
        )

    qualified: list[tuple[str, nn.Linear]] = []
    for name in target_names:
        linear = _target_linear(model, name)
        if module_family(name) in SUBLN_FAMILIES:
            qualified.append((name, linear))

    covered = {module_family(name) for name, _ in qualified}
    absent = [family for family in SUBLN_FAMILIES if family not in covered]
    if absent:
        seen = sorted({module_family(name) for name in target_names})
        raise ValueError(
            f"ternary.subln normalizes the {list(SUBLN_FAMILIES)} inputs, but {absent} "
            f"match none of the converted modules (families: {seen}); this "
            "architecture names its output/down projections differently, or "
            "ternary.target_modules does not cover them. Set ternary.subln: false"
        )

    inserted: list[str] = []
    for name, linear in qualified:
        existing = getattr(linear, SUBLN_ATTR, None)
        if existing is None:
            setattr(
                linear,
                SUBLN_ATTR,
                TernarySubNorm(
                    linear.in_features,
                    device=linear.weight.device,
                    dtype=linear.weight.dtype,
                ),
            )
        elif getattr(existing, "num_features", None) != linear.in_features:
            raise ValueError(
                f"{name} already carries a sub-norm over "
                f"{getattr(existing, 'num_features', '?')} features, but the linear "
                f"takes {linear.in_features}"
            )
        inserted.append(f"{name}.{SUBLN_ATTR}")

    restored = 0
    if not getattr(model, _INSERTED_ATTR, False):
        restored = _restore_saved_gains(model, cfg, inserted)
    config = getattr(model, "config", None)
    if config is not None:
        setattr(config, SUBLN_CONFIG_KEY, True)
    setattr(model, _INSERTED_ATTR, True)
    LOG.info(
        f"ternary: inserted {len(inserted)} sub-norms before the "
        f"{list(SUBLN_FAMILIES)} projections"
        + (f", restoring {restored} saved gains" if restored else "")
    )
    return inserted


def has_subln_marker(config: PretrainedConfig | None) -> bool:
    """Whether `config` came from a checkpoint saved with sub-norms."""
    return bool(getattr(config, SUBLN_CONFIG_KEY, False))


def assert_subln_reloadable(model: PreTrainedModel, cfg: DictDefault) -> None:
    """Refuse a sub-norm checkpoint that this run would load without its sub-norms.

    `from_pretrained` reports the saved `<linear>.sub_norm.weight` tensors as
    unexpected and drops them, so the model in memory is quietly a different one from
    the checkpoint it claims to be.

    Raises:
        ValueError: If the checkpoint is marked but `ternary.subln` is off.
    """
    if not has_subln_marker(getattr(model, "config", None)):
        return
    if resolve_ternary_config(cfg).subln:
        return
    raise ValueError(
        "this checkpoint was converted with ternary.subln: true, so its state dict "
        f"holds `<linear>.{SUBLN_ATTR}.weight` gains the plain architecture has no "
        "modules for — transformers dropped them as unexpected keys while loading it. "
        "Set ternary.subln: true to rebuild and restore them, or start from the "
        "original full-precision checkpoint"
    )


def assert_subln_exportable(manifest: SwapManifest, fmt: str) -> None:
    """Refuse an export format that would silently drop the sub-norm gains.

    Raises:
        ValueError: If `manifest.subln` is set and `fmt` cannot carry the gains.
    """
    if not manifest.subln:
        return
    reason = SUBLN_UNSUPPORTED_FORMATS.get(fmt)
    if reason is None:
        return
    raise ValueError(
        f"ternary.subln inserted {len(manifest.subln_modules)} sub-norms and {fmt} "
        f"cannot carry them: {reason}. Export master_bf16, or convert without "
        "ternary.subln"
    )


def _target_linear(model: PreTrainedModel, name: str) -> nn.Linear:
    try:
        module = model.get_submodule(name)
    except AttributeError as exc:
        raise ValueError(
            f"{name} is not a module of the model being converted"
        ) from exc
    if not isinstance(module, nn.Linear):
        raise ValueError(
            "ternary.subln expects the enumerated targets to still be Linears, but "
            f"{name} is a {type(module).__name__}; sub-norms go in before the swap"
        )
    return module


def _restore_saved_gains(
    model: PreTrainedModel, cfg: DictDefault, inserted: Sequence[str]
) -> int:
    """Copy the gains of a marked checkpoint back into the freshly built norms."""
    if not has_subln_marker(getattr(model, "config", None)):
        return 0

    from .export.bake import load_shard, shard_paths

    source = checkpoint_dir(model, cfg)
    if source is None:
        raise ValueError(
            "this checkpoint was converted with ternary.subln: true, but the local "
            "directory holding it could not be found, so its sub-norm gains cannot be "
            "restored; point `base_model` at the downloaded master directory"
        )
    wanted = {f"{name}.weight" for name in inserted}
    found: dict[str, torch.Tensor] = {}
    for path in shard_paths(source):
        tensors, _ = load_shard(path)
        found.update({key: tensors[key] for key in wanted & tensors.keys()})
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(
            f"{source} is marked as a ternary.subln checkpoint but {len(missing)} of "
            f"its sub-norm gains are missing from its shards (e.g. {missing[:3]}); the "
            "master and this run's ternary.target_modules disagree"
        )
    with torch.no_grad():
        for key, tensor in found.items():
            gain = model.get_parameter(key)
            gain.copy_(tensor.to(gain.dtype))
    return len(found)


def checkpoint_dir(model: PreTrainedModel, cfg: DictDefault) -> Path | None:
    """Return the local directory `model` was loaded from, if there is one."""
    candidates = [
        cfg.get("base_model") if hasattr(cfg, "get") else None,
        getattr(getattr(model, "config", None), "_name_or_path", None),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


__all__ = [
    "SUBLN_CONFIG_KEY",
    "SUBLN_EPS",
    "SUBLN_FAMILIES",
    "SUBLN_UNSUPPORTED_FORMATS",
    "TernarySubNorm",
    "assert_subln_exportable",
    "assert_subln_reloadable",
    "checkpoint_dir",
    "has_subln_marker",
    "insert_subln",
]
