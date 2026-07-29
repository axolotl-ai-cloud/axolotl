"""Registry of the auxiliary modules a ternary conversion must not quantize by default.

Every architecture bolts modules onto the transformer stack that are not part of the
residual path a heal repairs: MoE routers, vision towers, modality projectors,
multi-token-prediction heads, speculative-decoding draft heads, reward/value heads.
Together they are a rounding error in parameter count, and when quantized they do not
fail as a slightly worse loss — a router flips a near-tie and the expert mix drifts, a
draft head's acceptance rate collapses, an MTP head takes damage that no loss in the
run has any gradient to repair.

This module is the one place that knows their names. The schema's canary warning, the
strict-enumeration error, and the architecture presets all read from here, so a family
is described once and every seam learns about it.

Patterns are prefix-tolerant (`(.*\\.)?…`): the same module is named differently
depending on how deep in the tree it is loaded from — the qwen3_5 tower is
`model.visual.*` in the VL tree and `visual.*` when the tower is loaded alone.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from dataclasses import dataclass

# module names spelled out per family in an error before it truncates
_MAX_REPORTED: int = 5


@dataclass(frozen=True)
class AuxModuleFamily:
    """One family of auxiliary modules sharing a name shape and a keep-FP verdict."""

    key: str
    label: str
    patterns: tuple[str, ...]
    rationale: str
    keep_fp: bool = True
    best_effort: bool = False
    canaries: tuple[str, ...] = ()

    def matching_pattern(self, name: str) -> str | None:
        """Return the pattern of this family that fullmatches `name`, or `None`."""
        for index, compiled in enumerate(_compiled(self.key)):
            if compiled.fullmatch(name):
                return self.patterns[index]
        return None


AUX_MODULE_FAMILIES: dict[str, AuxModuleFamily] = {
    "routers": AuxModuleFamily(
        key="routers",
        label="MoE router",
        patterns=(
            r"(.*\.)?mlp\.gate",
            r"(.*\.)?block_sparse_moe\.gate",
            r"(.*\.)?router(\..*)?",
            r"(.*\.)?gate\.wg",
        ),
        rationale=(
            "router logits pick experts through a discrete top-k, so rounding flips "
            "near-ties and drifts utilization for ~0.1% of the parameters"
        ),
        canaries=(
            "model.layers.0.mlp.gate",
            "model.layers.0.block_sparse_moe.gate",
        ),
    ),
    "vision_towers": AuxModuleFamily(
        key="vision_towers",
        label="vision tower",
        patterns=(
            r"(.*\.)?visual\..*",
            r"(.*\.)?vision_tower\..*",
            r"(.*\.)?vision_model\..*",
        ),
        rationale=(
            "towers are a small fraction of the parameters and quantize far worse "
            "than the LM (the Prism 4-bit precedent)"
        ),
        canaries=("model.visual.blocks.0.attn.qkv",),
    ),
    "projectors": AuxModuleFamily(
        key="projectors",
        label="modality projector",
        patterns=(
            r"(.*\.)?multi_modal_projector\..*",
            r"(.*\.)?mm_projector\..*",
            r".*\.merger\..*",
        ),
        rationale=(
            "the single bridge between two pretrained representations; its error is "
            "not re-normalized by any later block"
        ),
        canaries=("model.multi_modal_projector.linear_1",),
    ),
    "mtp": AuxModuleFamily(
        key="mtp",
        label="multi-token-prediction head",
        patterns=(
            r"(.*\.)?mtp\..*",
            r"(.*\.)?nextn\..*",
        ),
        rationale=(
            "an MTP head is trained by an auxiliary loss the heal usually does not "
            "run, so nothing produces a gradient to repair the damage"
        ),
        best_effort=True,
        canaries=("model.layers.0.mtp.up_proj",),
    ),
    "draft_heads": AuxModuleFamily(
        key="draft_heads",
        label="draft head",
        patterns=(
            r"(.*\.)?medusa_head.*",
            r"(.*\.)?eagle.*",
            r"(.*\.)?draft_model\..*",
        ),
        rationale=(
            "a speculative-decoding draft head is judged by acceptance against the "
            "target model, so its damage shows up as lost throughput, not as loss"
        ),
        canaries=("medusa_head.0.linear",),
    ),
    "value_heads": AuxModuleFamily(
        key="value_heads",
        label="value head",
        patterns=(
            r"(.*\.)?value_head(\..*)?",
            r"(.*\.)?v_head(\..*)?",
            r"(.*\.)?score(\..*)?",
        ),
        rationale=(
            "a scalar head whose outputs are only ever compared to each other, where "
            "rounding a single output row is pure noise on the comparison"
        ),
        canaries=("score",),
    ),
}

# module names a router regex would grab if it is written unanchored (`.*gate.*`)
ROUTER_CANARY_NAMES: tuple[str, ...] = AUX_MODULE_FAMILIES["routers"].canaries


@functools.lru_cache(maxsize=None)
def _compiled(key: str) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(pattern) for pattern in AUX_MODULE_FAMILIES[key].patterns)


def classify(name: str) -> AuxModuleFamily | None:
    """Return the first family whose patterns fullmatch `name`, or `None`."""
    for family in AUX_MODULE_FAMILIES.values():
        if family.matching_pattern(name) is not None:
            return family
    return None


def family_patterns(*keys: str) -> tuple[str, ...]:
    """Return the keep-FP regexes of the named families, for an arch preset to reuse.

    Raises:
        KeyError: If a key names no registered family.
    """
    return tuple(
        pattern for key in keys for pattern in AUX_MODULE_FAMILIES[key].patterns
    )


def group_by_family(names: Iterable[str]) -> dict[str, list[str]]:
    """Group `names` by the family that claims them, in registry order.

    Names that match no family are dropped; the caller already knows about them.
    """
    grouped: dict[str, list[str]] = {}
    for name in names:
        family = classify(name)
        if family is not None and family.keep_fp:
            grouped.setdefault(family.key, []).append(name)
    return {key: grouped[key] for key in AUX_MODULE_FAMILIES if key in grouped}


def suggest_keep_fp(names: Iterable[str]) -> list[str]:
    """Return the registry patterns that would keep every recognized name in FP.

    Only the patterns that actually matched are returned, so the suggestion is
    paste-ready rather than a copy of the whole family.
    """
    suggested: list[str] = []
    for key, matched in group_by_family(names).items():
        family = AUX_MODULE_FAMILIES[key]
        for name in matched:
            pattern = family.matching_pattern(name)
            if pattern is not None and pattern not in suggested:
                suggested.append(pattern)
    return suggested


def explain(names: Iterable[str]) -> str:
    """Return the error-message section naming the recognized families, or `""`.

    Each family gets its rationale, the names it claimed, and the exact regexes to
    paste under `ternary.keep_fp_modules`.
    """
    grouped = group_by_family(names)
    if not grouped:
        return ""

    lines = [
        f"{sum(len(matched) for matched in grouped.values())} of them are known "
        "auxiliary modules that ternary conversion keeps full precision:"
    ]
    for key, matched in grouped.items():
        family = AUX_MODULE_FAMILIES[key]
        head = ", ".join(matched[:_MAX_REPORTED])
        extra = len(matched) - _MAX_REPORTED
        listed = f"{head} (+{extra} more)" if extra > 0 else head
        lines.append(f"  {family.label}: {listed}")
        lines.append(f"    {family.rationale}")
        if family.best_effort:
            lines.append(
                "    name-based detection only — this family is spelled out in the "
                "model config on some architectures and is indistinguishable from an "
                "ordinary block on others"
            )
    lines.append("")
    lines.append("ternary:")
    lines.append("  keep_fp_modules:")
    lines.extend(f"    - {pattern}" for pattern in suggest_keep_fp(names))
    return "\n".join(lines)


__all__ = [
    "AUX_MODULE_FAMILIES",
    "ROUTER_CANARY_NAMES",
    "AuxModuleFamily",
    "classify",
    "explain",
    "family_patterns",
    "group_by_family",
    "suggest_keep_fp",
]
