"""Decide whether a saved LoRA adapter maps onto a pinned base checkpoint."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .pinned_base import safetensors_layout

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
PREFIX = "base_model.model."
UNSUPPORTED_OPTIONS = (
    "use_dora",
    "modules_to_save",
    "trainable_token_indices",
    "layer_replication",
    "lora_bias",
    "alora_invocation_tokens",
    "arrow_config",
    "use_qalora",
)
# fused runtime parameter -> per-expert checkpoint projections it is stored as
EXPERT_PROJECTIONS = {
    "gate_up_proj": (("gate_proj", "up_proj"), ("w1", "w3")),
    "down_proj": (("down_proj",), ("w2",)),
}
_KEY_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])\.weight$")


@dataclass(frozen=True)
class AdapterTargets:
    """What a merge of this adapter must apply, to detect silently dropped LoRA."""

    pairs: int
    linear: int
    expert_groups: int
    fused: int


def load_adapter_config(directory: Path) -> dict:
    return json.loads((directory / ADAPTER_CONFIG).read_text(encoding="utf-8"))


def validate_adapter_config(config: dict):
    if (
        config.get("peft_type") != "LORA"
        or any(config.get(key) for key in UNSUPPORTED_OPTIONS)
        or config.get("bias", "none") != "none"
    ):
        raise ValueError("Only standard LoRA linear and expert adapters are supported")
    if config.get("init_lora_weights", True) not in (True, False, "gaussian"):
        raise ValueError("LoRA publication requires an unchanged original base")


def _suffix_match(module: str, suffix: str) -> bool:
    return module == suffix or module.endswith("." + suffix)


def _expert_indices(layout, module: str, projection: str) -> set[int]:
    pattern = re.compile(
        re.escape(module) + r"\.(\d+)\." + re.escape(projection) + r"\.weight$"
    )
    return {int(m[1]) for key in layout if (m := pattern.match(key))}


def _check_expert_parameter(layout, module, parameter, num_experts) -> str:
    """Return ``fused`` or ``per_expert`` for a targeted expert parameter."""
    fused = layout.get(f"{module}.{parameter}")
    if fused is not None:
        if len(fused[1]) != 3:
            raise ValueError(f"Fused expert tensor {module}.{parameter} is not 3D")
        return "fused"
    for projections in EXPERT_PROJECTIONS.get(parameter, ()):
        indices = [_expert_indices(layout, module, proj) for proj in projections]
        if not indices[0]:
            continue
        expected = set(range(num_experts or len(indices[0])))
        if any(found != expected for found in indices):
            raise ValueError(
                f"Base experts under {module} do not cover experts 0..{len(expected) - 1}"
            )
        for proj in projections:
            for expert in expected:
                stem = f"{module}.{expert}.{proj}"
                dtype = layout[f"{stem}.weight"][0]
                if dtype != "uint8":
                    raise ValueError(
                        f"Fused expert LoRA over per-expert {dtype} experts ({stem}) is "
                        "not supported by the sharded merge"
                    )
                if any(
                    f"{stem}.{leaf}" not in layout
                    for leaf in ("weight_scale", "weight_scale_2")
                ):
                    raise ValueError(f"NVFP4 expert {stem} is missing its scales")
        return "per_expert"
    raise ValueError(
        f"LoRA target parameter {module}.{parameter} does not match a base tensor"
    )


def validate_adapter(
    directory: Path,
    layout: dict[str, tuple[str, list[int]]],
    *,
    num_experts: int | None = None,
) -> AdapterTargets:
    """Check every adapter tensor maps onto ``layout``; refuse DoRA and friends."""
    from axolotl.cli.utils.param_wrapper_merge import strip_base_layers

    directory = Path(directory)
    config = load_adapter_config(directory)
    validate_adapter_config(config)
    adapter = safetensors_layout(directory / ADAPTER_WEIGHTS)
    if not adapter:
        raise ValueError("Empty LoRA adapter")
    modules: dict[str, set[str]] = {}
    for key, (_, shape) in adapter.items():
        match = _KEY_RE.match(key.removeprefix(PREFIX))
        if not key.startswith(PREFIX) or match is None or len(shape) != 2:
            raise ValueError(f"Unsupported LoRA tensor: {key}")
        other = "B" if match["side"] == "A" else "A"
        if f"{PREFIX}{match['module']}.lora_{other}.weight" not in adapter:
            raise ValueError(f"Incomplete LoRA pair: {key}")
        modules.setdefault(strip_base_layers(match["module"]), set()).add(
            match["module"]
        )
    linear = expert_groups = fused = 0
    parameters = config.get("target_parameters") or []
    for module, wrappers in modules.items():
        base = layout.get(f"{module}.weight")
        if wrappers == {module} and base is not None and len(base[1]) == 2:
            dtype, (rows, columns) = base
            a = adapter[f"{PREFIX}{module}.lora_A.weight"][1]
            b = adapter[f"{PREFIX}{module}.lora_B.weight"][1]
            packed = 2 if dtype == "uint8" else 1
            if a[0] != b[1] or b[0] != rows or a[1] != columns * packed:
                raise ValueError(f"LoRA shape does not match base matrix {module}")
            linear += 1
            continue
        targeted = sorted(
            {
                target.rsplit(".", 1)[1]
                for target in parameters
                if "." in target and _suffix_match(module, target.rsplit(".", 1)[0])
            }
        )
        if not targeted:
            raise ValueError(f"LoRA tensor does not match a base matrix: {module}")
        if len(targeted) != len(wrappers):
            raise ValueError(
                f"{module} has {len(wrappers)} LoRA wrappers but target_parameters "
                f"names {len(targeted)} parameters"
            )
        for parameter in targeted:
            if _check_expert_parameter(layout, module, parameter, num_experts) == (
                "fused"
            ):
                fused += 1
            else:
                expert_groups += 1
    return AdapterTargets(
        pairs=sum(len(wrappers) for wrappers in modules.values()),
        linear=linear,
        expert_groups=expert_groups,
        fused=fused,
    )
