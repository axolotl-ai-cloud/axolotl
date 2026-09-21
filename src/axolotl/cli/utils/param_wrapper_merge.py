"""Resolve saved expert adapters from PEFT wrapper identities on a meta model."""

import copy
import re
from dataclasses import dataclass

import torch
from packaging.version import Version
from peft import LoraConfig, __version__ as peft_version, get_peft_model
from peft.tuners.lora.layer import ParamWrapper

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass(frozen=True)
class ParamWrapperTarget:
    """A saved adapter's exact base parameter and merge convention."""

    a_key: str
    b_key: str
    shape: tuple[int, ...]
    is_transposed: bool
    alpha: float


def _reject_ambiguous_adapters(state: dict[str, torch.Tensor]) -> None:
    """Refuse a checkpoint in which shape matching alone could pick the wrong adapter."""
    seen = set()
    for key, a in state.items():
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key.removesuffix(".lora_A.weight")
        b = state.get(prefix + ".lora_B.weight")
        if b is None or a.ndim != 2 or b.ndim != 2:
            continue
        parent = re.sub(r"(?:\.base_layer)+$", "", prefix)
        signature = (parent, tuple(sorted((a.shape[1], b.shape[0]))))
        if signature in seen:
            raise ValueError(
                f"Ambiguous ParamWrapper adapters under {parent}; identity "
                "reconstruction from the base architecture is required to resolve them"
            )
        seen.add(signature)


def build_param_wrapper_map(
    model: torch.nn.Module | None,
    config_dict: dict,
    state: dict[str, torch.Tensor],
) -> dict[str, ParamWrapperTarget] | None:
    """Reconstruct adapter identities without allocating base weights or mutating the input."""
    if not config_dict.get("target_parameters"):
        _reject_ambiguous_adapters(state)
        return None
    if model is None:
        _reject_ambiguous_adapters(state)
        LOG.warning(
            "No base architecture is available to resolve ParamWrapper adapter identities; "
            "falling back to shape matching, which each adapter must resolve uniquely"
        )
        return None
    if any(not parameter.is_meta for parameter in model.parameters()):
        raise ValueError(
            "ParamWrapper identity reconstruction requires meta parameters"
        )
    recorded_version = config_dict.get("peft_version")
    if (
        recorded_version
        and Version(recorded_version).base_version != Version(peft_version).base_version
    ):
        LOG.warning(
            "Adapter was saved by PEFT %s and is being reconstructed with PEFT %s; "
            "a wrapper layout change between them fails the identity checks below",
            recorded_version,
            peft_version,
        )

    model = copy.deepcopy(model)
    original_names = {id(module): name for name, module in model.named_modules()}
    config = LoraConfig.from_peft_type(**copy.deepcopy(config_dict))
    config.init_lora_weights = False
    model = get_peft_model(model, config, low_cpu_mem_usage=True)
    targets = {}
    claimed = set()
    for name, module in model.named_modules():
        if not isinstance(module, ParamWrapper):
            continue
        base = module.get_base_layer()
        if id(base) not in original_names:
            raise ValueError(f"Cannot resolve original module for ParamWrapper {name}")
        parameter_name = f"{original_names[id(base)]}.{module.parameter_name}".lstrip(
            "."
        )
        a_key, b_key = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
        if parameter_name in targets or a_key in claimed or b_key in claimed:
            raise ValueError(f"Duplicate ParamWrapper mapping for {parameter_name}")
        for key, expected in (
            (a_key, module.lora_A["default"].weight),
            (b_key, module.lora_B["default"].weight),
        ):
            if key not in state:
                raise ValueError(f"Missing ParamWrapper adapter tensor {key}")
            if tuple(state[key].shape) != tuple(expected.shape):
                raise ValueError(f"ParamWrapper adapter shape mismatch for {key}")
            claimed.add(key)
        targets[parameter_name] = ParamWrapperTarget(
            a_key=a_key,
            b_key=b_key,
            shape=tuple(module.get_param().shape),
            is_transposed=bool(getattr(base, "is_transposed", False)),
            alpha=module.lora_alpha["default"],
        )
    if not targets:
        raise ValueError("No ParamWrapper identities were reconstructed")

    # Normal LoRA modules may share the checkpoint, but every A/B key must belong to an injected layer.
    expected_keys = {
        f"{name}.{factor}.weight"
        for name, module in model.named_modules()
        for factor in ("lora_A", "lora_B")
        if "default" in getattr(module, factor, {})
    }
    unexpected = {
        key for key in state if key.endswith((".lora_A.weight", ".lora_B.weight"))
    } - expected_keys
    if unexpected:
        raise ValueError(f"Unresolved adapter keys: {sorted(unexpected)}")
    return targets
