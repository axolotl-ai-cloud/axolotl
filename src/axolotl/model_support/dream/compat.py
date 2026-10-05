"""Narrow Transformers-5.17 compatibility adapter for pinned Dream remote code."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from functools import wraps
from pathlib import Path
from typing import Any, cast

from . import _DREAM_REVISIONS

DREAM_REVISION = _DREAM_REVISIONS["Dream-org/Dream-v0-Instruct-7B"]
PINNED_MODEL_SOURCE_SHA256 = (
    "3166e789f0d69beb1f4fbdee2317953d1c1c3b9bc21479b109afc5dd059de5b3"
)
PINNED_GENERATION_SOURCE_SHA256 = (
    "7f8ad01484898946b3c9c9d5ebc85e5ac726061d0fa7b4a522c4b8110c13a921"
)


@__import__("torch").compile(fullgraph=True, dynamic=False)
def _dream_flex_attention(query, key, value, block_mask, kernel_options=None):
    from torch.nn.attention.flex_attention import flex_attention

    return flex_attention(
        query, key, value, block_mask=block_mask, kernel_options=kernel_options
    )


def legacy_default_rope_parameters(
    config: Any, device: Any = None, seq_len: int | None = None, **kwargs: Any
) -> tuple[Any, float]:
    """Transformers 4.46's default RoPE frequencies for Dream's legacy config."""
    del seq_len
    if kwargs:
        raise ValueError("Dream legacy default RoPE accepts config, not rope kwargs")
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_type = (
        rope_scaling.get("rope_type", rope_scaling.get("type"))
        if isinstance(rope_scaling, dict)
        else None
    )
    if rope_scaling is not None and rope_type != "default":
        raise ValueError("Dream adapter supports only the pinned default RoPE")
    if getattr(config, "partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("Dream adapter does not support partial rotary embeddings")
    import torch

    head_dim = config.hidden_size // config.num_attention_heads
    declared_head_dim = getattr(config, "head_dim", None)
    if declared_head_dim is not None and declared_head_dim != head_dim:
        raise ValueError(
            "Dream config head_dim disagrees with hidden_size / num_attention_heads"
        )
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("Dream default RoPE requires a positive even head dimension")
    inv_freq = 1.0 / (
        config.rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device)
            / head_dim
        )
    )
    return inv_freq, 1.0


def patch_generation_validate(generation_config_class: type[Any]) -> bool:
    if getattr(generation_config_class, "_axolotl_validate_517", False):
        return False
    original_validate = generation_config_class.validate

    def validate(
        self: Any,
        is_init: bool = False,
        strict: bool = False,
        user_set_attributes: Any = None,
    ) -> None:
        del strict, user_set_attributes
        return original_validate(self, is_init=is_init)

    generation_config_class.validate = validate
    generation_config_class._axolotl_validate_517 = True
    return True


def load_patched_dream_model_class(model_dir: Path) -> tuple[type[Any], dict[str, Any]]:
    """Load Dream's local class and add only its missing legacy RoPE key.

    The assignment replaces the remote module's imported mapping, leaving
    ``transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS`` unchanged.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    source = model_dir / "modeling_dream.py"
    generation_source = model_dir / "generation_utils.py"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha256 != PINNED_MODEL_SOURCE_SHA256:
        raise ValueError(
            "Dream model source hash does not match the audited pinned source"
        )
    generation_source_sha256 = hashlib.sha256(
        generation_source.read_bytes()
    ).hexdigest()
    if generation_source_sha256 != PINNED_GENERATION_SOURCE_SHA256:
        raise ValueError(
            "Dream generation source hash does not match the audited pinned source"
        )
    global_mapping_before = dict(ROPE_INIT_FUNCTIONS)

    model_class = get_class_from_dynamic_module(
        "modeling_dream.DreamModel",
        str(model_dir),
        local_files_only=True,
    )
    module = importlib.import_module(model_class.__module__)
    module.DreamPreTrainedModel._supports_flex_attn = True
    original_mapping = getattr(module, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(original_mapping, dict):
        raise ValueError("Dream source does not expose a RoPE initializer mapping")
    remote_mapping_replaced = False
    if "default" not in original_mapping:
        cast(Any, module).ROPE_INIT_FUNCTIONS = {
            **original_mapping,
            "default": legacy_default_rope_parameters,
        }
        remote_mapping_replaced = True
    if dict(ROPE_INIT_FUNCTIONS) != global_mapping_before:
        raise RuntimeError(
            "Dream compatibility adapter modified the global RoPE registry"
        )
    generation_module = importlib.import_module(
        f"{model_class.__module__.rsplit('.', 1)[0]}.generation_utils"
    )
    generation_config_class = generation_module.DreamGenerationConfig
    generation_validate_replaced = patch_generation_validate(generation_config_class)

    class MaskAwareDream(model_class):  # type: ignore[valid-type, misc]
        _supports_flex_attn = True

        def __init__(self, config):
            super().__init__(config)
            enable_dream_block_mask(self)

        @wraps(model_class.forward)
        def forward(self, *args, **kwargs):
            kernel_options = kwargs.pop("kernel_options", None)
            for layer in self.model.layers:
                layer.self_attn._axolotl_kernel_options = kernel_options
            return super().forward(*args, **kwargs)

    MaskAwareDream.__name__ = model_class.__name__
    return MaskAwareDream, {
        "adapter_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "dream_revision": DREAM_REVISION,
        "remote_model_source_sha256": source_sha256,
        "remote_generation_source_sha256": generation_source_sha256,
        "model_class": model_class.__name__,
        "module": model_class.__module__,
        "remote_mapping_replaced": remote_mapping_replaced,
        "global_rope_registry_unchanged": dict(ROPE_INIT_FUNCTIONS)
        == global_mapping_before,
        "generation_validate_replaced": generation_validate_replaced,
        "forward_signature": str(inspect.signature(model_class.forward)),
    }


def enable_dream_block_mask(model: Any) -> None:
    """Route Flex BlockMask inputs through sparse FlexAttention per instance."""
    attention_type = type(model.model.layers[0].self_attn)
    source = importlib.import_module(attention_type.__module__)

    class MaskAwareAttention(attention_type):  # type: ignore[valid-type, misc]
        def forward(self, *args, **kwargs):
            mask = kwargs.get("attention_mask")
            if type(mask).__name__ != "BlockMask":
                return super().forward(*args, **kwargs)
            if kwargs.get("past_key_value") is not None:
                raise ValueError("Dream sparse attention does not support KV caching.")
            if self.training and getattr(self, "attention_dropout", 0.0):
                raise ValueError("Dream sparse attention requires attention_dropout=0.")
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                hidden_states = args[0]
            position_ids = kwargs.get("position_ids")
            position_embeddings = kwargs.get("position_embeddings")
            bsz, q_len, _ = hidden_states.size()
            query_states = (
                self.q_proj(hidden_states)
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            key_states = (
                self.k_proj(hidden_states)
                .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            value_states = (
                self.v_proj(hidden_states)
                .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            if position_embeddings is None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = source.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            key_states = source.repeat_kv(key_states, self.num_key_value_groups)
            value_states = source.repeat_kv(value_states, self.num_key_value_groups)
            output = _dream_flex_attention(
                query_states,
                key_states,
                value_states,
                mask,
                getattr(self, "_axolotl_kernel_options", None),
            )
            output = (
                output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
            )
            return self.o_proj(output), None, kwargs.get("past_key_value")

    for layer in model.model.layers:
        attention = layer.self_attn
        replacement = MaskAwareAttention(attention.config, attention.layer_idx)
        replacement.load_state_dict(attention.state_dict())
        layer.self_attn = replacement


def resolve_patched_dream_model_class(
    model_source: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> type[Any]:
    """Resolve audited Dream code from the requested local path or pinned Hub source."""

    source = Path(model_source)
    if not (source / "modeling_dream.py").is_file():
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                repo_id=str(model_source),
                revision=revision
                or _DREAM_REVISIONS.get(str(model_source), DREAM_REVISION),
                local_files_only=local_files_only,
            )
        )
    return load_patched_dream_model_class(source)[0]
