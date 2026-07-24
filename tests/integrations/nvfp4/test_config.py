"""CPU-runnable tests for the NVFP4 plugin: nested config schema, the flat
compat bridge, plugin registration/merge, and the CPU-only swap helpers.

These need no GPU — they exercise exactly the logic CI can run, which is where
the plugin's coverage should live (the FP4 GEMM kernels are GPU-gated elsewhere).
"""

import pytest
import torch.nn as nn
from pydantic import ValidationError

from axolotl.integrations.nvfp4 import NVFP4Args, NVFP4Plugin, NVFP4PluginArgs


# ----------------------------- nested schema ---------------------------------
def test_defaults():
    a = NVFP4Args()  # enabled unset -> the empty-quantize guard doesn't fire
    assert a.quantize == []
    assert a.cross_entropy.mode == "off" and a.cross_entropy.vocab_block == 4096
    assert a.recipe.stochastic_rounding and a.recipe.hadamard
    assert a.base is None and a.save_packed is False


def test_extra_forbidden_rejects_stale_or_removed_keys():
    # exclude_modules and fp8_lm_head were removed; the old flat keys never existed
    # on the nested schema. All must be rejected.
    for stale in (
        "quantize_lm_head",
        "stochastic_rounding",
        "skip_first_n_blocks",
        "fp8_lm_head",
        "exclude_modules",
        "target_modules",
    ):
        with pytest.raises(ValidationError):
            NVFP4Args(**{stale: [] if stale.endswith("modules") else True})


# ----------------------- the single `quantize` knob --------------------------
def test_quantize_is_pure_opt_in():
    # empty + enabled -> hard error (would quantize nothing)
    with pytest.raises(ValidationError):
        NVFP4Args(enabled=True, quantize=[])
    with pytest.raises(ValidationError):
        NVFP4Args(enabled=True)


def test_quantize_all():
    a = NVFP4Args(enabled=True, quantize=["all"])
    assert a.quantize_all_body is True
    assert a.quantize_body_fragments == ()
    assert not (a.quantize_lm_head or a.quantize_embeddings)


def test_quantize_body_fragments_only():
    a = NVFP4Args(enabled=True, quantize=["q_proj", "k_proj", "v_proj"])
    assert a.quantize_all_body is False
    assert a.quantize_body_fragments == ("q_proj", "k_proj", "v_proj")
    assert not a.quantize_lm_head


def test_quantize_keywords_and_all_combined():
    a = NVFP4Args(enabled=True, quantize=["all", "lm_head", "embeddings"])
    assert a.quantize_all_body and a.quantize_lm_head and a.quantize_embeddings
    assert a.quantize_vision_tower is False
    assert a.quantize_body_fragments == ()  # keywords excluded from body fragments


def test_quantize_keyword_only_no_body():
    a = NVFP4Args(enabled=True, quantize=["lm_head"])
    assert a.quantize_lm_head is True
    assert a.quantize_all_body is False and a.quantize_body_fragments == ()


# ----------------------------- compat bridge ---------------------------------
def test_bridge_recipe_and_base():
    a = NVFP4Args(base="storage", recipe={"stochastic_rounding": False})
    assert a.stochastic_rounding is False and a.hadamard is True
    assert a.base_mode == "storage" and a.quantize_base is True
    assert NVFP4Args(base="hp").quantize_base is False
    assert NVFP4Args().quantize_base is False


def test_bridge_cross_entropy():
    a = NVFP4Args(
        quantize=["all", "lm_head"],
        cross_entropy={"mode": "auto", "vocab_block": 8192},
    )
    assert a.lm_head_cross_entropy == "auto"
    assert a.fused_ce_vocab_block == 8192
    assert a.fp4_cross_entropy_active is True  # auto + lm_head quantized
    assert NVFP4Args().fp4_cross_entropy_active is False


def test_bridge_keep_hp_blocks_explicit_and_paper():
    a = NVFP4Args(keep_hp_blocks={"first": 1, "last": 8})
    assert a.skip_first_n_blocks == 1 and a.skip_last_n_blocks == 8
    assert a.keep_hp_paper_preset is False
    p = NVFP4Args(keep_hp_blocks="paper")
    assert p.keep_hp_paper_preset is True and p.skip_first_n_blocks == 1


def test_bridge_save():
    a = NVFP4Args(save_packed=True)
    assert a.save_nvfp4 is True


# ------------------------- plugin registration -------------------------------
def test_plugin_get_input_args():
    assert NVFP4Plugin().get_input_args() == (
        "axolotl.integrations.nvfp4.NVFP4PluginArgs"
    )


def test_plugin_enabled_gate():
    from axolotl.utils.dict import DictDefault

    on = NVFP4Args(enabled=True, quantize=["all"])
    assert NVFP4Plugin()._enabled(DictDefault({"nvfp4_training": on}))
    assert not NVFP4Plugin()._enabled(DictDefault({}))
    assert not NVFP4Plugin()._enabled(
        DictDefault({"nvfp4_training": NVFP4Args(enabled=None)})
    )


def test_merge_folds_into_input_config():
    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.config import merge_input_args

    manager = PluginManager.get_instance()
    saved_plugins = manager.plugins.copy()
    try:
        manager.register("axolotl.integrations.nvfp4.NVFP4Plugin")
        _, AxolotlInputConfig = merge_input_args()
        assert "nvfp4_training" in AxolotlInputConfig.model_fields
        base = dict(
            base_model="Qwen/Qwen3-1.7B",
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            datasets=[{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
            plugins=["axolotl.integrations.nvfp4.NVFP4Plugin"],
        )
        cfg = AxolotlInputConfig(
            **base,
            nvfp4_training={
                "enabled": True,
                "quantize": ["all", "lm_head"],
                "cross_entropy": {"mode": "auto"},
                "keep_hp_blocks": {"first": 1, "last": 8},
            },
        )
        assert (
            cfg.nvfp4_training.quantize_all_body and cfg.nvfp4_training.quantize_lm_head
        )
        assert cfg.nvfp4_training.skip_last_n_blocks == 8
    finally:
        manager.plugins = saved_plugins


def test_plugin_args_nested_validate():
    m = NVFP4PluginArgs(
        nvfp4_training={"enabled": True, "quantize": ["all"], "keep_hp_blocks": "paper"}
    )
    assert m.nvfp4_training.keep_hp_paper_preset


# --------------------------- CPU swap helpers --------------------------------
def test_warn_unfilled_gemms():
    """nvfp4 should warn (not error) when neither packing/padding is set, or when
    torch_compile is off — the two config states that make FP4 regress vs bf16."""
    from axolotl.integrations.nvfp4 import patches
    from axolotl.utils.dict import DictDefault

    calls = []
    orig = patches.LOG.warning
    patches.LOG.warning = lambda *a, **k: calls.append(a[0] if a else "")
    try:
        patches.warn_unfilled_gemms(
            DictDefault(
                {
                    "sample_packing": False,
                    "pad_to_sequence_len": False,
                    "torch_compile": False,
                }
            )
        )
        assert any("filled GEMMs" in c for c in calls)
        assert any("torch_compile" in c for c in calls)
        calls.clear()
        patches.warn_unfilled_gemms(
            DictDefault({"sample_packing": True, "torch_compile": True})
        )
        assert not calls  # packed + compiled -> silent
    finally:
        patches.LOG.warning = orig


def test_dynamo_flags_suppress_flash_attn_layer_idx_recompiles():
    """Regression: HF flash_attention_forward specializes on module.layer_idx and
    recompiles per layer (blows recompile_limit on >8-layer models). The plugin's
    pre_model_load must mark nn.Module ints unspecialized when flash-attn + nvfp4."""
    import torch

    from axolotl.integrations.nvfp4.patches import configure_dynamo_for_nvfp4
    from axolotl.utils.dict import DictDefault

    if not hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
        import pytest

        pytest.skip("this torch lacks allow_unspec_int_on_nn_module")
    saved = torch._dynamo.config.allow_unspec_int_on_nn_module
    torch._dynamo.config.allow_unspec_int_on_nn_module = False
    cfg = DictDefault(
        {
            "torch_compile": True,
            "attn_implementation": "flash_attention_2",
            "fsdp_config": None,
            "nvfp4_training": NVFP4Args(enabled=True, quantize=["all"]),
        }
    )
    try:
        configure_dynamo_for_nvfp4(cfg)
        assert torch._dynamo.config.allow_unspec_int_on_nn_module is True
    finally:
        torch._dynamo.config.allow_unspec_int_on_nn_module = saved


def test_swap_survives_dictdefault_roundtrip():
    """Regression: validate_config does NVFP4Args.model_dump() -> DictDefault, which
    drops the @property bridge. swap._as_args must rebuild NVFP4Args so the swap is
    not a silent no-op (quantize_all_body etc. must be truthy again)."""
    from axolotl.integrations.nvfp4 import swap
    from axolotl.utils.dict import DictDefault

    runtime = DictDefault(
        NVFP4Args(enabled=True, quantize=["all", "lm_head"]).model_dump(
            exclude_none=True
        )
    )
    # bridge is gone on the raw dict (the bug):
    assert not runtime.quantize_all_body
    # _as_args restores it -> the swap's want_body predicate is True again:
    a = swap._as_args(runtime)
    assert a.quantize_all_body is True and a.quantize_lm_head is True
    want_body = a.quantize_all_body or bool(a.quantize_body_fragments)
    assert want_body is True


def test_resolve_ce_mode_and_fp4_active():
    from axolotl.integrations.nvfp4 import swap

    assert swap.resolve_ce_mode(NVFP4Args()) == "off"
    a = NVFP4Args(quantize=["all", "lm_head"], cross_entropy={"mode": "auto"})
    assert swap.resolve_ce_mode(a) == "auto" and swap.fp4_ce_active(a) is True
    b = NVFP4Args(quantize=["all"], cross_entropy={"mode": "auto"})  # no lm_head
    assert swap.fp4_ce_active(b) is False


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)  # so named_modules yields "layers.<i>.proj"


class _TinyModel(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList(_Block() for _ in range(n_layers))


def test_block_exclusions():
    from axolotl.integrations.nvfp4 import swap

    model = _TinyModel(6)
    assert swap._block_exclusions(model, 0, 0) == ()
    frags = swap._block_exclusions(model, 1, 2)
    assert "layers.0." in frags and "layers.4." in frags and "layers.5." in frags
    assert "layers.2." not in frags


def test_resolve_keep_hp_counts_paper_preset():
    from axolotl.integrations.nvfp4 import swap

    model = _TinyModel(32)
    a = NVFP4Args(keep_hp_blocks="paper")
    first, last = swap._resolve_keep_hp_counts(a, model)
    assert first == 1 and last == round(0.13 * 32)  # == 4
    b = NVFP4Args(keep_hp_blocks={"first": 2, "last": 3})
    assert swap._resolve_keep_hp_counts(b, model) == (2, 3)
