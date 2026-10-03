"""Decision plugin configuration and registration tests."""

from collections import OrderedDict

import pytest
from pydantic import ValidationError

from axolotl.integrations.decision.args import (
    DecisionArgs,
    DecisionConfig,
)
from axolotl.integrations.decision.plugin import DecisionPlugin
from axolotl.integrations.diffusion.plugin import DiffusionPlugin

PLUGIN_PATH = "axolotl.integrations.decision.DecisionPlugin"
DIFFUSION_PLUGIN_PATH = "axolotl.integrations.diffusion.DiffusionPlugin"


def test_plugin_exposes_optional_nested_config():
    assert (
        DecisionPlugin().get_input_args()
        == "axolotl.integrations.decision.args.DecisionArgs"
    )
    assert DecisionArgs().decision is None


def test_config_preserves_no_slot_decision_layout():
    config = DecisionConfig.model_validate(
        {"layout": "thought_block", "latent": {"mode": "none"}}
    )
    assert config.layout == "thought_block"
    assert config.latent.mode == "none"


@pytest.mark.parametrize(
    "block,message",
    [
        ({"latent": {"mode": "learned"}}, "latent slots were removed"),
        ({"layout": "prompt_slots"}, "prompt-slot layout was removed"),
        ({"labels": {"full_ce_weighting": "dft"}}, "full_ce_weighting=dft was removed"),
        ({"eval": {"steps": [1, 1]}}, "unique"),
        ({"mixture": {"weights": {"source": 0.0}}}, "finite and positive"),
        (
            {"mixture": {"weights": {"source": 1.0}, "temperature": 0.5}},
            "explicit weights or temperature",
        ),
        ({"mixture": {"max_examples_per_source": {"source": 0}}}, "positive caps"),
        ({"labels": {"brier_weight": float("inf")}}, "finite"),
        ({"read_fraction": float("inf")}, "finite"),
    ],
)
def test_config_rejects_invalid_semantic_controls(block, message):
    with pytest.raises(ValidationError, match=message):
        DecisionConfig.model_validate(block)


def test_config_accepts_per_source_caps_without_numeric_union_comparison():
    config = DecisionConfig.model_validate(
        {"mixture": {"max_examples_per_source": {"source": 2}}}
    )
    assert config.mixture.max_examples_per_source == {"source": 2}


def test_labels_reject_removed_dft_weighting():
    with pytest.raises(ValidationError, match="full_ce_weighting=dft was removed"):
        DecisionConfig.model_validate({"labels": {"full_ce_weighting": "dft"}})


def test_labels_accept_hard_label_smoothing_only_inside_unit_interval():
    assert (
        DecisionConfig.model_validate(
            {"labels": {"hard_label_smoothing": 0.1}}
        ).labels.hard_label_smoothing
        == 0.1
    )
    with pytest.raises(ValidationError):
        DecisionConfig.model_validate({"labels": {"hard_label_smoothing": 1.0}})


def test_plugin_input_args_merge_into_config_schema(min_base_cfg, monkeypatch):
    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.config import merge_input_args

    manager = PluginManager.get_instance()
    monkeypatch.setattr(
        manager,
        "plugins",
        OrderedDict({PLUGIN_PATH: DecisionPlugin()}),
    )
    _capabilities, config_cls = merge_input_args()
    config = config_cls(
        **(
            min_base_cfg
            | {
                "plugins": [PLUGIN_PATH],
                "decision": {
                    "layout": "thought_block",
                    "latent": {"mode": "none"},
                },
            }
        )
    )
    assert config.decision.layout == "thought_block"
    assert config.decision.latent.mode == "none"


def test_native_registration_accepts_thought_block():
    plugin = DecisionPlugin()
    plugin.register(
        {
            "diffusion": {"from_causal_lm": False},
            "decision": {"layout": "thought_block"},
        }
    )


def test_registration_rejects_legacy_diffusion_compatibility():
    with pytest.raises(ValueError, match="native diffusion"):
        DecisionPlugin().register({"decision": {}})
    with pytest.raises(ValueError, match="from_causal_lm"):
        DecisionPlugin().register(
            {
                "diffusion": {"from_causal_lm": True},
                "decision": {},
            }
        )


def test_registration_allows_sample_packing_with_top_level_budget():
    DecisionPlugin().register(
        {
            "sample_packing": True,
            "micro_batch_size": 2,
            "sequence_len": 2048,
            "diffusion": {"from_causal_lm": False},
            "decision": {},
        }
    )


def test_sample_packing_passes_sequence_len_times_micro_batch_capacity():
    from types import SimpleNamespace

    from axolotl.integrations.decision.training_collator import (
        decision_collator_for_config,
    )
    from axolotl.utils.dict import DictDefault

    _collator, kwargs = decision_collator_for_config(
        DictDefault(
            {
                "model_config_type": "nemotron_labs_diffusion",
                "tokenizer": SimpleNamespace(pad_token_id=0),
                "sample_packing": True,
                "micro_batch_size": 3,
                "sequence_len": 128,
                "diffusion": {"from_causal_lm": False},
            }
        )
    )
    assert kwargs["physical_payload_capacity"] == 384


def test_eval_packing_uses_the_training_physical_budget():
    from types import SimpleNamespace

    from axolotl.integrations.decision.training_collator import (
        decision_collator_for_config,
    )
    from axolotl.utils.dict import DictDefault

    _collator, kwargs = decision_collator_for_config(
        DictDefault(
            {
                "model_config_type": "nemotron_labs_diffusion",
                "tokenizer": SimpleNamespace(pad_token_id=0),
                "sample_packing": True,
                "eval_sample_packing": True,
                "micro_batch_size": 16,
                "eval_batch_size": 8,
                "sequence_len": 128,
                "diffusion": {"from_causal_lm": False},
            }
        ),
        is_eval=True,
    )
    assert kwargs["physical_payload_capacity"] == 2048


def test_fixed_logical_eval_uses_the_eval_physical_budget():
    from types import SimpleNamespace

    from axolotl.integrations.decision.training_collator import (
        decision_collator_for_config,
    )
    from axolotl.utils.dict import DictDefault

    _collator, kwargs = decision_collator_for_config(
        DictDefault(
            {
                "model_config_type": "nemotron_labs_diffusion",
                "tokenizer": SimpleNamespace(pad_token_id=0),
                "sample_packing": False,
                "batch_flattening": True,
                "micro_batch_size": 16,
                "eval_batch_size": 8,
                "sequence_len": 128,
                "diffusion": {"from_causal_lm": False},
            }
        ),
        is_eval=True,
    )
    assert kwargs["physical_payload_capacity"] == 1024


def test_registration_rejects_explicit_unflattened_microbatches():
    config = {
        "diffusion": {"from_causal_lm": False},
        "decision": {},
        "sample_packing": False,
        "eval_sample_packing": True,
        "micro_batch_size": 8,
    }
    config["batch_flattening"] = False
    with pytest.raises(ValueError, match="require batch_flattening: true"):
        DecisionPlugin().register(config)


def test_fixed_logical_batch_requires_capacity():
    config = {
        "diffusion": {
            "from_causal_lm": False,
        },
        "decision": {},
        "batch_flattening": True,
        "sample_packing": False,
        "micro_batch_size": 8,
        "sequence_len": 2048,
    }
    DecisionPlugin().register(config)


def test_fixed_logical_batch_passes_physical_capacity_to_decision_collator():
    from types import SimpleNamespace

    from axolotl.integrations.decision.training_collator import (
        decision_collator_for_config,
    )
    from axolotl.utils.dict import DictDefault

    _collator, kwargs = decision_collator_for_config(
        DictDefault(
            {
                "model_config_type": "nemotron_labs_diffusion",
                "tokenizer": SimpleNamespace(pad_token_id=0),
                "sample_packing": False,
                "batch_flattening": True,
                "micro_batch_size": 8,
                "sequence_len": 2048,
                "diffusion": {
                    "from_causal_lm": False,
                    "canvas_width": 128,
                },
            }
        )
    )
    assert kwargs["physical_payload_capacity"] == 16384


@pytest.mark.parametrize("diffusion_first", [True, False])
def test_native_nemotron_plugin_routes_through_causal_builder(
    min_base_cfg, monkeypatch, diffusion_first
):
    """Exercise PluginManager and the causal builder with the resolved native profile."""
    from types import SimpleNamespace

    from axolotl.core.builders.causal import HFCausalTrainerBuilder
    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.decision.trainer import (
        DecisionTrainer,
    )
    from axolotl.integrations.decision.training_collator import (
        DecisionTrainingCollator,
    )
    from axolotl.model_support import (
        DiffusionLayout,
        MaskTokenPolicy,
        get_model_support_for_cfg,
        resolve_model_support,
    )
    from axolotl.utils.config import validate_config
    from axolotl.utils.dict import DictDefault

    manager = PluginManager.get_instance()
    plugins = [
        (DIFFUSION_PLUGIN_PATH, DiffusionPlugin()),
        (PLUGIN_PATH, DecisionPlugin()),
    ]
    if not diffusion_first:
        plugins.reverse()
    monkeypatch.setattr(manager, "plugins", OrderedDict(plugins))
    cfg = validate_config(
        DictDefault(
            min_base_cfg
            | {
                "plugins": [name for name, _ in plugins],
                "attn_implementation": "varlen",
                "diffusion": {
                    "from_causal_lm": False,
                    "canvas_width": 128,
                    "mask_token_id": 100,
                },
                "decision": {"layout": "thought_block"},
            }
        )
    )
    cfg.model_config_type = "nemotron_labs_diffusion"
    cfg.tokenizer = SimpleNamespace(pad_token_id=0)

    profile = resolve_model_support(get_model_support_for_cfg(cfg))
    assert profile is not None
    assert profile.diffusion is not None
    assert profile.diffusion.layout is DiffusionLayout.FULL_SEQUENCE
    assert profile.diffusion.mask_token_policy is MaskTokenPolicy.MODEL
    assert cfg.diffusion.mask_token_id == 100

    builder = HFCausalTrainerBuilder.__new__(HFCausalTrainerBuilder)
    builder.cfg = cfg
    builder.tokenizer = cfg.tokenizer
    assert builder._get_trainer_cls() is DecisionTrainer

    training_args = SimpleNamespace(
        pretraining=False,
        sample_packing=True,
        eval_sample_packing=False,
    )
    collator = builder.build_collator(
        training_args,
        padding=True,
        pad_to_multiple_of=8192,
    )
    eval_collator = builder.build_collator(
        training_args,
        is_eval=True,
        padding=True,
        pad_to_multiple_of=8192,
    )
    for candidate in (collator, eval_collator):
        assert isinstance(candidate, DecisionTrainingCollator)
        assert candidate.spec is profile.diffusion
        assert candidate.pad_token_id == 0


def test_config_rejects_removed_sampled_slot_policy():
    with pytest.raises(ValidationError, match="latent slots were removed"):
        DecisionConfig.model_validate(
            {"latent": {"mode": "pinned", "num_slots": 8, "sample_num_slots": True}}
        )
