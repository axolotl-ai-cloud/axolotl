"""Decision plugin configuration and registration tests."""

from collections import OrderedDict

import pytest
from pydantic import ValidationError

from axolotl.integrations.diffusion_decision.args import (
    DiffusionDecisionArgs,
    DiffusionDecisionConfig,
)
from axolotl.integrations.diffusion_decision.plugin import DiffusionDecisionPlugin

PLUGIN_PATH = "axolotl.integrations.diffusion_decision.DiffusionDecisionPlugin"


def test_plugin_exposes_optional_nested_config():
    assert (
        DiffusionDecisionPlugin().get_input_args()
        == "axolotl.integrations.diffusion_decision.args.DiffusionDecisionArgs"
    )
    assert DiffusionDecisionArgs().diffusion_decision is None


@pytest.mark.parametrize("layout", ("thought_block", "prompt_slots"))
def test_config_preserves_both_decision_layouts(layout):
    config = DiffusionDecisionConfig.model_validate(
        {
            "layout": layout,
            "read_fraction": 0.4,
            "mixture": {"weights": {"one": 1.0}},
            "labels": {"label_softmax": "both", "brier_weight": 0.1},
            "latent": {"mode": "learned", "num_slots": 8},
        }
    )
    assert config.layout == layout
    assert config.latent.mode == "learned"


@pytest.mark.parametrize(
    "block,message",
    [
        ({"latent": {"mode": "none", "num_slots": 1}}, "mode=none"),
        ({"latent": {"token_ids": [-1]}}, "nonnegative"),
        ({"carry": {"enabled": True}}, "carry is not wired"),
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
        DiffusionDecisionConfig.model_validate(block)


def test_config_accepts_per_source_caps_without_numeric_union_comparison():
    config = DiffusionDecisionConfig.model_validate(
        {"mixture": {"max_examples_per_source": {"source": 2}}}
    )
    assert config.mixture.max_examples_per_source == {"source": 2}


def test_labels_default_to_ce_weighting_and_accept_dft_only():
    assert DiffusionDecisionConfig().labels.full_ce_weighting == "ce"
    assert (
        DiffusionDecisionConfig.model_validate(
            {"labels": {"full_ce_weighting": "dft"}}
        ).labels.full_ce_weighting
        == "dft"
    )
    with pytest.raises(ValidationError):
        DiffusionDecisionConfig.model_validate(
            {"labels": {"full_ce_weighting": "other"}}
        )


def test_labels_accept_hard_label_smoothing_only_inside_unit_interval():
    assert (
        DiffusionDecisionConfig.model_validate(
            {"labels": {"hard_label_smoothing": 0.1}}
        ).labels.hard_label_smoothing
        == 0.1
    )
    with pytest.raises(ValidationError):
        DiffusionDecisionConfig.model_validate(
            {"labels": {"hard_label_smoothing": 1.0}}
        )
    with pytest.raises(ValidationError, match="requires full_ce_weighting=ce"):
        DiffusionDecisionConfig.model_validate(
            {"labels": {"hard_label_smoothing": 0.1, "full_ce_weighting": "dft"}}
        )


def test_plugin_input_args_merge_into_config_schema(min_base_cfg, monkeypatch):
    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.config import merge_input_args

    manager = PluginManager.get_instance()
    monkeypatch.setattr(
        manager,
        "plugins",
        OrderedDict({PLUGIN_PATH: DiffusionDecisionPlugin()}),
    )
    _capabilities, config_cls = merge_input_args()
    config = config_cls(
        **(
            min_base_cfg
            | {
                "plugins": [PLUGIN_PATH],
                "diffusion_decision": {
                    "layout": "prompt_slots",
                    "latent": {"mode": "prompt", "num_slots": 4},
                },
            }
        )
    )
    assert config.diffusion_decision.layout == "prompt_slots"
    assert config.diffusion_decision.latent.num_slots == 4


@pytest.mark.parametrize("layout", ("thought_block", "prompt_slots"))
def test_native_registration_is_layout_agnostic(layout):
    plugin = DiffusionDecisionPlugin()
    plugin.register(
        {
            "diffusion_lm": {"from_causal_lm": False},
            "diffusion_decision": {"layout": layout},
        }
    )


def test_registration_rejects_legacy_diffusion_compatibility():
    with pytest.raises(ValueError, match="native diffusion_lm"):
        DiffusionDecisionPlugin().register({"diffusion_decision": {}})
    with pytest.raises(ValueError, match="from_causal_lm"):
        DiffusionDecisionPlugin().register(
            {
                "diffusion_lm": {"from_causal_lm": True},
                "diffusion_decision": {},
            }
        )


def test_registration_allows_sample_packing_with_top_level_budget():
    DiffusionDecisionPlugin().register(
        {
            "sample_packing": True,
            "micro_batch_size": 2,
            "sequence_len": 2048,
            "diffusion_lm": {"from_causal_lm": False},
            "diffusion_decision": {},
        }
    )


def test_sample_packing_passes_sequence_len_times_micro_batch_capacity():
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision.training_collator import (
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
                "diffusion_lm": {"from_causal_lm": False},
            }
        )
    )
    assert kwargs["physical_payload_capacity"] == 384


def test_eval_packing_uses_the_training_physical_budget():
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision.training_collator import (
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
                "diffusion_lm": {"from_causal_lm": False},
            }
        ),
        is_eval=True,
    )
    assert kwargs["physical_payload_capacity"] == 2048


def test_fixed_logical_eval_uses_the_eval_physical_budget():
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision.training_collator import (
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
                "diffusion_lm": {"from_causal_lm": False},
            }
        ),
        is_eval=True,
    )
    assert kwargs["physical_payload_capacity"] == 1024


def test_registration_rejects_explicit_unflattened_microbatches():
    config = {
        "diffusion_lm": {"from_causal_lm": False},
        "diffusion_decision": {},
        "sample_packing": False,
        "eval_sample_packing": True,
        "micro_batch_size": 8,
    }
    config["batch_flattening"] = False
    with pytest.raises(ValueError, match="require batch_flattening: true"):
        DiffusionDecisionPlugin().register(config)


def test_fixed_logical_batch_requires_capacity():
    config = {
        "diffusion_lm": {
            "from_causal_lm": False,
        },
        "diffusion_decision": {},
        "batch_flattening": True,
        "sample_packing": False,
        "micro_batch_size": 8,
        "sequence_len": 2048,
    }
    DiffusionDecisionPlugin().register(config)


def test_fixed_logical_batch_passes_physical_capacity_to_decision_collator():
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision.training_collator import (
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
                "diffusion_lm": {
                    "from_causal_lm": False,
                    "canvas_width": 128,
                },
            }
        )
    )
    assert kwargs["physical_payload_capacity"] == 16384


def test_native_nemotron_plugin_routes_through_causal_builder(
    min_base_cfg, monkeypatch
):
    """Exercise PluginManager and the causal builder with the resolved native profile."""
    from types import SimpleNamespace

    from axolotl.core.builders.causal import HFCausalTrainerBuilder
    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.diffusion_decision.trainer import (
        DiffusionDecisionTrainer,
    )
    from axolotl.integrations.diffusion_decision.training_collator import (
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
    monkeypatch.setattr(
        manager,
        "plugins",
        OrderedDict({PLUGIN_PATH: DiffusionDecisionPlugin()}),
    )
    cfg = validate_config(
        DictDefault(
            min_base_cfg
            | {
                "plugins": [PLUGIN_PATH],
                "attn_implementation": "varlen",
                "diffusion_lm": {
                    "from_causal_lm": False,
                    "canvas_width": 128,
                    "mask_token_id": 100,
                },
                "diffusion_decision": {"layout": "thought_block"},
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
    assert cfg.diffusion_lm.mask_token_id == 100

    builder = HFCausalTrainerBuilder.__new__(HFCausalTrainerBuilder)
    builder.cfg = cfg
    builder.tokenizer = cfg.tokenizer
    assert builder._get_trainer_cls() is DiffusionDecisionTrainer

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


def test_config_accepts_versioned_sampled_slot_policy():
    config = DiffusionDecisionConfig.model_validate(
        {"latent": {"mode": "pinned", "num_slots": 8, "sample_num_slots": True}}
    )

    assert config.latent.sample_num_slots is True
    assert config.latent.num_slots == 8


@pytest.mark.parametrize(
    "latent,message",
    [
        (
            {"mode": "none", "sample_num_slots": True},
            "sample_num_slots requires a non-none",
        ),
        (
            {"mode": "pinned", "sample_num_slots": True},
            "sample_num_slots requires num_slots",
        ),
    ],
)
def test_config_rejects_invalid_sampled_slot_bounds(latent, message):
    with pytest.raises(ValidationError, match=message):
        DiffusionDecisionConfig.model_validate({"latent": latent})


def test_post_lora_merge_ignores_manifests_for_non_decision_runs(tmp_path):
    from axolotl.integrations.diffusion_decision.manifest import MANIFEST_FILENAME
    from axolotl.integrations.diffusion_decision.plugin import DiffusionDecisionPlugin

    adapter, merged = tmp_path / "adapter", tmp_path / "merged"
    adapter.mkdir()
    merged.mkdir()
    (adapter / MANIFEST_FILENAME).write_text("{not a manifest")

    DiffusionDecisionPlugin().post_lora_merge({}, str(adapter), str(merged))

    assert not (merged / MANIFEST_FILENAME).exists()


def test_config_caps_questions_per_canvas_at_the_template_limit():
    from axolotl.integrations.diffusion_decision.args import DiffusionDecisionConfig
    from axolotl.integrations.diffusion_decision.vendored.djev_template import (
        MAX_QUESTIONS,
    )

    DiffusionDecisionConfig.model_validate({"max_questions_per_canvas": MAX_QUESTIONS})
    with pytest.raises(ValueError, match="max_questions_per_canvas"):
        DiffusionDecisionConfig.model_validate(
            {"max_questions_per_canvas": MAX_QUESTIONS + 1}
        )


def test_collator_hook_defers_to_defaults_without_a_decision_block():
    from axolotl.integrations.diffusion_decision.plugin import DiffusionDecisionPlugin

    assert DiffusionDecisionPlugin().get_collator_cls_and_kwargs({}) is None
