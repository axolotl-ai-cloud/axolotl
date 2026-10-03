"""Native diffusion uses plugin hooks without changing the standard SFT path."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
from datasets import Dataset
from pydantic import ValidationError

from axolotl.integrations.diffusion import DiffusionPlugin, datasets as native_datasets
from axolotl.integrations.diffusion.args import DiffusionArgs
from axolotl.utils.dict import DictDefault


def _cfg():
    return DictDefault(
        {
            "model_config_type": "nemotron_labs_diffusion",
            "diffusion": DictDefault(
                {"from_causal_lm": False, "overflow_policy": "drop"}
            ),
            "sequence_len": 8,
            "sample_packing": False,
            "streaming": False,
            "pretraining_dataset": None,
            "test_datasets": None,
            "processor_type": None,
            "max_steps": None,
        }
    )


@pytest.mark.parametrize("preprocess", [False, True])
def test_native_filter_runs_before_step_count(monkeypatch, preprocess):
    dataset = Dataset.from_list(
        [
            {"input_ids": list(range(length)), "labels": list(range(length))}
            for length in (6, 10, 7)
        ]
    )

    class Loader:
        def __init__(self, cfg):
            pass

        def load(self, fn):
            return fn()

        def cleanup(self):
            pass

    observed = []

    def count(cfg, selected):
        observed.append(list(selected["input_ids"]))
        return len(selected)

    monkeypatch.setattr(native_datasets, "load_tokenizer", lambda cfg: object())
    monkeypatch.setattr(native_datasets, "FileLockLoader", Loader)
    monkeypatch.setattr(
        native_datasets,
        "_load_and_prepare_datasets",
        lambda *args, **kwargs: (dataset, None, []),
    )
    monkeypatch.setattr(native_datasets, "calculate_total_num_steps", count)
    monkeypatch.setenv("AXOLOTL_IS_PREPROCESS", "0")

    meta = native_datasets.load_native_datasets(_cfg(), preprocess=preprocess)
    assert len(meta.train_dataset) == 2
    assert meta.total_num_steps == (-1 if preprocess else 2)
    assert observed == ([] if preprocess else [list(meta.train_dataset["input_ids"])])


def test_native_plugin_yields_to_decision_in_either_plugin_order():
    cfg = _cfg()
    cfg.decision = SimpleNamespace()
    plugin = DiffusionPlugin()
    assert plugin.load_datasets(cfg) is None
    assert plugin.get_trainer_cls(cfg) is None
    assert plugin.get_collator_cls_and_kwargs(cfg) is None


def test_native_plugin_contributes_trainer_and_collator():
    cfg = _cfg()
    cfg.attn_implementation = "flex_attention"
    cfg.micro_batch_size = 1
    cfg.eval_batch_size = 1
    plugin = DiffusionPlugin()
    assert plugin.get_trainer_cls(cfg).__name__ == "AxolotlDiffusionTrainer"
    collator, kwargs = plugin.get_collator_cls_and_kwargs(cfg)
    instance = collator(SimpleNamespace(pad_token_id=0, eos_token_id=1), **kwargs)
    assert instance.layout == "full_sequence"
    assert plugin.get_training_args(cfg) == {"remove_unused_columns": False}


def test_varlen_torch_version_is_validated_by_plugin_args():
    class InputArgs(DiffusionArgs):
        attn_implementation: str = "varlen"
        env_capabilities: dict = {"torch_version": "2.13.0"}

    with pytest.raises(ValidationError, match="torch >= 2.14"):
        InputArgs(diffusion={"from_causal_lm": False})


def test_native_plugin_rejects_other_model_types():
    cfg = _cfg()
    cfg.model_config_type = "Dream"
    with pytest.raises(ValueError, match="Nemotron Labs Diffusion"):
        DiffusionPlugin().get_trainer_cls(cfg)


@pytest.mark.parametrize("reverse", [False, True])
def test_diffusion_schema_composes_with_decision_plugin(
    min_base_cfg, monkeypatch, reverse
):
    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.config import merge_input_args
    from axolotl.integrations.decision.plugin import DecisionPlugin
    from axolotl.utils.schemas.config import AxolotlInputConfig

    assert "diffusion" not in AxolotlInputConfig.model_fields
    entries = [
        ("axolotl.integrations.diffusion.DiffusionPlugin", DiffusionPlugin()),
        (
            "axolotl.integrations.decision.DecisionPlugin",
            DecisionPlugin(),
        ),
    ]
    manager = PluginManager.get_instance()
    monkeypatch.setattr(
        manager, "plugins", OrderedDict(reversed(entries) if reverse else entries)
    )
    _, merged = merge_input_args()
    config = merged(
        **(
            min_base_cfg
            | {
                "diffusion": {"from_causal_lm": False},
                "decision": {},
            }
        )
    )
    assert config.diffusion.from_causal_lm is False
    assert config.decision is not None
