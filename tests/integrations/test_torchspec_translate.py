"""Tests for the TorchSpec speculator-training config translation."""

import importlib.util

import pytest
from pydantic import ValidationError

from axolotl.integrations.torchspec.args import TorchSpecArgs
from axolotl.integrations.torchspec.translate import build_overrides
from axolotl.utils.dict import DictDefault

_HAS_TORCHSPEC = importlib.util.find_spec("torchspec") is not None


def _base_cfg(**speculator) -> DictDefault:
    spec = {
        "ttt_length": 7,
        "inference_engine": "sgl",
        "inference_num_gpus": 2,
        "training_num_gpus": 2,
        "mooncake_protocol": "tcp",
    }
    spec.update(speculator)
    return DictDefault(
        {
            "base_model": "Qwen/Qwen3-8B",
            "trust_remote_code": True,
            "chat_template": "qwen3",
            "sequence_len": 16384,
            "micro_batch_size": 1,
            "learning_rate": 1e-4,
            "num_epochs": 1,
            "output_dir": "./outputs/qwen3-8b-eagle3",
            "datasets": [{"path": "./conversations.jsonl", "type": "chat_template"}],
            "speculator": spec,
        }
    )


class TestTorchSpecArgs:
    def test_rdma_requires_device_name(self):
        with pytest.raises(ValidationError, match="mooncake_device_name"):
            TorchSpecArgs(mooncake_protocol="rdma")

    def test_rdma_with_device_name_ok(self):
        args = TorchSpecArgs(mooncake_protocol="rdma", mooncake_device_name="mlx5_0")
        assert args.mooncake_device_name == "mlx5_0"

    def test_ploss_weights_length_must_match_ttt(self):
        with pytest.raises(ValidationError, match="ploss_weights"):
            TorchSpecArgs(ttt_length=7, ploss_weights=[0.5, 0.5])


class TestBuildOverrides:
    def test_core_mapping(self):
        ov = build_overrides(_base_cfg())
        assert ov["model"]["target_model_path"] == "Qwen/Qwen3-8B"
        assert ov["model"]["target_model_backend"] == "sglang"
        assert ov["model"]["trust_remote_code"] is True
        assert ov["dataset"]["train_data_path"] == "./conversations.jsonl"
        assert ov["dataset"]["chat_template"] == "qwen"  # qwen3 -> qwen
        assert ov["training"]["max_seq_length"] == 16384
        assert ov["training"]["ttt_length"] == 7
        assert ov["training"]["training_num_gpus_per_node"] == 2
        assert ov["inference"]["inference_engine_type"] == "sgl"
        assert ov["inference"]["inference_num_gpus"] == 2
        assert ov["inference"]["sglang"]["tp_size"] == 1
        assert "vllm" not in ov["inference"]  # pruned for non-vllm engine
        assert ov["mooncake"]["protocol"] == "tcp"
        assert ov["output_dir"] == "./outputs/qwen3-8b-eagle3"

    def test_vllm_engine_selects_vllm_subsection(self):
        ov = build_overrides(_base_cfg(inference_engine="vllm"))
        assert ov["model"]["target_model_backend"] == "vllm"
        assert ov["inference"]["inference_engine_type"] == "vllm"
        assert "vllm" in ov["inference"]
        assert "sglang" not in ov["inference"]

    def test_speculator_chat_template_override(self):
        cfg = _base_cfg()
        cfg["speculator"]["chat_template"] = "custom-tpl"
        ov = build_overrides(cfg)
        assert ov["dataset"]["chat_template"] == "custom-tpl"

    def test_unmapped_chat_template_raises(self):
        cfg = _base_cfg()
        cfg["chat_template"] = "some_unknown_template"
        with pytest.raises(ValueError, match="no TorchSpec mapping"):
            build_overrides(cfg)

    def test_missing_speculator_block_raises(self):
        cfg = _base_cfg()
        del cfg["speculator"]
        with pytest.raises(ValueError, match="speculator"):
            build_overrides(cfg)

    def test_none_values_pruned(self):
        # mem_fraction_static unset -> should not appear in the sglang subsection
        ov = build_overrides(_base_cfg())
        assert "mem_fraction_static" not in ov["inference"]["sglang"]


@pytest.mark.skipif(not _HAS_TORCHSPEC, reason="torchspec not installed")
class TestBuildTorchSpecArgs:
    def test_flat_args_resolved(self):
        from axolotl.integrations.torchspec.translate import build_torchspec_args

        args = build_torchspec_args(_base_cfg())
        assert args.target_model_path == "Qwen/Qwen3-8B"
        # load_config absolutizes local data paths
        assert args.train_data_path.endswith("conversations.jsonl")
        assert args.chat_template == "qwen"
        assert args.inference_engine_type == "sgl"
        assert args.ttt_length == 7
        assert args.mooncake_protocol == "tcp"
        # computed by config_to_flat_args / _resolve_batch_size
        assert args.world_size == 2
        assert args.per_dp_rank_batch_size == 1

    def test_extra_overrides_applied(self):
        from axolotl.integrations.torchspec.translate import build_torchspec_args

        args = build_torchspec_args(
            _base_cfg(), extra_overrides=["training.num_train_steps=10"]
        )
        assert args.num_train_steps == 10
