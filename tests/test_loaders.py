"""Module for `axolotl.loaders`."""

from unittest.mock import MagicMock

import pytest
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils.import_utils import is_torch_mps_available

from axolotl.loaders import ModelLoader
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import _get_parallel_config_kwargs


class TestModelsUtils:
    """Testing module for `axolotl.loaders`."""

    def setup_method(self) -> None:
        # load config
        self.cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "load_in_8bit": True,
                "load_in_4bit": False,
                "adapter": "lora",
                "flash_attention": False,
                "sample_packing": True,
                "device_map": "auto",
            }
        )
        self.tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        self.inference = False
        self.reference_model = True

        # init ModelLoader
        self.model_loader = ModelLoader(
            cfg=self.cfg,
            tokenizer=self.tokenizer,
            inference=self.inference,
            reference_model=self.reference_model,
        )

    def test_set_device_map_config(self):
        # check device_map
        device_map = self.cfg.device_map
        if is_torch_mps_available():
            device_map = "mps"

        self.model_loader._set_device_map_config()
        if is_deepspeed_zero3_enabled():
            assert "device_map" not in self.model_loader.model_kwargs
        else:
            assert device_map in self.model_loader.model_kwargs["device_map"]

        # check torch_dtype
        assert self.cfg.torch_dtype == self.model_loader.model_kwargs["torch_dtype"]

    @pytest.mark.parametrize("adapter", ["lora", "qlora", None])
    @pytest.mark.parametrize("load_in_8bit", [True, False])
    @pytest.mark.parametrize("load_in_4bit", [True, False])
    @pytest.mark.parametrize("gptq", [True, False])
    def test_set_quantization_config(
        self,
        adapter,
        load_in_8bit,
        load_in_4bit,
        gptq,
    ):
        # init cfg as args
        self.cfg.load_in_8bit = load_in_8bit
        self.cfg.load_in_4bit = load_in_4bit
        self.cfg.gptq = gptq
        self.cfg.adapter = adapter

        self.model_loader._set_quantization_config()
        if "quantization_config" in self.model_loader.model_kwargs or self.cfg.gptq:
            assert not (
                hasattr(self.model_loader.model_kwargs, "load_in_8bit")
                and hasattr(self.model_loader.model_kwargs, "load_in_4bit")
            )

        if self.cfg.adapter == "qlora" and load_in_4bit:
            assert isinstance(
                self.model_loader.model_kwargs.get("quantization_config"),
                BitsAndBytesConfig,
            )

            assert (
                self.model_loader.model_kwargs["quantization_config"]._load_in_4bit
                is True
            )
        if self.cfg.adapter == "lora" and load_in_8bit:
            assert isinstance(
                self.model_loader.model_kwargs.get("quantization_config"),
                BitsAndBytesConfig,
            )

            assert (
                self.model_loader.model_kwargs["quantization_config"]._load_in_8bit
                is True
            )

    def test_message_property_mapping(self):
        """Test message property mapping configuration validation"""
        from axolotl.utils.schemas.datasets import SFTDataset

        # Test legacy fields are mapped orrectly
        dataset = SFTDataset(
            path="test_path",
            message_field_role="role_field",
            message_field_content="content_field",
        )
        assert dataset.message_property_mappings == {
            "role": "role_field",
            "content": "content_field",
        }

        # Test direct message_property_mapping works
        dataset = SFTDataset(
            path="test_path",
            message_property_mappings={
                "role": "custom_role",
                "content": "custom_content",
            },
        )
        assert dataset.message_property_mappings == {
            "role": "custom_role",
            "content": "custom_content",
        }

        # Test both legacy and new fields work when they match
        dataset = SFTDataset(
            path="test_path",
            message_field_role="same_role",
            message_property_mappings={"role": "same_role"},
        )
        assert dataset.message_property_mappings == {
            "role": "same_role",
            "content": "content",
        }

        # Test both legacy and new fields work when they don't overlap
        dataset = SFTDataset(
            path="test_path",
            message_field_role="role_field",
            message_property_mappings={"content": "content_field"},
        )
        assert dataset.message_property_mappings == {
            "role": "role_field",
            "content": "content_field",
        }

        # Test no role or content provided
        dataset = SFTDataset(
            path="test_path",
        )
        assert dataset.message_property_mappings == {
            "role": "role",
            "content": "content",
        }

        # Test error when legacy and new fields conflict
        with pytest.raises(ValueError) as exc_info:
            SFTDataset(
                path="test_path",
                message_field_role="legacy_role",
                message_property_mappings={"role": "different_role"},
            )
        assert "Conflicting message role fields" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            SFTDataset(
                path="test_path",
                message_field_content="legacy_content",
                message_property_mappings={"content": "different_content"},
            )
        assert "Conflicting message content fields" in str(exc_info.value)

    @pytest.mark.parametrize(
        "world_size, tensor_parallel_size, context_parallel_size, dp_shard_size, dp_replicate_size, is_fsdp, expected",
        [
            (16, 2, 2, 2, 2, True, (2, 2, 2, 2)),
            (16, 1, 1, None, None, True, (0, 0, 16, 1)),
            (16, 2, 2, 2, None, True, (2, 2, 2, 2)),
            (16, 2, 2, None, 2, True, (2, 2, 2, 2)),
            (16, 1, 1, None, 2, True, (0, 0, 8, 2)),
            (2, 1, 1, None, None, True, (0, 0, 2, 1)),
        ],
    )
    def test_get_parallel_config_kwargs(
        self,
        world_size,
        tensor_parallel_size,
        context_parallel_size,
        dp_shard_size,
        dp_replicate_size,
        is_fsdp,
        expected,
    ):
        res = _get_parallel_config_kwargs(
            world_size,
            tensor_parallel_size,
            context_parallel_size,
            dp_shard_size,
            dp_replicate_size,
            is_fsdp,
        )

        if expected[0] > 1:
            assert res["tp_size"] == expected[0]
        if expected[1] > 1:
            assert res["cp_size"] == expected[1]
        if expected[2] > 1:
            assert res["dp_shard_size"] == expected[2]
        if expected[3] > 1:
            assert res["dp_replicate_size"] == expected[3]
