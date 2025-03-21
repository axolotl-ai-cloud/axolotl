"""Module for testing models utils file."""

from unittest.mock import MagicMock, patch

import pytest
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils.import_utils import is_torch_mps_available

from axolotl.utils.dict import DictDefault
from axolotl.utils.models import ModelLoader, load_model


class TestModelsUtils:
    """Testing module for models utils."""

    def setup_method(self) -> None:
        # load config
        self.cfg = DictDefault(  # pylint: disable=attribute-defined-outside-init
            {
                "base_model": "JackFram/llama-68m",
                "model_type": "LlamaForCausalLM",
                "tokenizer_type": "LlamaTokenizer",
                "load_in_8bit": True,
                "load_in_4bit": False,
                "adapter": "lora",
                "flash_attention": False,
                "sample_packing": True,
                "device_map": "auto",
            }
        )
        self.tokenizer = MagicMock(  # pylint: disable=attribute-defined-outside-init
            spec=PreTrainedTokenizerBase
        )
        self.inference = False  # pylint: disable=attribute-defined-outside-init
        self.reference_model = True  # pylint: disable=attribute-defined-outside-init

        # init ModelLoader
        self.model_loader = (  # pylint: disable=attribute-defined-outside-init
            ModelLoader(
                cfg=self.cfg,
                tokenizer=self.tokenizer,
                inference=self.inference,
                reference_model=self.reference_model,
            )
        )

    def test_set_device_map_config(self):
        # check device_map
        device_map = self.cfg.device_map
        if is_torch_mps_available():
            device_map = "mps"
        self.model_loader.set_device_map_config()
        if is_deepspeed_zero3_enabled():
            assert "device_map" not in self.model_loader.model_kwargs
        else:
            assert device_map in self.model_loader.model_kwargs["device_map"]

        # check torch_dtype
        assert self.cfg.torch_dtype == self.model_loader.model_kwargs["torch_dtype"]

    def test_cfg_throws_error_with_s2_attention_and_sample_packing(self):
        cfg = DictDefault(
            {
                "s2_attention": True,
                "sample_packing": True,
                "base_model": "",
                "model_type": "LlamaForCausalLM",
            }
        )

        # Mock out call to HF hub
        with patch(
            "axolotl.utils.models.load_model_config"
        ) as mocked_load_model_config:
            mocked_load_model_config.return_value = {}
            with pytest.raises(ValueError) as exc:
                # Should error before hitting tokenizer, so we pass in an empty str
                load_model(cfg, tokenizer="")  # type: ignore
            assert (
                "shifted-sparse attention does not currently support sample packing"
                in str(exc.value)
            )

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

        self.model_loader.set_quantization_config()
        if "quantization_config" in self.model_loader.model_kwargs or self.cfg.gptq:
            assert not (
                hasattr(self.model_loader.model_kwargs, "load_in_8bit")
                and hasattr(self.model_loader.model_kwargs, "load_in_4bit")
            )
        elif load_in_8bit and self.cfg.adapter is not None:
            assert self.model_loader.model_kwargs["load_in_8bit"]
        elif load_in_4bit and self.cfg.adapter is not None:
            assert self.model_loader.model_kwargs["load_in_4bit"]

        if (self.cfg.adapter == "qlora" and load_in_4bit) or (
            self.cfg.adapter == "lora" and load_in_8bit
        ):
            assert self.model_loader.model_kwargs.get(
                "quantization_config", BitsAndBytesConfig
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
