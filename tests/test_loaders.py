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

    @pytest.mark.parametrize("adapter", ["lora", None])
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
        # init cfg as args. ``adapter: qlora`` no longer exists at this layer
        # — the validator demotes it to ``lora`` + ``load_in_4bit: True``
        # upstream, so the loader-side branches key off the flags.
        if load_in_4bit and load_in_8bit:
            # Validator blocks the combo, so the loader never sees it.
            return
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

        if self.cfg.adapter == "lora" and load_in_4bit:
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

    @pytest.mark.parametrize(
        "weight_dtype,adapter,quant_config_attr",
        [
            ("int4", "qlora", "Int4WeightOnlyConfig"),
            ("int8", "lora", "Int8WeightOnlyConfig"),
        ],
    )
    def test_set_quantization_config_torchao_qlora(
        self, weight_dtype, adapter, quant_config_attr
    ):
        """torchao backend installs a TorchAoConfig with the right quant_type."""
        pytest.importorskip("torchao")
        import torchao.quantization as tq
        from transformers import TorchAoConfig

        from axolotl.utils.schemas.model import (
            ModelQuantizationConfig,
            TorchAoBaseQuantConfig,
        )

        expected_cls = getattr(tq, quant_config_attr)

        self.cfg.load_in_8bit = False
        self.cfg.load_in_4bit = False
        self.cfg.adapter = adapter
        self.cfg.model_quantization_config = ModelQuantizationConfig(
            torchao=TorchAoBaseQuantConfig(weight_dtype=weight_dtype),
        )

        self.model_loader._set_quantization_config()
        quant_config = self.model_loader.model_kwargs.get("quantization_config")
        assert isinstance(quant_config, TorchAoConfig)
        assert isinstance(quant_config.quant_type, expected_cls)

    def test_set_quantization_config_torchao_nvfp4(self):
        """torchao NVFP4 installs an NVFP4WeightOnlyConfig inside TorchAoConfig."""
        pytest.importorskip("torchao")
        try:
            from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
        except ImportError:
            pytest.skip("torchao build lacks NVFP4WeightOnlyConfig")
        from transformers import TorchAoConfig

        from axolotl.utils.schemas.model import (
            ModelQuantizationConfig,
            TorchAoBaseQuantConfig,
        )

        self.cfg.load_in_8bit = False
        self.cfg.load_in_4bit = False
        self.cfg.adapter = "qlora"
        self.cfg.model_quantization_config = ModelQuantizationConfig(
            torchao=TorchAoBaseQuantConfig(weight_dtype="nvfp4"),
        )

        self.model_loader._set_quantization_config()
        quant_config = self.model_loader.model_kwargs.get("quantization_config")
        assert isinstance(quant_config, TorchAoConfig)
        assert isinstance(quant_config.quant_type, NVFP4WeightOnlyConfig)

    def test_set_quantization_config_torchao_fp8(self):
        """torchao FP8 installs a Float8WeightOnlyConfig inside TorchAoConfig."""
        pytest.importorskip("torchao")
        try:
            from torchao.quantization import Float8WeightOnlyConfig
        except ImportError:
            pytest.skip("torchao build lacks Float8WeightOnlyConfig")
        from transformers import TorchAoConfig

        from axolotl.utils.schemas.model import (
            ModelQuantizationConfig,
            TorchAoBaseQuantConfig,
        )

        self.cfg.load_in_8bit = False
        self.cfg.load_in_4bit = False
        self.cfg.adapter = "lora"
        self.cfg.model_quantization_config = ModelQuantizationConfig(
            torchao=TorchAoBaseQuantConfig(weight_dtype="fp8"),
        )

        self.model_loader._set_quantization_config()
        quant_config = self.model_loader.model_kwargs.get("quantization_config")
        assert isinstance(quant_config, TorchAoConfig)
        assert isinstance(quant_config.quant_type, Float8WeightOnlyConfig)

    @pytest.mark.parametrize(
        "ckpt_qcfg",
        [
            # gpt-oss native MXFP4
            {"quant_method": "mxfp4"},
            # AMD Quark MXFP4 with per-module exclusion list (real shape from
            # amd/Kimi-K2.6-MXFP4: experts MXFP4, attention/lm_head/vision
            # bf16). peft.backend must NOT re-quantize on top of this.
            {
                "quant_method": "quark",
                "exclude": [
                    "language_model.lm_head",
                    "language_model.model.layers.0.self_attn.q_a_proj",
                ],
            },
            # AWQ / GPTQ / BNB checkpoints — the earlier if-branch in
            # _set_quantization_config used to silently consume these and
            # drop peft.backend on the floor. Now caught upfront.
            {"quant_method": "awq", "bits": 4},
            {"quant_method": "gptq", "bits": 4, "group_size": 128},
            {"quant_method": "bitsandbytes", "load_in_4bit": True},
        ],
    )
    def test_set_quantization_config_torchao_rejects_quantized_checkpoint(
        self, ckpt_qcfg
    ):
        """torchao base-quant must not silently lose to any checkpoint quant_method."""
        pytest.importorskip("torchao")
        from axolotl.utils.schemas.model import (
            ModelQuantizationConfig,
            TorchAoBaseQuantConfig,
        )

        self.cfg.load_in_8bit = False
        self.cfg.load_in_4bit = False
        self.cfg.adapter = "qlora"
        self.cfg.model_quantization_config = ModelQuantizationConfig(
            torchao=TorchAoBaseQuantConfig(weight_dtype="int4"),
        )
        self.model_loader.model_config.quantization_config = ckpt_qcfg
        with pytest.raises(ValueError, match="already quantized"):
            self.model_loader._set_quantization_config()

    def test_set_quantization_config_torchao_mxfp4_errors(self):
        """mxfp4 has no weight-only flavor; loader points at quantize_moe_experts."""
        pytest.importorskip("torchao")
        from axolotl.utils.schemas.model import (
            ModelQuantizationConfig,
            TorchAoBaseQuantConfig,
        )

        self.cfg.load_in_8bit = False
        self.cfg.load_in_4bit = False
        self.cfg.adapter = "lora"
        self.cfg.model_quantization_config = ModelQuantizationConfig(
            torchao=TorchAoBaseQuantConfig(weight_dtype="mxfp4"),
        )

        with pytest.raises(ValueError, match="quantize_moe_experts"):
            self.model_loader._set_quantization_config()

    def test_set_quantization_config_torchao_nf4(self):
        """torchao NF4 installs an NF4WeightOnlyConfig inside TorchAoConfig."""
        pytest.importorskip("torchao")
        from transformers import TorchAoConfig

        try:
            from torchao.prototype._nf4tensor_api import NF4WeightOnlyConfig
        except ImportError:
            try:
                from torchao.dtypes._nf4tensor_api import (
                    NF4WeightOnlyConfig,
                )
            except ImportError:
                pytest.skip("torchao build lacks NF4WeightOnlyConfig")

        from axolotl.utils.schemas.model import (
            ModelQuantizationConfig,
            TorchAoBaseQuantConfig,
        )

        self.cfg.load_in_8bit = False
        self.cfg.load_in_4bit = False
        self.cfg.adapter = "qlora"
        self.cfg.model_quantization_config = ModelQuantizationConfig(
            torchao=TorchAoBaseQuantConfig(weight_dtype="nf4"),
        )

        self.model_loader._set_quantization_config()
        quant_config = self.model_loader.model_kwargs.get("quantization_config")
        assert isinstance(quant_config, TorchAoConfig)
        assert isinstance(quant_config.quant_type, NF4WeightOnlyConfig)

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
