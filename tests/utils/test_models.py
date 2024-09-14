"""Module for testing models utils file."""

import unittest
from unittest.mock import patch, MagicMock

import pytest
import torch
from transformers import (
    PreTrainedTokenizerBase,
    GPTQConfig,
    BitsAndBytesConfig,
    AwqConfig,
    AutoConfig,
)
from transformers.utils.import_utils import is_torch_mps_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from axolotl.utils.dict import DictDefault
from axolotl.utils.models import (
    load_model,
    load_tokenizer,
    load_model_config,
    ModelLoader,
)
from axolotl.cli import load_cfg


class TestModelsUtils:
    """Testing module for models utils."""

    def setup_method(self) -> None:
        # load config
        config_path = "examples/openllama-3b/config.yml"
        self.cfg = load_cfg(config_path)
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
                load_model(cfg, tokenizer="")
            assert (
                "shifted-sparse attention does not currently support sample packing"
                in str(exc.value)
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

    @pytest.mark.parametrize("adapter", ["lora", "qlora", None])
    @pytest.mark.parametrize("load_in_8bit", [True, False])
    @pytest.mark.parametrize("load_in_4bit", [True, False])
    @pytest.mark.parametrize("gptq", [True, False])
    # @pytest.mark.parametrize("gptq_disable_exllama", [True, False, None])
    # @pytest.mark.parametrize(
    #     "quant_method", ["gptq", "awq", "bitsandbytes", "bnb", "None"]
    # )
    def test_set_quantization_config(
        self,
        adapter,
        load_in_8bit,
        load_in_4bit,
        gptq,
        # gptq_disable_exllama,
        # quant_method,
    ):
        # init cfg as args
        self.cfg.load_in_8bit = load_in_8bit
        self.cfg.load_in_4bit = load_in_4bit
        self.cfg.gptq = gptq
        # self.cfg.gptq_disable_exllama = gptq_disable_exllama
        self.cfg.adapter = adapter

        # self.model_loader.model_config = load_model_config(self.cfg)

        self.model_loader.set_quantization_config()
        if "quantization_config" in self.model_loader.model_kwargs or self.cfg.gptq:
            assert not (
                hasattr(self.model_loader.model_kwargs, "load_in_8bit")
                and hasattr(self.model_loader.model_kwargs, "load_in_4bit")
            )
        elif load_in_8bit and self.cfg.adapter is not None:
            assert (
                self.model_loader.model_kwargs["load_in_8bit"],
                f'Expected self.model_loader.model_kwargs["load_in_8bit"] to be True, but got False',
            )
        elif load_in_4bit and self.cfg.adapter is not None:
            assert (
                self.model_loader.model_kwargs["load_in_4bit"],
                f'Expected self.model_loader.model_kwargs["load_in_4bit"] to be True, but got False',
            )

        if (self.cfg.adapter == "qlora" and load_in_4bit) or (
            self.cfg.adapter == "lora" and load_in_8bit
        ):
            assert (
                self.model_loader.model_kwargs.get(
                    "quantization_config", BitsAndBytesConfig
                ),
                f'Expected self.model_loader.model_kwargs["quantization_config"] to be BitsAndBytesConfig',
            )

        # # TODO (Mengqing Cao): fix these test cases
        # # AttributeError:'LlamaConfig' object has no attribute 'quantization_config'
        # if hasattr(self.model_loader.model_config, "quantization_config"):
        #     self.model_loader.model_config.quantization_config["quant_method"] = (
        #         quant_method
        #     )
        #     if gptq:
        #         if gptq_disable_exllama is not None:
        #             assert (
        #                 self.model_loader.model_kwargs.quantization_config[
        #                     "disable_exllama"
        #                 ]
        #                 == gptq_disable_exllama
        #             )
        #         else:
        #             assert (
        #                 self.model_loader.model_kwargs.get(
        #                     "quantization_config", GPTQConfig
        #                 ),
        #                 f'Expected self.model_loader.model_kwargs["quantization_config"] to be GPTQConfig',
        #             )

        #     quantization_map = {
        #         "gptq": GPTQConfig,
        #         "awq": AwqConfig,
        #         "bitsandbytes": BitsAndBytesConfig,
        #     }

        #     if self.cfg.adapter in [
        #         "qlora",
        #         "lora",
        #     ] and self.model_loader.model_config.quantization_config[
        #         "quant_method"
        #     ] in [
        #         "gptq",
        #         "awq",
        #         "bitsandbytes",
        #     ]:
        #         assert (
        #             self.model_loader.model_kwargs.quantization_config.get(
        #                 "quantization_config", quantization_map[quant_method]
        #             ),
        #             f"Expected quant_method to be configured as {quantization_map[quant_method]}",
        #         )

    @pytest.mark.parametrize("embedding_modules", ["embed_tokens", "lm_head"])
    @pytest.mark.parametrize("dist_dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("before_kbit_train_or_finetune", [True, False])
    def test_convert_embedding_modules_dtype(
        self, embedding_modules, dist_dtype, before_kbit_train_or_finetune
    ):
        tokenizer = load_tokenizer(self.cfg)
        self.model_loader.model, _ = load_model(self.cfg, tokenizer, inference=False)

        self.model_loader.convert_embedding_modules_dtype(
            embedding_modules, dist_dtype, before_kbit_train_or_finetune
        )
        for name, module in self.model_loader.model.named_modules():
            if (
                "norm" in name
                or (before_kbit_train_or_finetune and name.endswith(".gate"))
                or (
                    any(m in name for m in embedding_modules)
                    and hasattr(module, "weight")
                )
            ):
                for param_name, param in module.named_parameters(recurse=False):
                    assert (
                        param.dtype == dist_dtype,
                        f"Expected {dist_dtype}, but got {param.dtype} in {param_name} of {name}",
                    )
