"""cfg cases for benchmarking"""

from pytest_cases import case

from axolotl.utils.dict import DictDefault


class TestConfigs:  # pylint: disable=missing-class-docstring
    def model_llama2_7b(self):
        return (
            DictDefault(
                {
                    "base_model": "meta-llama/Llama-2-7b-chat-hf",
                    "base_model_config": "meta-llama/Llama-2-7b-chat-hf",
                    "model_type": "LlamaForCausalLM",
                    "is_llama_derived_model": True,
                    "pad_token": "<pad>",
                    "special_tokens": {
                        "bos_token": "<s>",
                        "eos_token": "</s>",
                        "unk_token": "<unk>",
                    },
                }
            )
            | self.ctx_1k()
            | self.train_simple()
        )

    def train_simple(self):
        return DictDefault(
            {
                "num_epochs": 1,
                "gradient_accumulation_steps": 1,
                "micro_batch_size": 1,
                "val_set_size": 0,
                "optimizer": "adamw_torch",
                "adam_beta2": 0.98,
                "max_grad_norm": 1.0,
                "learning_rate": 0.00005,
                "lr_scheduler": "cosine",
                "lr_quadratic_warmup": True,
            }
        )

    def ctx_256(self):
        return DictDefault(
            {
                "sequence_len": 256,
            }
        )

    def ctx_512(self):
        return DictDefault(
            {
                "sequence_len": 512,
            }
        )

    def ctx_1k(self):
        return DictDefault(
            {
                "sequence_len": 1024,
            }
        )

    def ctx_2k(self):
        return DictDefault(
            {
                "sequence_len": 2048,
            }
        )

    def ctx_4k(self):
        return DictDefault(
            {
                "sequence_len": 4096,
            }
        )

    def attn_base(self):
        return DictDefault({})

    def attn_xformers(self):
        return DictDefault(
            {
                "xformers_attention": True,
            }
        )

    def attn_flash(self):
        return DictDefault(
            {
                "flash_attention": True,
            }
        )

    def lora_params(self):
        return DictDefault(
            {
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_linear": True,
            }
        )

    def adapter_none(self):
        return DictDefault({})

    def adapter_lora(self):
        return (
            DictDefault(
                {
                    "adapter": "lora",
                }
            )
            | self.lora_params()
        )

    def adapter_qlora(self):
        return (
            DictDefault(
                {
                    "adapter": "qlora",
                }
            )
            | self.lora_params()
        )

    def dtype_fp32(self):
        return DictDefault(
            {
                "fp32": True,
            }
        )

    def dtype_tf32(self):
        return DictDefault(
            {
                "fp32": True,
                "tf32": True,
            }
        )

    @case(tags="quick")
    def dtype_bf16(self):
        return DictDefault(
            {
                "bf16": True,
            }
        )

    @case(tags="quick")
    def dtype_4bit(self):
        return (
            DictDefault(
                {
                    "load_in_4bit": True,
                    "bf16": True,
                }
            )
            | self.adapter_qlora()
        )

    def dtype_8bit(self):
        return (
            DictDefault(
                {
                    "load_in_8bit": True,
                    "bf16": True,
                }
            )
            | self.adapter_lora()
        )
