"""cfg cases for benchmarking"""

from pytest_cases import case, parametrize

from axolotl.utils.dict import DictDefault


class TestConfigs:  # pylint: disable=missing-class-docstring disable=too-many-public-methods
    def model_tinyllama(self):
        return (
            DictDefault(
                {
                    "base_model": "PY007/TinyLlama-1.1B-step-50K-105b",
                    "base_model_config": "PY007/TinyLlama-1.1B-step-50K-105b",
                    "model_type": "LlamaForCausalLM",
                    "sequence_len": 4096,
                    "special_tokens": {
                        "bos_token": "<s>",
                        "eos_token": "</s>",
                        "unk_token": "<unk>",
                    },
                }
            )
            | self.train_simple()
        )

    def model_llama2_7b(self):
        return (
            DictDefault(
                {
                    "base_model": "meta-llama/Llama-2-7b-chat-hf",
                    "base_model_config": "meta-llama/Llama-2-7b-chat-hf",
                    "model_type": "LlamaForCausalLM",
                    "sequence_len": 4096,
                    "special_tokens": {
                        "bos_token": "<s>",
                        "eos_token": "</s>",
                        "unk_token": "<unk>",
                    },
                }
            )
            | self.train_simple()
        )

    def train_simple(self):
        return DictDefault(
            {
                "warmup_steps": 3,
                "num_epochs": 2,
                "gradient_accumulation_steps": 1,
                "micro_batch_size": 1,
                "val_set_size": 0,
                "learning_rate": 0.00005,
                "lr_scheduler": "cosine",
            }
        )

    @parametrize(size=(128, 256, 512, 1024, 2048, 3072, 4096))
    def ctx_prompt(self, size):
        return min(4096 - 128, size)

    def opt_adamw_bnb_8bit(self):
        return DictDefault(
            {
                "optimizer": "adamw_bnb_8bit",
            }
        )

    def opt_adamw_torch(self):
        return DictDefault(
            {
                "optimizer": "adamw_torch",
            }
        )

    def opt_paged_adamw_8bit(self):
        return DictDefault(
            {
                "optimizer": "paged_adamw_8bit",
            }
        )

    def opt_paged_adamw_32bit(self):
        return DictDefault(
            {
                "optimizer": "paged_adamw_32bit",
            }
        )

    def opt_adamw_apex_fused(self):
        return DictDefault(
            {
                "optimizer": "adamw_apex_fused",
            }
        )

    def opt_lion_8bit(self):
        return DictDefault(
            {
                "optimizer": "lion_8bit",
            }
        )

    def opt_paged_lion_8bit(self):
        return DictDefault(
            {
                "optimizer": "paged_lion_8bit",
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

    def attn_sdp(self):
        return DictDefault(
            {
                "sdp_attention": True,
            }
        )

    def _attn_bettertransformer(self):
        return DictDefault(
            {
                "flash_optimum": True,
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
                "lora_r": 32,
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
