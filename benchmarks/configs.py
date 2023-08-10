"""cfg cases for benchmarking"""

from axolotl.utils.dict import DictDefault


class TestConfigs:  # pylint: disable=missing-class-docstring
    def model_llama2_7b(self):
        return DictDefault(
            {
                "base_model": "meta-llama/Llama-2-7b-chat-hf",
                "base_model_config": "meta-llama/Llama-2-7b-chat-hf",
                "model_type": "LlamaForCausalLM",
                "tokenizer_type": "LlamaTokenizer",
                "gradient_accumulation_steps": 1,
                "micro_batch_size": 1,
                "pad_token": "<pad>",
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

    def dtype_bf16(self):
        return DictDefault(
            {
                "bf16": True,
            }
        )

    def dtype_4bit(self):
        return (
            DictDefault(
                {
                    "load_in_4bit": True,
                }
            )
            | self.adapter_qlora()
        )

    def dtype_8bit(self):
        return (
            DictDefault(
                {
                    "load_in_8bit": True,
                }
            )
            | self.adapter_lora()
        )
