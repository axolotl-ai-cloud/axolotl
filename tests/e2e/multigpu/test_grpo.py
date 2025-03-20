"""
GRPO test suite
"""

import random
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from e2e.utils import require_vllm
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault


class TestGRPO:
    """
    Test case for GRPO training using multilpe GPUs
    """

    def _utils_write_yaml_and_rewards(self, cfg, temp_dir, suffix=""):
        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))
        with open(f"rewards_{suffix}.py", "w", encoding="utf-8") as fout:
            fout.write(
                """import random
def rand_reward_func(completions, **kwargs) -> list[float]:
    return [random.uniform(0, 1) for _ in completions]

def oai_gsm8k_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        label = example["answer"].split("####")[-1].strip().replace(",", "")
        return {
            "prompt": [{"role": "user", "content": example["question"]},],
            "answer": label,
        }
    return transform_fn, {"remove_columns": ["question"]}
"""
            )

    @pytest.mark.parametrize(
        "num_gpus",
        [1, 2],
    )
    @require_vllm
    def test_llama_dora(self, temp_dir, num_gpus):
        rnd_reward_suffix = str(random.randint(1000, 9999))
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "chat_template": "llama3",
                "rl": "grpo",
                "trl": {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": True,
                    "vllm_device": "auto" if num_gpus == 1 else "cuda:1",
                    "vllm_gpu_memory_utilization": 0.15,
                    "num_generations": 4,
                    "reward_funcs": [f"rewards_{rnd_reward_suffix}.rand_reward_func"],
                },
                "datasets": [
                    {
                        "path": "openai/gsm8k",
                        "name": "main",
                        "type": f"rewards_{rnd_reward_suffix}.oai_gsm8k_transform",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "peft_use_dora": True,
                "flash_attention": True,
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "max_steps": 5,
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 10,
                "val_set_size": 0.0,
                "output_dir": temp_dir,
                "learning_rate": 0.0001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "use_tensorboard": True,
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_reward_suffix)

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                str(num_gpus),
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

    @pytest.mark.parametrize(
        "num_gpus",
        [1, 2],
    )
    @require_vllm
    def test_llama_fft(self, temp_dir, num_gpus):
        rnd_reward_suffix = str(random.randint(1000, 9999))
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "chat_template": "llama3",
                "rl": "grpo",
                "trl": {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": True,
                    "vllm_device": "auto" if num_gpus == 1 else "cuda:1",
                    "vllm_gpu_memory_utilization": 0.15,
                    "num_generations": 4,
                    "reward_funcs": [f"rewards_{rnd_reward_suffix}.rand_reward_func"],
                },
                "datasets": [
                    {
                        "path": "openai/gsm8k",
                        "name": "main",
                        "type": f"rewards_{rnd_reward_suffix}.oai_gsm8k_transform",
                    },
                ],
                "flash_attention": True,
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "max_steps": 5,
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 10,
                "val_set_size": 0.0,
                "output_dir": temp_dir,
                "learning_rate": 0.0001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "use_tensorboard": True,
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_reward_suffix)

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                str(num_gpus),
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )
