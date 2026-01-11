"""
GDPO test suite

GDPO (Group Reward-Decoupled Normalization Policy Optimization) extends GRPO
with decoupled per-reward normalization for multi-reward RL training.
"""

import os
import random
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.multigpu.solo.test_grpo import recursive_kill, start_vllm
from tests.e2e.utils import require_vllm


@pytest.mark.skip(reason="flaky vllm tests in modal")
class TestGDPO:
    """
    Test case for GDPO training using multiple GPUs.

    GDPO is specifically designed for multi-reward RL training where it
    normalizes each reward function independently before combining them.
    """

    def _utils_write_yaml_and_rewards(self, cfg, temp_dir, suffix=""):
        """Write config and reward functions to temp directory."""
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))
        with open(f"rewards_gdpo_{suffix}.py", "w", encoding="utf-8") as fout:
            fout.write(
                """import random

def format_reward(prompts, completions, **kwargs) -> list[float]:
    '''Binary reward for format compliance (completion length > 10 chars).'''
    return [1.0 if len(c) > 10 else 0.0 for c in completions]

def correctness_reward(prompts, completions, **kwargs) -> list[float]:
    '''Continuous reward simulating correctness scoring.'''
    return [random.uniform(-1, 3) for _ in completions]

def safety_reward(prompts, completions, **kwargs) -> list[float]:
    '''Binary reward for safety compliance.'''
    return [1.0 if 'error' not in c.lower() else 0.0 for c in completions]

def single_reward(prompts, completions, **kwargs) -> list[float]:
    '''Single random reward for comparison with GRPO.'''
    return [random.uniform(0, 1) for _ in completions]

def oai_gsm8k_transform(cfg, *args, **kwargs):
    '''Transform function for GSM8K dataset.'''
    def transform_fn(example, tokenizer=None):
        label = example["answer"].split("####")[-1].strip().replace(",", "")
        return {
            "prompt": [{"role": "user", "content": example["question"]}],
            "answer": label,
        }
    return transform_fn, {"remove_columns": ["question"]}
"""
            )

    @pytest.mark.parametrize("num_gpus", [1, 2])
    @require_vllm
    def test_gdpo_multi_reward_lora(self, temp_dir, num_gpus):
        """Test GDPO with multiple reward functions using LoRA."""
        rnd_suffix = str(random.randint(1000, 9999))
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "chat_template": "llama3",
                "rl": "gdpo",
                "trl": {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": True,
                    "num_generations": 4,
                    # Multiple reward functions - GDPO's strength
                    "reward_funcs": [
                        f"rewards_gdpo_{rnd_suffix}.format_reward",
                        f"rewards_gdpo_{rnd_suffix}.correctness_reward",
                    ],
                    "reward_weights": [1.0, 2.0],
                    # GDPO-specific options
                    "gdpo_decoupled_norm": True,
                    "gdpo_batch_norm": False,
                    "gdpo_epsilon": 1e-4,
                    "scale_rewards": True,
                },
                "vllm": {
                    "max_model_len": 800,
                    "enable_prefix_caching": True,
                },
                "datasets": [
                    {
                        "path": "openai/gsm8k",
                        "name": "main",
                        "type": f"rewards_gdpo_{rnd_suffix}.oai_gsm8k_transform",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "flash_attention": True,
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "max_steps": 3,
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
                "save_first_step": False,
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_suffix)

        current_env = os.environ.copy()
        env = {
            "NCCL_P2P_LEVEL": "LOC",
            **current_env,
            "CUDA_VISIBLE_DEVICES": "1",
        }
        vllm_process = start_vllm(
            cfg.base_model,
            env=env,
            quiet=True,
            wait=300,
            gpu_memory_utilization=0.15,
            max_model_len=cfg.vllm.max_model_len,
            enable_prefix_caching=cfg.vllm.enable_prefix_caching,
            host="0.0.0.0",
            port=8000,
        )

        try:
            execute_subprocess_async(
                [
                    "axolotl",
                    "train",
                    str(Path(temp_dir) / "config.yaml"),
                    "--num-processes",
                    str(num_gpus),
                    "--main-process-port",
                    f"{get_torch_dist_unique_port()}",
                ],
                env={
                    "NCCL_P2P_LEVEL": "LOC",
                    "NCCL_DEBUG": "INFO",
                    **current_env,
                },
            )
        finally:
            recursive_kill(vllm_process)

    @require_vllm
    def test_gdpo_three_rewards(self, temp_dir):
        """Test GDPO with three reward functions (format, correctness, safety)."""
        rnd_suffix = str(random.randint(1000, 9999))
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "chat_template": "llama3",
                "rl": "gdpo",
                "trl": {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": True,
                    "num_generations": 4,
                    # Three reward functions
                    "reward_funcs": [
                        f"rewards_gdpo_{rnd_suffix}.format_reward",
                        f"rewards_gdpo_{rnd_suffix}.correctness_reward",
                        f"rewards_gdpo_{rnd_suffix}.safety_reward",
                    ],
                    "reward_weights": [1.0, 2.0, 1.5],
                    "gdpo_decoupled_norm": True,
                    "gdpo_batch_norm": True,  # Test with batch norm
                },
                "vllm": {
                    "max_model_len": 800,
                    "enable_prefix_caching": True,
                },
                "datasets": [
                    {
                        "path": "openai/gsm8k",
                        "name": "main",
                        "type": f"rewards_gdpo_{rnd_suffix}.oai_gsm8k_transform",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "flash_attention": True,
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "max_steps": 3,
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
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_suffix)

        current_env = os.environ.copy()
        env = {
            "NCCL_P2P_LEVEL": "LOC",
            **current_env,
            "CUDA_VISIBLE_DEVICES": "1",
        }
        vllm_process = start_vllm(
            cfg.base_model,
            env=env,
            quiet=True,
            wait=300,
            gpu_memory_utilization=0.15,
            max_model_len=cfg.vllm.max_model_len,
            enable_prefix_caching=cfg.vllm.enable_prefix_caching,
            host="0.0.0.0",
            port=8000,
        )

        try:
            execute_subprocess_async(
                [
                    "axolotl",
                    "train",
                    str(Path(temp_dir) / "config.yaml"),
                    "--num-processes",
                    "1",
                    "--main-process-port",
                    f"{get_torch_dist_unique_port()}",
                ],
                env={
                    "NCCL_P2P_LEVEL": "LOC",
                    "NCCL_DEBUG": "INFO",
                    **current_env,
                },
            )
        finally:
            recursive_kill(vllm_process)

    @require_vllm
    def test_gdpo_single_reward_fallback(self, temp_dir):
        """Test GDPO with single reward (should behave like GRPO)."""
        rnd_suffix = str(random.randint(1000, 9999))
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "chat_template": "llama3",
                "rl": "gdpo",
                "trl": {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": True,
                    "num_generations": 4,
                    # Single reward - GDPO falls back to GRPO behavior
                    "reward_funcs": [
                        f"rewards_gdpo_{rnd_suffix}.single_reward",
                    ],
                    "reward_weights": [1.0],
                    "gdpo_decoupled_norm": True,
                },
                "vllm": {
                    "max_model_len": 800,
                    "enable_prefix_caching": True,
                },
                "datasets": [
                    {
                        "path": "openai/gsm8k",
                        "name": "main",
                        "type": f"rewards_gdpo_{rnd_suffix}.oai_gsm8k_transform",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "flash_attention": True,
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "max_steps": 3,
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
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_suffix)

        current_env = os.environ.copy()
        env = {
            "NCCL_P2P_LEVEL": "LOC",
            **current_env,
            "CUDA_VISIBLE_DEVICES": "1",
        }
        vllm_process = start_vllm(
            cfg.base_model,
            env=env,
            quiet=True,
            wait=300,
            gpu_memory_utilization=0.15,
            max_model_len=cfg.vllm.max_model_len,
            enable_prefix_caching=cfg.vllm.enable_prefix_caching,
            host="0.0.0.0",
            port=8000,
        )

        try:
            execute_subprocess_async(
                [
                    "axolotl",
                    "train",
                    str(Path(temp_dir) / "config.yaml"),
                    "--num-processes",
                    "1",
                    "--main-process-port",
                    f"{get_torch_dist_unique_port()}",
                ],
                env={
                    "NCCL_P2P_LEVEL": "LOC",
                    "NCCL_DEBUG": "INFO",
                    **current_env,
                },
            )
        finally:
            recursive_kill(vllm_process)

    @require_vllm
    def test_gdpo_fft(self, temp_dir):
        """Test GDPO with full fine-tuning (no adapter)."""
        rnd_suffix = str(random.randint(1000, 9999))
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "chat_template": "llama3",
                "rl": "gdpo",
                "trl": {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": True,
                    "num_generations": 4,
                    "reward_funcs": [
                        f"rewards_gdpo_{rnd_suffix}.format_reward",
                        f"rewards_gdpo_{rnd_suffix}.correctness_reward",
                    ],
                    "reward_weights": [1.0, 2.0],
                    "gdpo_decoupled_norm": True,
                },
                "vllm": {
                    "max_model_len": 800,
                    "enable_prefix_caching": True,
                },
                "datasets": [
                    {
                        "path": "openai/gsm8k",
                        "name": "main",
                        "type": f"rewards_gdpo_{rnd_suffix}.oai_gsm8k_transform",
                    },
                ],
                # No adapter - full fine-tuning
                "flash_attention": True,
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "max_steps": 3,
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
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_suffix)

        current_env = os.environ.copy()
        env = {
            "NCCL_P2P_LEVEL": "LOC",
            **current_env,
            "CUDA_VISIBLE_DEVICES": "1",
        }
        vllm_process = start_vllm(
            cfg.base_model,
            env=env,
            quiet=True,
            wait=300,
            gpu_memory_utilization=0.15,
            max_model_len=cfg.vllm.max_model_len,
            enable_prefix_caching=cfg.vllm.enable_prefix_caching,
            host="0.0.0.0",
            port=8000,
        )

        try:
            execute_subprocess_async(
                [
                    "axolotl",
                    "train",
                    str(Path(temp_dir) / "config.yaml"),
                    "--num-processes",
                    "1",
                    "--main-process-port",
                    f"{get_torch_dist_unique_port()}",
                ],
                env={
                    "NCCL_P2P_LEVEL": "LOC",
                    "NCCL_DEBUG": "INFO",
                    **current_env,
                },
            )
        finally:
            recursive_kill(vllm_process)

    @require_vllm
    def test_gdpo_sequence_parallel(self, temp_dir):
        """Test GDPO with sequence parallelism."""
        rnd_suffix = str(random.randint(1000, 9999))
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "chat_template": "llama3",
                "rl": "gdpo",
                "context_parallel_size": 2,  # Enable sequence parallelism
                "trl": {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": True,
                    "num_generations": 4,
                    "reward_funcs": [
                        f"rewards_gdpo_{rnd_suffix}.format_reward",
                        f"rewards_gdpo_{rnd_suffix}.correctness_reward",
                    ],
                    "reward_weights": [1.0, 2.0],
                    "gdpo_decoupled_norm": True,
                },
                "vllm": {
                    "max_model_len": 800,
                    "enable_prefix_caching": True,
                },
                "datasets": [
                    {
                        "path": "openai/gsm8k",
                        "name": "main",
                        "type": f"rewards_gdpo_{rnd_suffix}.oai_gsm8k_transform",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "flash_attention": True,
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "max_steps": 3,
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 10,
                "val_set_size": 0.0,
                "output_dir": temp_dir,
                "dataset_prepared_path": temp_dir + "/last_run_prepared",
                "learning_rate": 0.0001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_suffix)

        current_env = os.environ.copy()
        env = {
            "NCCL_P2P_LEVEL": "LOC",
            **current_env,
            "CUDA_VISIBLE_DEVICES": "1",
        }
        vllm_process = start_vllm(
            cfg.base_model,
            env=env,
            quiet=True,
            wait=300,
            gpu_memory_utilization=0.15,
            max_model_len=cfg.vllm.max_model_len,
            enable_prefix_caching=cfg.vllm.enable_prefix_caching,
            host="0.0.0.0",
            port=8000,
        )

        try:
            execute_subprocess_async(
                [
                    "axolotl",
                    "train",
                    str(Path(temp_dir) / "config.yaml"),
                    "--num-processes",
                    "2",
                    "--main-process-port",
                    f"{get_torch_dist_unique_port()}",
                ],
                env={
                    "NCCL_P2P_LEVEL": "LOC",
                    "NCCL_DEBUG": "INFO",
                    **current_env,
                },
            )
        finally:
            recursive_kill(vllm_process)
