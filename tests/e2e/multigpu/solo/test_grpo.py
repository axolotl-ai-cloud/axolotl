"""
GRPO test suite
"""

import os
import random
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest
import requests
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import require_vllm


def start_vllm(
    model: str, env: dict, wait: int | None = None, quiet=False, **kwargs
) -> subprocess.Popen:
    """
    helper function to start the VLLM server in the background, mostly for testing purposes
    """
    cmd = [sys.executable, "-m", "trl.scripts.vllm_serve", "--model", model]

    if tensor_parallel_size := kwargs.get("tensor_parallel_size"):
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if host := kwargs.get("host"):
        cmd.extend(["--host", host])
    if port := kwargs.get("port"):
        cmd.extend(["--port", str(port)])
    if gpu_memory_utilization := kwargs.get("gpu_memory_utilization"):
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if dtype := kwargs.get("dtype"):
        cmd.extend(["--dtype", dtype])
    if max_model_len := kwargs.get("max_model_len"):
        cmd.extend(["--max-model-len", str(max_model_len)])
    if kwargs.get("enable_prefix_caching"):
        cmd.extend(["--enable-prefix-caching", "True"])

    # print out the command to be executed
    print(" ".join(cmd))

    vllm_logging_json = Path(tempfile.mkdtemp()) / "vllm_logging.json"
    with open(vllm_logging_json, "w", encoding="utf-8") as temp_file:
        temp_file.write(
            """{
  "formatters": {
    "json": {
      "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
    }
  },
  "handlers": {
    "file": {
      "class": "logging.FileHandler",
      "formatter": "json",
      "level": "DEBUG",
      "filename": "/tmp/vllm.log",
      "mode": "a"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["file"],
      "level": "DEBUG",
      "propagate": false
    }
  },
  "version": 1
}"""
        )

    cmd_env = env.copy()
    cmd_env.update({"VLLM_LOGGING_CONFIG_PATH": vllm_logging_json})
    # start `trl vllm-serve` command in the background and capture the process id
    process = subprocess.Popen(
        cmd,
        env=cmd_env,
        stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else subprocess.PIPE,
    )  # nosec B603

    # print out the process id so the user can easily kill it later
    print(f"VLLM server process started (PID: {process.pid})")

    # wait until the http server is ready, even if it 404s, but timeout after 60 seconds
    period_seconds = 5
    started = False
    if wait and host and port:
        for i in range(0, int(wait), period_seconds):
            try:
                response = requests.get(f"http://{host}:{port}", timeout=1)
                print(f"{i}: VLLM server (status: {response.status_code})")
                if int(response.status_code) in [200, 404]:
                    started = True
                    break
            except requests.exceptions.RequestException as exc:
                print(f"{i}: VLLM server failed to start: {str(exc)}")

            # also check if the process.pid is still running
            if process.poll() is not None:
                break

            time.sleep(period_seconds)

    if wait and not started:
        print(
            f"VLLM server process did not start within {wait} seconds. Please check your server logs."
        )
        recursive_kill(process)
        with open("/tmp/vllm.log", "r", encoding="utf-8") as log_file:
            print(log_file.read())
        try:
            os.remove("/tmp/vllm.log")
        except FileNotFoundError:
            pass
        raise RuntimeError(f"VLLM server process did not start within {wait} seconds.")

    # return the process
    return process


def recursive_kill(process: subprocess.Popen):
    """
    Recursively kill a process and its children
    """
    process = psutil.Process(process.pid)
    for child in psutil.Process(process.pid).children(recursive=True):
        child.terminate()
        child.kill()
        os.kill(child.pid, 9)
    process.terminate()
    process.kill()
    os.kill(process.pid, 9)


@pytest.mark.skip(reason="flaky vllm tests in modal")
class TestGRPO:
    """
    Test case for GRPO training using multiple GPUs
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
                    "num_generations": 4,
                    "reward_funcs": [f"rewards_{rnd_reward_suffix}.rand_reward_func"],
                },
                "vllm": {
                    "max_model_len": 800,
                    "enable_prefix_caching": True,
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

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_reward_suffix)

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
            (recursive_kill(vllm_process))

    @require_vllm
    def test_llama_lora_sp(self, temp_dir):
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
                    "num_generations": 4,
                    "reward_funcs": [f"rewards_{rnd_reward_suffix}.rand_reward_func"],
                },
                "vllm": {
                    "max_model_len": 800,
                    "enable_prefix_caching": True,
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
                "context_parallel_size": 2,
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
                "use_tensorboard": True,
                "save_first_step": False,
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_reward_suffix)

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
                    str(2),
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
                    "num_generations": 4,
                    "reward_funcs": [f"rewards_{rnd_reward_suffix}.rand_reward_func"],
                },
                "vllm": {
                    "max_model_len": 800,
                    "enable_prefix_caching": True,
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
                "use_tensorboard": True,
                "save_first_step": False,
            }
        )

        self._utils_write_yaml_and_rewards(cfg, temp_dir, suffix=rnd_reward_suffix)

        current_env = os.environ.copy()
        env = {
            "NCCL_P2P_LEVEL": "LOC",  # nccl can be brittle, assume P2P isn't reliable
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
