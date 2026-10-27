"""GRPO with ``lora_merge_aware: bf16`` and colocated vLLM, on one GPU and under FSDP2.

A probe plugin, written next to the config like the reward module, records on every
rank:

- how many LoRA linears run the merge-aware forward;
- at every vLLM weight sync, whether each LoRA linear's weight handed to vLLM equals
  ``merged_lora_weight`` of the current, unmerged weights (what the merge-aware forward
  computes);
- whether the base weights still equal their pre-training values at the end, and the
  final global step.
"""

import json
import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import require_vllm

pytestmark = pytest.mark.gpu

MAX_STEPS = 3

HELPERS = """
import json
import os
import random

import torch
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin


def rand_reward_func(completions, **kwargs):
    return [random.uniform(0, 1) for _ in completions]


def gsm8k_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [{"role": "user", "content": example["question"]}],
            "answer": example["answer"].split("####")[-1].strip(),
        }

    return transform_fn, {"remove_columns": ["question"]}


def _full(tensor):
    from torch.distributed.tensor import DTensor

    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


class _Report(dict):
    def __init__(self, path, **fields):
        super().__init__(**fields)
        self.path = path

    def write(self):
        with open(self.path, "w", encoding="utf-8") as fout:
            json.dump(self, fout)


class _CheckBaseAtTrainEnd(TrainerCallback):
    def __init__(self, probe):
        self.probe = probe

    def on_train_end(self, args, state, control, **kwargs):
        self.probe.finish(state.global_step)


class MergeAwareProbePlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        return [_CheckBaseAtTrainEnd(self)]

    def post_trainer_create(self, cfg, trainer):
        from peft.tuners.lora.layer import Linear as LoraLinear

        from axolotl.monkeypatch.lora_merge_aware import _merge_aware_forward
        from axolotl.utils.lora_merge_precision import merged_lora_weight

        generation = trainer.vllm_generation
        self.layers = {
            name: module
            for name, module in generation.model.named_modules()
            if isinstance(module, LoraLinear)
        }
        self.base = {
            name: _full(m.base_layer.weight).detach().cpu().clone()
            for name, m in self.layers.items()
        }
        rank = os.environ.get("RANK", "0")
        self.report = _Report(
            os.path.join(cfg.output_dir, f"merge_aware_probe.rank{rank}.json"),
            lora_linears=len(self.layers),
            merge_aware=sum(
                getattr(m, "_axolotl_lora_merge_aware_dtype", None) is not None
                and getattr(m.__dict__.get("forward"), "__func__", None)
                is _merge_aware_forward
                for m in self.layers.values()
            ),
            syncs=0,
            checked=0,
            mismatched=0,
            unseen=0,
        )
        expected = {}
        inner_iter = generation._iter_named_params

        def iter_and_check():
            for name, tensor in inner_iter():
                want = expected.pop(name, None)
                if want is not None:
                    self.report["checked"] += 1
                    self.report["mismatched"] += int(not torch.equal(tensor, want))
                yield name, tensor

        generation._iter_named_params = iter_and_check
        inner_sync = generation.sync_weights

        def sync_and_check(*args, **kwargs):
            with torch.no_grad():
                for name, m in self.layers.items():
                    served = name.removeprefix("base_model.model.").replace(
                        "_checkpoint_wrapped_module.", ""
                    )
                    expected[served + ".weight"] = merged_lora_weight(
                        _full(m.base_layer.weight),
                        _full(m.lora_A["default"].weight),
                        _full(m.lora_B["default"].weight),
                        m.scaling["default"],
                        torch.bfloat16,
                    )
            result = inner_sync(*args, **kwargs)
            self.report["syncs"] += 1
            self.report["unseen"] += len(expected)
            expected.clear()
            self.report.write()
            return result

        generation.sync_weights = sync_and_check

    def finish(self, global_step):
        with torch.no_grad():
            changed = sum(
                int((_full(m.base_layer.weight).cpu() != self.base[name]).sum())
                for name, m in self.layers.items()
            )
        self.report.update(global_step=global_step, base_changed=changed)
        self.report.write()
"""


@require_vllm
@pytest.mark.parametrize("num_gpus", [1, 2], ids=["single_gpu", "fsdp2"])
def test_grpo_lora_merge_aware_colocate(temp_dir, num_gpus):
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    (Path(temp_dir) / "merge_aware_grpo_helpers.py").write_text(HELPERS)
    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "chat_template": "llama3",
            "special_tokens": {"pad_token": "<|endoftext|>"},
            "rl": "grpo",
            "plugins": ["merge_aware_grpo_helpers.MergeAwareProbePlugin"],
            "trl": {
                "use_vllm": True,
                "vllm_mode": "colocate",
                "beta": 0.0,
                "num_generations": 4,
                "max_completion_length": 32,
                "reward_funcs": ["merge_aware_grpo_helpers.rand_reward_func"],
            },
            "vllm": {"gpu_memory_utilization": 0.25, "max_model_len": 512},
            "datasets": [
                {
                    "path": "openai/gsm8k",
                    "name": "main",
                    "split": "train[:64]",
                    "type": "merge_aware_grpo_helpers.gsm8k_transform",
                },
            ],
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_linear": True,
            "lora_merge_aware": "bf16",
            "sequence_len": 512,
            "max_steps": MAX_STEPS,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 0,
            "val_set_size": 0.0,
            "output_dir": temp_dir,
            "learning_rate": 1e-5,
            "optimizer": "adamw_torch_fused",
            "lr_scheduler": "constant",
            "attn_implementation": "sdpa",
            "bf16": True,
            "save_first_step": False,
        }
    )
    if num_gpus == 2:
        cfg.fsdp_version = 2
        cfg.fsdp_config = {
            "offload_params": False,
            "cpu_ram_efficient_loading": False,
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "state_dict_type": "FULL_STATE_DICT",
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "reshard_after_forward": True,
        }
    with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (temp_dir, env.get("PYTHONPATH")) if p
    )
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
        env=env,
    )

    assert (Path(temp_dir) / "adapter_model.safetensors").exists()
    for rank in range(num_gpus):
        report = json.loads(
            (Path(temp_dir) / f"merge_aware_probe.rank{rank}.json").read_text()
        )
        assert report["lora_linears"] > 0, report
        assert report["merge_aware"] == report["lora_linears"], report
        assert report["global_step"] == MAX_STEPS, report
        assert report["syncs"] >= MAX_STEPS, report
        assert report["checked"] == report["syncs"] * report["lora_linears"], report
        assert report["mismatched"] == 0 and report["unseen"] == 0, report
        assert report["base_changed"] == 0, report
