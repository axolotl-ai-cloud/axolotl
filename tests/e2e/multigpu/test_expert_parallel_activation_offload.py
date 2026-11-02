"""Activation offloading over the torch expert-parallel dispatch."""

from pathlib import Path

import pytest
import torch
import yaml
from accelerate.test_utils import execute_subprocess_async
from huggingface_hub import snapshot_download
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.multigpu.test_expert_parallel_sac_offload import (
    _load_safetensors,
    _read_grad_norm,
)

MODEL = "axolotl-ai-co/tiny-mixtral-30m"


def _train_one_step(out_dir, **overrides):
    cfg = DictDefault(
        {
            "base_model": MODEL,
            "experts_implementation": "grouped_mm",
            "sequence_len": 512,
            "val_set_size": 0,
            "datasets": [
                {"path": "tatsu-lab/alpaca", "type": "alpaca", "split": "train[:1%]"},
            ],
            "max_steps": 1,
            "warmup_steps": 0,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "output_dir": str(out_dir),
            "learning_rate": 10.0,
            "optimizer": "sgd",
            "lr_scheduler": "constant",
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
            "fsdp_version": 2,
            "fsdp_config": {
                "offload_params": False,
                "cpu_ram_efficient_loading": True,
                "transformer_layer_cls_to_wrap": "MixtralDecoderLayer",
                "state_dict_type": "FULL_STATE_DICT",
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "reshard_after_forward": True,
            },
            "plugins": ["axolotl.integrations.expert_parallel.ExpertParallelPlugin"],
            "expert_parallel_size": 2,
            "expert_parallel_backend": "torch",
            "dp_shard_size": 1,
            "seed": 42,
            "bf16": True,
            "save_strategy": "no",
            "save_only_model": True,
            "use_tensorboard": True,
            "logging_steps": 1,
            **overrides,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))
    execute_subprocess_async(
        [
            "axolotl",
            "train",
            str(out_dir / "config.yaml"),
            "--num-processes",
            "2",
            "--main-process-port",
            f"{get_torch_dist_unique_port()}",
        ]
    )
    return _load_safetensors(out_dir), _read_grad_norm(out_dir)


@pytest.fixture(scope="module")
def checkpointing_reference(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("ep_offload") / "checkpointing"
    return _train_one_step(
        out_dir,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


@pytest.mark.parametrize(
    "gradient_checkpointing", [True, False], ids=["recompute", "no_recompute"]
)
def test_activation_offloading_matches_checkpointing(
    temp_dir, checkpointing_reference, gradient_checkpointing
):
    """One SGD step with EP2 applies the same update with activation_offloading: true
    as under plain checkpointing. With recompute, the routing saves must still apply
    under the offloader: a re-run router topk would desync the all-to-all shapes."""
    ref, grad_norm_ref = checkpointing_reference
    overrides = {
        "activation_offloading": True,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if gradient_checkpointing:
        overrides["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    test, grad_norm_test = _train_one_step(Path(temp_dir) / "offload", **overrides)

    grad_norm_ratio = grad_norm_test / grad_norm_ref
    assert 0.95 < grad_norm_ratio < 1.05, (
        f"offload grad_norm {grad_norm_test:.4f} is {grad_norm_ratio:.3f}x "
        f"the reference grad_norm {grad_norm_ref:.4f}"
    )

    init = _load_safetensors(
        snapshot_download(MODEL, allow_patterns=["*.safetensors"]),
        glob="*.safetensors",
    )
    assert set(test) == set(ref), "checkpoint keys differ between the two arms"
    ratios = {}
    for key, init_w in init.items():
        init_w = init_w.float()
        delta_ref = (ref[key].float() - init_w).norm().item()
        delta_test = (test[key].float() - init_w).norm().item()
        if delta_ref == 0:
            assert delta_test == 0, f"{key} moved only under offload"
            continue
        ratios[key] = delta_test / delta_ref
    assert any("experts" in k for k in ratios), "no expert parameter moved"
    for key, ratio in ratios.items():
        assert 0.85 < ratio < 1.15, (
            f"{key}: offload update norm is {ratio:.3f}x the reference"
        )
    assert torch.isfinite(torch.tensor(grad_norm_test))
