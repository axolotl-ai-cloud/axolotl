"""Compare resumed EP training with uninterrupted full-parameter and LoRA runs."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from transformers.testing_utils import get_torch_dist_unique_port

pytestmark = pytest.mark.gpu

NUM_EXPERTS = 8  # axolotl-ai-co/tiny-mixtral-30m
LORA_R = 8


def _config(
    out_dir: Path, max_steps: int, resume: str | None, adapter: str | None
) -> Path:
    cfg = {
        "base_model": "axolotl-ai-co/tiny-mixtral-30m",
        "experts_implementation": "grouped_mm",
        "sequence_len": 256,
        "val_set_size": 0,
        "seed": 42,
        "datasets": [
            {"path": "tatsu-lab/alpaca", "type": "alpaca", "split": "train[:1%]"}
        ],
        "max_steps": max_steps,
        "warmup_steps": 0,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "output_dir": str(out_dir),
        "learning_rate": 1e-3,
        "optimizer": "adamw_torch",
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
        "bf16": True,
        "save_strategy": "steps",
        "save_steps": 1,
        "logging_steps": 1,
        "plugins": ["axolotl.integrations.expert_parallel.ExpertParallelPlugin"],
        "expert_parallel_size": 2,
        "expert_parallel_backend": "torch",
        "dp_shard_size": 1,
    }
    if adapter:
        cfg.update(
            {
                "adapter": adapter,
                "lora_r": LORA_R,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_target_parameters": [
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                ],
                "lora_mlp_kernel": False,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
            }
        )
    if resume:
        cfg["resume_from_checkpoint"] = resume
    path = out_dir.with_suffix(".yaml")
    path.write_text(yaml.safe_dump(cfg))
    return path


def _train(config: Path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "axolotl.cli.main",
            "train",
            str(config),
            "--num-processes",
            "2",
            "--main-process-port",
            str(get_torch_dist_unique_port()),
        ],
        check=True,
    )


def _experts(state_dict):
    return {k: v for k, v in state_dict.items() if "experts" in k}


def _experts_held(name, shape):
    """Experts along a tensor's experts axis (PEFT's expert LoRA packs ``r`` per expert:
    ``lora_A`` ``[E*r, in]``, ``lora_B`` ``[out, r*E]``)."""
    if "lora_B" in name:
        return shape[1] // LORA_R
    if "lora_A" in name:
        return shape[0] // LORA_R
    return shape[0]


def _assert_optimizer_matches(expected, resumed):
    assert expected["state"], "Uninterrupted run saved no optimizer state"
    assert resumed["state"].keys() == expected["state"].keys()
    assert resumed["param_groups"] == expected["param_groups"]
    for name, state in expected["state"].items():
        restored = resumed["state"][name]
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert key in state and key in restored, f"{name}: missing {key}"
            # Allow BF16 gradient roundoff while comparing moments at their own scale.
            torch.testing.assert_close(
                restored[key],
                state[key],
                rtol=0 if key == "step" else 0.02,
                atol=0 if key == "step" else 1e-8,
                msg=lambda message, n=name, k=key: f"{n} {k}: {message}",
            )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("adapter", [None, "lora"], ids=["full", "lora"])
def test_ep_checkpoint_keeps_every_ep_ranks_experts_and_optimizer_state(
    tmp_path, adapter
):
    from safetensors.torch import load_file

    failures = []
    straight = tmp_path / "straight"
    _train(_config(straight, 3, None, adapter))
    checkpoint = straight / "checkpoint-2"

    # the checkpoint's model and optimizer state must hold all experts, not one EP group's
    model_state = torch.load(checkpoint / "pytorch_model_fsdp.bin", weights_only=True)
    if not _experts(model_state):
        failures.append("pytorch_model_fsdp.bin holds no expert tensors")
    for name, value in _experts(model_state).items():
        if _experts_held(name, value.shape) != NUM_EXPERTS:
            failures.append(f"pytorch_model_fsdp.bin {name}: {tuple(value.shape)}")
    optimizer_state = torch.load(checkpoint / "optimizer.bin", weights_only=True)
    for key, state in optimizer_state["state"].items():
        for moment in ("exp_avg", "exp_avg_sq"):
            if (
                "experts" in str(key)
                and _experts_held(str(key), state[moment].shape) != NUM_EXPERTS
            ):
                failures.append(
                    f"optimizer.bin {key} {moment}: {tuple(state[moment].shape)}"
                )

    # resuming at step 2 and training to step 3 must match the uninterrupted run
    resumed_dir = tmp_path / "resumed"
    _train(_config(resumed_dir, 3, str(checkpoint), adapter))
    expected_optimizer = torch.load(
        straight / "checkpoint-3" / "optimizer.bin", weights_only=True
    )
    resumed_optimizer = torch.load(
        resumed_dir / "checkpoint-3" / "optimizer.bin", weights_only=True
    )
    _assert_optimizer_matches(expected_optimizer, resumed_optimizer)
    final = "adapter_model.safetensors" if adapter else "model.safetensors"
    expected = load_file(straight / final)
    resumed = load_file(resumed_dir / final)
    for name, value in expected.items():
        diff = (value.float() - resumed[name].float()).abs().max().item()
        if diff > 0.02:  # bf16 noise is ~2e-3; a lost EP group differs by ~0.2
            failures.append(f"resumed {name}: max diff {diff:.3g}")
    assert not failures, "\n".join(failures)
