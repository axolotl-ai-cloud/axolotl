"""Test module for FSDP2 multi-GPU functionality."""

import json
import os
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import (
    check_lora_b_fully_trained,
    check_tensorboard_loss_decreased,
    most_recent_subdir,
    require_torch_2_7_0,
    requires_flash_attn,
)

pytestmark = requires_flash_attn

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


def verify_training_success(temp_dir):
    """Verify that training completed successfully — artifacts, no-NaN, loss
    stayed in qwen2-pretraining scale (tiny-qwen2-129m final pretrain CE ~3.92).
    """
    output_path = Path(temp_dir)

    model_files = list(output_path.glob("*.bin")) + list(
        output_path.glob("*.safetensors")
    )
    assert len(model_files) > 0, "No model files found - training may have failed"

    checkpoint_files = list(output_path.glob("checkpoint-*"))
    assert len(checkpoint_files) > 0, (
        "No checkpoint files found - training may have failed"
    )

    check_tensorboard_loss_decreased(
        temp_dir + "/runs",
        initial_window=10,
        final_window=10,
        max_initial=5.0,
        max_final=4.7,
    )


class TestFSDP2:
    """Test class for FSDP2 functionality."""

    @require_torch_2_7_0
    @pytest.mark.parametrize(
        "fsdp_cpu_ram_efficient_loading",
        [True, False],
    )
    def test_fft_sft(self, temp_dir, fsdp_cpu_ram_efficient_loading):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:3%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 80,
                "warmup_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 2e-4,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": fsdp_cpu_ram_efficient_loading,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "bf16": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)

    @require_torch_2_7_0
    @pytest.mark.parametrize("peft_use_dora", [True, False])
    def test_lora_sft(self, temp_dir, peft_use_dora):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:3%]",
                    },
                ],
                "peft_use_dora": peft_use_dora,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 80,
                "warmup_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-3,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "bf16": True,
                # explicitly disable LORA kernels, as they may be auto-enabled
                "lora_mlp_kernel": False,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)
        check_lora_b_fully_trained(temp_dir)

    @require_torch_2_7_0
    @pytest.mark.parametrize(
        "save_only_model, unfrozen_parameters",
        [
            pytest.param(False, None, id="optimizer_state_save"),
            pytest.param(True, None, id="model_only_save"),
            # unfreezing runs after the loader; base params it promotes must get storage too
            pytest.param(
                False,
                [".+lora_.+", "^.+layers.0.input_layernorm.weight$"],
                id="unfrozen_parameters",
            ),
        ],
    )
    def test_lora_sft_cpu_ram_efficient_loading(
        self, temp_dir, save_only_model, unfrozen_parameters
    ):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:3%]",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 80,
                "warmup_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-3,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": True,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "bf16": True,
                "save_only_model": save_only_model,
                "unfrozen_parameters": unfrozen_parameters,
                # explicitly disable LORA kernels, as they may be auto-enabled
                "lora_mlp_kernel": False,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)
        check_lora_b_fully_trained(temp_dir)

    @require_torch_2_7_0
    @pytest.mark.parametrize(
        "ep_backend",
        [
            "torch",
            pytest.param(
                "deep_ep",
                marks=pytest.mark.skipif(
                    find_spec("deep_ep") is None, reason="deep_ep not installed"
                ),
            ),
        ],
    )
    def test_lora_sft_cpu_ram_efficient_loading_expert_parallel(
        self, temp_dir, ep_backend
    ):
        # ep == world_size builds no device mesh, so nothing else initializes the process
        # group before the load; the loader must do it or non-rank-0 buffers stay on meta
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-mixtral-30m",
                "plugins": [
                    "axolotl.integrations.expert_parallel.ExpertParallelPlugin",
                ],
                "expert_parallel_size": 2,
                "expert_parallel_backend": ep_backend,
                "experts_implementation": "grouped_mm",
                "sequence_len": 1024,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:3%]",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_target_parameters": [
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                ],
                "num_epochs": 1,
                "max_steps": 80,
                "warmup_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-3,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": True,
                    "transformer_layer_cls_to_wrap": "MixtralDecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "bf16": True,
                # a LoRA-shard regression then fails on the occupancy check, not a save-time hang
                "save_only_model": True,
                "lora_mlp_kernel": False,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)
        check_lora_b_fully_trained(temp_dir)

    @require_torch_2_7_0
    @pytest.mark.parametrize("adapter", [None, "lora"], ids=["fft", "lora"])
    def test_fft_sft_expert_parallel_gradient_parity(self, temp_dir, adapter):
        """One SGD step with FSDP alone and with FSDP + pure EP must apply the same update.

        SGD makes the saved weight delta the (clipped) gradient itself, so a wrong expert
        gradient scale (the EP combine sums over ranks where FSDP averages) shows up as a
        2x update norm; Adam would normalize it away and the loss curve would still match.
        The EP checkpoint must also come back with every expert, not one rank's slice.
        """
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        pattern = "adapter_model*.safetensors" if adapter else "model*.safetensors"

        def load_safetensors(directory, glob=pattern):
            state_dict = {}
            for file in sorted(Path(directory).glob(glob)):
                state_dict.update(load_file(file))
            return state_dict

        def run(variant, expert_parallel):
            out_dir = Path(temp_dir) / variant
            cfg = DictDefault(
                {
                    "base_model": "axolotl-ai-co/tiny-mixtral-30m",
                    "experts_implementation": "grouped_mm",
                    "sequence_len": 512,
                    "val_set_size": 0,
                    "datasets": [
                        {
                            "path": "tatsu-lab/alpaca",
                            "type": "alpaca",
                            "split": "train[:1%]",
                        },
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
                    "seed": 42,
                    "bf16": True,
                    "save_strategy": "no",
                    "save_only_model": True,
                    "use_tensorboard": True,
                    "logging_steps": 1,
                }
            )
            if expert_parallel:
                cfg["plugins"] = [
                    "axolotl.integrations.expert_parallel.ExpertParallelPlugin"
                ]
                cfg["expert_parallel_size"] = 2
                cfg["expert_parallel_backend"] = "torch"
                cfg["dp_shard_size"] = 1
            if adapter:
                cfg["adapter"] = adapter
                cfg["lora_r"] = 8
                cfg["lora_alpha"] = 16
                cfg["lora_dropout"] = 0.0
                cfg["lora_target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
                cfg["lora_target_parameters"] = [
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                ]
                cfg["lora_mlp_kernel"] = False
                cfg["lora_qkv_kernel"] = False
                cfg["lora_o_kernel"] = False
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
            return load_safetensors(out_dir), read_grad_norm(out_dir)

        def read_grad_norm(out_dir):
            from tbparse import SummaryReader

            tb_log_path = most_recent_subdir(out_dir / "runs")
            scalars = SummaryReader(str(tb_log_path)).scalars
            values = scalars[scalars.tag == "train/grad_norm"].value.values
            assert len(values), "no grad_norm logged"
            return float(values[-1])

        noep, grad_norm_noep = run("noep", expert_parallel=False)
        ep, grad_norm_ep = run("ep2", expert_parallel=True)
        # clipping to max_grad_norm cancels a uniform gradient scale in the update, so the
        # pre-clip global norm is the only place such an error is visible
        grad_norm_ratio = grad_norm_ep / grad_norm_noep
        assert 0.95 < grad_norm_ratio < 1.05, (
            f"EP grad_norm {grad_norm_ep:.4f} is {grad_norm_ratio:.3f}x "
            f"the FSDP grad_norm {grad_norm_noep:.4f}"
        )
        if adapter:
            # lora_A receives no gradient while lora_B is zero, so it must come back as the
            # same seeded init in both arms; lora_B alone is the one-step update
            for key in noep:
                if "lora_A" in key:
                    assert torch.equal(ep[key], noep[key]), f"{key} differs under EP"
            init = {k: torch.zeros_like(v) for k, v in noep.items() if "lora_B" in k}
            assert init, "no lora_B parameter saved"
        else:
            # compare in the on-disk key format the trainer saves (Mixtral's legacy layout)
            init = load_safetensors(
                snapshot_download(
                    "axolotl-ai-co/tiny-mixtral-30m", allow_patterns=["*.safetensors"]
                ),
                glob="*.safetensors",
            )

        assert set(ep) == set(noep), (
            "EP checkpoint keys differ from the FSDP checkpoint"
        )
        ratios = {}
        for key, init_w in init.items():
            init_w = init_w.float()
            delta_noep = (noep[key].float() - init_w).norm().item()
            delta_ep = (ep[key].float() - init_w).norm().item()
            assert ep[key].shape == noep[key].shape, key
            if delta_noep == 0:
                assert delta_ep == 0, f"{key} moved under EP but not under FSDP"
                continue
            ratios[key] = delta_ep / delta_noep
        assert ratios, "no parameter moved in one SGD step"
        experts = {k: r for k, r in ratios.items() if "experts" in k}
        assert experts, "no expert parameter moved"
        # The expert LoRA init is drawn per module from the seed over the global expert
        # count, so the LoRA arms are as directly comparable as the full-param ones.
        for key, ratio in ratios.items():
            assert 0.85 < ratio < 1.15, (
                f"{key}: EP update norm is {ratio:.3f}x the FSDP update norm"
            )
        # clipping fixes the total update norm at lr * max_grad_norm, so a wrong global grad
        # norm (a per-rank expert norm) scales every tensor and hides in the per-key band
        total_noep = sum(
            (noep[k].float() - init[k].float()).norm() ** 2 for k in ratios
        ).sqrt()
        total_ep = sum(
            (ep[k].float() - init[k].float()).norm() ** 2 for k in ratios
        ).sqrt()
        total_ratio = (total_ep / total_noep).item()
        assert 0.995 < total_ratio < 1.005, (
            f"EP total update norm is {total_ratio:.4f}x the FSDP total update norm"
        )

    @require_torch_2_7_0
    @pytest.mark.parametrize(
        "base_model, layer_cls, expert_parallel",
        [
            pytest.param(
                "axolotl-ai-co/tiny-qwen2-129m", "Qwen2DecoderLayer", False, id="dense"
            ),
            pytest.param(
                "axolotl-ai-co/tiny-mixtral-30m", "MixtralDecoderLayer", False, id="moe"
            ),
            # ep == world_size builds no device mesh, so the loader itself must initialize the
            # process group or transformers leaves non-rank-0 buffers on meta
            pytest.param(
                "axolotl-ai-co/tiny-mixtral-30m",
                "MixtralDecoderLayer",
                True,
                id="moe_expert_parallel",
            ),
        ],
    )
    def test_cpu_ram_efficient_loading_load_probe(
        self, temp_dir, base_model, layer_cls, expert_parallel
    ):
        """Launch-only: load + accelerator.prepare, no dataset prep (its rank checks would
        initialize the process group first) and no training; asserts every rank holds real
        storage and the optimizer was remapped onto its own sharded params."""
        cfg = DictDefault(
            {
                "base_model": base_model,
                "sequence_len": 512,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:1%]",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "micro_batch_size": 1,
                "num_epochs": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-3,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": True,
                    "transformer_layer_cls_to_wrap": layer_cls,
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                },
                "bf16": True,
                "lora_mlp_kernel": False,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
            }
        )
        if "mixtral" in base_model:
            cfg["lora_target_parameters"] = [
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
            ]
            cfg["experts_implementation"] = "grouped_mm"
        if expert_parallel:
            cfg["plugins"] = [
                "axolotl.integrations.expert_parallel.ExpertParallelPlugin",
            ]
            cfg["expert_parallel_size"] = 2
            cfg["expert_parallel_backend"] = "torch"

        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        probe_dir = Path(temp_dir) / "probe"
        env = dict(os.environ, CPU_RAM_LOAD_PROBE_DIR=str(probe_dir))
        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num_processes",
                "2",
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                str(Path(__file__).parent / "_cpu_ram_efficient_load_probe.py"),
                str(Path(temp_dir) / "config.yaml"),
            ],
            env=env,
        )

        facts = {
            rank: json.loads((probe_dir / f"rank{rank}.json").read_text())
            for rank in (0, 1)
        }
        for rank, f in facts.items():
            assert not f["meta_buffers"], (
                f"rank {rank} buffers left on meta: {f['meta_buffers']}"
            )
            assert not f["trainable_meta_params"], (
                f"rank {rank} trainable params left on meta: {f['trainable_meta_params']}"
            )
            assert not f["meta_params_after_prepare"], (
                f"rank {rank} params still on meta after prepare: "
                f"{f['meta_params_after_prepare']}"
            )
            # the bug signature: every optimizer slot collapsed onto one param
            assert f["optimizer_distinct_params"] == f["optimizer_slots"], (
                f"rank {rank} optimizer holds {f['optimizer_distinct_params']} distinct params "
                f"across {f['optimizer_slots']} slots; accelerate keyed its FSDP2 remap on "
                "data_ptr() and meta params collapsed onto one key"
            )
            assert f["optimizer_params_are_model_trainable"], (
                f"rank {rank} optimizer params are not the model's trainable params"
            )
            assert f["weight_checksums"] == facts[0]["weight_checksums"], (
                f"rank {rank} weights differ from rank 0 after the broadcast"
            )
        assert facts[0]["num_meta_params"] == 0
        # the non-rank-0 base model must still be on meta, or the CPU-RAM saving is gone
        assert 0 < facts[1]["num_meta_params"] < facts[1]["num_params"]
        # the probe only proves anything if nothing else initialized the process group first
        assert not facts[1]["dist_initialized_before_load"]

    @require_torch_2_7_0
    def test_lora_sft_kernels(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:3%]",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 80,
                "warmup_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-3,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "bf16": True,
                "lora_mlp_kernel": True,
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)

    @require_torch_2_7_0
    def test_qlora_sft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:3%]",
                    },
                ],
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 80,
                "warmup_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-3,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "bf16": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)

    @require_torch_2_7_0
    def test_qlora_sft_kernels(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:3%]",
                    },
                ],
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 80,
                "warmup_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-3,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "bf16": True,
                "lora_mlp_kernel": True,
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)

    @pytest.mark.skip(reason="slow test w cu129 + torch 2.9.1 + py3.12")
    @require_torch_2_7_0
    def test_dpo_fft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "rl": "dpo",
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "Intel/orca_dpo_pairs",
                        "split": "train[:100]",
                        "type": "chatml.intel",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 20,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 2e-4,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)

    @pytest.mark.skip(reason="slow test w cu129 + torch 2.9.1 + py3.12")
    @require_torch_2_7_0
    def test_dpo_lora(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "rl": "dpo",
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "Intel/orca_dpo_pairs",
                        "split": "train[:100]",
                        "type": "chatml.intel",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 20,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-3,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "seed": 42,
                "sample_packing": True,
                "pad_to_sequence_len": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        verify_training_success(temp_dir)
