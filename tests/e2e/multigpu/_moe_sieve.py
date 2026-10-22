"""Bounded CUDA verification of the committed MoE-Sieve branch."""

import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

ROOT = Path(os.environ.get("MOE_SIEVE_TEST_OUTPUT", "/tmp/moe-sieve-distributed"))


def command(name, args, timeout=360):
    path = ROOT / f"{name}.log"
    start = time.monotonic()
    with path.open("w") as out:
        result = subprocess.run(
            args,
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=os.environ.copy(),
        )
    info = {
        "exit_code": result.returncode,
        "seconds": round(time.monotonic() - start, 2),
        "log": path.read_text(),
    }
    if result.returncode:
        raise RuntimeError(f"{name} failed: {info['log'][-12000:]}")
    return info


def main():
    ROOT.mkdir(exist_ok=True)
    report = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "trl", "accelerate")
        },
        "gpus": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True
        ).strip(),
        "results": {},
    }
    try:
        assert report["gpus"]
        torch.manual_seed(42)
        base_dir = ROOT / "base"
        config = Qwen3MoeConfig(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=384,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=8,
            num_experts_per_tok=2,
            _attn_implementation="eager",
        )
        base = Qwen3MoeForCausalLM(config)
        report["model_parameters"] = sum(p.numel() for p in base.parameters())
        base.save_pretrained(base_dir)
        tokenizer = Tokenizer(
            WordLevel({f"t{i}": i for i in range(256)}, unk_token="t1")
        )
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="t0", unk_token="t1", eos_token="t2"
        ).save_pretrained(base_dir)
        generator = torch.Generator().manual_seed(123)
        rows = []
        for _ in range(32):
            ids = torch.randint(3, 256, (128,), generator=generator).tolist()
            rows.append(
                {"input_ids": ids, "labels": ids, "attention_mask": [1] * len(ids)}
            )
        data = ROOT / "train.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in rows))
        cfg = dict(
            base_model=str(base_dir),
            adapter="moe_sieve",
            plugins=["axolotl.integrations.moe_sieve.MoeSievePlugin"],
            moe_sieve=dict(
                selection_file=str(ROOT / "selection.json"),
                fraction=0.25,
                calibration_samples=8,
            ),
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0,
            lora_target_linear=True,
            lora_target_parameters=["gate.weight"],
            experts_implementation="eager",
            datasets=[dict(path=str(data), type="completion")],
            skip_prepare_dataset=True,
            dataset_num_proc=1,
            output_dir=str(ROOT / "single"),
            sequence_len=128,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            val_set_size=0,
            num_epochs=1,
            max_steps=2,
            sample_packing=False,
            bf16=True,
            fp16=False,
            flash_attention=False,
            attn_implementation="eager",
            save_steps=1,
            logging_steps=1,
            optimizer="adamw_torch",
            learning_rate=0.001,
            lr_scheduler="constant",
            warmup_steps=0,
            use_tensorboard=False,
            use_wandb=False,
            use_trackio=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=42,
            dataloader_num_workers=0,
        )
        cfg_path = ROOT / "train.yml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        report["results"]["calibration"] = command(
            "calibration",
            [
                sys.executable,
                "-m",
                "axolotl.integrations.moe_sieve.profile",
                str(cfg_path),
            ],
        )
        profile = json.loads((ROOT / "selection.json").read_text())
        assert len(profile["selection"]) == 4
        assert all(
            len(spec["selected_experts"]) == 2 for spec in profile["selection"].values()
        )
        report["selection"] = profile["selection"]
        count = torch.cuda.device_count()
        topologies = {
            2: {
                "fsdp2": {"dp_shard_size": 2},
                "ep": {"expert_parallel_size": 2},
                "cp": {"context_parallel_size": 2},
            },
            4: {
                "fsdp2_ep": {"dp_shard_size": 2, "expert_parallel_size": 2},
                "cp_ep": {"context_parallel_size": 2, "expert_parallel_size": 2},
                "hsdp": {"dp_replicate_size": 2, "dp_shard_size": 2},
            },
            8: {
                "hsdp_ep": {
                    "dp_replicate_size": 2,
                    "dp_shard_size": 2,
                    "expert_parallel_size": 2,
                },
                "fsdp2_cp_ep": {
                    "dp_shard_size": 2,
                    "context_parallel_size": 2,
                    "expert_parallel_size": 2,
                },
            },
        }[count]
        requested = os.environ.get("MOE_SIEVE_TEST_CASE")
        if requested:
            topologies = {requested: topologies[requested]}
        if os.environ.get("MOE_SIEVE_EMPTY_OWNER") == "1":
            for spec in profile["selection"].values():
                spec["selected_experts"] = [0, 1]
            (ROOT / "selection.json").write_text(json.dumps(profile))
        for mode, topology in topologies.items():
            current = copy.deepcopy(cfg)
            current["output_dir"] = str(ROOT / mode)
            current["ddp_find_unused_parameters"] = False
            current.update(topology)
            kernel = os.environ.get("MOE_SIEVE_EXPERT_KERNEL", "eager")
            if kernel != "eager":
                current["plugins"].append("axolotl.integrations.kernels.KernelsPlugin")
                current[f"use_{kernel}"] = True
                current["experts_implementation"] = kernel
            current["fsdp_version"] = 2
            current["fsdp_config"] = dict(
                offload_params=False,
                cpu_ram_efficient_loading=False,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_layer_cls_to_wrap="Qwen3MoeDecoderLayer",
                state_dict_type="FULL_STATE_DICT",
                reshard_after_forward=True,
            )
            current["expert_parallel_backend"] = "torch"
            current["attn_implementation"] = "sdpa"
            if topology.get("context_parallel_size", 1) > 1:
                current["context_parallel"] = dict(
                    size=2, backend="ulysses", load_balance="none"
                )
            path = ROOT / f"{mode}.yml"
            path.write_text(yaml.safe_dump(current))
            launcher = (
                [sys.executable, "-m", "axolotl.cli.train"]
                if count == 1
                else [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={count}",
                    "-m",
                    "axolotl.cli.train",
                ]
            )
            report["results"][f"{mode}_train"] = command(
                f"{mode}_train", launcher + [str(path)]
            )
            checkpoint = ROOT / mode / "checkpoint-2"
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            assert state["global_step"] == 2
            assert list(checkpoint.glob("*optimizer*"))
            saved_config = json.loads((checkpoint / "adapter_config.json").read_text())
            assert saved_config["moe_sieve_selection"] == profile["selection"]
            current["resume_from_checkpoint"] = str(checkpoint)
            current["max_steps"] = 3
            path.write_text(yaml.safe_dump(current))
            report["results"][f"{mode}_resume"] = command(
                f"{mode}_resume", launcher + [str(path)]
            )
            final_state = json.loads(
                (ROOT / mode / "checkpoint-3" / "trainer_state.json").read_text()
            )
            assert final_state["global_step"] == 3
            losses = [
                item["loss"] for item in final_state["log_history"] if "loss" in item
            ]
            assert losses and all(torch.isfinite(torch.tensor(loss)) for loss in losses)
            report["results"][f"{mode}_resume"]["losses"] = losses
            current.pop("resume_from_checkpoint")
            current["output_dir"] = str(ROOT / f"{mode}_uninterrupted")
            path.write_text(yaml.safe_dump(current))
            report["results"][f"{mode}_uninterrupted"] = command(
                f"{mode}_uninterrupted", launcher + [str(path)]
            )
            from safetensors.torch import load_file

            resumed = load_file(str(ROOT / mode / "adapter_model.safetensors"))
            uninterrupted = load_file(
                str(ROOT / f"{mode}_uninterrupted" / "adapter_model.safetensors")
            )
            assert resumed.keys() == uninterrupted.keys()
            for key in resumed:
                torch.testing.assert_close(
                    resumed[key], uninterrupted[key], atol=1e-6, rtol=1e-5, msg=key
                )
            report["results"][f"{mode}_resume"]["matches_uninterrupted"] = True
        report["status"] = "supported_paths_passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        report["logs"] = {path.name: path.read_text() for path in ROOT.glob("*.log")}
        (ROOT / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("logs", "results")
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
    if (
        json.loads((ROOT / "report.json").read_text())["status"]
        != "supported_paths_passed"
    ):
        raise SystemExit(1)
