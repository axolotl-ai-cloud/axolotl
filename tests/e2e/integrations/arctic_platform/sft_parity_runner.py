# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Single-run driver for the native-vs-AP SFT loss parity test.

Run once per side; writes per-step train losses to a JSON file:

* ``--mode native`` — axolotl baseline, launched under the ``deepspeed`` launcher
  (e.g. ``deepspeed --num_gpus=1 --no_local_rank sft_parity_runner.py
  --mode native ...``). Uses ``cfg.deepspeed`` = the shared DeepSpeed config.
* ``--mode arctic`` — Arctic Platform SFT via ``ArcticSFTPlugin``, run as a plain
  CPU-only client process (the server owns the GPU + its own DeepSpeed launch).

Both sides run DeepSpeed with the *identical* config (see ``shared_ds_config``):
bf16 mixed precision, ZeRO-0, DeepSpeed's default FusedAdam, same gradient
clipping / lr / betas / eps. Same config on the same batches → identical loss.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

# Shared hyperparameters (imported by the orchestrating pytest for assertions).
MAX_STEPS = 3
LEARNING_RATE = 1e-3
SEQUENCE_LEN = 128
SEED = 42
MAX_GRAD_NORM = 1.0
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
# FlashAttention-2 2.8.3 is ABI-incompatible with torch 2.12.x in this env
# (undefined symbol after the setup.sh torch upgrade). Use SDPA on both sides
# so parity is still apples-to-apples; switch back once a matching FA2/FA3
# wheel is installed.
ATTN_IMPLEMENTATION = "sdpa"


def shared_ds_config() -> dict:
    """DeepSpeed config used verbatim by BOTH sides.

    bf16 mixed precision + DeepSpeed default **FusedAdam** (no ``torch_adam``).
    ``grad_accum_dtype`` must be bf16 to match the bf16 model (DeepSpeed rejects
    a bf16 model with fp32 grad accumulation). The extra keys mirror the Arctic
    server's ``ds_training_config`` setdefaults so both effective configs match.
    """
    return {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 0},
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LEARNING_RATE,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },
        "gradient_clipping": MAX_GRAD_NORM,
        "torch_autocast": {"enabled": True, "dtype": "bfloat16"},
        "communication_data_type": "fp32",
        "data_types": {"grad_accum_dtype": "bf16"},
    }


def _base_cfg(output_dir: str) -> DictDefault:
    return DictDefault(
        {
            "base_model": BASE_MODEL,
            "tokenizer_type": "AutoTokenizer",
            "sequence_len": SEQUENCE_LEN,
            "sample_packing": False,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "val_set_size": 0.0,
            "special_tokens": {"pad_token": "<|endoftext|>"},
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                    "split": "train[:32]",
                }
            ],
            "dataset_num_proc": 1,
            "dataloader_num_workers": 0,
            "num_epochs": 1,
            "max_steps": MAX_STEPS,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": LEARNING_RATE,
            "optimizer": "adamw_torch",  # ignored: DeepSpeed config supplies FusedAdam
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": MAX_GRAD_NORM,
            "lr_scheduler": "constant",
            "warmup_steps": 0,
            "warmup_ratio": 0.0,
            "logging_steps": 1,
            "save_strategy": "no",
            "save_first_step": False,
            "seed": SEED,
            "output_dir": output_dir,
            "use_tensorboard": False,
            "bf16": True,
            "tf32": False,
            # Required once ArcticSFTPlugin installs a stub model whose forward
            # signature does not declare input_ids/labels (same as Hatchery).
            "remove_unused_columns": False,
        }
    )


def _step_losses(trainer) -> list[float]:
    losses = [
        float(entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry and "train_runtime" not in entry
    ]
    return losses[:MAX_STEPS]


def _run_native(output_dir: str) -> list[float]:
    cfg = _base_cfg(output_dir)
    cfg.deepspeed = DictDefault(deepcopy(shared_ds_config()))
    cfg = validate_config(cfg)
    normalize_config(cfg)
    dataset_meta = load_datasets(cfg=cfg)
    _, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
    return _step_losses(trainer)


def _run_arctic(output_dir: str, port: int) -> list[float]:
    cfg = _base_cfg(output_dir)
    # CPU-only client: it only dispatches, computing no loss, so its local HF
    # TrainingArguments must not request bf16 (HF rejects bf16 without a GPU).
    # Server-side bf16 mixed precision still comes from ds_config below — this
    # is a client-side no-op, not a disabled feature.
    cfg.bf16 = False
    cfg.tf32 = False
    cfg.plugins = ["axolotl.integrations.arctic_platform.sft.ArcticSFTPlugin"]
    cfg.arctic_sft = {
        "host": "127.0.0.1",
        "port": port,
        "training_gpus": 1,
        "launch_local_server": True,
        "server_cuda_visible_devices": "0",
        "loss_fn": "sft",
        "attn_implementation": ATTN_IMPLEMENTATION,
        "gradient_checkpointing": False,
        "checkpoint_path": f"{output_dir}/server_ckpt",
        "startup_timeout": 600.0,
        "job_ready_timeout": 600.0,
        # Same DeepSpeed config the axolotl baseline uses.
        "ds_config": deepcopy(shared_ds_config()),
        "training_config": {
            "optimizer": {
                "lr": LEARNING_RATE,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "gradient_clipping": MAX_GRAD_NORM,
            },
            "lr_scheduler": {"warmup_ratio": 0.0},
            "training_horizon": MAX_STEPS,
            "max_length": SEQUENCE_LEN,
            "gradient_accumulation_steps": 1,
        },
    }
    prepare_plugins(cfg)
    cfg = validate_config(cfg)
    normalize_config(cfg)
    dataset_meta = load_datasets(cfg=cfg)
    _, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
    losses = _step_losses(trainer)
    client = getattr(trainer, "_client", None)
    if client is not None:
        client.shutdown()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["native", "arctic"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--losses-out", required=True)
    parser.add_argument("--port", type=int, default=0)
    # The deepspeed launcher may still inject --local_rank; accept + ignore it.
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "native":
        losses = _run_native(args.output_dir)
    else:
        losses = _run_arctic(args.output_dir, args.port)

    with open(args.losses_out, "w", encoding="utf-8") as fout:
        json.dump({"mode": args.mode, "losses": losses}, fout)
    print(f"[{args.mode}] losses={losses}")


if __name__ == "__main__":
    main()
