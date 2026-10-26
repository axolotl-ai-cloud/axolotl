# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Async GRPO with LoRA on NVIDIA Nemotron 3.5 Lightning (30B-A3B), sharing prefixes.

Nemotron 3.5 Lightning is a hybrid: 23 Mamba2 layers, 6 attention layers and 23 MoE
layers (128 experts, 6 active). TRL 1.16 refuses it for its Mamba2 layers;
``tree_recurrence`` makes them follow TRL's prefix trie (``tree_recurrent_models``),
so each prompt is forwarded once for all its completions.

- ``use_kernels=True`` takes mamba-ssm's fused kernels from the Hub
  (``kernels-community/mamba-ssm``), so nothing has to be compiled. Every root-to-leaf
  path then runs through the fused Mamba2 mixer in one call.
- LoRA leaves the Mamba2 ``out_proj`` out: the fused kernel reads its weight
  directly, so an adapter there would be skipped in training.
- TRL's tree attention runs FlashAttention 3, which needs Hopper GPUs (H100/H200).
- TRL 1.16's FLOPs estimate reads ``decoder_sparse_step`` from every MoE config, which
  Nemotron-H lacks (huggingface/trl#7069), and would end the run at its first log;
  ``guarded_flops`` makes the MFU metrics read 0 instead.

This file runs on its own. Install, then, on two H200s (the bf16 weights are ~63 GB),
start the vLLM server on one and train on the other::

    pip install "tree-recurrence[transformers,trl]" peft vllm math-verify

    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve \\
        nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \\
        --max-model-len 4096 \\
        --mamba-ssm-cache-dtype float32 \\
        --logprobs-mode processed_logprobs \\
        --weight-transfer-config '{"backend":"nccl"}'

    CUDA_VISIBLE_DEVICES=1 python nemotron_3_5_lightning.py

``modal_async_grpo.py --example nemotron_3_5_lightning`` runs both on Modal for a few
steps.
"""

from __future__ import annotations

import argparse
import logging

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
# the Mamba2 out_proj is read by the fused kernel directly, past any adapter
LORA_TARGETS = ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


def main() -> None:
    from datasets import load_dataset
    from peft import LoraConfig
    from tree_recurrence.integrations.trl import guarded_flops, tree_recurrent_models
    from trl.experimental.async_grpo import (
        AsyncGRPOConfig,
        AsyncGRPOTrainer,
        async_grpo_trainer,
    )
    from trl.rewards import accuracy_reward

    parser = argparse.ArgumentParser(description="Async GRPO on Nemotron 3.5 Lightning")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default="trl-lib/DeepMath-103K")
    parser.add_argument("--output-dir", default="outputs/nemotron-3.5-lightning-grpo")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    # unique tokens per packed row: a shared prompt counts once
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger("tree_recurrence").setLevel(logging.INFO)
    async_grpo_trainer.compute_flops_per_token = guarded_flops(
        async_grpo_trainer.compute_flops_per_token
    )
    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        model_init_kwargs={"use_kernels": True},
        dtype="bfloat16",
        chat_template_kwargs={"enable_thinking": args.thinking},
        num_generations=args.num_generations,
        per_device_train_batch_size=args.prompts_per_step * args.num_generations,
        max_completion_length=args.max_completion_length,
        token_budget=args.token_budget,
        learning_rate=1e-5,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=100,
        report_to="none",
    )
    with tree_recurrent_models() as supports:
        trainer = AsyncGRPOTrainer(
            model=args.model,
            reward_funcs=accuracy_reward,
            args=config,
            train_dataset=load_dataset(args.dataset, split="train"),
            peft_config=LoraConfig(
                r=32, lora_alpha=64, lora_dropout=0.0, target_modules=LORA_TARGETS
            ),
        )
    print(f"tree_recurrence: {supports[0].reason}", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()
