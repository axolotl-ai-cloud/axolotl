# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Async GRPO with LoRA on LiquidAI LFM2.5-8B-A1B, sharing prefixes.

LFM2.5-8B-A1B (8.3B total, 1.5B active) has 18 gated short-conv layers and 6
attention layers, with MoE feed-forwards (32 experts, 4 active). TRL 1.16 refuses it
for its conv layers; ``tree_recurrence`` runs each conv over its own path in the
prefix trie (``tree_recurrent_models``). The conv is the layers' only recurrence, so
every shared token runs once.

- ``use_kernels=True`` takes the causal conv from the Hub
  (``kernels-community/mamba-ssm``), so nothing has to be compiled.
- It is a reasoning model that always thinks before answering, so completions need
  room: the server's ``--max-model-len`` and ``--max-completion-length`` below.
- TRL 1.16's FLOPs estimate fails on its MoE config (huggingface/trl#7069);
  ``guarded_flops`` makes the MFU metrics read 0 instead.

This file runs on its own. Install, then, on two GPUs (the bf16 weights are ~17 GB;
TRL's tree attention needs Hopper), start the vLLM server on one and train on the
other::

    pip install "tree-recurrence[transformers,trl]" peft vllm math-verify

    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve LiquidAI/LFM2.5-8B-A1B \\
        --max-model-len 8192 \\
        --logprobs-mode processed_logprobs \\
        --weight-transfer-config '{"backend":"nccl"}'

    CUDA_VISIBLE_DEVICES=1 python lfm2_5_8b_a1b.py

``modal_async_grpo.py --example lfm2_5_8b_a1b`` runs both on Modal for a few steps.
"""

from __future__ import annotations

import argparse
import logging

MODEL = "LiquidAI/LFM2.5-8B-A1B"
# the convs' and attention's projections and the dense feed-forwards; the experts are
# fused parameters
LORA_TARGETS = ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "w1", "w2", "w3"]


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

    parser = argparse.ArgumentParser(description="Async GRPO on LFM2.5-8B-A1B")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default="trl-lib/DeepMath-103K")
    parser.add_argument("--output-dir", default="outputs/lfm2.5-8b-a1b-grpo")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=6144)
    # unique tokens per packed row: a shared prompt counts once
    parser.add_argument("--token-budget", type=int, default=32768)
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
