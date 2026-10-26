# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Async GRPO with TRL on a hybrid recurrent model, keeping prefix sharing.

TRL's ``AsyncGRPOTrainer`` packs each rank's rollouts into a prefix trie, so the
prompt a group's completions share is forwarded once. TRL 1.16 refuses models with
recurrent layers (Gated DeltaNet, Mamba2, short convs), since they would scan across
the packed samples. ``tree_recurrent_models`` installs ``tree_recurrence`` on the
model TRL loads, which makes those layers follow the trie, and keeps the model's
``layer_types`` from TRL's check until the trainer is built.

The model loads with ``use_kernels=True``, which takes FLA's and mamba-ssm's kernels
from the Hub, so one install serves every family and nothing is compiled. This file
runs on its own; install, then start the vLLM server on one GPU and train on
another::

    pip install "tree-recurrence[transformers,trl]" vllm math-verify

    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-2B \\
        --max-model-len 4096 \\
        --logprobs-mode processed_logprobs \\
        --weight-transfer-config '{"backend":"nccl"}'

    CUDA_VISIBLE_DEVICES=1 accelerate launch trl_async_grpo.py --model Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    import tree_recurrence
    from datasets import load_dataset
    from tree_recurrence.integrations.trl import tree_recurrent_models
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
    from trl.rewards import accuracy_reward

    parser = argparse.ArgumentParser(description="Async GRPO with tree_recurrence")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", default="trl-lib/DeepMath-103K")
    parser.add_argument("--output-dir", default="outputs/async-grpo-tree-recurrence")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True
    )
    # prefix tokens a trie level must save to fork the state rather than rescan
    parser.add_argument("--fork-min-saved-tokens", type=int)
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger("tree_recurrence").setLevel(logging.INFO)
    policy = None
    if args.fork_min_saved_tokens is not None:
        policy = tree_recurrence.Policy(
            fork_min_saved_tokens=args.fork_min_saved_tokens
        )
    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        model_init_kwargs={"use_kernels": True},
        num_generations=8,
        per_device_train_batch_size=8,
        max_completion_length=args.max_completion_length,
        chat_template_kwargs={"enable_thinking": args.thinking},
        learning_rate=1e-6,
        gradient_checkpointing=True,
        logging_steps=1,
        report_to="none",
    )
    with tree_recurrent_models(policy) as supports:
        trainer = AsyncGRPOTrainer(
            model=args.model,
            reward_funcs=accuracy_reward,
            args=config,
            train_dataset=load_dataset(args.dataset, split="train"),
        )
    print(f"tree_recurrence: {supports[0].reason}", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()
