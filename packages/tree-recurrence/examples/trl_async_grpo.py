# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Async GRPO with TRL on a hybrid recurrent model, keeping prefix sharing.

TRL's ``AsyncGRPOTrainer`` packs each rank's rollouts into a prefix trie, so the
prompt a group's completions share is forwarded once. TRL 1.16 refuses models with
recurrent layers (Gated DeltaNet, Mamba2, short convs), since they would scan across
the packed samples; ``tree_recurrence.install`` makes them follow the trie instead.

TRL loads the model inside the trainer's constructor and checks its layer types
straight away, so ``tree_recurrent_models`` hooks that loader: the loaded model gets
``install``ed, and its ``layer_types`` stay hidden from the check until the trainer
is built.

The model loads with ``use_kernels=True``, which takes FLA's and mamba-ssm's kernels
from the Hub, so the same install serves every family and nothing is compiled::

    pip install "tree-recurrence[transformers,trl]" vllm math-verify

Then, with two GPUs, the vLLM server on one and training on the other::

    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-2B \\
        --max-model-len 4096 \\
        --logprobs-mode processed_logprobs \\
        --weight-transfer-config '{"backend":"nccl"}'

    CUDA_VISIBLE_DEVICES=1 accelerate launch examples/trl_async_grpo.py \\
        --model Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from collections.abc import Iterator

import tree_recurrence
from tree_recurrence.integrations.trl import hidden_layer_types
from trl.experimental.async_grpo import (
    AsyncGRPOConfig,
    AsyncGRPOTrainer,
    async_grpo_trainer,
)


@contextlib.contextmanager
def tree_recurrent_models(
    policy: tree_recurrence.Policy | None = None,
) -> Iterator[list[tree_recurrence.Support]]:
    """While TRL builds a trainer, make the model it loads follow tree-packed rows.

    Yields the ``Support`` of each model loaded meanwhile. A model whose recurrent
    layers cannot follow a trie is refused, as TRL would refuse it.
    """
    load = async_grpo_trainer.create_model_from_path
    supports: list[tree_recurrence.Support] = []
    with contextlib.ExitStack() as hidden:

        def create_model_from_path(*args, **kwargs):
            model = load(*args, **kwargs)
            support = tree_recurrence.install(model, policy=policy)
            if not support.shares_prefixes:
                raise ValueError(
                    f"{type(model).__name__} cannot share prefixes in a packed row: "
                    f"{support.reason}"
                )
            supports.append(support)
            hidden.enter_context(hidden_layer_types(model))
            return model

        async_grpo_trainer.create_model_from_path = create_model_from_path
        try:
            yield supports
        finally:
            async_grpo_trainer.create_model_from_path = load


def main() -> None:
    from datasets import load_dataset
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
