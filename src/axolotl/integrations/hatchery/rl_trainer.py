# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Remote RL trainer (GRPO/PPO) using Tinker or Hatchery API.

Full RL loop per step:
  1. Extract prompts from dataset batch
  2. Sample N completions per prompt via remote SamplingClient
  3. Score completions with local reward functions
  4. Compute GRPO-style advantages (per-group normalization)
  5. Send (prompt+completion, logprobs, advantages) as forward_backward
  6. Optimizer step
"""

from __future__ import annotations

import importlib
import inspect
import re
import time
from typing import Any, Callable, Optional

import torch
from transformers.trainer_utils import TrainOutput

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .args import HatcheryConfig
from .data import batch_to_datums_rl, datums_to_tinker
from .trainer import _create_training_client

LOG = get_logger(__name__)


def _load_reward_func(fqn: str) -> Callable:
    """Load a reward function from a fully qualified name like 'module.func'."""
    module_path = ".".join(fqn.split(".")[:-1])
    func_name = fqn.split(".")[-1]
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)
    if len(inspect.signature(func).parameters) < 2:
        raise ValueError(f"Reward function {fqn} must accept (prompts, completions)")
    return func


class HatcheryRLTrainer(AxolotlTrainer):
    """Remote RL trainer using Tinker/Hatchery for sampling and training."""

    hatchery_args: Optional[HatcheryConfig]
    _base_model_name: Optional[str]
    _training_client: Any
    _reward_functions: list[Callable]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hatchery_args = None
        self._base_model_name = None
        self._training_client = None
        self._reward_functions = []

    def _ensure_reward_functions(self):
        if self._reward_functions:
            return
        args = self.hatchery_args
        if not args or not args.reward_funcs:
            raise ValueError(
                "No reward functions configured. Set hatchery.reward_funcs "
                "in YAML, e.g. reward_funcs: ['my_module.my_reward']"
            )
        for fqn in args.reward_funcs:
            self._reward_functions.append(_load_reward_func(fqn))
        LOG.info(f"Loaded {len(self._reward_functions)} reward function(s)")

    def _get_training_client(self):
        if self._training_client is not None:
            return self._training_client

        self._training_client = _create_training_client(
            self.hatchery_args, self._base_model_name
        )
        LOG.info(
            f"Remote RL session created: backend={self.hatchery_args.backend}, "
            f"model={self._base_model_name}, rank={self.hatchery_args.lora_rank}"
        )
        return self._training_client

    def _sample_completions(self, prompt_ids_list: list[list[int]]):
        """Sample completions for prompts via remote API."""
        import tinker.types as tt

        tc = self._get_training_client()
        args = self.hatchery_args
        assert args is not None  # validated by _get_training_client
        results = []

        sc = tc.save_weights_and_get_sampling_client()

        for prompt_ids in prompt_ids_list:
            if hasattr(sc, "sampling_session_id"):
                sample_result = sc.sample(
                    prompt_ids,
                    max_tokens=args.max_sample_tokens,
                    temperature=args.sample_temperature,
                    n=args.num_samples,
                ).result(timeout=args.future_timeout)
            else:
                mi = tt.ModelInput.from_ints(prompt_ids)
                sp = tt.SamplingParams(
                    max_tokens=args.max_sample_tokens,
                    temperature=args.sample_temperature,
                    top_p=0.95,
                    top_k=-1,
                )
                sample_result = sc.sample(
                    prompt=mi,
                    num_samples=args.num_samples,
                    sampling_params=sp,
                ).result(timeout=args.future_timeout)

            sequences = (
                sample_result.sequences
                if hasattr(sample_result, "sequences")
                else sample_result.get("sequences", [])
            )
            for seq in sequences:
                tokens = (
                    list(seq.tokens)
                    if hasattr(seq, "tokens")
                    else seq.get("tokens", [])
                )
                logprobs = (
                    list(seq.logprobs)
                    if hasattr(seq, "logprobs") and seq.logprobs
                    else seq.get("logprobs", [])
                )
                results.append(
                    {
                        "tokens": list(prompt_ids) + tokens,
                        "completion_tokens": tokens,
                        "logprobs": logprobs,
                        "prompt_len": len(prompt_ids),
                    }
                )

        return results

    def _compute_rewards(
        self, prompts: list[str], completions: list[str]
    ) -> list[float]:
        total_rewards = [0.0] * len(completions)
        for reward_fn in self._reward_functions:
            rewards = reward_fn(prompts, completions)
            for i, r in enumerate(rewards):
                total_rewards[i] += r
        return total_rewards

    @staticmethod
    def _compute_advantages(rewards: list[float], group_size: int) -> list[float]:
        advantages = []
        for i in range(0, len(rewards), group_size):
            group = rewards[i : i + group_size]
            mean = sum(group) / len(group)
            var = sum((r - mean) ** 2 for r in group) / max(len(group), 1)
            std = var**0.5 if var > 1e-8 else 1.0
            advantages.extend([(r - mean) / std for r in group])
        return advantages

    def _do_optim_step(self):
        import tinker.types as tt

        tc = self._get_training_client()
        return tc.optim_step(tt.AdamParams(**self._optim_params))

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Any = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        args = self.hatchery_args
        if args is None:
            raise RuntimeError("hatchery_args not configured")

        self._ensure_reward_functions()

        train_dataloader = self.get_train_dataloader()
        num_train_epochs = int(self.args.num_train_epochs)
        max_steps = self.args.max_steps if self.args.max_steps > 0 else 1000

        LOG.info(
            f"Remote RL training: max_steps={max_steps}, "
            f"loss_fn={args.loss_fn}, samples/prompt={args.num_samples}"
        )

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = True
        self.state.is_world_process_zero = True

        self.control = self.callback_handler.on_train_begin(
            self.args,
            self.state,
            self.control,  # type: ignore[has-type]
        )

        tokenizer = self.processing_class
        global_step = 0
        total_loss = 0.0
        total_reward = 0.0
        start_time = time.time()

        for _epoch in range(num_train_epochs):
            if global_step >= max_steps:
                break

            for batch in train_dataloader:
                if global_step >= max_steps:
                    break

                self.control = self.callback_handler.on_step_begin(
                    self.args, self.state, self.control
                )

                prompt_ids_batch = batch["input_ids"]
                # Full prompt text (with gold tag) for reward scoring
                prompt_texts = tokenizer.batch_decode(
                    prompt_ids_batch, skip_special_tokens=False
                )

                # Strip <|gold|>...<|/gold|> from token ids before
                # sending to the model for sampling — the gold answer
                # must only be visible to the local reward function.
                sampling_prompts = []
                for prompt_text in prompt_texts:
                    clean = re.sub(r"<\|gold\|>.*?<\|/gold\|>", "", prompt_text)
                    clean_ids = tokenizer.encode(clean, add_special_tokens=False)
                    sampling_prompts.append(clean_ids)

                # 1. Sample completions (without gold answer)
                t0 = time.time()
                samples = self._sample_completions(sampling_prompts)
                t_sample = time.time() - t0

                if not samples:
                    LOG.warning("No samples generated, skipping step")
                    continue
                LOG.info(
                    f"Sampled {len(samples)} completions, "
                    f"avg_len={sum(len(s['completion_tokens']) for s in samples) / len(samples):.0f}tok"
                )

                # 2. Decode and score
                completion_texts = [
                    tokenizer.decode(s["completion_tokens"], skip_special_tokens=False)
                    for s in samples
                ]
                sample_prompts = []
                for prompt_text in prompt_texts:
                    sample_prompts.extend([prompt_text] * args.num_samples)

                rewards = self._compute_rewards(sample_prompts, completion_texts)

                # 3. GRPO advantages
                advantages_list = self._compute_advantages(
                    rewards, group_size=args.num_samples
                )

                # 4. Build training data
                all_datums = []
                for i, sample in enumerate(samples):
                    full_tokens = sample["tokens"]
                    prompt_len = sample["prompt_len"]
                    seq_len = len(full_tokens)

                    input_ids = torch.tensor([full_tokens], dtype=torch.long)
                    labels = torch.full((1, seq_len), -100, dtype=torch.long)
                    labels[0, prompt_len:] = torch.tensor(full_tokens[prompt_len:])

                    logprobs_t = torch.zeros(1, seq_len)
                    if sample["logprobs"]:
                        lp = sample["logprobs"][: seq_len - prompt_len]
                        logprobs_t[0, prompt_len : prompt_len + len(lp)] = torch.tensor(
                            lp
                        )

                    adv_t = torch.zeros(1, seq_len)
                    adv_t[0, prompt_len:] = advantages_list[i]

                    all_datums.extend(
                        batch_to_datums_rl(input_ids, labels, logprobs_t, adv_t)
                    )

                # 5. Forward backward (one datum at a time for memory) + optim
                t0 = time.time()
                tc = self._get_training_client()
                step_loss = 0.0
                for datum in all_datums:
                    fb_future = tc.forward_backward(
                        datums_to_tinker([datum]),
                        loss_fn=args.loss_fn,
                        loss_fn_config=args.loss_fn_config,
                    )
                    fb_result = fb_future.result(timeout=args.future_timeout)
                    if hasattr(fb_result, "metrics"):
                        step_loss += float(
                            (fb_result.metrics or {}).get("loss:sum", 0.0)
                        )
                    elif isinstance(fb_result, dict):
                        step_loss += float(
                            fb_result.get("metrics", {}).get("loss:sum", 0.0)
                        )
                optim_future = self._do_optim_step()
                if not args.pipeline:
                    optim_future.result(timeout=args.future_timeout)
                t_train = time.time() - t0

                mean_reward = sum(rewards) / len(rewards)
                accuracy = sum(1 for r in rewards if r > 0) / len(rewards)
                mean_adv = sum(abs(a) for a in advantages_list) / len(advantages_list)
                global_step += 1
                total_loss += step_loss
                total_reward += mean_reward
                self.state.global_step = global_step

                log_interval = self.args.logging_steps or 1
                if global_step % log_interval == 0:
                    elapsed = time.time() - start_time
                    LOG.info(
                        f"[step {global_step}/{max_steps}] "
                        f"acc={accuracy:.2f} reward={mean_reward:.3f} "
                        f"|adv|={mean_adv:.3f} loss:sum={step_loss:.1f} "
                        f"sample={t_sample:.1f}s train={t_train:.1f}s "
                        f"{elapsed / global_step:.1f}s/step"
                    )
                    self.log(
                        {
                            "loss": step_loss,
                            "reward": mean_reward,
                            "accuracy": accuracy,
                            "mean_abs_advantage": mean_adv,
                            "learning_rate": self._optim_params["learning_rate"],
                        }
                    )

                if args.save_steps and global_step % args.save_steps == 0:
                    self._save_remote_checkpoint(global_step)

                self.control = self.callback_handler.on_step_end(
                    self.args, self.state, self.control
                )
                if self.control.should_training_stop:
                    break

            if self.control.should_training_stop:
                break

        if global_step > 0:
            self._save_remote_checkpoint(global_step, name="final")

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(global_step, 1)
        avg_reward = total_reward / max(global_step, 1)

        LOG.info(
            f"RL training complete: {global_step} steps, {elapsed:.1f}s, "
            f"avg_reward={avg_reward:.4f}"
        )

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

        return TrainOutput(
            global_step=global_step,
            training_loss=avg_loss,
            metrics={
                "train_loss": avg_loss,
                "train_reward": avg_reward,
                "train_runtime": elapsed,
            },
        )

    def _save_remote_checkpoint(self, step: int, name: Optional[str] = None):
        tc = self._get_training_client()
        args = self.hatchery_args
        assert args is not None  # validated by _get_training_client
        ckpt_name = name or f"{args.save_name_prefix}-{step:06d}"
        try:
            future = tc.save_state(ckpt_name)
            future.result(timeout=args.future_timeout)
            LOG.info(f"Remote checkpoint saved: {ckpt_name}")
        except Exception:
            LOG.exception(f"Failed to save checkpoint {ckpt_name}")
            if name == "final":
                raise

    def save_model(self, output_dir=None, _internal_call=False):
        self._save_remote_checkpoint(
            step=self.state.global_step,
            name=output_dir or "hf-save",
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raise NotImplementedError(
            "HatcheryRLTrainer uses remote API; compute_loss not called locally."
        )
