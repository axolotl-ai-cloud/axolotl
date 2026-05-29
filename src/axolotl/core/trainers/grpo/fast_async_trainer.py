# Copyright 2020-2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Experimental GRPO extensions: parallel reward workers, replay buffer,
deferred re-roll, and zero-advantage skipping.

These features are built as subclasses of GRPOTrainer and GRPODataProducer,
using the hook system (_compute_rewards_for_batch, _post_advantage_hook,
_pre_produce_hook) defined in the base classes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field

import torch
from torch import nn
from trl import GRPOTrainer

from axolotl.core.trainers.grpo.async_trainer import (
    AsyncGRPOConfig,
    AsyncGRPOTrainer,
    GRPODataProducer,
)
from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended config
# ---------------------------------------------------------------------------


@dataclass
class FastAsyncGRPOConfig(AsyncGRPOConfig):
    """GRPOConfig with additional experimental parameters."""

    reward_num_workers: int = field(
        default=1,
        metadata={
            "help": "Number of persistent subprocess workers for parallel reward computation. Each worker has its "
            "own main thread so signal.alarm() (used by math_verify) works correctly. Work is sharded across "
            "workers by prompt groups. Only used with use_data_producer=True and non-nn.Module reward functions."
        },
    )
    replay_buffer_size: int = field(
        default=0,
        metadata={
            "help": "[Experimental, disabled by default] Size of the replay buffer for storing high-signal rollout "
            "groups. When > 0, groups with reward variance are cached and used to replace zero-signal groups "
            "(where all rewards are identical). Set to 0 to disable. Only used with use_data_producer=True."
        },
    )
    replay_recompute_logps: bool = field(
        default=True,
        metadata={
            "help": "When True (default), recompute old_per_token_logps for replayed groups using the current "
            "training model. This fixes the importance sampling mismatch that occurs when replaying stale data. "
            "Only relevant when replay_buffer_size > 0."
        },
    )
    reroll_start_fraction: float = field(
        default=0.5,
        metadata={
            "help": "Fraction of total training steps after which deferred re-rolling begins. Zero-signal prompts "
            "(where all rewards in a group are identical) are buffered and re-injected into later batches when the "
            "model is more likely to solve them. Set to 1.0 to disable. Only used with use_data_producer=True."
        },
    )
    reroll_max_groups: int = field(
        default=1,
        metadata={
            "help": "Maximum number of prompt groups to replace with re-roll candidates per batch. Higher values "
            "increase data utilization but reduce prompt diversity. Only used with use_data_producer=True."
        },
    )
    skip_zero_advantage_batches: bool = field(
        default=True,
        metadata={
            "help": "When True, skip gradient computation for micro-batches where all advantages are zero (no learning "
            "signal). This avoids the forward/backward pass entirely when no learning signal is present. The step is "
            "logged with skipped_zero_adv_batches=1 for monitoring."
        },
    )
    vllm_lora_sync: bool = field(
        default=False,
        metadata={
            "help": "When True, sync LoRA adapter weights to vLLM via filesystem instead of merging into base model "
            "and NCCL-broadcasting all parameters. vLLM loads the adapter natively using Punica kernels. "
            "Requires vllm_serve_lora serve module (auto-selected when this is True). "
            "Syncs only LoRA adapter weights (much smaller) vs full merged model. Legacy merge behavior is used when False."
        },
    )


# ---------------------------------------------------------------------------
# Extended data producer with re-roll injection
# ---------------------------------------------------------------------------


class RerollDataProducer(GRPODataProducer):
    """GRPODataProducer that injects re-roll candidates into prompt batches.

    Reads from the trainer's ``_reroll_buffer`` (populated by
    ``GRPOExperimentalTrainer._post_advantage_hook``) and replaces the
    last N prompt groups with previously-failed prompts.
    """

    def _pre_produce_hook(self, inputs: list, global_step: int) -> list:
        trainer = self._trainer
        reroll_buf = getattr(trainer, "_reroll_buffer", None)
        reroll_lock = getattr(trainer, "_reroll_lock", None)
        if reroll_buf is None or reroll_lock is None:
            return inputs

        max_steps = getattr(trainer.args, "max_steps", -1)
        start_frac = getattr(trainer.args, "reroll_start_fraction", 1.0)
        max_groups = getattr(trainer.args, "reroll_max_groups", 1)
        reroll_start_step = (
            max(1, int(max_steps * start_frac)) if max_steps > 0 else float("inf")
        )

        if global_step < reroll_start_step:
            return inputs

        with reroll_lock:
            n_to_take = min(max_groups, len(reroll_buf))
            reroll_prompts = [reroll_buf.pop(0) for _ in range(n_to_take)]

        if reroll_prompts:
            num_gen = self._num_generations
            n_groups = len(inputs) // num_gen
            for i, reroll_prompt in enumerate(reroll_prompts):
                group_idx = n_groups - 1 - i
                if group_idx < 0:
                    break
                start = group_idx * num_gen
                for j in range(num_gen):
                    inputs[start + j] = reroll_prompt
            logger.info(
                f"[REROLL] Step {global_step}: replaced {len(reroll_prompts)}/{n_groups} prompt groups "
                f"with deferred re-roll candidates ({len(reroll_buf)} remaining)"
            )

        return inputs


# ---------------------------------------------------------------------------
# Persistent reward subprocess pool
# ---------------------------------------------------------------------------


def _persistent_reward_worker(conn):
    """Long-lived reward worker. Receives work items, returns results."""
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:  # Shutdown signal
            break
        (
            reward_funcs,
            prompts,
            completions,
            completion_ids_list,
            inputs,
            reward_func_names,
        ) = msg
        try:
            keys = [
                key
                for key in inputs[0]
                if key not in ["prompt", "completion", "completion_ids"]
            ]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            results = []
            for reward_func, _reward_func_name in zip(
                reward_funcs, reward_func_names, strict=True
            ):
                output = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    **reward_kwargs,
                )
                results.append(
                    [float(r) if r is not None else float("nan") for r in output]
                )
            conn.send(results)
        except Exception:
            conn.send(None)


# ---------------------------------------------------------------------------
# Extended trainer
# ---------------------------------------------------------------------------


class FastAsyncGRPOTrainer(AsyncGRPOTrainer):
    """GRPOTrainer with experimental extensions.

    Adds:
    - Parallel reward subprocess workers (``reward_num_workers``)
    - Replay buffer for high-signal group reuse (``replay_buffer_size``)
    - Deferred re-roll of failed prompts (``reroll_start_fraction``)
    - Zero-advantage micro-batch skipping
    """

    def __init__(self, *args, **kwargs):
        # These must be initialized before super().__init__() because
        # _create_data_producer (called during super().__init__) needs them.
        self._reroll_buffer: list = []
        self._reroll_lock = threading.Lock()

        # Temporarily suppress the base class's Liger + OPSM validation check,
        # since this subclass supports it via a custom compute_liger_loss override.
        grpo_args = kwargs.get("args")
        if grpo_args is None:
            for a in args:
                if hasattr(a, "off_policy_mask_threshold"):
                    grpo_args = a
                    break
        saved_threshold = None
        if grpo_args is not None and getattr(grpo_args, "use_liger_kernel", False):
            saved_threshold = grpo_args.off_policy_mask_threshold
            grpo_args.off_policy_mask_threshold = None

        super().__init__(*args, **kwargs)

        if saved_threshold is not None:
            grpo_args.off_policy_mask_threshold = saved_threshold
            self.off_policy_mask_threshold = saved_threshold

        # Replay buffer
        if getattr(self.args, "replay_buffer_size", 0) > 0:
            self._replay_buffer = ReplayBuffer(max_size=self.args.replay_buffer_size)
        else:
            self._replay_buffer = None
        self._replay_recompute_logps = getattr(
            self.args, "replay_recompute_logps", True
        )

        # Reward worker pool (lazy-initialized)
        self._reward_workers = None

    # -- Factory override: use RerollDataProducer ----------------------------

    def _create_data_producer(self, args, train_dataset):
        """Override to use RerollDataProducer for re-roll prompt injection."""
        from axolotl.core.trainers.grpo.async_trainer import (
            AsyncDataProducer,
            ProducerConfig,
        )

        producer_config = ProducerConfig(
            mini_epochs=args.num_iterations,
            max_rollouts=None,
            eval_during_produce=False,
            empty_cache_before_produce=True,
            empty_cache_after_produce=True,
            async_prefetch=args.async_prefetch,
            prefetch_depth=args.prefetch_depth,
        )
        data_producer = RerollDataProducer(
            config=producer_config,
            prompt_dataset=train_dataset,
            num_generations=self.num_generations,
            generation_batch_size=args.generation_batch_size,
            train_batch_size=args.per_device_train_batch_size,
            steps_per_generation=args.steps_per_generation,
            shuffle_dataset=self.shuffle_dataset,
            seed=args.seed,
        )
        data_producer.set_trainer(self)
        if args.async_prefetch:
            data_producer = AsyncDataProducer(
                data_producer,
                background_produce_kwargs={"skip_policy_logps": True},
            )
        return data_producer

    # -- Reward worker pool --------------------------------------------------

    def _get_reward_workers(self):
        """Return a list of persistent reward worker subprocesses (lazy-initialized)."""
        import multiprocessing as _mp

        num_workers = getattr(self.args, "reward_num_workers", 1)
        if num_workers < 1:
            num_workers = 1

        if self._reward_workers is not None:
            alive = all(proc.is_alive() for conn, proc in self._reward_workers)
            if alive and len(self._reward_workers) == num_workers:
                return self._reward_workers
            self._shutdown_reward_workers()

        workers = []
        for _ in range(num_workers):
            parent_conn, child_conn = _mp.Pipe()
            proc = _mp.Process(
                target=_persistent_reward_worker, args=(child_conn,), daemon=True
            )
            proc.start()
            child_conn.close()
            workers.append((parent_conn, proc))

        self._reward_workers = workers
        return workers

    def _shutdown_reward_workers(self):
        """Shut down all persistent reward workers."""
        if self._reward_workers is None:
            return
        for conn, proc in self._reward_workers:
            try:
                conn.send(None)
                proc.join(timeout=5)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        self._reward_workers = None

    # -- Hook overrides ------------------------------------------------------

    def _compute_rewards_for_batch(
        self, inputs, prompts, completions, completion_ids_list
    ):
        """Dispatch rewards to parallel subprocess workers (synchronous wrapper)."""
        self._launch_reward_workers(inputs, prompts, completions, completion_ids_list)
        return self._collect_reward_workers(
            inputs, prompts, completions, completion_ids_list
        )

    def _launch_reward_workers(self, inputs, prompts, completions, completion_ids_list):
        """Send reward work to subprocess workers (non-blocking).

        Results are collected later by _collect_reward_workers, allowing GPU
        logprob computation to overlap with CPU reward computation.
        """
        reward_can_bg = all(
            callable(rf)
            and not isinstance(rf, nn.Module)
            and not asyncio.iscoroutinefunction(rf)
            for rf in self.reward_funcs
        )
        num_workers = getattr(self.args, "reward_num_workers", 1)

        if not reward_can_bg or num_workers <= 1:
            # Can't parallelize — store args for sync fallback in collect
            self._reward_workers_used = None
            self._pending_reward_args = (
                inputs,
                prompts,
                completions,
                completion_ids_list,
            )
            return

        workers = self._get_reward_workers()
        num_generations = self.num_generations
        num_prompts = len(prompts)
        num_groups = num_prompts // num_generations

        # Shard by prompt groups across workers
        groups_per_worker = max(1, (num_groups + len(workers) - 1) // len(workers))
        workers_used = []
        for w_idx, (conn, _proc) in enumerate(workers):
            g_start = w_idx * groups_per_worker
            g_end = min((w_idx + 1) * groups_per_worker, num_groups)
            if g_start >= num_groups:
                break
            s_start = g_start * num_generations
            s_end = g_end * num_generations
            conn.send(
                (
                    self.reward_funcs,
                    prompts[s_start:s_end],
                    completions[s_start:s_end],
                    completion_ids_list[s_start:s_end],
                    inputs[s_start:s_end],
                    self.reward_func_names,
                )
            )
            workers_used.append(conn)

        self._reward_workers_used = workers_used
        self._pending_reward_args = (inputs, prompts, completions, completion_ids_list)

    def _collect_reward_workers(
        self, inputs, prompts, completions, completion_ids_list
    ):
        """Collect reward results from subprocess workers (blocks until done)."""
        from accelerate.utils import gather

        workers_used = getattr(self, "_reward_workers_used", None)
        args = getattr(self, "_pending_reward_args", None)
        self._reward_workers_used = None
        self._pending_reward_args = None

        if workers_used is None:
            # Sync fallback — compute on main thread
            if args is not None:
                return self._calculate_rewards(*args)
            return self._calculate_rewards(
                inputs, prompts, completions, completion_ids_list
            )

        device = self.accelerator.device
        num_prompts = len(args[1]) if args else len(prompts)

        # Collect results from workers
        all_worker_results = []
        any_failed = False
        for conn in workers_used:
            result = conn.recv()
            if result is None:
                any_failed = True
                # Drain remaining workers to prevent stale results in pipes
                for remaining_conn in workers_used:
                    if remaining_conn is not conn:
                        try:
                            remaining_conn.recv()
                        except Exception:
                            pass
                break
            all_worker_results.append(result)

        if not any_failed:
            rewards_per_func = torch.zeros(
                num_prompts, len(self.reward_funcs), device=device
            )
            offset = 0
            for worker_result in all_worker_results:
                chunk_size = len(worker_result[0])
                for i, result in enumerate(worker_result):
                    rewards_per_func[offset : offset + chunk_size, i] = torch.tensor(
                        result, dtype=torch.float32, device=device
                    )
                offset += chunk_size
            return gather(rewards_per_func)

        # Fallback to main thread on failure
        if args is not None:
            return self._calculate_rewards(*args)
        return self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )

    def _post_advantage_hook(
        self,
        data: dict,
        rewards_per_func,
        advantages,
        inputs: list,
        num_generations: int,
        mode: str,
        s_start: int | None = None,
        s_end: int | None = None,
        is_last_chunk: bool = True,
    ) -> None:
        """Replay buffer store/replace + re-roll buffering."""
        from trl.models.utils import disable_gradient_checkpointing

        # -- Replay buffer: store high-signal groups --
        if self._replay_buffer is not None:
            local_grouped = rewards_per_func.view(
                -1, num_generations, len(self.reward_funcs)
            )
            per_group_std = local_grouped.std(dim=1)
            has_signal = (per_group_std > 0).any(dim=1)
            offset = s_start or 0

            if has_signal.any():
                grouped_adv = advantages.view(-1, num_generations)
                replay_scores = grouped_adv.abs().sum(dim=1) * per_group_std.sum(dim=1)
                for group_idx in has_signal.nonzero(as_tuple=True)[0]:
                    gi = group_idx.item()
                    start = offset + gi * num_generations
                    end = start + num_generations
                    group_data = {}
                    for key in data:
                        val = data[key]
                        if (
                            isinstance(val, torch.Tensor)
                            and val.dim() > 0
                            and val.size(0) >= end
                        ):
                            group_data[key] = val[start:end].clone()
                    self._replay_buffer.add(replay_scores[gi].item(), group_data)

            # Replace zero-signal groups with high-signal replay buffer entries
            # Only in non-streaming path (s_start is None) — streaming scores
            # groups incrementally, so replacement + logprob recompute would be
            # too expensive per chunk.
            n_replaced = 0
            if s_start is None:
                no_signal = ~has_signal
                replaced_ranges = []
                if no_signal.any() and len(self._replay_buffer) > 0:
                    for group_idx in no_signal.nonzero(as_tuple=True)[0]:
                        sampled = self._replay_buffer.sample(1)
                        if sampled is None:
                            break
                        sampled_group = sampled[0]
                        gi = group_idx.item()
                        start = offset + gi * num_generations
                        end = start + num_generations
                        for key, val in sampled_group.items():
                            if key in data and isinstance(data[key], torch.Tensor):
                                src = val.to(data[key].device)
                                tgt_seq_len = (
                                    data[key].size(1) if data[key].dim() > 1 else None
                                )
                                if start >= data[key].size(0) or end > data[key].size(
                                    0
                                ):
                                    continue
                                if tgt_seq_len is not None:
                                    if src.size(1) <= tgt_seq_len:
                                        data[key][start:end] = 0
                                        data[key][start:end, : src.size(1)] = src
                                    else:
                                        data[key][start:end] = src[:, :tgt_seq_len]
                                else:
                                    data[key][start:end] = src
                        replaced_ranges.append((start, end))
                        n_replaced += 1

                # Recompute old_per_token_logps for replayed groups
                if (
                    n_replaced > 0
                    and self._replay_recompute_logps
                    and "old_per_token_logps" in data
                ):
                    with (
                        torch.no_grad(),
                        disable_gradient_checkpointing(
                            self.model, self.args.gradient_checkpointing_kwargs
                        ),
                    ):
                        for r_start, r_end in replaced_ranges:
                            r_ids = torch.cat(
                                [
                                    data["prompt_ids"][r_start:r_end],
                                    data["completion_ids"][r_start:r_end],
                                ],
                                dim=1,
                            )
                            r_mask = torch.cat(
                                [
                                    data["prompt_mask"][r_start:r_end],
                                    data["completion_mask"][r_start:r_end],
                                ],
                                dim=1,
                            )
                            r_logits_to_keep = data["completion_ids"].size(1)
                            r_fwd_kwargs = {}
                            for fk in (
                                "pixel_values",
                                "image_grid_thw",
                                "pixel_attention_mask",
                                "image_sizes",
                                "token_type_ids",
                                "mm_token_type_ids",
                            ):
                                if fk in data:
                                    r_fwd_kwargs[fk] = data[fk]
                            r_logps, _ = self._get_per_token_logps_and_entropies(
                                self.model,
                                r_ids,
                                r_mask,
                                r_logits_to_keep,
                                r_end - r_start,
                                **r_fwd_kwargs,
                            )
                            data["old_per_token_logps"][r_start:r_end] = r_logps

                if n_replaced > 0:
                    self._metrics[mode]["replay_buffer_replacements"].append(
                        float(n_replaced)
                    )

            if is_last_chunk:
                self._metrics[mode]["replay_buffer_size"].append(
                    float(len(self._replay_buffer))
                )

        # -- Re-roll buffer: store failed prompts --
        if getattr(self.args, "reroll_start_fraction", 1.0) < 1.0:
            grouped_rewards = rewards_per_func.view(
                -1, num_generations, len(self.reward_funcs)
            )
            per_group_std = grouped_rewards.std(dim=1)
            per_group_mean = grouped_rewards.mean(dim=1)
            zero_signal = (per_group_std == 0).all(dim=1)
            all_failed = (per_group_mean.abs() < 1e-6).all(dim=1)
            should_reroll = zero_signal & all_failed
            _n_buffered = 0
            with self._reroll_lock:
                for group_idx in should_reroll.nonzero(as_tuple=True)[0]:
                    idx = group_idx.item() * num_generations
                    if idx >= len(inputs):
                        continue
                    prompt_input = inputs[idx]
                    self._reroll_buffer.append(prompt_input)
                    _n_buffered += 1
            if _n_buffered > 0:
                self._metrics[mode]["reroll_buffered"].append(float(_n_buffered))
            if is_last_chunk:
                self._metrics[mode]["reroll_buffer_size"].append(
                    float(len(self._reroll_buffer))
                )

    # -- Zero-advantage skipping + Liger OPSM ---------------------------------

    def compute_liger_loss(self, unwrapped_model, inputs):
        """Liger loss with zero-adv skipping and off-policy sequence masking (OPSM).

        The base class Liger path doesn't support OPSM because the fused kernel
        doesn't expose per-token logprobs needed for the KL computation. This
        override computes them via chunked lm_head matmul (no grad, low memory)
        and applies the OPSM to the loss mask before calling the kernel.
        """
        if self.args.skip_zero_advantage_batches and torch.all(
            inputs["advantages"] == 0
        ):
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["skipped_zero_adv_batches"].append(1.0)
            return torch.tensor(
                0.0, device=inputs["advantages"].device, requires_grad=True
            )

        if self.off_policy_mask_threshold is None:
            return super().compute_liger_loss(unwrapped_model, inputs)

        # OPSM path: need per_token_logps for KL, which Liger kernel doesn't provide
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        loss_mask = (
            completion_mask
            if "tool_mask" not in inputs
            else completion_mask * inputs["tool_mask"]
        )

        # Compute per_token_logps via chunked lm_head matmul (no grad, low memory)
        lm_weight = unwrapped_model.lm_head.weight
        lm_bias = unwrapped_model.lm_head.bias
        with torch.no_grad():
            per_token_logps_chunks = []
            for i in range(last_hidden_state.size(0)):
                chunk_logits = torch.matmul(last_hidden_state[i : i + 1], lm_weight.t())
                if lm_bias is not None:
                    chunk_logits = chunk_logits + lm_bias
                chunk_lps = (
                    chunk_logits.float()
                    .log_softmax(-1)
                    .gather(-1, completion_ids[i : i + 1].unsqueeze(-1))
                    .squeeze(-1)
                )
                per_token_logps_chunks.append(chunk_lps)
                del chunk_logits
            per_token_logps = torch.cat(per_token_logps_chunks, dim=0)

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages_2d = advantages.unsqueeze(1)
        else:
            advantages_2d = advantages

        sampling_per_token_logps = inputs.get("sampling_per_token_logps")
        if sampling_per_token_logps is None:
            sampling_per_token_logps = inputs.get("old_per_token_logps")
        if sampling_per_token_logps is None:
            sampling_per_token_logps = per_token_logps

        off_policy_mask = GRPOTrainer.get_off_policy_mask(
            advantages=advantages_2d,
            per_token_logps=per_token_logps,
            sampling_per_token_logps=sampling_per_token_logps,
            mask=loss_mask,
            off_policy_threshold=self.off_policy_mask_threshold,
        )
        loss_mask = loss_mask * off_policy_mask

        # Call the Liger fused kernel with OPSM-modified mask
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=loss_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
            vllm_is_ratio=inputs.get("importance_sampling_ratio"),
        )

        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).mean().item()
            )
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather(clip_ratio).mean().item()
        )
        normalizer = (
            self.current_gradient_accumulation_steps if mode == "train" else 1.0
        )
        return loss / normalizer

    def _compute_loss(self, model, inputs):
        if self.args.skip_zero_advantage_batches and torch.all(
            inputs["advantages"] == 0
        ):
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["skipped_zero_adv_batches"].append(1.0)
            # Create zero loss with grad_fn. DeepSpeed requires grad_fn != None.
            # With ZeRO-3, parameters are partitioned (shape=[0], requires_grad=False)
            # so we can't just do `(p * 0).sum()`. Instead, do a tiny forward pass
            # with a single token to create a proper computation graph.
            prompt_ids = inputs["prompt_ids"][:1, :1]  # (1, 1)
            attn = torch.ones_like(prompt_ids)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=prompt_ids, attention_mask=attn)
            return out.logits.sum() * 0
        return super()._compute_loss(model, inputs)
