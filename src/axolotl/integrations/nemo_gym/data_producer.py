# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
NeMo Gym Data Producer for async GRPO training.

Replaces GRPODataProducer to generate rollouts via NeMo Gym agent /run endpoints
instead of vLLM. The agent handles generation, tool execution, and reward computation.
Returns RolloutDataset in the same format as the standard producer, so all downstream
components (deferred scoring, IS correction, streaming, replay, re-roll) work unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from trl.trainer.utils import pad

from axolotl.core.trainers.grpo.async_trainer import GRPODataProducer, RolloutDataset
from axolotl.utils.logging import get_logger

from .multi_turn import _call_agents, _parse_agent_response

LOG = get_logger(__name__)


class NemoGymDataProducer(GRPODataProducer):
    """Produces GRPO rollouts by calling NeMo Gym agent /run endpoints.

    Drop-in replacement for GRPODataProducer. Instead of calling vLLM for generation,
    sends prompts to NeMo Gym agents which handle generation + tool execution + reward.
    Returns the same RolloutDataset format so deferred scoring, IS correction,
    replay buffer, and re-roll all work unchanged.
    """

    def __init__(
        self,
        *args,
        agent_servers: dict[str, str],
        dataset_lookup: dict,
        request_timeout: float = 10800,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._agent_servers = agent_servers
        self._dataset_lookup = dataset_lookup
        self._request_timeout = request_timeout

    def produce(
        self,
        model: Any,
        global_step: int,
        *,
        skip_policy_logps: bool = False,
        processing_class: Any = None,
        accelerator: Any = None,
        args: Any = None,
        _rank0_only: bool = False,
        **kwargs,
    ) -> RolloutDataset | None:
        """Generate rollouts via NeMo Gym agents.

        Calls agent /run endpoints, parses responses into padded tensors,
        and returns a RolloutDataset for deferred scoring on the main thread.
        """
        trainer = self._trainer
        is_main = trainer.accelerator.is_main_process
        device = trainer.accelerator.device

        if _rank0_only and not is_main:
            return None

        # Get prompt batch from iterator
        try:
            inputs = next(self._prompt_iter)
        except StopIteration:
            self._prompt_iter = iter(self._prompt_dl)
            inputs = next(self._prompt_iter)

        # Extract dataset items for agent calls
        dataset_items = []
        for inp in inputs:
            prompt_text = ""
            prompt = inp.get("prompt", [])
            if isinstance(prompt, list) and prompt:
                prompt_text = (
                    prompt[-1].get("content", "")
                    if isinstance(prompt[-1], dict)
                    else str(prompt[-1])
                )
            elif isinstance(prompt, str):
                prompt_text = prompt

            # Find the full dataset item, preserving agent_ref for routing
            full_item = self._dataset_lookup.get(prompt_text, {})
            item = full_item.get("verify_extra", {})
            if not item:
                item = {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": prompt_text}]
                    }
                }
            # Preserve agent_ref from the dataset row for _call_agents routing
            if "agent_ref" in full_item and "agent_ref" not in item:
                item["agent_ref"] = full_item["agent_ref"]
            dataset_items.append(item)

        # NOTE: do NOT re-expand by num_generations here.
        # ``RepeatSampler(mini_repeat_count=num_generations)`` already
        # yields ``num_generations`` consecutive copies of each unique
        # prompt, so ``inputs`` is a list of ``(unique_prompts_per_rank *
        # num_generations)`` items — one entry per rollout. Expanding
        # again here would fire ``num_generations^2`` rollouts per
        # prompt per rank and make every step dogpile on a handful of
        # tasks.
        expanded_items = dataset_items

        # Diagnostic: log what this rank is about to fire.
        try:
            import collections
            iid_counts = collections.Counter()
            for it in dataset_items:
                iid_counts[
                    (it.get("responses_create_params", {}).get("metadata") or {}).get(
                        "instance_id"
                    )
                ] += 1
            LOG.info(
                "[RANK:%d] produce(): firing %d agent /run calls covering %d unique prompts: %s",
                trainer.accelerator.process_index,
                len(dataset_items),
                len(iid_counts),
                list(iid_counts.most_common(5)),
            )
        except Exception:
            pass

        # Call NeMo Gym agents
        loop = asyncio.new_event_loop()
        try:
            responses = loop.run_until_complete(
                _call_agents(
                    dataset_items=expanded_items,
                    agent_servers=self._agent_servers,
                    timeout=self._request_timeout,
                    max_completion_length=trainer.max_completion_length,
                    temperature=trainer.temperature,
                    top_p=getattr(trainer, "top_p", None) or 0.999,
                )
            )
        finally:
            loop.close()

        # Parse responses
        eos_token_id = trainer.processing_class.eos_token_id
        prompt_ids_list = []
        completion_ids_list = []
        env_mask_list = []
        logprobs_list = []
        rewards_list = []

        num_turns_list: list[int] = []
        for resp in responses:
            parsed = _parse_agent_response(resp, eos_token_id)
            prompt_ids_list.append(parsed["prompt_ids"])
            completion_ids_list.append(parsed["completion_ids"])
            env_mask_list.append(parsed["env_mask"])
            logprobs_list.append(parsed["logprobs"])
            rewards_list.append(parsed["reward"])
            num_turns_list.append(parsed.get("num_turns", 0))

        # Pad to tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids, padding_value=trainer.pad_token_id, padding_side="left"
        )
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [
            torch.tensor(ids, device=device) for ids in completion_ids_list
        ]
        completion_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in completion_ids
        ]
        completion_ids = pad(
            completion_ids, padding_value=trainer.pad_token_id, padding_side="right"
        )
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # Sampling logprobs from agent (used for IS correction)
        sampling_logps = [
            torch.tensor(lp, dtype=torch.float32, device=device) for lp in logprobs_list
        ]
        sampling_per_token_logps = pad(
            sampling_logps, padding_value=0.0, padding_side="right"
        )

        # env_mask as tool_mask (1=model tokens, 0=tool tokens)
        tool_mask = [torch.tensor(m, device=device) for m in env_mask_list]
        tool_mask = pad(tool_mask, padding_value=1, padding_side="right")

        # Inject per-rollout reward + num_turns into each input. Since
        # ``RepeatSampler`` already yields ``num_generations`` copies of
        # each prompt, ``inputs`` has ONE entry per rollout (matching
        # ``rewards_list`` 1:1). No per-prompt grouping happens here —
        # GRPO advantage normalization is the trainer's job downstream.
        assert len(inputs) == len(rewards_list), (
            f"rewards/inputs length mismatch: "
            f"{len(rewards_list)} rewards vs {len(inputs)} inputs"
        )
        for i, inp in enumerate(inputs):
            inp["env_reward"] = rewards_list[i]
            inp["num_turns"] = num_turns_list[i]

        # One expanded_input per rollout (already correct count because
        # inputs has num_generations copies baked in by the sampler).
        expanded_inputs = [dict(inp) for inp in inputs]

        # Log rollout-level stats to wandb from rank 0. These are the
        # true agent-side metrics (not the tokenized TRL view) — so
        # num_turns reflects how many /run iterations each rollout
        # actually took before finishing or hitting max_turns.
        if is_main and num_turns_list:
            try:
                import wandb

                if wandb.run is not None:
                    import statistics as _stats

                    nonzero = sum(1 for r in rewards_list if r > 0)
                    log_payload = {
                        "rollout/num_turns/mean": float(_stats.mean(num_turns_list)),
                        "rollout/num_turns/min": float(min(num_turns_list)),
                        "rollout/num_turns/max": float(max(num_turns_list)),
                        "rollout/reward/mean": float(_stats.mean(rewards_list)),
                        "rollout/reward/nonzero_frac": (
                            nonzero / len(rewards_list) if rewards_list else 0.0
                        ),
                        "rollout/n_samples": float(len(rewards_list)),
                    }
                    wandb.log(log_payload, commit=False)
            except Exception as exc:  # never let metric logging break training
                LOG.warning("rollout wandb log failed: %s", exc)


        # Decode completions for reward functions
        completions = trainer.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Build total token count
        num_items_in_batch = completion_mask.sum()

        # Build output dict (same shape as _generate_only)
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": num_items_in_batch,
            "advantages": torch.zeros(completion_ids.size(0), device=device),
            "sampling_per_token_logps": sampling_per_token_logps,
            "tool_mask": tool_mask,
            # Deferred scoring markers
            "_pending_policy_logps": True,
            "_deferred_inputs": expanded_inputs,
            "_deferred_prompts": [inp.get("prompt", "") for inp in expanded_inputs],
            "_deferred_completions": completions,
            "_deferred_completion_ids_list": completion_ids_list,
            "_rank0_only": _rank0_only,
        }

        return RolloutDataset(output)
