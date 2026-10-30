"""EBFT rollouts in async GRPO's rollout process.

EBFT's reward needs the feature network's hidden states, which this CUDA-free process
cannot compute, so samples leave with their advantage pending (``trainer_scored``) and
carry what the trainer's ``EBFTScorer`` needs: each rollout's feature sequence and its
group's reference. Only the first assistant turn is trained; with ``remaining_turns``
the later assistant turns are generated as reward context only.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Any

import numpy as np

from axolotl.core.trainers.async_grpo.rollout import AxolotlRolloutLoop
from axolotl.core.trainers.async_grpo.trl_compat import (
    RolloutGroup,
    TrainingSequence,
    unsampled_sequence_fields,
)

_NO_DRIFT = {
    "clean": 0.0,
    "realign": 0.0,
    "fork": 0.0,
    "transitions": 0.0,
    "drift_tokens": 0.0,
    "drift_max": 0.0,
}


def reference_ids(
    tokenizer,
    prompt: list[dict],
    ground_truth: str,
    remaining_turns: list[dict] | None = None,
    **template_kwargs: Any,
) -> tuple[list[int], int]:
    """Token IDs of the reference answer and where its pooled part starts.

    Tokenized from text exactly as ``EBFTMixin._feature_matching_reward`` does.
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    if remaining_turns:
        text = tokenizer.apply_chat_template(
            [*prompt, {"role": "assistant", "content": ground_truth}, *remaining_turns],
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        )
    else:
        text = prompt_text + ground_truth
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    pool_start = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    return ids, pool_start


def adaptive_max_tokens(
    tokenizer, ground_truth: str | None, max_tokens: int, multiplier: float
) -> int:
    """The ``ebft_adaptive_max_tokens`` budget for a prompt with ``ground_truth``."""
    if not ground_truth:
        return max_tokens
    gt_tokens = len(tokenizer.encode(ground_truth, add_special_tokens=False))
    return max(min(int(gt_tokens * multiplier), max_tokens), 64)


class EBFTRolloutLoop(AxolotlRolloutLoop):
    """Generates EBFT rollouts and ships their feature sequences to the trainer.

    Args:
        adaptive_max_tokens: Size each prompt's completions from its ground truth.
        gt_length_multiplier: Ground-truth tokens times this is the adaptive budget.
        generation_kwargs: Extra ``/v1/completions`` sampling parameters, e.g.
            ``stop_token_ids``.
    """

    _parses_responses = False

    def __init__(
        self,
        *,
        adaptive_max_tokens: bool = True,
        gt_length_multiplier: float = 1.5,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._adaptive_max_tokens = adaptive_max_tokens
        self._gt_length_multiplier = gt_length_multiplier
        self._generation_kwargs = dict(generation_kwargs or {})
        self._stop_ids = {
            *self.eos_token_ids,
            self.tokenizer.pad_token_id,
            *self._generation_kwargs.get("stop_token_ids", []),
        }
        self._features: dict[str, dict[str, Any]] = {}
        self._reference: tuple[int, dict[str, Any]] | None = None

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            return_dict=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )

    async def _complete(
        self, prompt_ids: list[int], max_tokens: int
    ) -> tuple[list[int], list[float]]:
        # TRL's `_generate_one_turn` with a per-prompt budget and generation kwargs.
        payload = {
            "model": self._request_model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": 1,
            "return_token_ids": True,  # nosec B105
            "logprobs": 0,
            "seed": random.getrandbits(63),  # nosec B311
        }
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        payload.update(self._generation_kwargs)
        output = await self._retry(
            lambda: self._post("/v1/completions", payload, self.request_timeout),
            max_attempts=30,
            label="vllm /v1/completions",
        )
        choice = output["choices"][0]
        return choice["token_ids"], choice["logprobs"]["token_logprobs"]

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        t_dispatch = time.monotonic()
        row = self._group_rows[group_id]
        max_tokens = self.max_tokens
        if self._adaptive_max_tokens:
            max_tokens = adaptive_max_tokens(
                self.tokenizer,
                row.get("ground_truth"),
                self.max_tokens,
                self._gt_length_multiplier,
            )
        prompt_ids = self._render(prompt, add_generation_prompt=True)
        ids, logprobs = await self._complete(prompt_ids, max_tokens)
        text = self._decode(ids)

        kept = len(ids)
        while kept and ids[kept - 1] in self._stop_ids:
            kept -= 1
        feature_ids = prompt_ids + list(ids[:kept])
        if row.get("remaining_turns"):
            conversation = [*prompt, {"role": "assistant", "content": text}]
            for turn in row["remaining_turns"]:
                if turn["role"] == "user":
                    conversation.append(turn)
                elif turn["role"] == "assistant":
                    turn_ids, _ = await self._complete(
                        self._render(conversation, add_generation_prompt=True),
                        self.max_tokens,
                    )
                    conversation.append(
                        {"role": "assistant", "content": self._decode(turn_ids)}
                    )
            feature_ids = self._render(conversation, add_generation_prompt=False)

        sequence = TrainingSequence(
            input_ids=prompt_ids + list(ids),
            completion_mask=[0] * len(prompt_ids) + [1] * len(ids),
            old_log_probs=[0.0] * len(prompt_ids) + list(logprobs),
            **unsampled_sequence_fields(len(prompt_ids) + len(ids)),
            rollout_id=uuid.uuid4().hex,
        )
        self._features[sequence.rollout_id] = {
            "feature_ids": np.asarray(feature_ids, dtype=np.int64),
            "pool_start": len(prompt_ids),
        }
        self._push_rollout_metrics(
            turns=1,
            sequences=1,
            completion_ids=list(ids),
            tally=_NO_DRIFT,
            loop_exhausted=False,
            duration_s=time.monotonic() - t_dispatch,
        )
        return (
            [{"role": "assistant", "content": text}],
            list(ids),
            [sequence],
            0,
            0,
            None,
        )

    def _trainer_score_payload(
        self, group: RolloutGroup, index: int, sequence: TrainingSequence
    ) -> dict[str, Any]:
        if self._reference is None or self._reference[0] != group.group_id:
            ground_truth = group.reward_kwargs.get("ground_truth", [None])[index]
            if ground_truth is None:
                raise ValueError("EBFT needs a `ground_truth` column in the dataset")
            turns = group.reward_kwargs.get("remaining_turns", [None])[index]
            ids, pool_start = reference_ids(
                self.tokenizer,
                group.prompts[index],
                ground_truth,
                turns,
                **self.chat_template_kwargs,
            )
            self._reference = (
                group.group_id,
                {
                    "reference_ids": np.asarray(ids, dtype=np.int64),
                    "reference_pool_start": pool_start,
                },
            )
        return {**self._features.pop(sequence.rollout_id), **self._reference[1]}
