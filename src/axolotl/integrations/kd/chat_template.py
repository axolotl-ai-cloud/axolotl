# Copyright 2024 Axolotl AI. All rights reserved.
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
Chat template prompt strategy loader with KD support
"""

from typing import Any, Dict, List, Tuple

import torch

from axolotl.integrations.kd.utils import LOGPROB_PAD_VALUE, rescale_topk_logprobs
from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy, StrategyLoader
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ChatTemplateStrategyWithKD(ChatTemplateStrategy):
    """
    Handle fields for logprob KD
    """

    def __init__(
        self,
        prompter,
        tokenizer,
        train_on_inputs,
        sequence_len,
        roles_to_train=None,
        train_on_eos=None,
        train_on_eot=None,
        eot_tokens=None,
        split_thinking: bool | None = False,
        logprobs_field="logprobs",
        gen_temperature=1.0,
        kd_temperature=1.0,
    ):
        self.logprobs_field = logprobs_field
        self.gen_temperature = gen_temperature
        self.kd_temperature = kd_temperature

        super().__init__(
            prompter,
            tokenizer,
            train_on_inputs,
            sequence_len,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            train_on_eot=train_on_eot,
            eot_tokens=eot_tokens,
            split_thinking=split_thinking,
        )

    @property
    def supports_batched(self) -> bool:
        # batching doesn't work well for logprob data
        return False

    @staticmethod
    def _infer_top_k(logprobs) -> int:
        # prune None logprobs from vllm data step
        top_k_vals = [
            len(logprobs[i])
            for i in range(len(logprobs))
            if logprobs[i] is not None and len(logprobs[i])
        ]
        if not top_k_vals:
            raise ValueError("No non-zero top-k logprobs found.")
        max_top_k = max(set(top_k_vals), key=top_k_vals.count)
        min_top_k = min(set(top_k_vals), key=top_k_vals.count)
        top_k = min(max_top_k, min_top_k)
        if top_k == 0:
            raise ValueError("No non-zero top-k logprobs found.")
        return top_k

    def _extract_rows(
        self, sample, logprobs
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Split the raw logprob rows into parallel (logprob, token_id) rows.
        """
        row_logprobs: List[List[float]] = []
        row_token_ids: List[List[int]] = []

        for token_pos_logprobs in logprobs:
            position_logprobs = []
            position_token_ids = []
            for entry in token_pos_logprobs or []:
                position_logprobs.append(entry["logprob"])
                # token is stored in the "token_id:###" format
                position_token_ids.append(int(entry["token"].split(":")[1]))
            row_logprobs.append(position_logprobs)
            row_token_ids.append(position_token_ids)

        return row_logprobs, row_token_ids

    def transform_logprobs(self, sample):
        """
        Transform logprobs to target format for KD training.

        Row ``r`` of the dataset logprobs is the teacher distribution over the token at
        absolute position ``front_pad + r``; the loss pairs teacher row ``j`` with the
        student position that predicts ``input_ids[j + 1]``, so rows land one position
        to the left of the token they describe.
        """

        logprobs = sample.pop(self.logprobs_field)
        input_seq_len = len(sample["input_ids"])
        top_k = self._infer_top_k(logprobs)

        row_logprobs, row_token_ids = self._extract_rows(sample, logprobs)

        front_pad = input_seq_len - len(row_logprobs)
        if front_pad <= 0:
            # over-length: tokenization kept the head of the sequence, so keep the head
            # rows, minus row 0 which describes a token no student position predicts
            row_logprobs = row_logprobs[1:input_seq_len]
            row_token_ids = row_token_ids[1:input_seq_len]
            front_pad = 0
        else:
            front_pad -= 1

        if not row_logprobs:
            raise ValueError(
                f"no teacher logprob rows left for a sequence of {input_seq_len} tokens"
            )

        # the final position predicts a token outside the sequence, so it is always padded
        tail_pad = input_seq_len - front_pad - len(row_logprobs)
        if tail_pad < 1:
            raise ValueError(
                f"teacher logprobs overflow the sequence: {front_pad} + "
                f"{len(row_logprobs)} rows > {input_seq_len - 1} predictable tokens"
            )

        pad_logprobs = [LOGPROB_PAD_VALUE] * top_k
        pad_token_ids = [0] * top_k
        pad_mask = [0] * top_k

        target_logprobs: List[List[float]] = [
            list(pad_logprobs) for _ in range(front_pad)
        ]
        target_token_ids: List[List[int]] = [
            list(pad_token_ids) for _ in range(front_pad)
        ]
        target_mask: List[List[int]] = [list(pad_mask) for _ in range(front_pad)]

        labels = sample["labels"]
        for offset, (position_logprobs, position_token_ids) in enumerate(
            zip(row_logprobs, row_token_ids, strict=True)
        ):
            position_logprobs = position_logprobs[:top_k]
            position_token_ids = position_token_ids[:top_k]

            if not position_logprobs:
                target_logprobs.append(list(pad_logprobs))
                target_token_ids.append(list(pad_token_ids))
                target_mask.append(list(pad_mask))
                continue

            scaled = rescale_topk_logprobs(
                torch.tensor(position_logprobs, dtype=torch.float),
                gen_temperature=self.gen_temperature,
                kd_temperature=self.kd_temperature,
            ).tolist()

            valid = 0 if labels[front_pad + offset + 1] == -100 else 1
            position_mask = [valid] * len(position_token_ids)

            slot_pad = top_k - len(position_token_ids)
            if slot_pad > 0:
                scaled = scaled + [LOGPROB_PAD_VALUE] * slot_pad
                position_token_ids = position_token_ids + [0] * slot_pad
                position_mask = position_mask + [0] * slot_pad

            target_logprobs.append(scaled)
            target_token_ids.append(position_token_ids)
            target_mask.append(position_mask)

        for _ in range(tail_pad):
            target_logprobs.append(list(pad_logprobs))
            target_token_ids.append(list(pad_token_ids))
            target_mask.append(list(pad_mask))

        if (
            not len(target_logprobs)
            == len(target_token_ids)
            == len(target_mask)
            == input_seq_len
        ):
            raise ValueError(
                f"teacher target rows ({len(target_logprobs)}) do not match the "
                f"{input_seq_len} input tokens"
            )

        # Update sample with transformed logprobs
        sample["target_logprobs"] = target_logprobs
        sample["target_token_ids"] = target_token_ids
        sample["target_mask"] = target_mask

        return sample

    def _tokenize_single_prompt(self, prompt):
        logprobs = prompt.pop(self.logprobs_field)
        tokenized_prompt = super()._tokenize_single_prompt(prompt)
        tokenized_prompt[self.logprobs_field] = logprobs

        # let subclasses add fields before transform
        tokenized_prompt = self._prepare_kd_fields(tokenized_prompt, prompt)

        tokenized_prompt = self.transform_logprobs(tokenized_prompt)
        return tokenized_prompt

    def _prepare_kd_fields(self, tokenized_prompt, original_prompt):
        """
        Hook for subclasses to prepare additional KD fields before transform
        """
        return tokenized_prompt


class ChatTemplateStrategyWithKDv2(ChatTemplateStrategyWithKD):
    """
    Strat for datasets with complete structured KD logprob data
    """

    def _extract_rows(
        self, sample, logprobs
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        v2 rows are already split: logprob values in the logprobs field and their token
        ids in a parallel pre-tokenized field.
        """
        row_token_ids = sample["target_token_ids"]
        if len(row_token_ids) != len(logprobs):
            raise ValueError(
                f"target_token_ids has {len(row_token_ids)} rows but logprobs has "
                f"{len(logprobs)}"
            )
        return (
            [list(row or []) for row in logprobs],
            [list(row or []) for row in row_token_ids],
        )

    def _prepare_kd_fields(self, tokenized_prompt, original_prompt):
        """
        Add pre-tokenized target_token_ids for v2 format
        """
        target_token_ids = original_prompt.pop("target_token_ids", None)
        if target_token_ids is not None:
            tokenized_prompt["target_token_ids"] = target_token_ids
        return tokenized_prompt


class KDStrategyLoader(StrategyLoader):
    """
    Load ChatTemplateStrategy with KD support using StrategyLoader.
    """

    def _get_strategy_cls(self, cfg):
        return ChatTemplateStrategyWithKD

    def _get_strategy_params(self, cfg, ds_cfg: Dict[str, Any]):
        strategy_params = super()._get_strategy_params(cfg, ds_cfg)
        if logprobs_field := ds_cfg.get("logprobs_field"):
            strategy_params["logprobs_field"] = logprobs_field
        if gen_temperature := ds_cfg.get("temperature"):
            strategy_params["gen_temperature"] = gen_temperature
        if kd_temperature := cfg.get("kd_temperature"):
            strategy_params["kd_temperature"] = kd_temperature

        return strategy_params


class KDStrategyLoaderV2(KDStrategyLoader):
    """
    Load KD chat template datasets with pre-tokenized logprob data
    """

    def _get_strategy_cls(self, cfg):
        return ChatTemplateStrategyWithKDv2


load_legacy = KDStrategyLoader()
load = KDStrategyLoaderV2()
