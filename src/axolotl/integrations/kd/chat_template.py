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

import logging
from typing import Any, Dict

import torch

from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy, StrategyLoader

LOG = logging.getLogger(__name__)


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

    def transform_logprobs(self, sample):
        """
        Transform logprobs to target format for KD training
        """

        logprobs = sample.pop(self.logprobs_field)
        target_seq_len = len(logprobs)
        input_seq_len = len(sample["input_ids"])
        input_padding_len = input_seq_len - target_seq_len
        # get non-zero top-k (prune None logprobs from vllm data step)
        top_k_vals = [
            len(logprobs[i])
            for i in range(len(logprobs))
            if logprobs[i] is not None and len(logprobs[i])
        ]
        max_top_k = max(set(top_k_vals), key=top_k_vals.count)
        min_top_k = min(set(top_k_vals), key=top_k_vals.count)
        top_k = min(max_top_k, min_top_k)
        if top_k == 0:
            raise ValueError("No non-zero top-k logprobs found.")

        target_logprobs = []
        target_token_ids = []
        target_mask = []

        if input_padding_len < 0:
            # logprobs is longer than target_seq_len,
            # so we need to slice from the left/beginning of logprobs
            logprobs = logprobs[:-input_seq_len]
            input_padding_len = 0
            # target_seq_len = input_seq_len

        # truncate the second dimension of the logprobs to top_k
        logprobs = [row[:top_k] for row in logprobs]

        # fill with -inf for padding_len tokens for top_k tokens
        # extend target_logprobs with a padding_len x top_k 2D list filled with -inf

        # we shift for causal models in the trainer, so start the range from 0
        for _ in range(0, input_padding_len):
            target_logprobs.append([-float("inf")] * top_k)
            target_token_ids.append(list(range(top_k)))
            target_mask.append([0] * top_k)

        for position in range(input_padding_len, input_seq_len):
            if sample["labels"][position] == -100:
                target_mask.append([0] * top_k)
            else:
                target_mask.append([1] * top_k)

        for _, token_pos_logprobs in enumerate(logprobs):
            # Initialize collections for logprobs and token_ids
            position_logprobs = []
            position_token_ids = []

            # Process each token probability entry
            for entry in token_pos_logprobs:
                # Extract logprob value
                logprob = entry["logprob"]

                # Parse token_id from the "token_id:###" format
                token_id = int(entry["token"].split(":")[1])

                # Append to our collections
                position_logprobs.append(logprob)
                position_token_ids.append(token_id)

            # Convert to a tensor for easier manipulation
            position_logprobs_tensor = torch.tensor(
                position_logprobs, dtype=torch.float
            )

            # Now we have distribution at T1 in log form, i.e. log p_{T1}(k).
            # Next, re-scale to T2 = self.kd_temperature via exponent-based trick
            # p_{T2}(k) = [p_{T1}(k)]^(T1 / T2) / Z
            #
            # Convert from log to probability
            teacher_probs_t1 = position_logprobs_tensor.exp()
            # normalize probabilities to sum to 1 in case they aren't already
            teacher_probs_t1_sum = teacher_probs_t1.sum(dim=0, keepdim=True)
            if teacher_probs_t1_sum > 1e-9:
                teacher_probs_t1 = teacher_probs_t1 / teacher_probs_t1_sum
            if self.kd_temperature != self.gen_temperature:
                # Exponentiate by factor (T1 / T2)
                exponent = self.gen_temperature / self.kd_temperature
                teacher_probs_t2 = teacher_probs_t1**exponent
            else:
                teacher_probs_t2 = teacher_probs_t1
            # Re-normalize
            teacher_probs_t2 = teacher_probs_t2 / teacher_probs_t2.sum(
                dim=0, keepdim=True
            )
            # Convert back to log
            position_logprobs_tensor = torch.log(teacher_probs_t2)

            # Now we have log p_{teacher, T2}(k) stored in position_logprobs_tensor
            position_logprobs_scaled = position_logprobs_tensor.tolist()

            target_logprobs.append(position_logprobs_scaled)
            target_token_ids.append(position_token_ids)

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

    def transform_logprobs(self, sample):
        """
        Transform logprobs to target format for KD training
        """

        logprobs = sample.pop(self.logprobs_field)
        target_seq_len = len(logprobs)
        input_seq_len = len(sample["input_ids"])
        input_padding_len = input_seq_len - target_seq_len
        # get non-zero top-k (prune None logprobs from vllm data step)
        top_k_vals = [
            len(logprobs[i])
            for i in range(len(logprobs))
            if logprobs[i] is not None and len(logprobs[i])
        ]
        max_top_k = max(set(top_k_vals), key=top_k_vals.count)
        min_top_k = min(set(top_k_vals), key=top_k_vals.count)
        top_k = min(max_top_k, min_top_k)
        if top_k == 0:
            raise ValueError("No non-zero top-k logprobs found.")

        target_logprobs = []
        target_token_ids = []
        target_mask = []

        if input_padding_len < 0:
            # logprobs is longer than target_seq_len,
            # so we need to slice from the left/beginning of logprobs
            logprobs = logprobs[:-input_seq_len]
            input_padding_len = 0
            # target_seq_len = input_seq_len

        # truncate the second dimension of the logprobs to top_k
        logprobs = [row[:top_k] for row in logprobs]

        # fill with -inf for padding_len tokens for top_k tokens
        # extend target_logprobs with a padding_len x top_k 2D list filled with -inf

        # we shift for causal models in the trainer, so start the range from 0
        for _ in range(0, input_padding_len):
            target_logprobs.append([-float("inf")] * top_k)
            target_token_ids.append(list(range(top_k)))
            target_mask.append([0] * top_k)

        for position in range(input_padding_len, input_seq_len):
            if sample["labels"][position] == -100:
                target_mask.append([0] * top_k)
            else:
                target_mask.append([1] * top_k)

        for token_pos_logprobs, pos_target_token_ids in zip(
            logprobs, sample["target_token_ids"], strict=False
        ):
            # Convert to a tensor for easier manipulation
            position_logprobs_tensor = torch.tensor(
                token_pos_logprobs, dtype=torch.float
            )

            # Now we have distribution at T1 in log form, i.e. log p_{T1}(k).
            # Next, re-scale to T2 = self.kd_temperature via exponent-based trick
            # p_{T2}(k) = [p_{T1}(k)]^(T1 / T2) / Z
            #
            # Convert from log to probability
            teacher_probs_t1 = position_logprobs_tensor.exp()
            # normalize probabilities to sum to 1 in case they aren't already
            teacher_probs_t1_sum = teacher_probs_t1.sum(dim=0, keepdim=True)
            if teacher_probs_t1_sum > 1e-9:
                teacher_probs_t1 = teacher_probs_t1 / teacher_probs_t1_sum
            if self.kd_temperature != self.gen_temperature:
                # Exponentiate by factor (T1 / T2)
                exponent = self.gen_temperature / self.kd_temperature
                teacher_probs_t2 = teacher_probs_t1**exponent
            else:
                teacher_probs_t2 = teacher_probs_t1
            # Re-normalize
            teacher_probs_t2 = teacher_probs_t2 / teacher_probs_t2.sum(
                dim=0, keepdim=True
            )
            # Convert back to log
            position_logprobs_tensor = torch.log(teacher_probs_t2)

            # Now we have log p_{teacher, T2}(k) stored in position_logprobs_tensor
            position_logprobs_scaled = position_logprobs_tensor.tolist()

            target_logprobs.append(position_logprobs_scaled)
            target_token_ids.append(pos_target_token_ids)

        # Update sample with transformed logprobs
        sample["target_logprobs"] = target_logprobs
        sample["target_token_ids"] = target_token_ids
        sample["target_mask"] = target_mask

        return sample

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
