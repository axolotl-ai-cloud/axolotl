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
from typing import Any, Dict

import torch

from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy, StrategyLoader


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
        )

    def transform_logprobs(self, sample):
        logprobs = sample.pop(self.logprobs_field)
        target_seq_len = len(logprobs)
        input_seq_len = len(sample["input_ids"])
        input_padding_len = input_seq_len - target_seq_len
        top_k = len(logprobs[0])
        target_logprobs = []
        target_token_ids = []
        target_mask = []

        # fill with -inf for padding_len tokens for top_k tokens
        # extend target_logprobs with a padding_len x top_k 2D list filled with -inf
        for _ in range(1, input_padding_len):  # start at 1 since this is causal
            target_logprobs.append([-float("inf")] * top_k)
            target_token_ids.append(list(range(top_k)))
            target_mask.append([0] * top_k)

        for _ in range(target_seq_len):
            # TODO also check against sample["labels"]
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
            # Convert to tensor
            position_logprobs_tensor = torch.tensor(
                position_logprobs, dtype=torch.float
            )

            if self.kd_temperature != self.gen_temperature:
                #
                # Now we have distribution at T1 in log form, i.e. log p_{T1}(k).
                # Next, re-scale to T2 = self.kd_temperature via exponent-based trick
                # p_{T2}(k) = [p_{T1}(k)]^(T1 / T2) / Z
                #
                # Convert from log to probability
                teacher_probs_t1 = position_logprobs_tensor.exp()
                # Exponentiate by factor (T1 / T2)
                exponent = self.gen_temperature / self.kd_temperature
                teacher_probs_t2 = teacher_probs_t1**exponent
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

        # since we started at index 1 for causal, we need one more padding token
        target_logprobs.append([-float("inf")] * top_k)
        target_token_ids.append(list(range(top_k)))
        target_mask.append([0] * top_k)

        # Update sample with transformed logprobs
        sample["target_logprobs"] = target_logprobs
        sample["target_token_ids"] = target_token_ids
        sample["target_mask"] = target_mask

        return sample

    def tokenize_prompt(self, prompt):
        logprobs = prompt.pop(self.logprobs_field)
        tokenized_prompt = super().tokenize_prompt(prompt)
        tokenized_prompt[self.logprobs_field] = logprobs
        tokenized_prompt = self.transform_logprobs(tokenized_prompt)

        return tokenized_prompt


class KDStrategyLoader(StrategyLoader):
    """
    Load ChatTemplateStrategy with KD support using StrategyLoader.
    """

    def _get_strategy_cls(self):
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


load = KDStrategyLoader()
