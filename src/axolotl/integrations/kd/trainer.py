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
KD trainer
"""

import os
from typing import Any, Optional, Union

import requests
import torch
from torch import nn
from transformers import GenerationConfig
from trl.models import unwrap_model_for_generation
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer

from .kernels.liger import LigerFusedLinearKLTopKLogprobLoss


class AxolotlKDTrainer(AxolotlTrainer):
    """
    Custom trainer subclass for Knowledge Distillation (KD)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True

        loss_fn = LigerFusedLinearKLTopKLogprobLoss(
            self.args.kd_ce_alpha,  # hard label loss
            self.args.kd_alpha,  # kd loss
            self.args.kd_temperature,
            self.args.kd_beta or 0.0,
            compute_ce_loss=bool(self.args.kd_ce_alpha),
            normalize_topk=self.args.kd_normalize_topk,
        )
        target = self.model

        # Unwrap PEFT wrapper
        if hasattr(target, "get_base_model"):
            target = target.get_base_model()

        # Set on the actual model instance
        target._loss_function = loss_fn

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        columns_to_add = []
        if self._signature_columns:
            if "target_logprobs" not in self._signature_columns:
                columns_to_add.append("target_logprobs")
            if "target_token_ids" not in self._signature_columns:
                columns_to_add.append("target_token_ids")
            if "target_mask" not in self._signature_columns:
                columns_to_add.append("target_mask")
            if columns_to_add:
                self._signature_columns += columns_to_add

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (
            self.args.sample_packing
            and hasattr(inputs, "attention_mask")
            and hasattr(inputs, "position_ids")
        ):
            del inputs["attention_mask"]

        if num_items_in_batch is None and "labels" in inputs:
            num_items_in_batch = (inputs["labels"] != -100).sum().item()

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs["loss"]
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs

        return (loss, outputs) if return_outputs else loss


class AxolotlOnlineKDTrainer(AxolotlKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generation_config = GenerationConfig(
            max_new_tokens=kwargs.get("kd_online_max_new_tokens"),
            temperature=1.0,
            do_sample=True,
            top_k=0,
            use_cache=False if kwargs.get("gradient_checkpointing") else True,
            pad_token_id=self.processing_class.pad_token_id,
        )
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = (
                self.model.generation_config.eos_token_id
            )

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        # Generate output with respect to the prompt-only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs.get("prompt_attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the original inputs.
        """
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            new_input_ids, new_attention_mask, new_labels = (
                self.generate_on_policy_outputs(
                    unwrapped_model,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
            )
        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_attention_mask
        inputs["labels"] = new_labels

        target_token_ids, target_logprobs, target_mask = self.get_teacher_logprobs(
            inputs["input_ids"], inputs["labels"]
        )
        inputs["target_token_ids"] = target_token_ids
        inputs["target_logprobs"] = target_logprobs
        inputs["target_mask"] = target_mask

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss

    def get_teacher_logprobs(self, input_ids, labels):
        request_body = {
            "model": self.axolotl_cfg.kd_online_server_model,
            "prompt": input_ids,
            "logprobs": self.axolotl_cfg.kd_online_topk,
            "echo": True,
            "skip_special_tokens": False,
            "n": 1,
            "max_tokens": 0,
            "temperature": 1.0,
        }
        base_url = self.args.kd_online_server_base_url
        api_url = f"{base_url}/v1/completions"
        bearer_token = os.getenv("OPENAI_API_KEY")

        headers = {"Authorization": f"Bearer {bearer_token}"}
        response = requests.post(
            api_url, json=request_body, headers=headers, timeout=30
        )
        prompt_logprobs = response.choices[0].logprobs.top_logprobs[
            1:
        ]  # prune first null position
        return self.transform_logprobs(input_ids, labels, prompt_logprobs)

    def transform_logprobs(self, input_ids, labels, logprobs):
        """
        Transform logprobs to target format for KD training
        """

        target_seq_len = len(logprobs)
        input_seq_len = len(input_ids)
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
            if labels[position] == -100:
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
            # teacher_probs_t2 = teacher_probs_t2 / teacher_probs_t2.sum(
            #     dim=0, keepdim=True
            # )
            # Convert back to log
            position_logprobs_tensor = torch.log(teacher_probs_t2)

            # Now we have log p_{teacher, T2}(k) stored in position_logprobs_tensor
            position_logprobs_scaled = position_logprobs_tensor.tolist()

            target_logprobs.append(position_logprobs_scaled)
            target_token_ids.append(position_token_ids)

        # Update sample with transformed logprobs
        return target_token_ids, target_logprobs, target_mask
