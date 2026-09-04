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
On-policy distillation (GKD) trainer. An ``AxolotlTrainer`` subclass over the
standard pre-tokenized pipeline: ``training_step`` swaps the ground-truth
completion for a student (or teacher, under ``seq_kd``) rollout with probability
``lmbda`` (Axis B); ``compute_loss`` scores student vs teacher logits with the
configured divergence over the completion tokens (Axis A). ``_resolve_teacher``
(Axis C) and ``_token_weights`` (Axis D, uniform in v1) are extension seams.
"""

from __future__ import annotations

import random

import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from trl.models import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import disable_dropout_in_model
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer

from .divergence import resolve_divergence
from .rollout import extract_prompt_batch


class AxolotlGKDTrainer(AxolotlTrainer):
    """On-policy (generalized) knowledge distillation trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We own the loss (divergence vs teacher), so opt out of HF's CE loss-kwargs path.
        self.model_accepts_loss_kwargs = False

        a = self.args
        self.gkd_lmbda = a.gkd_lmbda
        self.gkd_temperature = a.gkd_temperature
        self.gkd_seq_kd = bool(a.gkd_seq_kd)
        self.divergence_fn = resolve_divergence(a.gkd_divergence, a.gkd_beta)

        teacher_model = self._resolve_teacher()
        self._check_shared_vocab(teacher_model)
        disable_dropout_in_model(self.model)
        disable_dropout_in_model(teacher_model)
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(
                teacher_model, evaluation_mode=True
            )

        self.generation_kwargs = {
            "max_new_tokens": a.gkd_max_new_tokens,
            "temperature": a.gkd_temperature,
            "do_sample": True,
            "top_k": a.gkd_top_k or 0,
            "use_cache": not self.args.gradient_checkpointing,
            "pad_token_id": self.processing_class.pad_token_id,
        }
        if self.processing_class.eos_token_id is not None:
            self.generation_kwargs["eos_token_id"] = self.processing_class.eos_token_id
        self.generation_config = GenerationConfig(**self.generation_kwargs)

    def _resolve_teacher(self) -> torch.nn.Module:
        """Axis C seam: the supervising model. v1 loads one external HF model; v3
        (self-distill) overrides to return the student under a privileged context."""
        init_kwargs = dict(self.args.gkd_teacher_init_kwargs or {})
        dtype = init_kwargs.get("dtype", next(self.model.parameters()).dtype)
        init_kwargs["dtype"] = (
            getattr(torch, dtype) if isinstance(dtype, str) else dtype
        )
        init_kwargs.setdefault(
            "trust_remote_code", getattr(self.args, "trust_remote_code", False)
        )
        return AutoModelForCausalLM.from_pretrained(
            self.args.gkd_teacher, **init_kwargs
        )

    def _check_shared_vocab(self, teacher_model: torch.nn.Module) -> None:
        student, teacher = self.model.config.vocab_size, teacher_model.config.vocab_size
        if student != teacher:
            raise ValueError(
                f"GKD needs a shared vocabulary: student vocab_size={student} but teacher "
                f"vocab_size={teacher}. Use a teacher with the same tokenizer/vocab."
            )

    def _token_weights(self, shift_labels, shift_student_logits, shift_teacher_logits):
        """Axis D seam: per-token loss weights. Uniform (``None``) in v1."""
        return None

    def _generate_on_policy(self, model, prompt_ids, prompt_mask):
        tokens = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
        ).sequences
        attn = torch.ones_like(tokens)
        labels = tokens.clone()
        pad = self.processing_class.pad_token_id
        if pad is not None:
            labels[labels == pad] = -100
            attn[tokens == pad] = 0
        # generate echoes the prompt back; mask it so only the completion is scored.
        labels[:, : prompt_ids.shape[1]] = -100
        return tokens, attn, labels

    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        gen_model = None
        if random.random() <= self.gkd_lmbda:  # nosec B311
            gen_model = model
        elif self.gkd_seq_kd:
            gen_model = self.teacher_model

        if gen_model is not None:
            prompt_ids, prompt_mask = extract_prompt_batch(
                inputs["input_ids"],
                inputs["labels"],
                inputs.get("attention_mask"),
                self.processing_class.pad_token_id,
            )
            with unwrap_model_for_generation(
                gen_model, self.accelerator, generation_kwargs=self.generation_kwargs
            ) as unwrapped_model:
                new_ids, new_attn, new_labels = self._generate_on_policy(
                    unwrapped_model, prompt_ids, prompt_mask
                )
            inputs = dict(inputs)
            inputs["input_ids"] = new_ids
            inputs["attention_mask"] = new_attn
            inputs["labels"] = new_labels
            inputs.pop("position_ids", None)  # stale for the regenerated sequence

        return super().training_step(model, inputs, num_items_in_batch)

    @override
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs = dict(inputs)
        inputs.pop("position_ids", None)
        inputs.pop("num_items_in_batch", None)

        labels = inputs["labels"]
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum().clamp_min(1)

        attention_mask = inputs.get("attention_mask")
        student_outputs = model(
            input_ids=inputs["input_ids"], attention_mask=attention_mask
        )
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"], attention_mask=attention_mask
            )

        # Causal shift: logits at i predict token i+1; the -100 label mask drops prompt/pad.
        loss = self.divergence_fn(
            student_logits=student_outputs.logits[:, :-1, :],
            teacher_logits=teacher_outputs.logits[:, :-1, :],
            labels=labels[:, 1:],
            temperature=self.gkd_temperature,
            num_items_in_batch=num_items_in_batch,
        )
        return (loss, student_outputs) if return_outputs else loss
