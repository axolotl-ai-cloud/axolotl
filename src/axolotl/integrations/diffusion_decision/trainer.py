"""Model-agnostic native diffusion trainer for typed decision labels."""

from __future__ import annotations

__ci_config_keys__ = ("diffusion_decision",)

import random
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.data import BatchSampler, Dataset, SequentialSampler

from axolotl.core.trainers.diffusion_lm.backends.encoder_canvas import (
    EncoderCanvasBackend,
)
from axolotl.core.trainers.diffusion_lm.batch import DiffusionBatch
from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer
from axolotl.core.trainers.diffusion_lm.unroll import run_unroll
from axolotl.model_support import DiffusionLayout, DiffusionNoise, LogitAlignment

from ._util import _value
from .args import DiffusionDecisionConfig
from .loss import (
    DecisionLabelExample,
    DecisionLossResult,
    decision_label_loss,
    decision_label_loss_from_hidden,
)
from .slot_sampling import DecisionDraw


class DiffusionDecisionTrainer(AxolotlDiffusionTrainer):
    """Train typed decisions through the backend selected by ``DiffusionSpec``."""

    reports_token_perplexity = False
    _decision_metrics: dict[str, dict[str, dict[str, float]]]

    def post_set_axolotl_cfg(self):
        super().post_set_axolotl_cfg()
        self._decision_config()
        self.model_accepts_loss_kwargs = True

    def _prepare_inputs(self, inputs):
        prepared = super()._prepare_inputs(inputs)
        batch = prepared.get("diffusion_batch")
        if isinstance(batch, DiffusionBatch):
            prepared = dict(prepared)
            prepared["diffusion_batch"] = batch.to(self.args.device)
        return prepared

    def _get_train_sampler(self, train_dataset: Dataset | None = None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        manifest = getattr(dataset, "manifest", None)
        sampled_draws = self._uses_sampled_slot_draws()
        if getattr(self.args, "sample_packing", False):
            if sampled_draws:
                raise ValueError(
                    "diffusion_decision sample_packing does not support sampled decision slots"
                )
            if isinstance(manifest, Mapping) and manifest.get(
                "per_batch_stratified", False
            ):
                raise ValueError(
                    "diffusion_decision sample_packing requires per_batch_stratified: false"
                )
            return super()._get_train_sampler(dataset)
        if not isinstance(manifest, Mapping) or not manifest.get(
            "per_batch_stratified", False
        ):
            if not sampled_draws:
                return super()._get_train_sampler(dataset)
            configured_drop_last = getattr(self.args, "dataloader_drop_last", None)
            return _DecisionDrawBatchSampler(
                len(dataset),
                self.args.per_device_train_batch_size,
                self._slot_draw_seed(),
                self.args.world_size,
                drop_last=True
                if configured_drop_last is None
                else configured_drop_last,
            )
        expected_batch_size = manifest.get("stratified_micro_batch_size")
        if not isinstance(expected_batch_size, int) or expected_batch_size < 1:
            raise ValueError("decision dataset is missing stratified_micro_batch_size")
        if self.args.per_device_train_batch_size != expected_batch_size:
            raise ValueError(
                "decision dataset microbatch size does not match per_device_train_batch_size"
            )
        seed = manifest.get("mixture_seed")
        if not isinstance(seed, int):
            raise ValueError("decision dataset is missing mixture_seed")
        return _StratifiedDecisionBatchSampler(
            len(dataset),
            expected_batch_size,
            seed,
            self.args.world_size,
            emit_draw_descriptors=sampled_draws,
        )

    def _uses_sampled_slot_draws(self) -> bool:
        return self._decision_config().latent.sample_num_slots

    def _slot_draw_seed(self) -> int:
        cfg_seed = _value(self.axolotl_cfg, "seed")
        seed = cfg_seed if cfg_seed is not None else getattr(self.args, "seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sampled decision slots require a nonnegative run seed")
        return seed

    def _get_dataloader(
        self,
        dataset,
        description,
        batch_size,
        sampler_fn=None,
        is_training=False,
        dataloader_key=None,
    ):
        manifest = getattr(dataset, "manifest", None)
        typed_decision_dataset = isinstance(manifest, Mapping)
        stratified_training = (
            is_training
            and typed_decision_dataset
            and manifest.get("per_batch_stratified", False)
        )
        sampled_slot_training = (
            is_training and typed_decision_dataset and self._uses_sampled_slot_draws()
        )
        if not hasattr(dataset, "column_names"):
            dataset.column_names = ()
        sample_packing = self.args.sample_packing
        drop_last = self.args.dataloader_drop_last
        packed_typed_batch = sample_packing and (
            is_training or self.args.eval_sample_packing is not False
        )
        if typed_decision_dataset and not packed_typed_batch:
            # Core packing would discard typed metadata and partial evaluation batches.
            self.args.sample_packing = False
            if not is_training:
                self.args.dataloader_drop_last = False
        if stratified_training or sampled_slot_training:
            self.accelerator.even_batches = True
        try:
            dataloader = super()._get_dataloader(
                dataset,
                description,
                batch_size,
                sampler_fn=sampler_fn,
                is_training=is_training,
                dataloader_key=dataloader_key,
            )
        finally:
            self.args.sample_packing = sample_packing
            self.args.dataloader_drop_last = drop_last
        return dataloader

    def _get_eval_sampler(self, eval_dataset: Dataset | None = None):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if self.args.sample_packing and self.args.eval_sample_packing is not False:
            return super()._get_eval_sampler(dataset)
        if isinstance(getattr(dataset, "manifest", None), Mapping):
            return SequentialSampler(dataset)
        return super()._get_eval_sampler(dataset)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        spec = self._native_spec
        if spec is None:
            raise ValueError(
                "diffusion_decision requires a resolved native DiffusionSpec"
            )
        decision = self._decision_config()
        k_max, grad_through_steps = self._native_unroll_settings()
        self._validate_decision_config(decision, k_max=k_max, spec=spec)
        self._validate_sampled_slot_metadata(inputs, decision, training=model.training)
        if (
            k_max > 1
            and grad_through_steps
            and (
                spec.layout is not DiffusionLayout.ENCODER_CANVAS
                or not spec.self_conditioning
            )
        ):
            raise NotImplementedError(
                "decision grad-through-steps requires encoder-canvas self-conditioning"
            )
        packed = None
        coordinates = None
        use_cce = bool(getattr(self.axolotl_cfg, "cut_cross_entropy", False))
        loss_function = decision_label_loss
        if use_cce:
            if (
                spec.layout is not DiffusionLayout.FULL_SEQUENCE
                or spec.logit_alignment is not LogitAlignment.ALIGNED
                or spec.self_conditioning
            ):
                raise ValueError("Decision CCE requires aligned full-sequence Nemotron")
            from axolotl.model_support.nemotron_diffusion.cut_cross_entropy import (
                get_cce_head,
                get_cce_options,
                linear_token_loss,
            )

            options = get_cce_options(model)
            if options is None:
                raise ValueError("Decision CCE requires the Nemotron CCE model patch")
            use_cce = model.training or not options.train_only
            if use_cce:
                head = get_cce_head(model)
                loss_function = partial(
                    decision_label_loss_from_hidden,
                    head=head,
                    linear_token_loss=partial(linear_token_loss, options=options),
                )
        if spec.layout is DiffusionLayout.FULL_SEQUENCE:
            logits, outputs = self._full_sequence_logits(
                model, inputs, spec, decision, return_hidden_states=use_cce
            )
        elif spec.layout is DiffusionLayout.ENCODER_CANVAS:
            logits, outputs, packed, coordinates = self._encoder_canvas_logits(
                model, inputs, spec, decision
            )
        else:
            raise ValueError(f"unsupported decision diffusion layout: {spec.layout}")
        if getattr(outputs, "axolotl_selected_logits", False):
            supervision = self._question_supervision(inputs)
            selected = logits
        else:
            selected, supervision = self._select_question_logits(
                logits, inputs, coordinates=coordinates
            )
        examples = inputs.get("decision_examples")
        if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
            raise TypeError(
                "decision_examples must be a sequence of DecisionLabelExample"
            )
        if not all(isinstance(example, DecisionLabelExample) for example in examples):
            raise TypeError(
                "decision_examples must contain DecisionLabelExample values"
            )
        result = loss_function(
            selected,
            examples=examples,
            supervision_mask=supervision,
            label_softmax=decision.labels.label_softmax,
            full_ce_weighting=decision.labels.full_ce_weighting,
            brier_weight=decision.labels.brier_weight,
            hard_label_smoothing=(
                decision.labels.hard_label_smoothing if model.training else 0.0
            ),
        )
        self._store_decision_metrics(
            result,
            selected,
            examples,
            supervision,
            decision,
            inputs.get("decision_sources"),
            train_eval="train" if model.training else "eval",
            loss_function=loss_function,
        )
        loss = self._scale_global_example_mean(
            result, len(examples), num_items_in_batch
        )
        if packed is not None:
            loss = loss + self._encoder_ar_loss(
                model, outputs, packed, num_items_in_batch
            )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            computed_loss = self.compute_loss(model, inputs)
        loss = computed_loss[0] if isinstance(computed_loss, tuple) else computed_loss
        loss = loss.detach().mean()
        if prediction_loss_only:
            return loss, None, None
        return loss, None, None

    def _store_decision_metrics(
        self,
        result: DecisionLossResult,
        selected: torch.Tensor,
        examples: Sequence[DecisionLabelExample],
        supervision: torch.Tensor,
        decision: DiffusionDecisionConfig,
        sources: object,
        *,
        train_eval: str,
        loss_function=decision_label_loss,
    ) -> None:
        totals = self._decision_metric_totals(train_eval)
        _accumulate_decision_metrics(
            totals.setdefault("decision/all", _empty_decision_metric_totals()),
            result,
            len(examples),
        )
        if sources is None:
            return
        if (
            not isinstance(sources, Sequence)
            or isinstance(sources, (str, bytes))
            or len(sources) != len(examples)
            or any(not isinstance(source, str) or not source for source in sources)
        ):
            raise ValueError(
                "decision_sources must be nonempty strings aligned with decision_examples"
            )
        groups: dict[str, list[int]] = {}
        for index, source in enumerate(sources):
            groups.setdefault(source, []).append(index)
        per_example = _per_example_decision_results(result, len(examples))
        with torch.no_grad():
            for source, indices in groups.items():
                if per_example is not None:
                    source_result = _subset_decision_result(per_example, indices)
                else:
                    index = torch.tensor(
                        indices, device=selected.device, dtype=torch.long
                    )
                    source_result = loss_function(
                        selected.detach().index_select(0, index),
                        examples=[examples[position] for position in indices],
                        supervision_mask=supervision.index_select(0, index),
                        label_softmax=decision.labels.label_softmax,
                        full_ce_weighting=decision.labels.full_ce_weighting,
                        brier_weight=decision.labels.brier_weight,
                        hard_label_smoothing=(
                            decision.labels.hard_label_smoothing
                            if train_eval == "train"
                            else 0.0
                        ),
                    )
                prefix = f"decision/{source}"
                _accumulate_decision_metrics(
                    totals.setdefault(prefix, _empty_decision_metric_totals()),
                    source_result,
                    len(indices),
                )

    def _decision_metric_totals(self, train_eval: str) -> dict[str, dict[str, float]]:
        if not hasattr(self, "_decision_metrics"):
            self._decision_metrics = {"train": {}, "eval": {}}
        return self._decision_metrics[train_eval]

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        train_eval = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        totals = _reduce_decision_metric_totals(
            self._decision_metric_totals(train_eval)
        )
        key_prefix = "eval_" if train_eval == "eval" else ""
        for prefix, values in totals.items():
            count = values["examples"]
            if count:
                for name in (
                    "loss",
                    "restricted_loss",
                    "full_vocab_loss",
                    "effective_full_vocab_loss",
                    "brier_loss",
                ):
                    logs[f"{key_prefix}{prefix}/{name}"] = values[name] / count
                hard_count = values["full_vocab_dft_hard_count"]
                logs[f"{key_prefix}{prefix}/full_vocab_dft_hard_count"] = hard_count
                logs[f"{key_prefix}{prefix}/full_vocab_dft_mean_hard_weight"] = (
                    values["full_vocab_dft_hard_weight_sum"] / hard_count
                    if hard_count
                    else 0.0
                )
                logs[f"{key_prefix}{prefix}/examples"] = count
        self._decision_metric_totals(train_eval).clear()
        return super().log(logs, start_time)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """Count logical examples, never native corruption or raw label tokens."""
        batches = []
        for _ in range(num_batches):
            try:
                batches.append(next(epoch_iterator))
            except StopIteration:
                break
        local_examples = sum(
            len(batch.get("decision_examples", ())) for batch in batches
        )
        if not local_examples:
            raise ValueError("decision gradient window contains no logical examples")
        count = torch.as_tensor(local_examples, device=device, dtype=torch.long)
        if self.args.world_size > 1:
            count = self.accelerator.gather(count).sum()
        spec = self._native_spec
        if spec is not None and spec.layout is DiffusionLayout.ENCODER_CANVAS:
            ar_count = sum(
                (
                    batch["diffusion_batch"].encoder_ar_valid_mask[:, 1:]
                    & batch["diffusion_batch"].encoder_ar_valid_mask[:, :-1]
                    & (
                        batch["diffusion_batch"].encoder_document_ids[:, 1:]
                        == batch["diffusion_batch"].encoder_document_ids[:, :-1]
                    )
                ).sum()
                for batch in batches
            ).to(device)
            if self.args.world_size > 1:
                ar_count = self.accelerator.gather(ar_count).sum()
            return batches, {"examples": count, "encoder_ar": ar_count}
        return batches, count

    def _decision_config(self) -> DiffusionDecisionConfig:
        value = _value(self.axolotl_cfg, "diffusion_decision")
        if isinstance(value, DiffusionDecisionConfig):
            return value
        if isinstance(value, Mapping):
            return DiffusionDecisionConfig.model_validate(value)
        raise ValueError(
            "DiffusionDecisionTrainer requires diffusion_decision settings"
        )

    @staticmethod
    def _validate_decision_config(
        decision: DiffusionDecisionConfig,
        *,
        k_max: int = 1,
        spec=None,
    ) -> None:
        latent = decision.latent
        if latent.mode not in {
            "none",
            "pad",
            "pinned",
            "learned",
            "prompt",
            "mask",
            "free",
        }:
            raise ValueError(f"unsupported decision latent mode: {latent.mode!r}")
        if latent.mode == "free":
            if latent.free_update_policy != "argmax":
                raise ValueError(
                    "decision free latent slots require free_update_policy=argmax"
                )
            if k_max < 2:
                raise ValueError("decision free latent slots require unroll.k_max > 1")
        if latent.mode == "mask" and getattr(spec, "noise", None) is not None:
            if spec.noise.value != "absorbing":
                raise ValueError(
                    "decision mask latent slots require absorbing diffusion"
                )

    def _validate_sampled_slot_metadata(
        self,
        inputs: Mapping[str, Any],
        decision: DiffusionDecisionConfig,
        *,
        training: bool,
    ) -> None:
        if not decision.latent.sample_num_slots:
            return
        examples = inputs.get("decision_examples")
        counts = inputs.get("decision_slot_counts")
        draws = inputs.get("decision_draws")
        slots = inputs.get("decision_slot_mask")
        prompt_slots = inputs.get("decision_prompt_slot_mask")
        if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
            raise TypeError("sampled decision slots require decision_examples")
        if not isinstance(counts, torch.Tensor) or counts.dtype is not torch.long:
            raise TypeError("sampled decision slots require long decision_slot_counts")
        if counts.ndim != 1 or counts.shape[0] != len(examples):
            raise ValueError(
                "decision_slot_counts must contain one count for each logical example"
            )
        if not isinstance(draws, tuple) or len(draws) != len(examples):
            raise ValueError(
                "sampled decision slots require one DecisionDraw for each logical example"
            )
        static_evaluation = all(draw is None for draw in draws)
        if static_evaluation:
            if training:
                raise ValueError(
                    "sampled-slot training batches require DecisionDraw metadata"
                )
            torch._assert_async(
                counts.eq(decision.latent.num_slots).all(),
                "sampled-slot evaluation must use the configured maximum count",
            )
        elif not all(isinstance(draw, DecisionDraw) for draw in draws):
            raise TypeError("sampled decision slots require DecisionDraw metadata")
        if not isinstance(slots, torch.Tensor) or slots.dtype is not torch.bool:
            raise TypeError("sampled decision slots require bool decision_slot_mask")
        torch._assert_async(
            (counts >= 0).all(), "sampled decision slot counts must be nonnegative"
        )
        torch._assert_async(
            (counts <= decision.latent.num_slots).all(),
            "sampled decision slot counts exceed configured maximum",
        )
        prompt_mode = decision.latent.mode == "prompt"
        if (
            self._native_spec is not None
            and self._native_spec.layout is DiffusionLayout.FULL_SEQUENCE
        ):
            documents = inputs.get("document_ids")
            loss_mask = inputs.get("canvas_loss_mask")
            pinned_mask = inputs.get("canvas_input_pinned_mask")
            if not isinstance(documents, torch.Tensor) or documents.ndim != 2:
                raise TypeError("sampled full-sequence slots require document_ids")
            if slots.shape != documents.shape:
                raise ValueError(
                    "sampled full-sequence slot mask must align with documents"
                )
            if (
                not isinstance(loss_mask, torch.Tensor)
                or not isinstance(pinned_mask, torch.Tensor)
                or loss_mask.shape != slots.shape
                or pinned_mask.shape != slots.shape
            ):
                raise ValueError(
                    "sampled full-sequence slots require aligned loss and pinned masks"
                )
            if prompt_mode:
                positions = inputs.get("position_ids")
                if (
                    not isinstance(prompt_slots, torch.Tensor)
                    or prompt_slots.dtype is not torch.bool
                    or prompt_slots.shape != documents.shape
                    or not isinstance(positions, torch.Tensor)
                    or positions.shape != documents.shape
                ):
                    raise ValueError(
                        "sampled prompt slots require aligned prompt-slot positions"
                    )
                semantic = inputs.get("semantic_validity")
                if (
                    not isinstance(semantic, torch.Tensor)
                    or semantic.shape != documents.shape
                ):
                    raise ValueError(
                        "sampled prompt slots require aligned semantic validity"
                    )
                safe_documents = documents.clamp_min(0)
                expected = semantic.bool() & (positions < counts[safe_documents])
                torch._assert_async(
                    prompt_slots.eq(expected).all(),
                    "sampled prompt slots must be each logical prompt prefix",
                )
                torch._assert_async(
                    (~slots).all(),
                    "sampled prompt slots cannot occupy canvas positions",
                )
                observed = torch.zeros_like(counts)
                observed.scatter_add_(
                    0,
                    safe_documents.reshape(-1),
                    (prompt_slots & semantic.bool()).reshape(-1).long(),
                )
                slots = prompt_slots
            else:
                observed = torch.zeros_like(counts)
                observed.scatter_add_(
                    0, documents.reshape(-1), slots.reshape(-1).long()
                )
        else:
            batch = inputs.get("diffusion_batch")
            if not isinstance(batch, DiffusionBatch):
                raise TypeError("sampled encoder-canvas slots require diffusion_batch")
            if slots.ndim != 2 or slots.shape[0] != len(examples):
                raise ValueError(
                    "sampled encoder-canvas slot mask must have one row per logical example"
                )
            if (
                slots.shape != batch.canvas_loss_mask.shape
                or slots.shape != batch.canvas_input_pinned_mask.shape
            ):
                raise ValueError(
                    "sampled encoder-canvas slots require aligned loss and pinned masks"
                )
            if prompt_mode:
                prompt_pinned = inputs.get("decision_prompt_input_pinned_mask")
                if (
                    not isinstance(prompt_slots, torch.Tensor)
                    or prompt_slots.dtype is not torch.bool
                    or prompt_slots.shape != batch.encoder_input_ids.shape
                    or not isinstance(prompt_pinned, torch.Tensor)
                    or prompt_pinned.dtype is not torch.bool
                    or prompt_pinned.shape != batch.encoder_input_ids.shape
                ):
                    raise ValueError(
                        "sampled prompt slots require aligned prompt pinned metadata"
                    )
                positions = torch.arange(
                    prompt_slots.shape[1], device=prompt_slots.device
                )[None]
                expected = batch.encoder_validity & (positions < counts[:, None])
                torch._assert_async(
                    prompt_slots.eq(expected).all(),
                    "sampled prompt slots must be each logical prompt prefix",
                )
                torch._assert_async(
                    (~slots).all(),
                    "sampled prompt slots cannot occupy canvas positions",
                )
                torch._assert_async(
                    ~(prompt_slots & batch.encoder_ar_valid_mask).any(),
                    "sampled prompt slots cannot receive encoder AR loss",
                )
                loss_mask = torch.zeros_like(prompt_slots)
                pinned_mask = prompt_pinned
                observed = prompt_slots.sum(dim=1, dtype=torch.long)
                slots = prompt_slots
            else:
                loss_mask = batch.canvas_loss_mask
                pinned_mask = batch.canvas_input_pinned_mask
                observed = slots.sum(dim=1, dtype=torch.long)
        torch._assert_async(
            observed.eq(counts).all(),
            "sampled decision slot masks do not match descriptor counts",
        )
        torch._assert_async(
            ~(slots & loss_mask).any(),
            "sampled decision slots cannot receive direct supervised loss",
        )
        if decision.latent.mode != "free":
            torch._assert_async(
                ~(slots & ~pinned_mask).any(),
                "fixed sampled decision slots must remain input-pinned",
            )

    def _full_sequence_logits(
        self,
        model,
        inputs: dict[str, Any],
        spec,
        decision: DiffusionDecisionConfig,
        *,
        return_hidden_states: bool = False,
    ):
        required = (
            "input_ids",
            "document_ids",
            "semantic_validity",
            "canvas_corruptible_mask",
            "canvas_input_pinned_mask",
            "canvas_update_mask",
        )
        _require_fields(inputs, required, "full-sequence decision batch")
        backend = self._full_sequence_backend()
        input_ids = inputs["input_ids"].long()
        packed = backend.pack(
            input_ids,
            inputs["document_ids"].long(),
            inputs["semantic_validity"].bool(),
            inputs.get("position_ids"),
        )
        logical_rows, logical_count = self._logical_row_indices(
            packed["document_ids"], packed["semantic_validity"]
        )
        times = self._decision_times(
            logical_count, packed["input_ids"].device, spec.time_floor, decision
        )
        corruptible = _pad_to_packed(
            inputs["canvas_corruptible_mask"].bool(), packed, False
        )
        pinned = _pad_to_packed(
            inputs["canvas_input_pinned_mask"].bool(), packed, False
        )
        update_mask = _pad_to_packed(inputs["canvas_update_mask"].bool(), packed, False)
        state, _, _ = backend.corrupt_native(
            packed,
            corruptible & ~pinned,
            times,
            document_time_indices=logical_rows,
        )
        free_slot_mask = None
        if decision.latent.mode == "free":
            if "decision_slot_mask" not in inputs:
                raise ValueError("free decision slots require decision_slot_mask")
            free_slot_mask = (
                _pad_to_packed(inputs["decision_slot_mask"].bool(), packed, False)
                & ~pinned
            )
            state = self._fresh_free_slot_state(
                model,
                state,
                free_slot_mask,
                spec,
                mask_token_id=backend.mask_token_id,
            )
        k_max, grad_through_steps = self._native_unroll_settings()
        final_kwargs = {}
        selected_rows, selected_positions, _ = self._question_coordinates(
            inputs,
            rows=packed["input_ids"].shape[0],
            positions=packed["input_ids"].shape[1],
        )
        capability_model = model
        while hasattr(capability_model, "module"):
            capability_model = capability_model.module
        use_selected_logits = (
            not return_hidden_states
            and spec.logit_alignment is LogitAlignment.ALIGNED
            and bool(getattr(capability_model, "supports_selected_logits", False))
        )
        if use_selected_logits:
            final_kwargs = {
                "forward_final": lambda current_state, _conditioning, _mask: (
                    backend.forward(
                        model,
                        packed,
                        current_state,
                        kernel_options=getattr(
                            self.axolotl_cfg, "flex_attn_compile_kwargs", None
                        ),
                        model_kwargs={
                            "axolotl_selected_logits": (
                                selected_rows,
                                selected_positions,
                            )
                        },
                    )
                ),
                "final_logits_from_outputs": lambda output: output.logits,
            }
        if return_hidden_states:
            final_kwargs = {
                "forward_final": lambda current_state, _conditioning, _mask: (
                    backend.forward(
                        model,
                        packed,
                        current_state,
                        kernel_options=getattr(
                            self.axolotl_cfg, "flex_attn_compile_kwargs", None
                        ),
                        model_kwargs={"cce_return_hidden_states": True},
                    )
                ),
                "final_logits_from_outputs": lambda output: output.last_hidden_state,
            }
        if k_max == 1:
            outputs, logits, _, _ = self._run_native_unroll(
                state=state,
                update_mask=update_mask & ~pinned,
                supports_self_conditioning=spec.self_conditioning,
                k1_conditioning_mask=None,
                recurrent_conditioning_mask=None,
                forward_step=lambda current_state, _conditioning, _conditioning_mask: (
                    backend.forward(
                        model,
                        packed,
                        current_state,
                        kernel_options=getattr(
                            self.axolotl_cfg, "flex_attn_compile_kwargs", None
                        ),
                    )
                ),
                logits_from_outputs=lambda current_outputs: backend.canvas_logits(
                    current_outputs,
                    packed,
                    aligned=spec.logit_alignment is LogitAlignment.ALIGNED,
                ),
                update_state=backend.update,
                **final_kwargs,
            )
        else:
            if grad_through_steps:
                raise NotImplementedError(
                    "decision grad-through-steps is not validated for multistep training"
                )
            if spec.self_conditioning:
                raise NotImplementedError(
                    "full-sequence decision K-step self-conditioning is not implemented"
                )
            # Non-SC full-sequence reads retain the same noisy labels each step.
            steps = self._sample_native_unroll_steps(k_max, packed["input_ids"].device)
            if free_slot_mask is not None:
                update_mask = free_slot_mask
            else:
                update_mask = torch.zeros_like(update_mask)
            outputs, logits, _ = run_unroll(
                state=state,
                update_mask=update_mask,
                steps=steps,
                grad_through_steps=False,
                supports_self_conditioning=False,
                k1_conditioning_mask=None,
                recurrent_conditioning_mask=None,
                forward_step=lambda current_state, _conditioning, _conditioning_mask: (
                    backend.forward(
                        model,
                        packed,
                        current_state,
                        kernel_options=getattr(
                            self.axolotl_cfg, "flex_attn_compile_kwargs", None
                        ),
                    )
                ),
                logits_from_outputs=lambda current_outputs: backend.canvas_logits(
                    current_outputs,
                    packed,
                    aligned=spec.logit_alignment is LogitAlignment.ALIGNED,
                ),
                update_state=backend.update,
                pilot_for_single_step=False,
                **final_kwargs,
            )
        return logits, outputs

    def _encoder_canvas_logits(
        self,
        model,
        inputs: dict[str, Any],
        spec,
        decision: DiffusionDecisionConfig,
    ):
        _require_fields(
            inputs,
            (
                "diffusion_batch",
                "decision_logical_rows",
                "decision_label_positions",
                "decision_question_mask",
                "decision_slot_mask",
            ),
            "encoder-canvas decision batch",
        )
        batch = inputs["diffusion_batch"]
        model_config = getattr(model, "config", None)
        if model_config is None:
            model_config = getattr(getattr(model, "module", None), "config", None)
        if model_config is None:
            raise TypeError("native encoder-canvas model must expose config")
        text_config = getattr(model_config, "text_config", model_config)
        backend = EncoderCanvasBackend(
            vocab_size=int(text_config.vocab_size),
            sliding_window=int(getattr(text_config, "sliding_window", 1024)),
            attention_backend=(
                "flex_attention"
                if getattr(self.axolotl_cfg, "attn_implementation", None)
                == "flex_attention"
                else "dense"
            ),
        )
        packed = backend.pack(batch)
        times = self._decision_times(
            batch.logical_ids.numel(), packed.device, spec.time_floor, decision
        )
        corrupted = backend.corrupt(packed, times)
        state = corrupted.input_ids
        k1, recurrent = self._conditioning_masks(packed, batch.logical_ids.numel())
        k_max, grad_through_steps = self._native_unroll_settings()
        steps = self._sample_native_unroll_steps(k_max, packed.device)
        update_mask = packed.canvas_update_mask & ~packed.canvas_input_pinned_mask
        if decision.latent.mode == "free":
            free_slot_mask = (
                self._packed_slot_mask(inputs["decision_slot_mask"], packed)
                & ~packed.canvas_input_pinned_mask
            )
            state = self._fresh_free_slot_state(model, state, free_slot_mask, spec)
        if k_max > 1:
            update_mask = torch.zeros_like(update_mask)
            recurrent = recurrent | self._packed_slot_mask(
                inputs["decision_slot_mask"], packed
            )
            if decision.latent.mode == "free":
                update_mask = free_slot_mask
        outputs = backend.forward(
            model,
            packed,
            state,
            unroll_steps=steps,
            grad_through_steps=grad_through_steps,
            pilot_for_single_step=k_max == 1,
            k1_conditioning_mask=k1,
            recurrent_conditioning_mask=recurrent,
            update_mask=update_mask,
            kernel_options=getattr(self.axolotl_cfg, "flex_attn_compile_kwargs", None),
        )
        logical_rows = inputs["decision_logical_rows"].long()
        positions = inputs["decision_label_positions"].long()
        question_mask = inputs["decision_question_mask"].bool()
        _validate_question_coordinates(
            logical_rows, positions, question_mask, batch.logical_ids.numel()
        )
        safe_rows = logical_rows.clamp_min(0)
        safe_positions = positions.clamp_min(0)
        physical_positions = packed.canvas_offsets[safe_rows] + safe_positions
        physical_rows = torch.zeros_like(physical_positions)
        inactive = torch.full_like(physical_rows, -1)
        coordinates = (
            torch.where(question_mask, physical_rows, inactive),
            torch.where(question_mask, physical_positions, inactive),
        )
        return outputs.logits, outputs, packed, coordinates

    @staticmethod
    def _fresh_free_slot_state(model, state, slot_mask, spec, *, mask_token_id=None):
        config = getattr(model, "config", None)
        if config is None or getattr(config, "vocab_size", None) is None:
            base_model = getattr(model, "get_base_model", lambda: None)()
            config = getattr(base_model, "config", config)
        text_config = getattr(config, "text_config", config)
        if spec.noise is DiffusionNoise.ABSORBING:
            if mask_token_id is None:
                mask_token_id = getattr(
                    config, "mask_token_id", getattr(text_config, "mask_token_id", None)
                )
            if not isinstance(mask_token_id, int):
                raise ValueError("absorbing free slots require a mask_token_id")
            values = torch.full_like(state, mask_token_id)
        else:
            values = torch.randint(
                int(text_config.vocab_size),
                state.shape,
                dtype=state.dtype,
                device=state.device,
            )
        return torch.where(slot_mask, values, state)

    @staticmethod
    def _packed_slot_mask(slot_mask: torch.Tensor, packed) -> torch.Tensor:
        if slot_mask.ndim != 2:
            raise ValueError("decision_slot_mask must be [logical_examples, canvas]")
        if slot_mask.shape[0] != packed.batch.logical_ids.numel():
            raise ValueError(
                "decision_slot_mask must have one row for each logical example"
            )
        logical_rows = packed.canvas_logical_row_indices
        physical_positions = torch.arange(
            logical_rows.shape[1], device=logical_rows.device
        )[None]
        valid_rows = logical_rows >= 0
        safe_rows = logical_rows.clamp_min(0)
        local_positions = physical_positions - packed.canvas_offsets[safe_rows]
        valid_positions = (local_positions >= 0) & (
            local_positions < slot_mask.shape[1]
        )
        safe_positions = local_positions.clamp(0, slot_mask.shape[1] - 1)
        slots = slot_mask.to(device=packed.device, dtype=torch.bool)[
            safe_rows, safe_positions
        ]
        return slots & valid_rows & valid_positions & packed.canvas_semantic_validity

    def _decision_times(
        self,
        count: int,
        device: torch.device,
        time_floor: float,
        decision: DiffusionDecisionConfig,
    ) -> torch.Tensor:
        times = self._sample_native_times(count, device, time_floor)
        if decision.read_fraction == 1.0:
            return torch.ones_like(times)
        if decision.read_fraction == 0.0:
            return times
        reads = torch.rand(count, device=device) < decision.read_fraction
        return torch.where(reads, torch.ones_like(times), times)

    def _encoder_ar_loss(
        self, model, outputs, packed, num_items_in_batch
    ) -> torch.Tensor:
        weight = self._native_value("encoder_ar_weight")
        if weight is None:
            weight = 1.0
        if not weight:
            return outputs.logits.new_zeros(())
        encoder_logits = getattr(outputs, "encoder_logits", None)
        if encoder_logits is None:
            if hasattr(model, "module"):
                raise RuntimeError(
                    "packed encoder-canvas DDP forward must return encoder_logits"
                )
            encoder_logits = self._native_lm_logits(
                model, outputs.encoder_last_hidden_state
            )
        token_loss = torch.nn.functional.cross_entropy(
            encoder_logits[:, :-1].float().flatten(0, -2),
            packed.encoder_input_ids[:, 1:].flatten(),
            reduction="none",
        )
        support = (
            packed.encoder_ar_valid_mask[:, 1:]
            & packed.encoder_ar_valid_mask[:, :-1]
            & (
                packed.encoder_document_ids[:, 1:]
                == packed.encoder_document_ids[:, :-1]
            )
        )
        denominator = _global_count(num_items_in_batch, "encoder_ar")
        if denominator is None:
            denominator = support.sum().detach().to(token_loss.dtype)
        else:
            denominator = torch.as_tensor(
                denominator, device=token_loss.device, dtype=token_loss.dtype
            )
            if denominator.numel() != 1:
                raise ValueError("decision encoder AR count must be scalar")
            if self.args.world_size > 1:
                denominator = denominator / self.args.world_size
        numerator = (token_loss * support.flatten()).sum()
        zero_denominator = denominator == 0
        safe_denominator = torch.where(
            zero_denominator, torch.ones_like(denominator), denominator
        )
        return (
            float(weight)
            * numerator
            / safe_denominator
            * (~zero_denominator).to(numerator.dtype)
        )

    def _conditioning_masks(self, packed, logical_count: int):
        sc_cfg = self._native_value("self_conditioning")
        probability = float(getattr(sc_cfg, "p", 0.5) if sc_cfg is not None else 0.5)
        k1 = None
        if probability:
            gates = torch.rand(logical_count, device=packed.device) < probability
            k1 = (
                packed.canvas_sc_eligible_mask
                & ~packed.canvas_input_pinned_mask
                & gates[packed.canvas_logical_row_indices]
            )
        recurrent = packed.canvas_sc_eligible_mask & ~packed.canvas_input_pinned_mask
        return k1, recurrent

    @staticmethod
    def _question_supervision(inputs: Mapping[str, Any]) -> torch.Tensor:
        question_mask = inputs["decision_question_mask"].bool()
        supervision = inputs["decision_supervision_mask"].bool()
        if supervision.shape != question_mask.shape or not torch.equal(
            question_mask, supervision
        ):
            raise ValueError(
                "decision supervision mask must exactly match question mask"
            )
        return supervision

    @staticmethod
    def _question_coordinates(
        inputs: Mapping[str, Any], *, rows: int, positions: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require_fields(
            inputs,
            (
                "decision_label_rows",
                "decision_label_positions",
                "decision_question_mask",
                "decision_supervision_mask",
            ),
            "decision label batch",
        )
        label_rows = inputs["decision_label_rows"].long()
        label_positions = inputs["decision_label_positions"].long()
        question_mask = inputs["decision_question_mask"].bool()
        supervision = inputs["decision_supervision_mask"].bool()
        if (
            label_rows.shape != label_positions.shape
            or label_rows.shape != question_mask.shape
        ):
            raise ValueError("decision label coordinates and question mask must align")
        if supervision.shape != question_mask.shape or not torch.equal(
            question_mask, supervision
        ):
            raise ValueError(
                "decision supervision mask must exactly match question mask"
            )
        _validate_question_coordinates(
            label_rows, label_positions, question_mask, rows, positions
        )
        return label_rows, label_positions, supervision

    @staticmethod
    def _select_question_logits(
        logits: torch.Tensor,
        inputs: Mapping[str, Any],
        *,
        coordinates: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_fields(
            inputs,
            (
                "decision_label_rows",
                "decision_label_positions",
                "decision_question_mask",
                "decision_supervision_mask",
            ),
            "decision label batch",
        )
        rows = (
            inputs["decision_label_rows"].long()
            if coordinates is None
            else coordinates[0]
        )
        positions = (
            inputs["decision_label_positions"].long()
            if coordinates is None
            else coordinates[1]
        )
        question_mask = inputs["decision_question_mask"].bool()
        supervision = inputs["decision_supervision_mask"].bool()
        if rows.shape != positions.shape or rows.shape != question_mask.shape:
            raise ValueError("decision label coordinates and question mask must align")
        if supervision.shape != question_mask.shape:
            raise ValueError("decision supervision mask must align with question mask")
        if not torch.equal(question_mask, supervision):
            raise ValueError(
                "decision supervision mask must exactly match question mask"
            )
        _validate_question_coordinates(
            rows, positions, question_mask, logits.shape[0], logits.shape[1]
        )
        safe_rows = rows.clamp_min(0)
        safe_positions = positions.clamp_min(0)
        return logits[safe_rows, safe_positions], question_mask

    def _scale_global_example_mean(
        self,
        result: DecisionLossResult,
        local_examples: int,
        global_examples: torch.Tensor | int | None,
    ) -> torch.Tensor:
        if global_examples is None:
            return result.loss
        denominator = torch.as_tensor(
            _global_count(global_examples, "examples"),
            device=result.loss.device,
            dtype=result.loss.dtype,
        )
        if denominator.numel() != 1:
            raise ValueError("decision global example count must be scalar")
        if self.args.world_size > 1:
            denominator = denominator / self.args.world_size
        return result.loss * local_examples / denominator


def _require_fields(
    inputs: Mapping[str, Any], names: tuple[str, ...], context: str
) -> None:
    missing = [name for name in names if name not in inputs]
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")


def _global_count(
    value: torch.Tensor | int | Mapping[str, torch.Tensor | int] | None,
    name: str,
) -> torch.Tensor | int | None:
    if not isinstance(value, Mapping):
        return value
    if name not in value:
        raise ValueError(f"decision global counts are missing {name!r}")
    return value[name]


def _empty_decision_metric_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "restricted_loss": 0.0,
        "full_vocab_loss": 0.0,
        "effective_full_vocab_loss": 0.0,
        "full_vocab_dft_hard_weight_sum": 0.0,
        "full_vocab_dft_hard_count": 0.0,
        "brier_loss": 0.0,
        "examples": 0.0,
    }


def _per_example_decision_results(
    result: DecisionLossResult, examples: int
) -> (
    tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]
    | None
):
    values = (
        result.per_example_loss,
        result.per_example_restricted_loss,
        result.per_example_full_vocab_loss,
        result.per_example_effective_full_vocab_loss,
        result.per_example_full_vocab_dft_hard_weight_sum,
        result.per_example_full_vocab_dft_hard_count,
        result.per_example_brier_loss,
    )
    if all(value is None for value in (values[0], values[1], values[2], values[6])):
        return None
    if any(
        value is None or len(value) != examples
        for value in (values[0], values[1], values[2], values[6])
    ):
        raise ValueError("decision loss result has invalid per-example metrics")
    full = values[2]
    assert full is not None
    effective = values[3] if values[3] is not None else full
    dft_hard_weight_sum = (
        values[4]
        if values[4] is not None
        else tuple(value.new_zeros(()) for value in full)
    )
    dft_hard_count = (
        values[5]
        if values[5] is not None
        else tuple(value.new_zeros(()) for value in full)
    )
    return (  # type: ignore[return-value]
        values[0],
        values[1],
        values[2],
        effective,
        dft_hard_weight_sum,
        dft_hard_count,
        values[6],
    )


def _subset_decision_result(
    values: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ],
    indices: Sequence[int],
) -> DecisionLossResult:
    if not indices:
        raise ValueError("source metric group must not be empty")
    selected = tuple(
        tuple(component[index] for index in indices) for component in values
    )
    means = tuple(torch.stack(component).mean() for component in selected)
    return DecisionLossResult(
        loss=means[0],
        restricted_loss=means[1],
        full_vocab_loss=means[2],
        brier_loss=means[6],
        effective_full_vocab_loss=means[3],
        full_vocab_dft_hard_weight_sum=torch.stack(selected[4]).sum(),
        full_vocab_dft_hard_count=torch.stack(selected[5]).sum(),
        per_example_loss=selected[0],
        per_example_restricted_loss=selected[1],
        per_example_full_vocab_loss=selected[2],
        per_example_effective_full_vocab_loss=selected[3],
        per_example_full_vocab_dft_hard_weight_sum=selected[4],
        per_example_full_vocab_dft_hard_count=selected[5],
        per_example_brier_loss=selected[6],
    )


def _accumulate_decision_metrics(
    totals: dict[str, float], result: DecisionLossResult, examples: int
) -> None:
    for name in (
        "loss",
        "restricted_loss",
        "full_vocab_loss",
        "effective_full_vocab_loss",
        "brier_loss",
    ):
        value = getattr(result, name)
        if value is None:
            if name == "effective_full_vocab_loss":
                value = result.full_vocab_loss
        totals[name] += float(value.detach()) * examples
    hard_weight_sum = result.full_vocab_dft_hard_weight_sum
    hard_count = result.full_vocab_dft_hard_count
    if hard_weight_sum is not None:
        totals["full_vocab_dft_hard_weight_sum"] += float(hard_weight_sum.detach())
    if hard_count is not None:
        totals["full_vocab_dft_hard_count"] += float(hard_count.detach())
    totals["examples"] += examples


def _reduce_decision_metric_totals(
    totals: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Merge interval numerators across ranks before emitting source metrics."""
    local = {name: dict(values) for name, values in totals.items()}
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local
    gathered: list[dict[str, dict[str, float]] | None] = [
        None for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather_object(gathered, local)
    merged: dict[str, dict[str, float]] = {}
    for rank_totals in gathered:
        if rank_totals is None:
            continue
        for prefix, values in rank_totals.items():
            target = merged.setdefault(prefix, _empty_decision_metric_totals())
            for name in target:
                target[name] += values.get(name, 0.0)
    return merged


def _pad_to_packed(
    value: torch.Tensor, packed: Mapping[str, torch.Tensor], fill: bool
) -> torch.Tensor:
    expected = packed["input_ids"].shape
    if value.shape == expected:
        return value
    if value.shape[0] != expected[0] or value.shape[1] > expected[1]:
        raise ValueError("full-sequence decision metadata must fit the packed row")
    return torch.nn.functional.pad(value, (0, expected[1] - value.shape[1]), value=fill)


def _validate_question_coordinates(
    rows: torch.Tensor,
    positions: torch.Tensor,
    question_mask: torch.Tensor,
    row_count: int,
    position_count: int | None = None,
) -> None:
    if rows.dtype is not torch.long or positions.dtype is not torch.long:
        raise TypeError("decision label rows and positions must be long")
    if question_mask.dtype is not torch.bool:
        raise TypeError("decision question mask must be bool")
    if position_count is None:
        torch._assert_async(
            (rows[question_mask] >= 0).all(), "active logical rows must be nonnegative"
        )
        torch._assert_async(
            (rows[question_mask] < row_count).all(),
            "active logical rows are out of range",
        )
        torch._assert_async(
            (positions[question_mask] >= 0).all(),
            "active label positions must be nonnegative",
        )
        return
    torch._assert_async(
        (rows[question_mask] >= 0).all(), "active label rows must be nonnegative"
    )
    torch._assert_async(
        (rows[question_mask] < row_count).all(), "active label rows are out of range"
    )
    torch._assert_async(
        (positions[question_mask] >= 0).all(),
        "active label positions must be nonnegative",
    )
    torch._assert_async(
        (positions[question_mask] < position_count).all(),
        "active label positions are out of range",
    )


class _DecisionDrawBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        world_size: int = 1,
        *,
        drop_last: bool = True,
    ) -> None:
        _validate_decision_sampler_args(dataset_size, batch_size, seed, world_size)
        self._dataset_size = dataset_size
        self._batch_size = batch_size
        self._seed = seed
        self._world_size = world_size
        self._drop_last = drop_last
        self._epoch = 0
        super().__init__(SequentialSampler(range(dataset_size)), batch_size, drop_last)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("decision sampler epoch must be a nonnegative integer")
        self._epoch = epoch

    def __iter__(self):
        indices = list(range(self._dataset_size))
        random.Random(f"decision-draw-v1:{self._seed}:{self._epoch}").shuffle(  # nosec B311 - Run-seeded training order.
            indices
        )
        groups = [
            indices[start : start + self._batch_size]
            for start in range(0, len(indices), self._batch_size)
            if not self._drop_last or start + self._batch_size <= len(indices)
        ]
        yield from _decision_draw_groups(
            groups,
            epoch=self._epoch,
            batch_size=self._batch_size,
            world_size=self._world_size,
        )

    def __len__(self) -> int:
        groups = self._dataset_size // self._batch_size
        if not self._drop_last and self._dataset_size % self._batch_size:
            groups += 1
        return groups + (-groups) % self._world_size


class _StratifiedDecisionBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        world_size: int = 1,
        *,
        emit_draw_descriptors: bool = False,
    ) -> None:
        _validate_decision_sampler_args(dataset_size, batch_size, seed, world_size)
        if dataset_size % batch_size:
            raise ValueError(
                "stratified decision dataset must contain complete microbatches"
            )
        self._dataset_size = dataset_size
        self._batch_size = batch_size
        self._seed = seed
        self._world_size = world_size
        self._emit_draw_descriptors = emit_draw_descriptors
        self._epoch = 0
        super().__init__(SequentialSampler(range(dataset_size)), batch_size, True)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("decision sampler epoch must be a nonnegative integer")
        self._epoch = epoch

    def __iter__(self):
        groups = list(range(self._dataset_size // self._batch_size))
        random.Random(f"decision-draw-v1:{self._seed}:{self._epoch}").shuffle(  # nosec B311 - Run-seeded training order.
            groups
        )
        batches = [
            list(range(group * self._batch_size, (group + 1) * self._batch_size))
            for group in groups
        ]
        if not self._emit_draw_descriptors:
            extra = (-len(batches)) % self._world_size
            batches.extend(batches[index % len(batches)] for index in range(extra))
            yield from batches
            return
        yield from _decision_draw_groups(
            batches,
            epoch=self._epoch,
            batch_size=self._batch_size,
            world_size=self._world_size,
        )

    def __len__(self) -> int:
        groups = self._dataset_size // self._batch_size
        return groups + (-groups) % self._world_size


def _decision_draw_groups(
    groups: Sequence[Sequence[int]],
    *,
    epoch: int,
    batch_size: int,
    world_size: int,
):
    if not groups:
        return
    padded = list(groups)
    extra = (-len(padded)) % world_size
    padded.extend(groups[index % len(groups)] for index in range(extra))
    for group_ordinal, group in enumerate(padded):
        yield [
            DecisionDraw(
                index=index,
                epoch=epoch,
                global_draw_ordinal=group_ordinal * batch_size + position,
            )
            for position, index in enumerate(group)
        ]


def _validate_decision_sampler_args(
    dataset_size: int, batch_size: int, seed: int, world_size: int
) -> None:
    values = {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "seed": seed,
        "world_size": world_size,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values.values()
    ):
        raise ValueError("decision sampler arguments must be integers")
    if dataset_size < 1 or batch_size < 1:
        raise ValueError(
            "decision sampler dataset size and batch size must be positive"
        )
    if seed < 0:
        raise ValueError("decision sampler seed must be nonnegative")
    if world_size < 1:
        raise ValueError("decision sampler world_size must be positive")
