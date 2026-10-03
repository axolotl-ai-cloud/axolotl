"""Model-agnostic native diffusion trainer for typed decision labels."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.data import BatchSampler, Dataset, SequentialSampler

from axolotl.integrations.diffusion.lm.batch import DiffusionBatch
from axolotl.integrations.diffusion.lm.trainer import AxolotlDiffusionTrainer
from axolotl.model_support import DiffusionLayout, LogitAlignment

from .args import DecisionConfig
from .loss import (
    DecisionLabelExample,
    DecisionLossResult,
    decision_label_loss,
    decision_label_loss_from_hidden,
)
from .unroll import run_decision_unroll


class DecisionTrainer(AxolotlDiffusionTrainer):
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
        if getattr(self.args, "sample_packing", False):
            if isinstance(manifest, Mapping) and manifest.get(
                "per_batch_stratified", False
            ):
                raise ValueError(
                    "decision sample_packing requires per_batch_stratified: false"
                )
            return super()._get_train_sampler(dataset)
        if not isinstance(manifest, Mapping) or not manifest.get(
            "per_batch_stratified", False
        ):
            return super()._get_train_sampler(dataset)
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
        )

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
        if stratified_training:
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
                "decision requires a resolved native DiffusionSpec"
            )
        decision = self._decision_config()
        k_max, grad_through_steps = self._native_unroll_settings()
        self._validate_decision_config(decision, k_max=k_max, spec=spec)
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
        if spec.layout is not DiffusionLayout.FULL_SEQUENCE:
            raise ValueError("decision supports full-sequence Nemotron only")
        logits, outputs = self._full_sequence_logits(
            model, inputs, spec, decision, return_hidden_states=use_cce
        )
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
        decision: DecisionConfig,
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
        return batches, count

    def _decision_config(self) -> DecisionConfig:
        cfg = self.axolotl_cfg
        value = (
            cfg.get("decision")
            if isinstance(cfg, Mapping)
            else getattr(cfg, "decision", None)
        )
        if isinstance(value, DecisionConfig):
            return value
        if isinstance(value, Mapping):
            return DecisionConfig.model_validate(value)
        raise ValueError(
            "DecisionTrainer requires decision settings"
        )

    @staticmethod
    def _validate_decision_config(self, decision, *, k_max, spec):
        if spec.layout is not DiffusionLayout.FULL_SEQUENCE:
            raise ValueError("decision supports full-sequence Nemotron only")

    def _full_sequence_logits(
        self,
        model,
        inputs: dict[str, Any],
        spec,
        decision: DecisionConfig,
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
            update_mask = torch.zeros_like(update_mask)
            outputs, logits, _ = run_decision_unroll(
                state=state,
                steps=steps,
                grad_through_steps=False,
                supports_self_conditioning=False,
                conditioning_mask=None,
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
                update_mask=update_mask,
                **final_kwargs,
            )
        return logits, outputs

    def _decision_times(
        self,
        count: int,
        device: torch.device,
        time_floor: float,
        decision: DecisionConfig,
    ) -> torch.Tensor:
        times = self._sample_native_times(count, device, time_floor)
        if decision.read_fraction == 1.0:
            return torch.ones_like(times)
        if decision.read_fraction == 0.0:
            return times
        reads = torch.rand(count, device=device) < decision.read_fraction
        return torch.where(reads, torch.ones_like(times), times)

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
