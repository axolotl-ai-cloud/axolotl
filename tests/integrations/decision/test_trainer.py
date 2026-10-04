"""CPU coverage for the model-agnostic decision diffusion trainer."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler

import axolotl.core.trainers.base as trainer_base
from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.training_args import AxolotlTrainingArguments
from axolotl.integrations.decision.datasets import DecisionDataset
from axolotl.integrations.decision.loss import (
    DecisionLabelExample,
    DecisionLabelQuestion,
    DecisionLossResult,
    DistributionLabel,
    HardLabel,
    SetLabel,
)
from axolotl.integrations.decision.records import DecisionCanvas
from axolotl.integrations.decision.trainer import (
    DecisionTrainer,
    _reduce_decision_metric_totals,
    _StratifiedDecisionBatchSampler,
)
from axolotl.integrations.decision.training_collator import (
    DecisionTrainingCollator,
)
from axolotl.integrations.diffusion.lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.integrations.diffusion.lm.trainer import AxolotlDiffusionTrainer
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
)


class _TinyNativeModel(nn.Module):
    def __init__(self, vocab_size: int = 11) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.linspace(-0.4, 0.6, vocab_size))
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            sliding_window=8,
            text_config=SimpleNamespace(vocab_size=vocab_size, sliding_window=8),
        )
        self.calls: list[dict[str, object]] = []

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if input_ids is not None else decoder_input_ids
        assert ids is not None
        self.calls.append(kwargs)
        token_slopes = torch.arange(
            self.logit_bias.numel(), device=ids.device, dtype=self.logit_bias.dtype
        )
        positions = torch.arange(
            ids.shape[1], device=ids.device, dtype=self.logit_bias.dtype
        )
        logits = self.logit_bias + positions[None, :, None] * token_slopes
        logits = logits.expand(ids.shape[0], -1, -1)
        encoder_ids = kwargs.get("encoder_input_ids")
        encoder_logits = None
        if isinstance(encoder_ids, torch.Tensor):
            encoder_positions = torch.arange(
                encoder_ids.shape[1],
                device=encoder_ids.device,
                dtype=self.logit_bias.dtype,
            )
            encoder_logits = self.logit_bias + (
                encoder_positions[None, :, None] * token_slopes
            )
        return SimpleNamespace(logits=logits, encoder_logits=encoder_logits)


class _SelectedTinyNativeModel(_TinyNativeModel):
    supports_selected_logits = True

    def __init__(self, vocab_size: int = 11) -> None:
        super().__init__(vocab_size)
        self.selected_shapes: list[torch.Size] = []

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        selected = kwargs.pop("axolotl_selected_logits", None)
        output = super().forward(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, **kwargs
        )
        if selected is None:
            return output
        rows, positions = selected
        output.logits = output.logits[rows.clamp_min(0), positions.clamp_min(0)]
        output.axolotl_selected_logits = True
        self.selected_shapes.append(output.logits.shape)
        return output


class _TraceNativeModel(_TinyNativeModel):
    def __init__(self, vocab_size: int = 11) -> None:
        super().__init__(vocab_size)
        self.forward_states: list[torch.Tensor] = []
        self.forward_grad_enabled: list[bool] = []
        self.forward_logits: list[torch.Tensor] = []

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if input_ids is not None else decoder_input_ids
        assert ids is not None
        self.forward_states.append(ids.detach().clone())
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        outputs = super().forward(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, **kwargs
        )
        if torch.is_grad_enabled():
            outputs.logits.retain_grad()
            self.forward_logits.append(outputs.logits)
        return outputs


class _LoopNativeModel(nn.Module):
    """Small native model whose logits expose whether labels were corrupted."""

    def __init__(self, vocab_size: int = 11) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.linspace(-0.4, 0.6, vocab_size))
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            sliding_window=128,
            text_config=SimpleNamespace(vocab_size=vocab_size, sliding_window=128),
        )
        self.observed_input_ids: list[torch.Tensor] = []

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if input_ids is not None else decoder_input_ids
        assert ids is not None
        self.observed_input_ids.append(ids.detach().cpu().clone())
        encoder_ids = kwargs.get("encoder_input_ids")
        encoder_logits = None
        if isinstance(encoder_ids, torch.Tensor):
            encoder_logits = self.logit_bias.expand(
                encoder_ids.shape[0], encoder_ids.shape[1], -1
            )
        return SimpleNamespace(
            logits=self.logit_bias.expand(ids.shape[0], ids.shape[1], -1),
            encoder_logits=encoder_logits,
        )


class _TrainerHarness(DecisionTrainer):
    def __init__(
        self,
        spec: DiffusionSpec,
        decision: dict[str, object] | None = None,
        *,
        world_size: int = 1,
        native_values: dict[str, object] | None = None,
    ) -> None:
        self._spec = spec
        self.axolotl_cfg = {"decision": decision or {}}
        self.args = SimpleNamespace(world_size=world_size)
        self._special_token_ids: set[int] = set()
        self._native_values = native_values or {}

    @property
    def _native_spec(self) -> DiffusionSpec:
        return self._spec

    def _full_sequence_backend(self) -> FullSequenceBackend:
        return FullSequenceBackend(mask_token_id=10, attention_backend="dense")

    def _native_value(self, name: str, default=None):
        values: dict[str, object] = {"t_eps": 0.5, "self_conditioning": None}
        values.update(self._native_values)
        return values.get(name, default)

    def _native_unroll_settings(self) -> tuple[int, bool]:
        return 1, False

    @staticmethod
    def _sample_native_unroll_steps(k_max: int, device: torch.device) -> int:
        del k_max, device
        return 1

    def _sample_native_times(
        self, count: int, device: torch.device, default_eps: float
    ) -> torch.Tensor:
        del default_eps
        return torch.full((count,), 0.5, device=device)


class _MultistepTrainerHarness(_TrainerHarness):
    _sampled_steps_value = 1

    def __init__(
        self,
        spec: DiffusionSpec,
        *,
        k_max: int,
        sampled_steps: int,
        grad_through_steps: bool = False,
        decision: dict[str, object] | None = None,
    ) -> None:
        super().__init__(spec, decision)
        self._k_max = k_max
        _MultistepTrainerHarness._sampled_steps_value = sampled_steps
        self._grad_through_steps = grad_through_steps

    def _native_unroll_settings(self) -> tuple[int, bool]:
        return self._k_max, self._grad_through_steps

    @staticmethod
    def _sample_native_unroll_steps(k_max: int, device: torch.device) -> int:
        del k_max, device
        return _MultistepTrainerHarness._sampled_steps_value


class _NoCommitMultistepTrainerHarness(_MultistepTrainerHarness):
    def _run_native_unroll(self, *args, **kwargs):
        del args, kwargs
        pytest.fail("K>1 decision configuration must not use committing native unroll")


class _TrainingStepHarness(_TrainerHarness):
    def _prepare_context_parallel_inputs(self, model, inputs):
        del model
        return contextlib.nullcontext, inputs

    @staticmethod
    def _prepare_inputs(inputs):
        return inputs

    @staticmethod
    def compute_loss_context_manager():
        return contextlib.nullcontext()

    @staticmethod
    def compute_loss(model, inputs, num_items_in_batch=None):
        del inputs, num_items_in_batch
        return model.logit_bias.sum()


class _LoopTrainer(DecisionTrainer):
    def __init__(self, *args, spec: DiffusionSpec, **kwargs) -> None:
        self._loop_spec = spec
        super().__init__(*args, **kwargs)

    @property
    def _native_spec(self) -> DiffusionSpec:
        return self._loop_spec

    def _full_sequence_backend(self) -> FullSequenceBackend:
        return FullSequenceBackend(mask_token_id=10, attention_backend="dense")

    def _native_value(self, name: str, default=None):
        return {"t_eps": 0.0, "self_conditioning": None}.get(name, default)

    def _native_unroll_settings(self) -> tuple[int, bool]:
        return 1, False

    @staticmethod
    def _sample_native_unroll_steps(k_max: int, device: torch.device) -> int:
        del k_max, device
        return 1

    def _sample_native_times(
        self, count: int, device: torch.device, default_eps: float
    ) -> torch.Tensor:
        del default_eps
        return torch.ones(count, device=device)

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(
            self.train_dataset if train_dataset is None else train_dataset
        )

    def create_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.learning_rate
        )
        return self.optimizer


class _PredictionHarness(_TrainerHarness):
    def __init__(self, spec: DiffusionSpec) -> None:
        super().__init__(spec)
        self.prepared = False
        self.seen_inputs = None

    def _prepare_inputs(self, inputs):
        self.prepared = True
        return {"prepared": inputs}

    @staticmethod
    def compute_loss_context_manager():
        return contextlib.nullcontext()

    def compute_loss(self, model, inputs, **kwargs):
        del kwargs
        self.seen_inputs = inputs
        return model.logit_bias.square().sum()


def _spec(layout: DiffusionLayout, alignment: LogitAlignment) -> DiffusionSpec:
    return DiffusionSpec(
        noise=DiffusionNoise.ABSORBING,
        layout=layout,
        logit_alignment=alignment,
        first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
        self_conditioning=layout is DiffusionLayout.ENCODER_CANVAS,
        max_canvas=16,
        max_context=32,
        eos_handling=EosHandling.INDEPENDENT,
        mask_token_policy=MaskTokenPolicy.MODEL,
        default_time_weighting=TimeWeighting.NONE,
        objective_reduction=ObjectiveReduction.EXAMPLE_MEAN,
        generation_adapter=GenerationAdapter.FULL_SEQUENCE,
        reduction_scope=ReductionScope.GLOBAL_WINDOW,
    )


def _example(question_count: int, *, weight: float = 1.0) -> DecisionLabelExample:
    return DecisionLabelExample(
        questions=tuple(
            DecisionLabelQuestion(
                position=index,
                allowed_token_ids=(1, 2, 3),
                target=HardLabel(index % 3),
            )
            for index in range(question_count)
        ),
        source_weight=weight,
    )


@pytest.mark.parametrize("alignment", [LogitAlignment.SHIFTED, LogitAlignment.ALIGNED])
def test_full_sequence_trainer_uses_raw_input_columns_and_spec_alignment(alignment):
    trainer = _TrainerHarness(_spec(DiffusionLayout.FULL_SEQUENCE, alignment))
    model = _TinyNativeModel()
    inputs = {
        "input_ids": torch.tensor([[1, 4, 5, 2, 6, 7]]),
        "document_ids": torch.tensor([[0, 0, 0, 1, 1, 1]]),
        "semantic_validity": torch.ones((1, 6), dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
        "canvas_corruptible_mask": torch.tensor(
            [[False, True, False, False, False, True]]
        ),
        "canvas_input_pinned_mask": torch.zeros((1, 6), dtype=torch.bool),
        "canvas_update_mask": torch.tensor([[False, True, False, False, False, True]]),
        "decision_examples": (_example(2),),
        "decision_question_mask": torch.tensor([[True, True]]),
        "decision_supervision_mask": torch.tensor([[True, True]]),
        "decision_label_rows": torch.tensor([[0, 0]], dtype=torch.long),
        "decision_label_positions": torch.tensor([[1, 5]], dtype=torch.long),
    }

    aligned_logits, raw_outputs = trainer._full_sequence_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )
    if alignment is LogitAlignment.SHIFTED:
        torch.testing.assert_close(aligned_logits[0, 1], raw_outputs.logits[0, 0])
        torch.testing.assert_close(aligned_logits[0, 3], raw_outputs.logits[0, 3])
    else:
        torch.testing.assert_close(aligned_logits, raw_outputs.logits)

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    loss.backward()

    assert outputs.logits.shape == (1, 6, 11)
    assert model.logit_bias.grad is not None
    assert torch.isfinite(model.logit_bias.grad).all()
    assert torch.count_nonzero(model.logit_bias.grad)
    torch.testing.assert_close(
        trainer._decision_times(
            2, torch.device("cpu"), 0.0, trainer._decision_config()
        ),
        torch.ones(2),
    )


def _full_sequence_multistep_inputs() -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[1, 4, 5, 2, 6, 7]]),
        "document_ids": torch.tensor([[0, 0, 0, 1, 1, 1]]),
        "semantic_validity": torch.ones((1, 6), dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
        "canvas_corruptible_mask": torch.tensor(
            [[False, True, False, False, False, True]]
        ),
        "canvas_input_pinned_mask": torch.tensor(
            [[False, False, False, False, False, True]]
        ),
        "canvas_update_mask": torch.tensor([[False, True, False, False, False, True]]),
        "decision_examples": (_example(2),),
        "decision_question_mask": torch.tensor([[True, True]]),
        "decision_supervision_mask": torch.tensor([[True, True]]),
        "decision_label_rows": torch.tensor([[0, 0]], dtype=torch.long),
        "decision_label_positions": torch.tensor([[1, 5]], dtype=torch.long),
    }


@pytest.mark.parametrize("steps", [1, 2])
def test_selected_logits_match_dense_full_ce_brier_for_multistep_padded_questions(
    steps,
):
    decision = {"labels": {"label_softmax": "full", "brier_weight": 0.1}}
    dense_trainer = _MultistepTrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=2,
        sampled_steps=steps,
        decision=decision,
    )
    selected_trainer = _MultistepTrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=2,
        sampled_steps=steps,
        decision=decision,
    )
    dense_model = _TinyNativeModel()
    selected_model = _SelectedTinyNativeModel()
    selected_model.load_state_dict(dense_model.state_dict())
    inputs = _full_sequence_multistep_inputs()
    inputs.update(
        {
            "input_ids": torch.tensor([[1, 4, 5, 2, 6, 7], [3, 4, 6, 1, 5, 2]]),
            "document_ids": torch.tensor([[0] * 6, [1] * 6]),
            "semantic_validity": torch.ones((2, 6), dtype=torch.bool),
            "position_ids": torch.arange(6)[None].expand(2, -1),
            "canvas_corruptible_mask": torch.tensor(
                [
                    [False, True, False, False, False, True],
                    [False, False, True, False, False, False],
                ]
            ),
            "canvas_input_pinned_mask": torch.tensor(
                [[False, False, False, False, False, True], [False] * 6]
            ),
            "canvas_update_mask": torch.tensor(
                [
                    [False, True, False, False, False, True],
                    [False, False, True, False, False, False],
                ]
            ),
            "decision_examples": (_example(2), _example(1)),
            "decision_question_mask": torch.tensor([[True, True], [True, False]]),
            "decision_supervision_mask": torch.tensor([[True, True], [True, False]]),
            "decision_label_rows": torch.tensor([[0, 0], [1, -1]]),
            "decision_label_positions": torch.tensor([[1, 5], [2, -1]]),
        }
    )

    torch.manual_seed(19)
    dense_loss = dense_trainer.compute_loss(dense_model, inputs)
    dense_loss.backward()
    dense_grad = dense_model.logit_bias.grad.detach().clone()
    torch.manual_seed(19)
    selected_loss, selected_outputs = selected_trainer.compute_loss(
        selected_model, inputs, return_outputs=True
    )
    selected_loss.backward()

    assert selected_outputs.axolotl_selected_logits is True
    assert selected_model.selected_shapes == [torch.Size([2, 2, 11])]
    torch.testing.assert_close(selected_loss, dense_loss)
    torch.testing.assert_close(selected_model.logit_bias.grad, dense_grad)
    assert all("axolotl_selected_logits" not in call for call in dense_model.calls)


@pytest.mark.parametrize("steps", [2, 3])
def test_packed_multistep_noise_is_per_document_and_holds_each_canvas_state(steps):
    trainer = _MultistepTrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=steps,
        sampled_steps=steps,
    )
    trainer._decision_times = lambda count, device, default_eps, decision: torch.tensor(
        [0.0, 1.0], device=device
    )
    inputs = _full_sequence_multistep_inputs()
    inputs["canvas_input_pinned_mask"] = torch.zeros((1, 6), dtype=torch.bool)
    model = _TraceNativeModel()

    trainer._full_sequence_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )

    expected_state = torch.tensor([[1, 4, 5, 2, 6, 10]])
    assert len(model.forward_states) == steps
    assert all(torch.equal(state, expected_state) for state in model.forward_states)


def test_full_sequence_multistep_config_sampled_to_one_never_uses_native_commit():
    trainer = _NoCommitMultistepTrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=1,
    )
    model = _TraceNativeModel()

    trainer._full_sequence_logits(
        model,
        _full_sequence_multistep_inputs(),
        trainer._native_spec,
        trainer._decision_config(),
    )

    assert len(model.forward_states) == 1
    torch.testing.assert_close(
        model.forward_states[0], torch.tensor([[1, 10, 5, 2, 6, 7]])
    )


def _canvas(prompt, canvas, labels, identifier: str) -> DecisionCanvas:
    return DecisionCanvas(
        prompt_ids=tuple(prompt),
        canvas_ids=tuple(canvas),
        label_positions=tuple(labels),
        allowed_ids=tuple((1, 2, 3) for _ in labels),
        question_ids=tuple(f"{identifier}-{index}" for index in range(len(labels))),
        targets=tuple({"kind": "hard", "gold_idx": 0} for _ in labels),
        pinned_mask=(False,) * len(canvas),
        semantic_mask=(True,) * len(canvas),
        slot_mask=(False,) * len(canvas),
        template_length=len(canvas),
    )


def test_global_example_scaling_matches_ddp_average_for_unequal_windows():
    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED), world_size=2
    )
    result = DecisionLossResult(
        loss=torch.tensor(2.0),
        restricted_loss=torch.tensor(0.0),
        full_vocab_loss=torch.tensor(0.0),
        brier_loss=torch.tensor(0.0),
    )

    scaled = trainer._scale_global_example_mean(
        result, local_examples=3, global_examples=8
    )

    torch.testing.assert_close(scaled, torch.tensor(1.5))


def test_post_config_prevents_training_step_from_dividing_global_loss_again(
    monkeypatch,
):
    trainer = _TrainingStepHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    monkeypatch.setattr(
        AxolotlDiffusionTrainer,
        "post_set_axolotl_cfg",
        lambda self: setattr(self, "model_accepts_loss_kwargs", False),
    )
    trainer.post_set_axolotl_cfg()
    assert trainer.model_accepts_loss_kwargs

    class _Accelerator:
        distributed_type = "NO"

        def backward(self, loss, **kwargs):
            del kwargs
            self.loss = loss
            loss.backward()

    trainer.accelerator = _Accelerator()
    trainer.args = SimpleNamespace(
        world_size=1,
        n_gpu=1,
        optim="adamw_torch",
        torch_empty_cache_steps=None,
    )
    trainer.current_gradient_accumulation_steps = 4
    trainer.state = SimpleNamespace(global_step=1)
    trainer.optimizer = object()
    trainer.compute_loss_func = None
    trainer._layer_offload_ctx = contextlib.nullcontext()
    trainer.activation_offload_context = contextlib.nullcontext()
    model = _TinyNativeModel()
    trainer.model = model

    reported = trainer.training_step(
        model, {}, num_items_in_batch=torch.tensor(2, dtype=torch.long)
    )

    torch.testing.assert_close(reported, model.logit_bias.detach().sum())
    torch.testing.assert_close(trainer.accelerator.loss, model.logit_bias.sum())
    torch.testing.assert_close(model.logit_bias.grad, torch.ones_like(model.logit_bias))


def test_stratified_sampler_shuffles_whole_microbatches_per_epoch():
    sampler = _StratifiedDecisionBatchSampler(dataset_size=12, batch_size=3, seed=17)

    first = list(sampler)
    sampler.set_epoch(0)
    repeated = list(sampler)
    sampler.set_epoch(1)
    next_epoch = list(sampler)

    assert first == repeated
    assert next_epoch != first
    assert sorted(index for batch in next_epoch for index in batch) == list(range(12))
    assert all(batch == list(range(batch[0], batch[0] + 3)) for batch in next_epoch)


def test_stratified_sampler_pads_whole_batches_for_even_distributed_steps():
    sampler = _StratifiedDecisionBatchSampler(
        dataset_size=2, batch_size=2, seed=17, world_size=8
    )
    batches = list(sampler)

    assert len(batches) == 8
    assert all(batch == list(range(batch[0], batch[0] + 2)) for batch in batches)
    assert set(index for batch in batches for index in batch) == {0, 1}


def test_stratified_batch_sampler_is_accepted_by_a_real_dataloader():
    class _Dataset:
        def __len__(self):
            return 6

        def __getitem__(self, index):
            return index

    sampler = _StratifiedDecisionBatchSampler(
        dataset_size=6, batch_size=2, seed=17, world_size=2
    )

    batches = list(DataLoader(_Dataset(), batch_sampler=sampler))

    assert isinstance(sampler, BatchSampler)
    assert len(batches) == 4
    assert all(batch.shape == (2,) for batch in batches)


def test_trainer_uses_core_multipack_for_typed_packing(monkeypatch):
    class _Dataset:
        manifest = {
            "per_batch_stratified": False,
        }

        def __len__(self):
            return 6

    dataset = _Dataset()
    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    trainer.train_dataset = dataset
    trainer.args.per_device_train_batch_size = 2
    trainer.args.sample_packing = True
    trainer.accelerator = SimpleNamespace(even_batches=True)

    observed = []

    def _base_sampler(_self, received):
        observed.append(received)
        return "multipack"

    monkeypatch.setattr(AxolotlDiffusionTrainer, "_get_train_sampler", _base_sampler)
    assert trainer._get_train_sampler() == "multipack"
    assert observed == [dataset]


def test_trainer_rejects_packing_that_would_bypass_stratified_quotas():
    class _Dataset:
        manifest = {"per_batch_stratified": True}

    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    trainer.train_dataset = _Dataset()
    trainer.args.sample_packing = True

    with pytest.raises(ValueError, match="per_batch_stratified: false"):
        trainer._get_train_sampler()


def test_decision_dataset_exposes_canvas_lengths_to_multipack():
    from axolotl.integrations.decision.datasets import DecisionDataset
    from axolotl.utils.samplers import get_dataset_lengths

    dataset = DecisionDataset(
        [{"length": 5}, {"length": 9}], {"per_batch_stratified": False}
    )
    assert get_dataset_lengths(dataset).tolist() == [5, 9]
    assert dataset[[1, 0]] == [{"length": 9}, {"length": 5}]


@pytest.mark.parametrize(
    ("is_training", "eval_packing"), [(True, True), (False, True), (False, False)]
)
def test_core_dataloader_removes_only_a_lengthless_decision_dataset_copy(
    is_training, eval_packing
):
    class _CoreDataLoaderHarness(_TrainerHarness):
        def _get_collator_with_removed_columns(self, collator, description):
            del description
            return collator

    dataset = DecisionDataset(
        [{"length": 5}, {"length": 7}], {"per_batch_stratified": False}
    )
    trainer = _CoreDataLoaderHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    trainer.data_collator = lambda rows: rows
    trainer.eval_data_collator = trainer.data_collator
    trainer.args = SimpleNamespace(
        sample_packing=True,
        eval_sample_packing=eval_packing,
        dataloader_drop_last=None,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=None,
        process_index=0,
        pretraining=False,
        sample_packing_drop_attention_mask=False,
    )
    trainer.accelerator = SimpleNamespace(
        even_batches=True, prepare=lambda dataloader: dataloader
    )
    sampler = BatchSampler(SequentialSampler(dataset), batch_size=1, drop_last=False)

    dataloader = trainer._get_dataloader(
        dataset,
        "training" if is_training else "evaluation",
        1,
        sampler_fn=lambda _dataset: sampler,
        is_training=is_training,
    )

    assert isinstance(dataloader, DataLoader)
    assert dataloader.dataset.column_names == ()
    assert dataset.column_names == ("length",)
    assert dataset["length"] == [5, 7]
    assert trainer.args.sample_packing is True


def test_train_dataloader_accepts_typed_dataset_and_restores_even_batches(
    monkeypatch,
):
    class _Dataset:
        manifest = {"per_batch_stratified": True}

    dataset = _Dataset()
    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    trainer.accelerator = SimpleNamespace(even_batches=False)
    trainer.args.sample_packing = True
    trainer.args.dataloader_drop_last = None
    observed_packing = []

    def _base_loader(_self, received, *args, **kwargs):
        del received, args, kwargs
        observed_packing.append(_self.args.sample_packing)
        return dataset

    monkeypatch.setattr(
        AxolotlDiffusionTrainer,
        "_get_dataloader",
        _base_loader,
    )

    result = trainer._get_dataloader(
        dataset,
        "training",
        2,
        is_training=True,
    )

    assert result is dataset
    assert dataset.column_names == ()
    assert trainer.accelerator.even_batches is True
    assert observed_packing == [True]
    assert trainer.args.sample_packing is True


def test_eval_sample_packing_false_keeps_typed_eval_unpacked(monkeypatch):
    class _Dataset:
        manifest = {}

        def __len__(self):
            return 9

        def __getitem__(self, index):
            return index

    dataset = _Dataset()
    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    trainer.eval_dataset = dataset
    trainer.accelerator = SimpleNamespace(even_batches=True)
    trainer.args.sample_packing = True
    trainer.args.eval_sample_packing = False
    trainer.args.dataloader_drop_last = None
    observed: list[tuple[bool, bool | None]] = []

    def _base_loader(_self, received, *args, **kwargs):
        del received, args, kwargs
        observed.append((_self.args.sample_packing, _self.args.dataloader_drop_last))
        return dataset

    monkeypatch.setattr(AxolotlDiffusionTrainer, "_get_dataloader", _base_loader)

    result = trainer._get_dataloader(dataset, "evaluation", 8, is_training=False)
    assert result is dataset
    assert observed == [(False, False)]
    assert trainer.args.sample_packing is True
    assert trainer.args.dataloader_drop_last is None


def test_prediction_step_uses_typed_loss_without_retaining_full_logits():
    trainer = _PredictionHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    model = _TinyNativeModel()

    loss, logits, labels = trainer.prediction_step(
        model, {"decision_examples": (_example(1),)}, prediction_loss_only=False
    )

    torch.testing.assert_close(loss, model.logit_bias.detach().square().sum())
    assert trainer.prepared
    assert trainer.seen_inputs == {"prepared": {"decision_examples": (_example(1),)}}
    assert logits is None
    assert labels is None
    assert model.logit_bias.grad is None


def test_eval_metric_keys_are_prefixed_and_distributed_totals_merge(monkeypatch):
    local = {
        "decision/alpha": {
            "loss": 2.0,
            "restricted_loss": 1.0,
            "full_vocab_loss": 1.0,
            "brier_loss": 0.0,
            "examples": 2.0,
        }
    }

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def _gather(output, value):
        output[:] = [
            value,
            {
                "decision/beta": {
                    "loss": 3.0,
                    "restricted_loss": 2.0,
                    "full_vocab_loss": 1.0,
                    "brier_loss": 0.5,
                    "examples": 1.0,
                }
            },
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _gather)
    merged = _reduce_decision_metric_totals(local)
    assert merged["decision/alpha"]["examples"] == 2
    assert merged["decision/beta"]["brier_loss"] == 0.5

    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    trainer._decision_metrics = {"train": {}, "eval": local}
    captured: dict[str, float] = {}
    monkeypatch.setattr(
        AxolotlDiffusionTrainer,
        "log",
        lambda _self, logs, start_time=None: captured.update(logs),
    )
    trainer.log({"eval_loss": 1.0, "train_loss": 4.0})

    assert captured["eval_decision/alpha/loss"] == 1.0
    assert captured["eval_decision/beta/loss"] == 3.0
    assert "decision/alpha/loss" not in captured


def test_base_reports_token_perplexity_but_decision_objective_does_not(monkeypatch):
    forwarded: list[dict[str, float]] = []
    monkeypatch.setattr(trainer_base, "is_main_process", lambda: False)
    monkeypatch.setattr(
        trainer_base.Trainer,
        "log",
        lambda _self, logs, start_time=None: forwarded.append(dict(logs)),
    )

    legacy = object.__new__(AxolotlDiffusionTrainer)
    legacy._stored_metrics = {"train": {}, "eval": {}}
    legacy.args = SimpleNamespace(include_tkps=False)
    AxolotlTrainer.log(legacy, {"loss": 2.0})

    decision = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    decision._stored_metrics = {"train": {}, "eval": {}}
    decision.args.include_tkps = False
    AxolotlTrainer.log(decision, {"eval_loss": 2.0})

    assert forwarded[0]["ppl"] == pytest.approx(torch.exp(torch.tensor(2.0)).item())
    assert "eval_ppl" not in forwarded[1]


def test_get_batch_samples_counts_logical_examples_and_handles_last_short_window():
    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    batches, count = trainer.get_batch_samples(
        iter(
            (
                {"decision_examples": (_example(1), _example(2))},
                {"decision_examples": (_example(1),)},
            )
        ),
        num_batches=4,
        device=torch.device("cpu"),
    )

    assert len(batches) == 2
    assert count.item() == 3


class _LoopDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _typed_loop_example(index: int) -> DecisionLabelExample:
    questions = [
        DecisionLabelQuestion(
            position=0,
            allowed_token_ids=(1, 3, 5),
            target=HardLabel(index % 3),
        )
    ]
    if index % 2:
        questions.append(
            DecisionLabelQuestion(
                position=1,
                allowed_token_ids=(2, 4, 6),
                target=DistributionLabel((0.2, 0.3, 0.5)),
            )
        )
    if index % 3 == 0:
        questions.append(
            DecisionLabelQuestion(
                position=2,
                allowed_token_ids=(3, 7, 9),
                target=SetLabel((0, 2)),
            )
        )
    return DecisionLabelExample(tuple(questions), source_weight=1.0 + (index % 4) / 4)


def _loop_rows(count: int):
    rows = []
    for index in range(count):
        example = _typed_loop_example(index)
        canvas = _canvas(
            (1, 2 + (index % 3)),
            (4, 5, 6, 7, 8),
            tuple(range(len(example.questions))),
            f"loop-{index}",
        )
        rows.append(
            {
                "canvas": canvas,
                "decision_example": example,
                "source": "alpha" if index % 2 else "beta",
            }
        )
    return rows


def _manual_typed_objective(bias: torch.Tensor, examples):
    """Independent typed-label objective for the fixed-bias loop model."""
    per_example = []
    for example in examples:
        per_question = []
        for question in example.questions:
            allowed = torch.tensor(question.allowed_token_ids, device=bias.device)
            restricted_logprobs = torch.log_softmax(
                bias.index_select(0, allowed), dim=0
            )
            restricted_probs = restricted_logprobs.exp()
            target = question.target
            if isinstance(target, HardLabel):
                restricted = -restricted_logprobs[target.gold_index]
                desired = torch.nn.functional.one_hot(
                    torch.tensor(target.gold_index, device=bias.device),
                    num_classes=len(question.allowed_token_ids),
                ).to(bias.dtype)
                full_target = allowed[target.gold_index]
            elif isinstance(target, DistributionLabel):
                desired = torch.tensor(target.probabilities, device=bias.device)
                restricted = torch.where(
                    desired > 0,
                    desired * (desired.log() - restricted_logprobs),
                    torch.zeros_like(desired),
                ).sum()
                full_target = allowed[desired.argmax()]
            else:
                assert isinstance(target, SetLabel)
                members = torch.tensor(target.allowed_indices, device=bias.device)
                member_logprobs = restricted_logprobs.index_select(0, members)
                restricted = -torch.logsumexp(member_logprobs, dim=0)
                desired = None
                full_target = allowed[members[member_logprobs.detach().argmax()]]
            full_logprobs = torch.log_softmax(bias, dim=0)
            full = (
                -full_logprobs[full_target]
                if isinstance(target, (HardLabel, SetLabel))
                else -(desired * full_logprobs.index_select(0, allowed)).sum()
            )
            if desired is None:
                brier = (1.0 - restricted_probs.index_select(0, members).sum()).square()
            else:
                brier = (restricted_probs - desired).square().sum()
            per_question.append(restricted + full + 0.1 * brier)
        per_example.append(torch.stack(per_question).mean() * example.source_weight)
    return torch.stack(per_example).mean()


def _run_actual_decision_loop(
    tmp_path, *, microbatch_size, accumulation_steps, row_count=67, max_steps=2
):
    rows = _loop_rows(row_count)
    spec = _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    model = _LoopNativeModel()
    args = AxolotlTrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=microbatch_size,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=0.05,
        max_steps=max_steps,
        lr_scheduler_type="constant",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        seed=0,
        data_seed=0,
        optim="sgd",
    )
    trainer = _LoopTrainer(
        model=model,
        args=args,
        train_dataset=_LoopDataset(rows),
        data_collator=DecisionTrainingCollator(spec),
        spec=spec,
    )
    trainer.axolotl_cfg = {"decision": {"read_fraction": 1.0}}
    trainer.post_set_axolotl_cfg()
    trainer.train()
    return model, rows


def test_actual_trainer_loop_matches_manual_typed_labels_under_gradient_accumulation(
    tmp_path,
):
    """The Trainer loop uses logical examples across full and short windows."""
    torch.manual_seed(0)
    accumulated, rows = _run_actual_decision_loop(
        tmp_path / "microbatch", microbatch_size=8, accumulation_steps=8
    )
    torch.manual_seed(0)
    full, _ = _run_actual_decision_loop(
        tmp_path / "full", microbatch_size=64, accumulation_steps=1
    )

    expected = torch.linspace(-0.4, 0.6, 11).requires_grad_(True)
    for window in (rows[:64], rows[64:]):
        objective = _manual_typed_objective(
            expected, [row["decision_example"] for row in window]
        )
        (gradient,) = torch.autograd.grad(objective, expected)
        expected = (expected - 0.05 * gradient).detach().requires_grad_(True)

    torch.testing.assert_close(
        accumulated.logit_bias, expected.detach(), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(full.logit_bias, expected.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        accumulated.logit_bias, full.logit_bias, rtol=1e-5, atol=1e-6
    )
    assert len(accumulated.observed_input_ids) == 9
    assert len(full.observed_input_ids) == 2
    assert all(torch.any(ids.eq(10)) for ids in accumulated.observed_input_ids)
    assert all(torch.any(ids.eq(10)) for ids in full.observed_input_ids)


def test_actual_trainer_final_six_microbatch_window_matches_one_48_example_update(
    tmp_path,
):
    """A six-microbatch terminal window is one 48-example mean, not a 64-example mean."""
    torch.manual_seed(0)
    accumulated, rows = _run_actual_decision_loop(
        tmp_path / "six-microbatches",
        microbatch_size=8,
        accumulation_steps=8,
        row_count=48,
        max_steps=1,
    )
    torch.manual_seed(0)
    full, _ = _run_actual_decision_loop(
        tmp_path / "one-batch",
        microbatch_size=48,
        accumulation_steps=1,
        row_count=48,
        max_steps=1,
    )

    expected = torch.linspace(-0.4, 0.6, 11).requires_grad_(True)
    objective = _manual_typed_objective(
        expected, [row["decision_example"] for row in rows]
    )
    (gradient,) = torch.autograd.grad(objective, expected)
    expected = (expected - 0.05 * gradient).detach()

    torch.testing.assert_close(accumulated.logit_bias, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full.logit_bias, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        accumulated.logit_bias, full.logit_bias, rtol=1e-5, atol=1e-6
    )
    assert len(accumulated.observed_input_ids) == 6
    assert len(full.observed_input_ids) == 1


@pytest.mark.parametrize(
    "decision",
    [
        {"latent": {"mode": "free", "num_slots": 1}},
        {"carry": {"enabled": True}},
    ],
)
def test_unwired_decision_controls_fail_explicitly(decision):
    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED), decision
    )

    with pytest.raises((NotImplementedError, ValueError)):
        trainer._decision_config()


def test_fractional_read_times_preserve_endpoint_rng_and_select_logical_examples():
    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        {"read_fraction": 0.4},
    )
    native = torch.full((5,), 0.5)

    torch.manual_seed(19)
    before = torch.get_rng_state()
    zero = trainer._decision_times(
        5,
        torch.device("cpu"),
        0.0,
        trainer._decision_config().model_copy(update={"read_fraction": 0.0}),
    )
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(zero, native)

    torch.manual_seed(19)
    before = torch.get_rng_state()
    one = trainer._decision_times(
        5,
        torch.device("cpu"),
        0.0,
        trainer._decision_config().model_copy(update={"read_fraction": 1.0}),
    )
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(one, torch.ones(5))

    torch.manual_seed(19)
    expected_reads = torch.rand(5) < 0.4
    torch.manual_seed(19)
    fractional = trainer._decision_times(
        5, torch.device("cpu"), 0.0, trainer._decision_config()
    )
    torch.testing.assert_close(
        fractional, torch.where(expected_reads, torch.ones(5), native)
    )
    assert torch.equal(fractional.eq(1), expected_reads)


def test_fractional_read_times_preserve_native_sampler_rng_contract():
    trainer = _TrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        native_values={"t_eps": 0.2},
    )
    trainer._sample_native_times = AxolotlDiffusionTrainer._sample_native_times.__get__(
        trainer, type(trainer)
    )
    count = 4
    device = torch.device("cpu")
    base = trainer._decision_config()

    torch.manual_seed(31)
    expected_native = torch.rand(count) * 0.8 + 0.2
    native_state = torch.get_rng_state()
    torch.manual_seed(31)
    one = trainer._decision_times(
        count, device, 0.0, base.model_copy(update={"read_fraction": 1.0})
    )
    torch.testing.assert_close(one, torch.ones(count))
    assert torch.equal(torch.get_rng_state(), native_state)

    torch.manual_seed(31)
    zero = trainer._decision_times(
        count, device, 0.0, base.model_copy(update={"read_fraction": 0.0})
    )
    torch.testing.assert_close(zero, expected_native)
    assert torch.equal(torch.get_rng_state(), native_state)

    torch.manual_seed(31)
    expected_native = torch.rand(count) * 0.8 + 0.2
    expected_reads = torch.rand(count) < 0.5
    expected_state = torch.get_rng_state()
    torch.manual_seed(31)
    fractional = trainer._decision_times(
        count, device, 0.0, base.model_copy(update={"read_fraction": 0.5})
    )
    torch.testing.assert_close(
        fractional,
        torch.where(expected_reads, torch.ones_like(expected_native), expected_native),
    )
    assert torch.equal(torch.get_rng_state(), expected_state)


def test_decision_multistep_rejects_grad_through_steps():
    trainer = _MultistepTrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=2,
        sampled_steps=2,
        grad_through_steps=True,
    )

    with pytest.raises(NotImplementedError, match="grad-through-steps"):
        trainer.compute_loss(_TinyNativeModel(), _full_sequence_multistep_inputs())


def test_full_sequence_multistep_rejects_unsupported_self_conditioning():
    spec = replace(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        self_conditioning=True,
    )
    trainer = _MultistepTrainerHarness(spec, k_max=2, sampled_steps=2)

    with pytest.raises(NotImplementedError, match="self-conditioning"):
        trainer.compute_loss(_TinyNativeModel(), _full_sequence_multistep_inputs())


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("train_only_eval", [False, True])
def test_decision_cce_final_read_matches_dense_loss_and_gradients(
    monkeypatch, steps, train_only_eval
):
    from axolotl.model_support.nemotron_diffusion import cut_cross_entropy as cce

    class ProjectionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(11, 5)
            self.diffusion_head = nn.Linear(5, 11, bias=False)
            self.config = SimpleNamespace(model_type="nemotron_labs_diffusion")
            self.reads = []

        def forward(self, input_ids, cce_return_hidden_states=False, **kwargs):
            del kwargs
            hidden = self.embedding(input_ids)
            self.reads.append((cce_return_hidden_states, torch.is_grad_enabled()))
            if cce_return_hidden_states:
                return SimpleNamespace(last_hidden_state=hidden, logits=None)
            return SimpleNamespace(logits=self.diffusion_head(hidden))

    torch.manual_seed(5)
    dense_model, cce_model = ProjectionModel(), ProjectionModel()
    cce_model.load_state_dict(dense_model.state_dict())
    if train_only_eval:
        dense_model.eval()
        cce_model.eval()
    options = SimpleNamespace(train_only=train_only_eval)
    monkeypatch.setattr(cce, "get_cce_options", lambda model: options)
    monkeypatch.setattr(cce, "get_cce_head", lambda model: model.diffusion_head)
    calls = []

    def reference_loss(hidden, head, targets, options):
        assert options.train_only == train_only_eval
        calls.append(targets.detach().clone())
        return torch.nn.functional.cross_entropy(
            head(hidden).float(), targets, reduction="none"
        )

    monkeypatch.setattr(cce, "linear_token_loss", reference_loss)
    losses, trainers = [], []
    for enabled, model in ((False, dense_model), (True, cce_model)):
        trainer = _MultistepTrainerHarness(
            _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
            k_max=steps,
            sampled_steps=steps,
        )
        trainer.axolotl_cfg = SimpleNamespace(decision={}, cut_cross_entropy=enabled)
        inputs = _full_sequence_multistep_inputs()
        inputs["decision_sources"] = ["source"]
        loss = trainer.compute_loss(model, inputs)
        loss.backward()
        losses.append(loss)
        trainers.append(trainer)
    torch.testing.assert_close(losses[0], losses[1])
    for left, right in zip(
        dense_model.parameters(), cce_model.parameters(), strict=True
    ):
        torch.testing.assert_close(left.grad, right.grad)
    assert all(not hidden for hidden, _ in cce_model.reads[:-1])
    assert cce_model.reads[-1][0] is (not train_only_eval)
    assert bool(calls) is (not train_only_eval)
    for phase, groups in trainers[0]._decision_metrics.items():
        assert groups.keys() == trainers[1]._decision_metrics[phase].keys()
        for source, values in groups.items():
            assert trainers[1]._decision_metrics[phase][source] == pytest.approx(
                values, rel=1e-6, abs=1e-7
            )


@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("mode", ["full", "selected", "hidden"])
def test_multimodal_inputs_reach_each_decision_forward(steps, mode):
    class ImageModel(_SelectedTinyNativeModel):
        supports_selected_logits = mode == "selected"

        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            if kwargs.get("cce_return_hidden_states"):
                output.last_hidden_state = output.logits
            return output

    trainer = _MultistepTrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=steps,
        sampled_steps=steps,
    )
    model = ImageModel()
    inputs = _full_sequence_multistep_inputs()
    media = {
        "pixel_values": torch.ones(2, 3, 2, 2),
        "image_sizes": torch.tensor([[2, 2], [2, 2]]),
    }
    inputs["model_inputs"] = media
    logits, _ = trainer._full_sequence_logits(
        model,
        inputs,
        trainer._native_spec,
        trainer._decision_config(),
        return_hidden_states=mode == "hidden",
    )
    logits.sum().backward()
    assert model.calls
    for call in model.calls:
        torch.testing.assert_close(call["pixel_values"], media["pixel_values"])
        torch.testing.assert_close(call["image_sizes"], media["image_sizes"])
    assert model.logit_bias.grad is not None
