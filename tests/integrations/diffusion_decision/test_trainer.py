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
from axolotl.core.trainers.diffusion_lm.backends.encoder_canvas import (
    EncoderCanvasBackend,
)
from axolotl.core.trainers.diffusion_lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer
from axolotl.core.training_args import AxolotlTrainingArguments
from axolotl.integrations.diffusion_decision.datasets import DecisionDataset
from axolotl.integrations.diffusion_decision.loss import (
    DecisionLabelExample,
    DecisionLabelQuestion,
    DecisionLossResult,
    DistributionLabel,
    HardLabel,
    SetLabel,
    decision_example_from_canvas,
    decision_label_loss,
)
from axolotl.integrations.diffusion_decision.records import DecisionCanvas
from axolotl.integrations.diffusion_decision.slot_sampling import DecisionDraw
from axolotl.integrations.diffusion_decision.slots import SlotPlan
from axolotl.integrations.diffusion_decision.trainer import (
    DiffusionDecisionTrainer,
    _DecisionDrawBatchSampler,
    _reduce_decision_metric_totals,
    _StratifiedDecisionBatchSampler,
)
from axolotl.integrations.diffusion_decision.training_collator import (
    DecisionTrainingCollator,
)
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    LogitAlignment,
    MaskTokenPolicy,
)

from tests.integrations.diffusion_decision.helpers import (
    MultistepTrainerHarness,
    TrainerHarness,
    make_canvas,
    tiny_gemma_config,
    trainer_spec,
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


class _NoCommitMultistepTrainerHarness(MultistepTrainerHarness):
    def _run_native_unroll(self, *args, **kwargs):
        del args, kwargs
        pytest.fail("K>1 decision configuration must not use committing native unroll")


class _TrainingStepHarness(TrainerHarness):
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


class _LoopTrainer(DiffusionDecisionTrainer):
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


class _SampledLoopTrainer(_LoopTrainer):
    def _get_train_sampler(self, train_dataset=None):
        return DiffusionDecisionTrainer._get_train_sampler(self, train_dataset)


class _PredictionHarness(TrainerHarness):
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
    trainer = TrainerHarness(trainer_spec(DiffusionLayout.FULL_SEQUENCE, alignment))
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
    dense_trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=2,
        sampled_steps=steps,
        decision=decision,
    )
    selected_trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
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
def test_full_sequence_multistep_holds_noisy_labels_and_pinned_slots(steps):
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=steps,
        decision={"latent": {"mode": "pinned", "num_slots": 1}},
    )
    model = _TraceNativeModel()
    inputs = _full_sequence_multistep_inputs()

    logits, _ = trainer._full_sequence_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )

    expected_state = torch.tensor([[1, 10, 5, 2, 6, 7]])
    assert len(model.forward_states) == steps
    assert all(torch.equal(state, expected_state) for state in model.forward_states)
    assert model.forward_grad_enabled == [False] * (steps - 1) + [True]
    logits.sum().backward()
    assert model.logit_bias.grad is not None
    assert torch.count_nonzero(model.logit_bias.grad)
    assert torch.equal(inputs["input_ids"], torch.tensor([[1, 4, 5, 2, 6, 7]]))


def test_packed_multistep_noise_is_per_document_and_holds_each_canvas_state():
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=2,
        sampled_steps=2,
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
    assert len(model.forward_states) == 2
    assert all(torch.equal(state, expected_state) for state in model.forward_states)


def test_full_sequence_multistep_config_sampled_to_one_never_uses_native_commit():
    trainer = _NoCommitMultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
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


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_native_nemotron_peft_learned_slot_rows_receive_final_step_gradients(
    steps, device
):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

    from tests.native_source_fixtures import native_source_fixture_path

    source = native_source_fixture_path("nemotron")
    if source is None:
        pytest.skip("native Nemotron source fixture unavailable")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    config._attn_implementation = "eager"
    base = resolve_nemotron_model_class(str(source))(config).to(device)
    calls: list[bool] = []
    original_forward = base.forward

    def traced_forward(*args, **kwargs):
        calls.append(torch.is_grad_enabled())
        return original_forward(*args, **kwargs)

    base.forward = traced_forward
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj"],
            r=2,
            lora_alpha=2,
            trainable_token_indices=[120],
        ),
    )
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=steps,
        decision={"latent": {"mode": "learned", "num_slots": 1}},
    )
    inputs = {
        "input_ids": torch.tensor([[1, 120, 4, 5, 2]], device=device),
        "document_ids": torch.zeros((1, 5), dtype=torch.long, device=device),
        "semantic_validity": torch.ones((1, 5), dtype=torch.bool, device=device),
        "position_ids": torch.arange(5, device=device)[None],
        "canvas_corruptible_mask": torch.tensor(
            [[False, False, True, False, False]], device=device
        ),
        "canvas_input_pinned_mask": torch.tensor(
            [[False, True, False, False, False]], device=device
        ),
        "canvas_update_mask": torch.tensor(
            [[False, True, True, False, False]], device=device
        ),
        "decision_examples": (_example(1),),
        "decision_question_mask": torch.tensor([[True]], device=device),
        "decision_supervision_mask": torch.tensor([[True]], device=device),
        "decision_label_rows": torch.tensor([[0]], dtype=torch.long, device=device),
        "decision_label_positions": torch.tensor(
            [[2]], dtype=torch.long, device=device
        ),
    }

    loss = trainer.compute_loss(model, inputs)
    loss.backward()

    assert calls == [False] * (steps - 1) + [True]
    lora_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if ".lora_" in name
    ]
    token_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "trainable_tokens_delta" in name
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient)
        for gradient in lora_grads
    )
    assert any(
        gradient is not None and torch.count_nonzero(gradient)
        for gradient in token_grads
    )


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_native_gemma_peft_multistep_holds_canvas_and_trains_final_read(
    steps, device, monkeypatch
):
    from peft import LoraConfig, get_peft_model

    import axolotl.model_support.diffusion_gemma.modeling as gemma_modeling
    from axolotl.model_support.diffusion_gemma.modeling import (
        AxolotlDiffusionGemmaForBlockDiffusion,
    )

    torch.manual_seed(17)
    base = AxolotlDiffusionGemmaForBlockDiffusion(tiny_gemma_config()).to(device)
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=(
                r"^(model\.encoder\.language_model\.layers|model\.decoder\.layers)"
                r"\.0\.self_attn\.q_proj$"
            ),
            task_type=None,
            trainable_token_indices=[25],
        ),
    ).train()
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=steps,
        decision={"latent": {"mode": "learned", "num_slots": 1}},
    )
    canvas = DecisionCanvas(
        prompt_ids=(2, 3),
        canvas_ids=(25, 6, 7, 8, 0, 0, 0, 0),
        label_positions=(1, 3),
        allowed_ids=((1, 2, 3), (4, 5, 6)),
        question_ids=("q0", "q1"),
        targets=({"kind": "hard", "gold_idx": 0},) * 2,
        pinned_mask=(True, False, False, False, True, True, True, True),
        semantic_mask=(True,) * 8,
        slot_mask=(True, False, False, False, False, False, False, False),
        template_length=4,
    )
    inputs = DecisionTrainingCollator(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )([{"canvas": canvas, "source": "native"}])
    inputs["diffusion_batch"] = inputs["diffusion_batch"].to(device)
    for name, value in tuple(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[name] = value.to(device)

    observed: list[
        tuple[torch.Tensor, bool, torch.Tensor | None, torch.Tensor | None]
    ] = []
    original_decode = gemma_modeling.decode_packed_canvas

    def traced_decode(model, decoder_input_ids, *args, **kwargs):
        conditioning = args[3] if len(args) >= 4 else kwargs.get("logits")
        token_gate = args[4] if len(args) >= 5 else kwargs.get("token_gate")
        observed.append(
            (
                decoder_input_ids.detach().clone(),
                torch.is_grad_enabled(),
                conditioning,
                token_gate,
            )
        )
        return original_decode(model, decoder_input_ids, *args, **kwargs)

    monkeypatch.setattr(gemma_modeling, "decode_packed_canvas", traced_decode)
    torch.manual_seed(29)
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert outputs.denoised_input_ids is not None
    assert len(observed) == steps
    assert all(
        torch.equal(state, outputs.denoised_input_ids) for state, _, _, _ in observed
    )
    assert [enabled for _, enabled, _, _ in observed] == [False] * (steps - 1) + [True]
    assert observed[0][2] is None
    assert all(conditioning is not None for _, _, conditioning, _ in observed[1:])
    assert all(gate is not None for _, _, _, gate in observed)
    assert all(gate[0, 0] for _, _, _, gate in observed if gate is not None)
    assert outputs.denoised_input_ids[0, 0].item() == 25

    loss.backward()
    lora_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if ".lora_" in name
    ]
    token_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "trainable_tokens_delta" in name
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient)
        for gradient in lora_grads
    )
    assert any(
        gradient is not None and torch.count_nonzero(gradient)
        for gradient in token_grads
    )


@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_native_gemma_peft_free_slots_evolve_without_supervising_slots(
    steps, device, monkeypatch
):
    from peft import LoraConfig, get_peft_model

    import axolotl.model_support.diffusion_gemma.modeling as gemma_modeling
    from axolotl.model_support.diffusion_gemma.modeling import (
        AxolotlDiffusionGemmaForBlockDiffusion,
    )

    torch.manual_seed(17)
    base = AxolotlDiffusionGemmaForBlockDiffusion(tiny_gemma_config())
    base.config.mask_token_id = 2
    model = (
        get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=(
                    r"^(model\.encoder\.language_model\.layers|model\.decoder\.layers)"
                    r"\.0\.self_attn\.q_proj$"
                ),
                task_type=None,
            ),
        )
        .to(device)
        .train()
    )
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=steps,
        decision={
            "latent": {
                "mode": "free",
                "num_slots": 1,
                "free_update_policy": "argmax",
            }
        },
    )
    canvas = DecisionCanvas(
        prompt_ids=(2, 3),
        canvas_ids=(25, 6, 7, 8, 0, 0, 0, 0),
        label_positions=(1, 3),
        allowed_ids=((1, 2, 3), (4, 5, 6)),
        question_ids=("q0", "q1"),
        targets=({"kind": "hard", "gold_idx": 0},) * 2,
        pinned_mask=(False, False, False, False, True, True, True, True),
        semantic_mask=(True,) * 8,
        slot_mask=(True, False, False, False, False, False, False, False),
        template_length=4,
    )
    inputs = DecisionTrainingCollator(trainer._native_spec)(
        [{"canvas": canvas, "source": "native"}]
    )
    inputs["diffusion_batch"] = inputs["diffusion_batch"].to(device)
    for name, value in tuple(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[name] = value.to(device)
    observed: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    original_decode = gemma_modeling.decode_packed_canvas

    def traced_decode(model, decoder_input_ids, *args, **kwargs):
        token_gate = args[4] if len(args) >= 5 else kwargs.get("token_gate")
        observed.append((decoder_input_ids.detach().clone(), token_gate))
        logits = original_decode(model, decoder_input_ids, *args, **kwargs)
        forced = logits.clone()
        forced[:, 0] = -torch.inf
        forced[:, 0, 24] = 0
        return forced

    monkeypatch.setattr(gemma_modeling, "decode_packed_canvas", traced_decode)
    loss, _ = trainer.compute_loss(model, inputs, return_outputs=True)
    assert len(observed) == steps
    assert all(
        torch.equal(state[0, [1, 3]], observed[0][0][0, [1, 3]])
        for state, _ in observed
    )
    assert all(
        torch.equal(state[0, 4:], observed[0][0][0, 4:]) for state, _ in observed
    )
    assert all(gate is not None and gate[0, 0] for _, gate in observed)
    assert observed[0][0][0, 0].item() != 24
    assert all(state[0, 0].item() == 24 for state, _ in observed[1:])
    assert not inputs["diffusion_batch"].canvas_loss_mask[0, 0]
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for name, parameter in model.named_parameters()
        if ".lora_" in name
    )


def test_encoder_canvas_trainer_remaps_logical_positions_after_packing():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )
    model = _TinyNativeModel()
    collator = DecisionTrainingCollator(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )
    inputs = collator(
        [
            {"canvas": make_canvas((1, 2), (4, 5, 6, 7), (1,), "first"), "source": "a"},
            {
                "canvas": make_canvas((3, 4, 5), (6, 7, 8, 9), (0, 3), "second"),
                "source": "b",
            },
        ]
    )

    _, pre_outputs, packed, _ = trainer._encoder_canvas_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )
    ar_loss = trainer._encoder_ar_loss(model, pre_outputs, packed, None)
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    loss.backward()

    assert outputs.logits.shape == (1, 8, 11)
    assert inputs["decision_label_rows"].tolist() == [[-1, -1], [-1, -1]]
    assert inputs["decision_label_positions"].tolist() == [[1, -1], [0, 3]]
    assert ar_loss > 0
    assert model.logit_bias.grad is not None
    assert torch.isfinite(model.logit_bias.grad).all()
    assert model.calls[0]["unroll_steps"] == 1
    assert model.calls[0]["update_mask"] is not None


def _encoder_ar_packed(prompt_ids, prompt_slot_mask):
    canvas = DecisionCanvas(
        prompt_ids=prompt_ids,
        canvas_ids=(1, 2, 3, 0),
        label_positions=(1,),
        allowed_ids=((1, 2),),
        question_ids=("encoder-ar",),
        targets=({"kind": "hard", "gold_idx": 0},),
        pinned_mask=(False,) * 4,
        semantic_mask=(True,) * 4,
        slot_mask=(False,) * 4,
        template_length=4,
        prompt_slot_mask=prompt_slot_mask,
    )
    batch = DecisionTrainingCollator(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )([{"canvas": canvas, "source": "encoder-ar"}])
    return EncoderCanvasBackend(vocab_size=11, sliding_window=8).pack(
        batch["diffusion_batch"]
    )


@pytest.mark.parametrize(
    ("global_count", "world_size"),
    [(None, 1), ({"encoder_ar": 0}, 2)],
)
def test_encoder_ar_loss_is_finite_zero_without_supported_prompt_targets(
    global_count, world_size
):
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        world_size=world_size,
    )
    packed = _encoder_ar_packed((7,), (True,))
    logits = torch.randn((*packed.encoder_input_ids.shape, 11), requires_grad=True)
    outputs = SimpleNamespace(encoder_logits=logits, logits=logits)

    loss = trainer._encoder_ar_loss(SimpleNamespace(), outputs, packed, global_count)
    loss.backward()

    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.isfinite(logits.grad).all()
    assert not torch.count_nonzero(logits.grad)


def test_encoder_ar_loss_preserves_positive_fractional_global_denominator():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        world_size=2,
    )
    packed = _encoder_ar_packed((7, 8, 9), (False, False, False))
    logits = torch.randn((*packed.encoder_input_ids.shape, 11), requires_grad=True)
    outputs = SimpleNamespace(encoder_logits=logits, logits=logits)
    token_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().flatten(0, -2),
        packed.encoder_input_ids[:, 1:].flatten(),
        reduction="none",
    )
    support = (
        packed.encoder_ar_valid_mask[:, 1:]
        & packed.encoder_ar_valid_mask[:, :-1]
        & (packed.encoder_document_ids[:, 1:] == packed.encoder_document_ids[:, :-1])
    )
    expected = (token_loss * support.flatten()).sum() / 0.5

    loss = trainer._encoder_ar_loss(
        SimpleNamespace(), outputs, packed, {"encoder_ar": 1}
    )
    loss.backward()

    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad)


@pytest.mark.parametrize("steps", [2, 3])
def test_encoder_canvas_multistep_holds_canvas_and_uses_outer_unroll(steps):
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=steps,
        decision={"latent": {"mode": "learned", "num_slots": 1}},
    )
    model = _TinyNativeModel()
    collator = DecisionTrainingCollator(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )
    inputs = collator(
        [
            {
                "canvas": make_canvas((1, 2), (4, 5, 6, 7), (1,), "first"),
                "source": "a",
            }
        ]
    )

    trainer._encoder_canvas_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )

    call = model.calls[-1]
    assert call["unroll_steps"] == steps
    update_mask = call["update_mask"]
    assert isinstance(update_mask, torch.Tensor)
    assert not update_mask.any()
    recurrent = call["recurrent_conditioning_mask"]
    assert isinstance(recurrent, torch.Tensor)
    assert torch.equal(
        recurrent,
        (~inputs["diffusion_batch"].canvas_input_pinned_mask)
        & inputs["diffusion_batch"].canvas_sc_eligible_mask,
    )


def test_encoder_canvas_multistep_config_sampled_to_one_holds_update_mask():
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=1,
    )
    model = _TinyNativeModel()
    collator = DecisionTrainingCollator(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )
    inputs = collator(
        [
            {
                "canvas": make_canvas((1, 2), (4, 5, 6, 7), (1,), "first"),
                "source": "a",
            }
        ]
    )

    trainer._encoder_canvas_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )

    assert model.calls[-1]["unroll_steps"] == 1
    assert model.calls[-1]["pilot_for_single_step"] is False
    update_mask = model.calls[-1]["update_mask"]
    assert isinstance(update_mask, torch.Tensor)
    assert not update_mask.any()


@pytest.mark.parametrize("attention_backend", ["dense", "flex_attention"])
def test_packed_slot_mask_remaps_multiple_canvases_and_excludes_bucket_padding(
    attention_backend,
):
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=2,
    )
    first = DecisionCanvas(
        prompt_ids=(1, 2),
        canvas_ids=(9, 3, 4, 0, 0, 0, 0, 0),
        label_positions=(1,),
        allowed_ids=((1, 2),),
        question_ids=("first",),
        targets=({"kind": "hard", "gold_idx": 0},),
        pinned_mask=(True, False, True, True, True, True, True, True),
        semantic_mask=(True,) * 8,
        slot_mask=(True, False, False, False, False, False, False, False),
        template_length=3,
    )
    second = DecisionCanvas(
        prompt_ids=(3,),
        canvas_ids=(5, 6, 7, 0, 0, 0, 0, 0),
        label_positions=(1,),
        allowed_ids=((1, 2),),
        question_ids=("second",),
        targets=({"kind": "hard", "gold_idx": 0},),
        pinned_mask=(True, False, True, True, True, True, True, True),
        semantic_mask=(True,) * 8,
        slot_mask=(False,) * 8,
        template_length=3,
    )
    inputs = DecisionTrainingCollator(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )(
        [
            {"canvas": first, "source": "a"},
            {"canvas": second, "source": "b"},
        ]
    )
    backend = EncoderCanvasBackend(
        vocab_size=11,
        sliding_window=8,
        attention_backend=attention_backend,
        physical_bucket_size=32,
    )
    packed = backend.pack(inputs["diffusion_batch"])
    slots = trainer._packed_slot_mask(inputs["decision_slot_mask"], packed)

    expected = torch.zeros_like(slots)
    expected[0, 0] = True
    assert torch.equal(slots, expected)
    assert not slots[:, 16:].any()


@pytest.mark.parametrize("steps", [2, 3])
def test_full_sequence_free_slots_use_flattened_collator_mask_for_two_documents(steps):
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=steps,
        decision={
            "latent": {
                "mode": "free",
                "num_slots": 1,
                "free_update_policy": "argmax",
            }
        },
    )
    first = DecisionCanvas(
        prompt_ids=(1, 2),
        canvas_ids=(9, 3, 4, 0),
        label_positions=(1,),
        allowed_ids=((1, 2),),
        question_ids=("first",),
        targets=({"kind": "hard", "gold_idx": 0},),
        pinned_mask=(False, False, True, True),
        semantic_mask=(True,) * 4,
        slot_mask=(True, False, False, False),
        template_length=3,
    )
    second = DecisionCanvas(
        prompt_ids=(3,),
        canvas_ids=(8, 5, 6, 0),
        label_positions=(1,),
        allowed_ids=((1, 2),),
        question_ids=("second",),
        targets=({"kind": "hard", "gold_idx": 0},),
        pinned_mask=(False, False, True, True),
        semantic_mask=(True,) * 4,
        slot_mask=(True, False, False, False),
        template_length=3,
    )
    inputs = DecisionTrainingCollator(trainer._native_spec)(
        [{"canvas": first, "source": "a"}, {"canvas": second, "source": "b"}]
    )
    assert inputs["decision_slot_mask"].shape == inputs["input_ids"].shape
    assert inputs["decision_slot_mask"].tolist() == [
        [
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ]
    ]

    model = _TraceNativeModel()
    logits, _ = trainer._full_sequence_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )

    assert len(model.forward_states) == steps
    initial = model.forward_states[0]
    assert all(
        torch.equal(state[0, [3, 8]], initial[0, [3, 8]])
        for state in model.forward_states
    )
    if steps > 1:
        assert all(
            torch.equal(state[0, [2, 7]], torch.tensor([10, 10]))
            for state in model.forward_states[1:]
        )
    assert torch.equal(inputs["canvas_loss_mask"], inputs["canvas_corruptible_mask"])
    assert not torch.any(inputs["decision_slot_mask"] & inputs["canvas_loss_mask"])
    logits.sum().backward()
    assert model.logit_bias.grad is not None
    assert torch.count_nonzero(model.logit_bias.grad)


def test_full_sequence_free_slots_use_effective_backend_mask_at_k1():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        {
            "latent": {
                "mode": "free",
                "num_slots": 1,
                "free_update_policy": "argmax",
            }
        },
    )
    canvas = DecisionCanvas(
        prompt_ids=(1,),
        canvas_ids=(7, 3, 4, 0),
        label_positions=(1,),
        allowed_ids=((1, 2),),
        question_ids=("q",),
        targets=({"kind": "hard", "gold_idx": 0},),
        pinned_mask=(False, False, True, True),
        semantic_mask=(True,) * 4,
        slot_mask=(True, False, False, False),
        template_length=3,
    )
    inputs = DecisionTrainingCollator(trainer._native_spec)(
        [{"canvas": canvas, "source": "test"}]
    )
    model = _TraceNativeModel()
    model.config.mask_token_id = 2

    trainer._full_sequence_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )

    assert model.forward_states[0][0, 1].item() == 10


def test_global_example_scaling_matches_ddp_average_for_unequal_windows():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        world_size=2,
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


def test_encoder_gradient_window_counts_examples_and_ar_tokens_globally():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        world_size=2,
    )
    trainer.accelerator = SimpleNamespace(
        gather=lambda count: torch.stack((count, count + 1))
    )
    collator = DecisionTrainingCollator(trainer._native_spec)
    batch = collator(
        [
            {"canvas": make_canvas((1, 2), (4, 5, 6, 7), (1,), "first"), "source": "a"},
            {
                "canvas": make_canvas((3, 4, 5), (6, 7, 8, 9), (0, 3), "second"),
                "source": "b",
            },
        ]
    )

    _batches, counts = trainer.get_batch_samples(
        iter((batch,)), num_batches=2, device=torch.device("cpu")
    )

    assert counts["examples"].item() == 5
    assert counts["encoder_ar"].item() == 3


def test_post_config_prevents_training_step_from_dividing_global_loss_again(
    monkeypatch,
):
    trainer = _TrainingStepHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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


def test_prepare_inputs_moves_encoder_canvas_dataclass(monkeypatch):
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )
    trainer.args.device = torch.device("cpu")
    collator = DecisionTrainingCollator(trainer._native_spec)
    batch = collator(
        [{"canvas": make_canvas((1, 2), (4, 5, 6, 7), (1,), "first"), "source": "a"}]
    )["diffusion_batch"]
    monkeypatch.setattr(
        AxolotlDiffusionTrainer, "_prepare_inputs", lambda _self, inputs: inputs
    )

    prepared = trainer._prepare_inputs({"diffusion_batch": batch})

    assert prepared["diffusion_batch"] is not batch
    assert prepared["diffusion_batch"].device == torch.device("cpu")


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


def test_stratified_draw_descriptors_are_unique_before_distributed_sharding():
    sampler = _StratifiedDecisionBatchSampler(
        dataset_size=2,
        batch_size=2,
        seed=17,
        world_size=8,
        emit_draw_descriptors=True,
    )
    first = list(sampler)
    sampler.set_epoch(0)
    repeated = list(sampler)
    sampler.set_epoch(1)
    next_epoch = list(sampler)
    draws = [draw for batch in first for draw in batch]

    assert first == repeated
    assert next_epoch != first
    assert all(isinstance(draw, DecisionDraw) for draw in draws)
    assert [draw.global_draw_ordinal for draw in draws] == list(range(16))
    assert {draw.index for draw in draws} == {0, 1}
    assert {draw.epoch for draw in draws} == {0}


def test_non_stratified_draw_descriptors_are_deterministic_and_resume_stable():
    sampler = _DecisionDrawBatchSampler(
        dataset_size=6, batch_size=2, seed=23, world_size=4
    )
    first = list(sampler)
    sampler.set_epoch(0)
    repeated = list(sampler)
    sampler.set_epoch(1)
    next_epoch = list(sampler)
    draws = [draw for batch in first for draw in batch]

    assert first == repeated
    assert next_epoch != first
    assert [draw.global_draw_ordinal for draw in draws] == list(range(8))
    assert [draw.global_draw_ordinal for batch in first[2:] for draw in batch] == [
        4,
        5,
        6,
        7,
    ]
    assert len({draw.global_draw_ordinal for draw in draws}) == len(draws)


def test_draw_batch_sampler_is_accepted_by_a_real_dataloader():
    class _Dataset:
        def __len__(self):
            return 4

        def __getitem__(self, draw):
            assert isinstance(draw, DecisionDraw)
            return {
                "index": draw.index,
                "epoch": draw.epoch,
                "ordinal": draw.global_draw_ordinal,
            }

    sampler = _DecisionDrawBatchSampler(dataset_size=4, batch_size=2, seed=13)
    batches = list(DataLoader(_Dataset(), batch_sampler=sampler))

    assert [batch["ordinal"].tolist() for batch in batches] == [[0, 1], [2, 3]]


def test_trainer_uses_draw_batch_sampler_for_non_stratified_sampled_slots():
    class _Dataset:
        manifest = {}

        def __len__(self):
            return 6

    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        decision={
            "latent": {"mode": "pinned", "num_slots": 2, "sample_num_slots": True}
        },
        world_size=2,
    )
    trainer.train_dataset = _Dataset()
    trainer.args.per_device_train_batch_size = 2
    trainer.args.seed = 29
    trainer.args.dataloader_drop_last = None

    sampler = trainer._get_train_sampler()

    assert isinstance(sampler, _DecisionDrawBatchSampler)
    assert [draw.global_draw_ordinal for batch in sampler for draw in batch] == list(
        range(8)
    )


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
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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

    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    trainer.train_dataset = _Dataset()
    trainer.args.sample_packing = True

    with pytest.raises(ValueError, match="per_batch_stratified: false"):
        trainer._get_train_sampler()


def test_decision_dataset_exposes_canvas_lengths_to_multipack():
    from axolotl.integrations.diffusion_decision.datasets import DecisionDataset
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
    class _CoreDataLoaderHarness(TrainerHarness):
        def _get_collator_with_removed_columns(self, collator, description):
            del description
            return collator

    dataset = DecisionDataset(
        [{"length": 5}, {"length": 7}], {"per_batch_stratified": False}
    )
    trainer = _CoreDataLoaderHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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


def test_interval_metrics_are_source_weighted_and_cleared_after_logging(monkeypatch):
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    decision = trainer._decision_config()
    selected = torch.tensor(
        [
            [[0.0, 0.5, 1.0, -0.5]],
            [[0.3, -0.2, 0.7, -0.1]],
            [[-0.4, 0.6, 0.1, 0.2]],
        ],
        requires_grad=True,
    )
    examples = (_example(1, weight=1.0), _example(1, weight=2.0), _example(1))
    supervision = torch.ones((3, 1), dtype=torch.bool)
    result = decision_label_loss(
        selected, examples, supervision, label_softmax="both", brier_weight=0.1
    )
    trainer._store_decision_metrics(
        result,
        selected,
        examples,
        supervision,
        decision,
        ("alpha", "alpha", "beta"),
        train_eval="train",
        loss_function=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source metrics must reuse the initial loss result")
        ),
    )
    alpha = decision_label_loss(
        selected[:2],
        examples[:2],
        supervision[:2],
        label_softmax="both",
        brier_weight=0.1,
    )
    captured: dict[str, float] = {}

    def _base_log(_self, logs, start_time=None):
        del _self, start_time
        captured.update(logs)

    monkeypatch.setattr(AxolotlDiffusionTrainer, "log", _base_log)
    trainer.log({"loss": 1.0})

    for name in (
        "loss",
        "restricted_loss",
        "full_vocab_loss",
        "effective_full_vocab_loss",
        "brier_loss",
    ):
        torch.testing.assert_close(
            torch.tensor(captured[f"decision/all/{name}"]),
            getattr(result, name).detach(),
        )
        torch.testing.assert_close(
            torch.tensor(captured[f"decision/alpha/{name}"]),
            getattr(alpha, name).detach(),
        )
    assert captured["decision/all/examples"] == 3
    assert captured["decision/alpha/examples"] == 2
    assert captured["decision/beta/examples"] == 1
    assert captured["decision/all/full_vocab_dft_hard_count"] == 0
    assert captured["decision/all/full_vocab_dft_mean_hard_weight"] == 0
    assert selected.grad is None

    trainer.log({"loss": 2.0})
    assert not trainer._decision_metric_totals("train")


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

    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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


def test_decision_objective_does_not_report_token_perplexity(monkeypatch):
    forwarded: list[dict[str, float]] = []
    monkeypatch.setattr(trainer_base, "is_main_process", lambda: False)
    monkeypatch.setattr(
        trainer_base.Trainer,
        "log",
        lambda _self, logs, start_time=None: forwarded.append(dict(logs)),
    )

    decision = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )
    decision._stored_metrics = {"train": {}, "eval": {}}
    decision.args.include_tkps = False
    AxolotlTrainer.log(decision, {"loss": 2.0, "eval_loss": 2.0})

    assert "ppl" not in forwarded[0]
    assert "eval_ppl" not in forwarded[0]


def test_get_batch_samples_counts_logical_examples_and_handles_last_short_window():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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
        canvas = make_canvas(
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


def _sampled_slot_rows(count: int) -> list[dict[str, object]]:
    plan = SlotPlan(
        ids=(7, 8, 9),
        placement="thought",
        pinned_mask=(True, True, True),
        update_mask=(False, False, False),
        loss_mask=(False, False, False),
        trainable_token_ids=(),
    )
    rows: list[dict[str, object]] = []
    for index in range(count):
        canvas = DecisionCanvas(
            prompt_ids=(1, 2 + (index % 3)),
            canvas_ids=(7, 8, 9, 4, 5, 6),
            label_positions=(3, 4),
            allowed_ids=((1, 2, 3), (2, 4, 6)),
            question_ids=(f"sampled-{index}-a", f"sampled-{index}-b"),
            targets=(
                {"kind": "hard", "gold_idx": index % 3},
                {"kind": "dist", "probs": (0.2, 0.3, 0.5)},
            ),
            pinned_mask=(True, True, True, False, False, False),
            semantic_mask=(True,) * 6,
            slot_mask=(True, True, True, False, False, False),
            template_length=6,
        )
        rows.append(
            {
                "canvas": canvas,
                "slot_plan": plan,
                "slot_sampling": {
                    "seed": 29,
                    "max_slots": 3,
                    "pad_token_id": 0,
                    "padding_pinned": True,
                },
                "decision_example": decision_example_from_canvas(
                    canvas, source_weight=1.0 + (index % 2) / 2
                ),
                "source": "alpha" if index % 2 else "beta",
            }
        )
    return rows


def _sampled_prompt_slot_rows(count: int) -> list[dict[str, object]]:
    plan = SlotPlan(
        ids=(7, 8, 9),
        placement="prompt",
        pinned_mask=(True, True, True),
        update_mask=(False, False, False),
        loss_mask=(False, False, False),
        trainable_token_ids=(7, 8, 9),
    )
    rows: list[dict[str, object]] = []
    for index in range(count):
        canvas = DecisionCanvas(
            prompt_ids=(7, 8, 9, 1, 2 + (index % 3)),
            canvas_ids=(4, 5, 6, 0),
            label_positions=(1, 2),
            allowed_ids=((1, 2, 3), (2, 4, 6)),
            question_ids=(f"prompt-{index}-a", f"prompt-{index}-b"),
            targets=(
                {"kind": "hard", "gold_idx": index % 3},
                {"kind": "dist", "probs": (0.2, 0.3, 0.5)},
            ),
            pinned_mask=(False,) * 4,
            semantic_mask=(True,) * 4,
            slot_mask=(False,) * 4,
            template_length=4,
            prompt_slot_mask=(True, True, True, False, False),
        )
        rows.append(
            {
                "canvas": canvas,
                "slot_plan": plan,
                "slot_sampling": {
                    "seed": 29,
                    "max_slots": 3,
                    "pad_token_id": 0,
                    "padding_pinned": True,
                },
                "decision_example": decision_example_from_canvas(
                    canvas, source_weight=1.0 + (index % 2) / 2
                ),
                "source": "alpha" if index % 2 else "beta",
            }
        )
    return rows


def _sampled_loop_trainer(
    tmp_path,
    dataset,
    *,
    stratified: bool,
    mode: str = "pinned",
    layout: DiffusionLayout = DiffusionLayout.FULL_SEQUENCE,
):
    spec = trainer_spec(layout, LogitAlignment.ALIGNED)
    args = AxolotlTrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=0.05,
        max_steps=1,
        lr_scheduler_type="constant",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        seed=29,
        data_seed=29,
        optim="sgd",
    )
    trainer = _SampledLoopTrainer(
        model=_LoopNativeModel(),
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DecisionTrainingCollator(spec),
        spec=spec,
    )
    trainer.axolotl_cfg = {
        "seed": 29,
        "diffusion_decision": {
            "read_fraction": 1.0,
            "latent": {
                "mode": mode,
                "num_slots": 3,
                "sample_num_slots": True,
            },
        },
    }
    if stratified:
        dataset.manifest.update(
            {
                "per_batch_stratified": True,
                "stratified_micro_batch_size": 2,
                "mixture_seed": 29,
            }
        )
    trainer.post_set_axolotl_cfg()
    return trainer


@pytest.mark.parametrize("stratified", [False, True])
def test_sampled_slots_survive_trainer_dataloader_and_drive_typed_loss(
    tmp_path, stratified
):
    dataset = DecisionDataset(_sampled_slot_rows(12), {})
    trainer = _sampled_loop_trainer(tmp_path, dataset, stratified=stratified)

    loader = trainer.get_train_dataloader()
    first_epoch = list(loader)
    sampler = trainer._get_train_sampler()
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)
    draws = [draw for batch in first_epoch for draw in batch["decision_draws"]]
    counts = torch.cat([batch["decision_slot_counts"] for batch in first_epoch])

    assert all(isinstance(draw, DecisionDraw) for draw in draws)
    assert len({draw.global_draw_ordinal for draw in draws}) == len(draws)
    assert len(set(counts.tolist())) > 1
    assert second_epoch != epoch_zero
    for batch in first_epoch:
        slots = batch["decision_slot_mask"]
        docs = batch["document_ids"]
        observed = torch.zeros_like(batch["decision_slot_counts"])
        observed.scatter_add_(0, docs.reshape(-1), slots.reshape(-1).long())
        assert torch.equal(observed, batch["decision_slot_counts"])
        assert not torch.any(slots & batch["canvas_loss_mask"])
        loss = trainer.compute_loss(
            trainer.model,
            trainer._prepare_inputs(batch),
            num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
        )
        loss.backward()
        assert trainer.model.logit_bias.grad is not None
        assert torch.count_nonzero(trainer.model.logit_bias.grad)
        trainer.model.logit_bias.grad = None
    if not stratified:
        before = trainer.model.logit_bias.detach().clone()
        trainer.train()
        assert not torch.equal(trainer.model.logit_bias.detach(), before)


def test_sampled_slots_evaluation_uses_static_maximum_canvases(tmp_path):
    dataset = DecisionDataset(_sampled_slot_rows(4), {})
    trainer = _sampled_loop_trainer(tmp_path, dataset, stratified=False)

    loader = trainer.get_eval_dataloader()
    batch = next(iter(loader))

    assert batch["decision_draws"] == (None,) * len(dataset)
    assert batch["decision_slot_counts"].tolist() == [3] * len(dataset)
    observed = torch.zeros_like(batch["decision_slot_counts"])
    observed.scatter_add_(
        0,
        batch["document_ids"].reshape(-1),
        batch["decision_slot_mask"].reshape(-1).long(),
    )
    assert torch.equal(batch["decision_slot_counts"], observed)
    with pytest.raises(ValueError, match="training batches require DecisionDraw"):
        trainer.compute_loss(
            trainer.model,
            trainer._prepare_inputs(batch),
            num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
        )
    trainer.model.eval()
    loss = trainer.compute_loss(
        trainer.model,
        trainer._prepare_inputs(batch),
        num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
    )
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "layout", [DiffusionLayout.FULL_SEQUENCE, DiffusionLayout.ENCODER_CANVAS]
)
def test_sampled_prompt_slots_validate_positions_and_train(tmp_path, layout):
    dataset = DecisionDataset(_sampled_prompt_slot_rows(8), {})
    trainer = _sampled_loop_trainer(
        tmp_path, dataset, stratified=False, mode="prompt", layout=layout
    )
    batch = next(iter(trainer.get_train_dataloader()))
    counts = batch["decision_slot_counts"]
    prompt_slots = batch["decision_prompt_slot_mask"]
    if layout is DiffusionLayout.FULL_SEQUENCE:
        positions = batch["position_ids"]
        documents = batch["document_ids"]
        assert torch.equal(prompt_slots, positions < counts[documents])
    else:
        logical = batch["diffusion_batch"]
        positions = torch.arange(prompt_slots.shape[1])[None]
        assert torch.equal(
            prompt_slots, logical.encoder_validity & (positions < counts[:, None])
        )
        assert not torch.any(prompt_slots & logical.encoder_ar_valid_mask)
    assert not torch.any(batch["decision_slot_mask"])
    if layout is DiffusionLayout.FULL_SEQUENCE:
        assert torch.all(prompt_slots <= batch["canvas_input_pinned_mask"])
    else:
        assert torch.all(prompt_slots <= batch["decision_prompt_input_pinned_mask"])

    loss = trainer.compute_loss(
        trainer.model,
        trainer._prepare_inputs(batch),
        num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
    )
    loss.backward()
    assert trainer.model.logit_bias.grad is not None
    assert torch.count_nonzero(trainer.model.logit_bias.grad)

    malformed = dict(batch)
    malformed["decision_prompt_slot_mask"] = prompt_slots.roll(1, dims=1)
    with pytest.raises(RuntimeError, match="logical prompt prefix"):
        trainer.compute_loss(
            trainer.model,
            trainer._prepare_inputs(malformed),
            num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
        )

    missing = dict(batch)
    missing.pop("decision_prompt_slot_mask")
    with pytest.raises(ValueError, match="sampled prompt slots require"):
        trainer.compute_loss(
            trainer.model,
            trainer._prepare_inputs(missing),
            num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
        )
    if layout is DiffusionLayout.ENCODER_CANVAS:
        missing = dict(batch)
        missing.pop("decision_prompt_input_pinned_mask")
        with pytest.raises(ValueError, match="prompt pinned metadata"):
            trainer.compute_loss(
                trainer.model,
                trainer._prepare_inputs(missing),
                num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
            )


def test_sampled_slot_runtime_rejects_descriptor_mask_mismatch(tmp_path):
    dataset = DecisionDataset(_sampled_slot_rows(4), {})
    trainer = _sampled_loop_trainer(tmp_path, dataset, stratified=False)
    batch = next(iter(trainer.get_train_dataloader()))
    malformed = dict(batch)
    malformed["decision_slot_counts"] = batch["decision_slot_counts"] + 1

    with pytest.raises(RuntimeError, match="slot masks do not match descriptor counts"):
        trainer.compute_loss(
            trainer.model,
            trainer._prepare_inputs(malformed),
            num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
        )

    malformed = dict(batch)
    malformed["canvas_loss_mask"] = batch["decision_slot_mask"].clone()
    with pytest.raises(RuntimeError, match="cannot receive direct supervised loss"):
        trainer.compute_loss(
            trainer.model,
            trainer._prepare_inputs(malformed),
            num_items_in_batch=torch.tensor(len(batch["decision_examples"])),
        )


def test_sampled_slot_draws_are_resume_stable_through_dataloader_workers():
    dataset = DecisionDataset(_sampled_slot_rows(12), {})
    collator = DecisionTrainingCollator(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
    )

    def batches(epoch: int, workers: int):
        sampler = _DecisionDrawBatchSampler(
            dataset_size=len(dataset), batch_size=2, seed=29
        )
        sampler.set_epoch(epoch)
        dataloader_kwargs = {
            "batch_sampler": sampler,
            "collate_fn": collator,
            "num_workers": workers,
        }
        if workers:
            dataloader_kwargs["multiprocessing_context"] = "spawn"
        return [
            (
                tuple(batch["decision_slot_counts"].tolist()),
                tuple(batch["decision_draws"]),
            )
            for batch in DataLoader(dataset, **dataloader_kwargs)
        ]

    epoch_zero = batches(0, workers=0)
    assert epoch_zero == batches(0, workers=0)
    assert epoch_zero == batches(0, workers=2)
    assert epoch_zero != batches(1, workers=0)


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
    spec = trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED)
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
    trainer.axolotl_cfg = {"diffusion_decision": {"read_fraction": 1.0}}
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


def test_unwired_decision_controls_fail_explicitly():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        {"latent": {"mode": "free", "num_slots": 1}},
    )

    with pytest.raises((NotImplementedError, ValueError)):
        trainer._validate_decision_config(trainer._decision_config())


def test_fractional_read_times_preserve_endpoint_rng_and_select_logical_examples():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
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
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
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


@pytest.mark.parametrize("mode", ["pad", "pinned", "learned", "prompt", "mask"])
def test_fixed_decision_slots_are_supported_for_one_step(mode):
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        {"latent": {"mode": mode, "num_slots": 1}},
    )

    trainer._validate_decision_config(
        trainer._decision_config(), k_max=1, spec=trainer._native_spec
    )


def test_fixed_decision_slots_support_multistep_unroll():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        {"latent": {"mode": "pinned", "num_slots": 1}},
    )

    trainer._validate_decision_config(
        trainer._decision_config(), k_max=2, spec=trainer._native_spec
    )


def test_decision_multistep_rejects_grad_through_steps():
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=2,
        sampled_steps=2,
        grad_through_steps=True,
    )

    with pytest.raises(NotImplementedError, match="grad-through-steps"):
        trainer.compute_loss(_TinyNativeModel(), _full_sequence_multistep_inputs())


def test_full_sequence_multistep_rejects_unsupported_self_conditioning():
    spec = replace(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        self_conditioning=True,
    )
    trainer = MultistepTrainerHarness(spec, k_max=2, sampled_steps=2)

    with pytest.raises(NotImplementedError, match="self-conditioning"):
        trainer.compute_loss(_TinyNativeModel(), _full_sequence_multistep_inputs())


def test_mask_decision_slots_require_absorbing_diffusion():
    trainer = TrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        {"latent": {"mode": "mask", "num_slots": 1}},
    )
    uniform_spec = replace(
        trainer._native_spec,
        noise=DiffusionNoise.UNIFORM,
        mask_token_policy=MaskTokenPolicy.NONE,
    )

    with pytest.raises(ValueError, match="absorbing"):
        trainer._validate_decision_config(
            trainer._decision_config(), k_max=1, spec=uniform_spec
        )


@pytest.mark.parametrize(
    "layout", [DiffusionLayout.FULL_SEQUENCE, DiffusionLayout.ENCODER_CANVAS]
)
def test_fractional_reads_change_actual_forward_inputs_per_logical_example(
    monkeypatch, layout
):
    trainer = TrainerHarness(
        trainer_spec(layout, LogitAlignment.ALIGNED), {"read_fraction": 0.5}
    )
    model = _TraceNativeModel()
    calls = 0

    def controlled_rand(*args, **kwargs):
        nonlocal calls
        calls += 1
        shape = args[0] if args else kwargs["size"]
        if calls == 1:
            return torch.tensor([0.1, 0.9], device=kwargs.get("device"))
        size = (shape,) if isinstance(shape, int) else shape
        return torch.full(size, 0.75, device=kwargs.get("device"))

    monkeypatch.setattr(torch, "rand", controlled_rand)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, **kwargs: torch.zeros(
            size, dtype=kwargs.get("dtype", torch.long), device=kwargs.get("device")
        ),
    )
    if layout is DiffusionLayout.FULL_SEQUENCE:
        inputs = {
            "input_ids": torch.tensor([[1, 4, 5, 2, 6, 7]]),
            "document_ids": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "semantic_validity": torch.ones((1, 6), dtype=torch.bool),
            "canvas_corruptible_mask": torch.tensor(
                [[False, True, False, False, False, True]]
            ),
            "canvas_input_pinned_mask": torch.zeros((1, 6), dtype=torch.bool),
            "canvas_update_mask": torch.zeros((1, 6), dtype=torch.bool),
            "decision_examples": (_example(2),),
            "decision_question_mask": torch.tensor([[True, True]]),
            "decision_supervision_mask": torch.tensor([[True, True]]),
            "decision_label_rows": torch.tensor([[0, 0]]),
            "decision_label_positions": torch.tensor([[1, 5]]),
        }
        trainer._full_sequence_logits(
            model, inputs, trainer._native_spec, trainer._decision_config()
        )
        state = model.forward_states[-1]
        assert state[0, 1].item() == 10
        assert state[0, 5].item() == 7
    else:
        inputs = DecisionTrainingCollator(trainer._native_spec)(
            [
                {"canvas": make_canvas((1,), (4, 5, 6), (0,), "first"), "source": "a"},
                {"canvas": make_canvas((2,), (7, 8, 9), (0,), "second"), "source": "b"},
            ]
        )
        trainer._encoder_canvas_logits(
            model, inputs, trainer._native_spec, trainer._decision_config()
        )
        state = model.forward_states[-1]
        assert state[0, 0].item() == 0
        assert state[0, 3].item() == 7


@pytest.mark.parametrize("steps", [2, 3])
def test_fractional_reads_hold_actual_label_states_across_full_sequence_unroll(
    monkeypatch, steps
):
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=steps,
        decision={"read_fraction": 0.5, "latent": {"mode": "pinned", "num_slots": 1}},
    )
    calls = 0

    def controlled_rand(*args, **kwargs):
        nonlocal calls
        calls += 1
        shape = args[0] if args else kwargs["size"]
        if calls == 1:
            return torch.tensor([0.1, 0.9], device=kwargs.get("device"))
        return torch.full(
            (shape,) if isinstance(shape, int) else shape,
            0.75,
            device=kwargs.get("device"),
        )

    monkeypatch.setattr(torch, "rand", controlled_rand)
    model = _TraceNativeModel()
    inputs = _full_sequence_multistep_inputs()
    inputs["canvas_input_pinned_mask"] = torch.zeros((1, 6), dtype=torch.bool)
    loss, _ = trainer.compute_loss(model, inputs, return_outputs=True)
    expected = torch.tensor([[1, 10, 5, 2, 6, 7]])
    assert len(model.forward_states) == steps
    assert all(torch.equal(state, expected) for state in model.forward_states)
    loss.backward()
    final_grad = model.forward_logits[-1].grad
    assert final_grad is not None
    assert torch.count_nonzero(final_grad[0, 1])
    assert torch.count_nonzero(final_grad[0, 5])
    assert not torch.count_nonzero(final_grad[0, [0, 2, 3, 4]])
    assert model.logit_bias.grad is not None and torch.count_nonzero(
        model.logit_bias.grad
    )


@pytest.mark.parametrize("steps", [2, 3])
def test_fractional_reads_hold_encoder_canvas_labels_and_supervise_only_labels(
    monkeypatch, steps
):
    import axolotl.model_support.diffusion_gemma.modeling as gemma_modeling

    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        k_max=3,
        sampled_steps=steps,
        decision={"read_fraction": 0.5},
    )
    model = gemma_modeling.AxolotlDiffusionGemmaForBlockDiffusion(
        tiny_gemma_config()
    ).train()
    observed = []
    original_decode = gemma_modeling.decode_packed_canvas

    def traced_decode(model, decoder_input_ids, *args, **kwargs):
        observed.append((decoder_input_ids.detach().clone(), torch.is_grad_enabled()))
        return original_decode(model, decoder_input_ids, *args, **kwargs)

    monkeypatch.setattr(gemma_modeling, "decode_packed_canvas", traced_decode)
    calls = 0

    def controlled_rand(*args, **kwargs):
        nonlocal calls
        calls += 1
        shape = args[0] if args else kwargs["size"]
        if calls == 1:
            return torch.tensor([0.1, 0.9], device=kwargs.get("device"))
        return torch.full(
            (shape,) if isinstance(shape, int) else shape,
            0.75,
            device=kwargs.get("device"),
        )

    monkeypatch.setattr(torch, "rand", controlled_rand)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, **kwargs: torch.zeros(
            size, dtype=kwargs.get("dtype", torch.long), device=kwargs.get("device")
        ),
    )
    inputs = DecisionTrainingCollator(trainer._native_spec)(
        [
            {"canvas": make_canvas((1,), (4, 5, 6), (0,), "first"), "source": "a"},
            {"canvas": make_canvas((2,), (7, 8, 9), (0,), "second"), "source": "b"},
        ]
    )
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert len(observed) == steps
    expected = torch.tensor([[0, 5, 6, 7, 8, 9]])
    assert all(torch.equal(state, expected) for state, _ in observed)
    assert [enabled for _, enabled in observed] == [False] * (steps - 1) + [True]
    outputs.logits.retain_grad()
    loss.backward()
    grad = outputs.logits.grad
    assert grad is not None
    assert torch.count_nonzero(grad[0, 0]) and torch.count_nonzero(grad[0, 3])
    assert not torch.count_nonzero(grad[0, [1, 2, 4, 5]])


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
        trainer = MultistepTrainerHarness(
            trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
            k_max=steps,
            sampled_steps=steps,
        )
        trainer.axolotl_cfg = SimpleNamespace(
            diffusion_decision={}, cut_cross_entropy=enabled
        )
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
