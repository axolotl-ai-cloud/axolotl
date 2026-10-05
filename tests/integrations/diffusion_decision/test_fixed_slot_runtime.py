"""Fixed decision slots retain their prepared masks through training and reads."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from axolotl.core.trainers.diffusion_lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.integrations.diffusion_decision.loss import decision_label_loss
from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.readers.hf import HFReader
from axolotl.integrations.diffusion_decision.slots import SlotInit, SlotMode
from axolotl.integrations.diffusion_decision.trainer import DiffusionDecisionTrainer
from axolotl.integrations.diffusion_decision.training_collator import (
    DecisionTrainingCollator,
)
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


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


class _EchoModel(nn.Module):
    def __init__(self, vocab_size=256) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=vocab_size, mask_token_id=9)
        self.seen_input_ids: torch.Tensor | None = None

    def forward(self, input_ids, attention_mask, position_ids, use_cache, **kwargs):
        del attention_mask, position_ids, use_cache, kwargs
        self.seen_input_ids = input_ids.detach().clone()
        logits = torch.nn.functional.one_hot(input_ids, num_classes=256).float()
        return SimpleNamespace(logits=logits + self.anchor)


class _TrainerHarness(DiffusionDecisionTrainer):
    def __init__(self, spec: DiffusionSpec, decision: dict[str, object]) -> None:
        self._spec = spec
        self.axolotl_cfg = {"diffusion_decision": decision}
        self.args = SimpleNamespace(world_size=1)
        self._special_token_ids: set[int] = set()

    @property
    def _native_spec(self):
        return self._spec

    @staticmethod
    def _full_sequence_backend():
        return FullSequenceBackend(mask_token_id=9, attention_backend="dense")

    @staticmethod
    def _native_unroll_settings():
        return 1, False

    @staticmethod
    def _sample_native_unroll_steps(k_max, device):
        del k_max, device
        return 1

    @staticmethod
    def _sample_native_times(count, device, default_eps):
        del default_eps
        return torch.ones(count, device=device)


def _spec(noise: DiffusionNoise) -> DiffusionSpec:
    return DiffusionSpec(
        noise=noise,
        layout=DiffusionLayout.FULL_SEQUENCE,
        logit_alignment=LogitAlignment.ALIGNED,
        first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
        self_conditioning=False,
        max_canvas=128,
        max_context=1024,
        eos_handling=EosHandling.INDEPENDENT,
        mask_token_policy=(
            MaskTokenPolicy.MODEL
            if noise is DiffusionNoise.ABSORBING
            else MaskTokenPolicy.NONE
        ),
        default_time_weighting=TimeWeighting.NONE,
        objective_reduction=ObjectiveReduction.EXAMPLE_MEAN,
        generation_adapter=GenerationAdapter.FULL_SEQUENCE,
        reduction_scope=ReductionScope.GLOBAL_WINDOW,
    )


def _record():
    return {
        "id": "fixed-slot",
        "source": "test",
        "group": "test",
        "state": "state",
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Pick.",
                "options": ["one", "two"],
            }
        },
        "labels": {"choice": {"kind": "hard", "gold_idx": 0}},
    }


def _slot_config(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "num_slots": 2,
        "token_ids": list(
            {"pinned": (7,), "learned": (7, 8), "prompt": (7, 8)}.get(mode, ())
        ),
    }


def _slot_token_ids(mode: SlotMode) -> tuple[int, ...]:
    return {"pinned": (7,), "learned": (7, 8), "prompt": (7, 8)}.get(mode, ())


def _prepared_canvas(mode: str, noise: DiffusionNoise):
    spec = _spec(noise)
    slot_mode = cast(SlotMode, mode)
    plan = SlotInit(
        slot_mode,
        token_ids=_slot_token_ids(slot_mode),
        num_slots=2,
        vocab_size=256,
        pad_id=0,
        spec=spec,
        mask_token_id=9,
    ).build()
    canvas = build_decision_canvas(
        _CharacterTokenizer(),
        _record(),
        prompt_ids=(90,),
        scaffold_ids=(),
        turn_close_id=106,
        pad_id=0,
        vocab_size=256,
        width=128,
        seed=23,
        steps=1,
        noise_kind=noise.value,
        mask_token_id=9 if noise is DiffusionNoise.ABSORBING else None,
        slot_plan=plan,
        thought_open_ids=(70,),
        thought_close_ids=(71,),
    )
    return spec, plan, canvas


@pytest.mark.parametrize(
    ("mode", "noise"),
    [
        ("pad", DiffusionNoise.UNIFORM),
        ("pinned", DiffusionNoise.UNIFORM),
        ("learned", DiffusionNoise.UNIFORM),
        ("prompt", DiffusionNoise.UNIFORM),
        ("mask", DiffusionNoise.ABSORBING),
    ],
)
def test_prepared_fixed_slots_stay_pinned_through_trainer_and_reader(mode, noise):
    spec, plan, canvas = _prepared_canvas(mode, noise)
    fixed_canvas = torch.tensor(canvas.slot_mask, dtype=torch.bool)
    label_mask = torch.zeros(len(canvas.canvas_ids), dtype=torch.bool)
    label_mask[torch.tensor(canvas.label_positions)] = True
    assert not torch.any(fixed_canvas & label_mask)

    batch = DecisionTrainingCollator(spec)([{"canvas": canvas, "source": "test"}])
    assert batch["decision_label_positions"][0, 0].item() == (
        len(canvas.prompt_ids) + canvas.label_positions[0]
    )
    fixed_positions = (
        torch.arange(len(plan.ids))
        if mode == "prompt"
        else torch.where(fixed_canvas)[0] + len(canvas.prompt_ids)
    )
    batch["canvas_corruptible_mask"][0, fixed_positions] = True
    batch["canvas_update_mask"][0, fixed_positions] = True
    trainer = _TrainerHarness(spec, {"latent": _slot_config(mode)})
    trainer._validate_decision_config(trainer._decision_config(), k_max=1, spec=spec)
    train_model = _EchoModel()
    trainer._full_sequence_logits(train_model, batch, spec, trainer._decision_config())
    assert train_model.seen_input_ids is not None
    expected = torch.tensor(
        (
            plan.ids
            if mode == "prompt"
            else torch.tensor(canvas.canvas_ids)[fixed_canvas].tolist()
        ),
        dtype=torch.long,
    )
    torch.testing.assert_close(train_model.seen_input_ids[0, fixed_positions], expected)

    read_model = _EchoModel()
    read = HFReader(vocab_size=256, mask_token_id=9, attention_backend="dense").read(
        read_model, spec, canvas, steps=1, seed=17, diagnostics=True
    )
    assert read.diagnostics is not None
    if mode == "prompt":
        assert read_model.seen_input_ids is not None
        torch.testing.assert_close(
            read_model.seen_input_ids[0, : len(plan.ids)], expected
        )
    else:
        positions = torch.where(fixed_canvas)[0]
        clean = torch.tensor(canvas.canvas_ids)
        torch.testing.assert_close(
            read.diagnostics.initial_canvas_ids[positions], clean[positions]
        )
        torch.testing.assert_close(
            read.diagnostics.final_canvas_ids[positions], clean[positions]
        )


@pytest.mark.parametrize("mode", ["pad", "pinned", "learned", "mask"])
def test_reader_rejects_initial_canvas_override_of_prepared_fixed_slot(mode):
    noise = DiffusionNoise.ABSORBING if mode == "mask" else DiffusionNoise.UNIFORM
    spec, _plan, canvas = _prepared_canvas(mode, noise)
    override = torch.tensor(canvas.canvas_ids)
    fixed = torch.where(torch.tensor(canvas.slot_mask))[0][0]
    override[fixed] = (override[fixed] + 1) % 256

    with pytest.raises(ValueError, match="cannot alter pinned"):
        HFReader(vocab_size=256, mask_token_id=9, attention_backend="dense").read(
            _EchoModel(),
            spec,
            canvas,
            steps=1,
            initial_canvas_ids=override,
        )


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_tiny_native_dream_mask_slots_are_pinned_and_train_through_lora(
    monkeypatch: pytest.MonkeyPatch, steps: int
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig

    from axolotl.model_support.dream import _model_class

    from tests.native_source_fixtures import native_source_fixture_path

    source = native_source_fixture_path("dream")
    if source is None:
        pytest.skip("native Dream source fixture unavailable")
    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    for name, value in {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 128,
        "max_window_layers": 1,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "mask_token_id": 9,
        "use_cache": False,
    }.items():
        setattr(config, name, value)
    config._name_or_path = str(source)
    model = (
        cast(Any, _model_class()).from_config(config, trust_remote_code=True).train()
    )
    model = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    )
    spec, plan, canvas = _prepared_canvas("mask", DiffusionNoise.ABSORBING)
    spec = replace(spec, logit_alignment=LogitAlignment.SHIFTED)
    batch = DecisionTrainingCollator(spec)([{"canvas": canvas, "source": "test"}])
    slots = torch.where(torch.tensor(canvas.slot_mask))[0] + len(canvas.prompt_ids)
    assert batch["canvas_input_pinned_mask"][0, slots].all()
    assert not batch["canvas_loss_mask"][0, slots].any()
    assert not batch["canvas_corruptible_mask"][0, slots].any()
    assert not batch["canvas_update_mask"][0, slots].any()
    seen: list[torch.Tensor] = []
    raw_logits: list[torch.Tensor] = []
    selected_inputs: list[torch.Tensor] = []
    selected_outputs: list[torch.Tensor] = []

    def capture_input(_module, args, kwargs):
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
        seen.append(input_ids.detach().clone())

    def capture_output(_module, _args, _kwargs, output):
        raw_logits.append(output.logits.detach().clone())

    original_select = DiffusionDecisionTrainer._select_question_logits

    def capture_selected(logits, inputs, *, coordinates=None):
        selected_inputs.append(logits.detach().clone())
        selected, supervision = original_select(logits, inputs, coordinates=coordinates)
        selected_outputs.append(selected.detach().clone())
        return selected, supervision

    monkeypatch.setattr(
        DiffusionDecisionTrainer,
        "_select_question_logits",
        staticmethod(capture_selected),
    )
    base_model = model.get_base_model()
    input_hook = base_model.register_forward_pre_hook(capture_input, with_kwargs=True)
    output_hook = base_model.register_forward_hook(capture_output, with_kwargs=True)

    class KStepHarness(_TrainerHarness):
        @staticmethod
        def _native_unroll_settings():
            return steps, False

        @staticmethod
        def _sample_native_unroll_steps(k_max, device):
            del k_max, device
            return steps

    trainer = KStepHarness(spec, {"latent": _slot_config("mask")})
    try:
        loss = cast(torch.Tensor, trainer.compute_loss(model, batch))
    finally:
        input_hook.remove()
        output_hook.remove()
    assert torch.isfinite(loss)
    assert len(seen) == len(raw_logits) == steps
    assert selected_inputs and selected_outputs
    assert all(
        torch.equal(value[0, slots], torch.full_like(slots, 9)) for value in seen
    )
    labels = torch.tensor(canvas.label_positions) + len(canvas.prompt_ids)
    assert all(
        torch.equal(value[0, labels], torch.full_like(labels, 9)) for value in seen
    )
    pinned = torch.tensor(canvas.pinned_mask, dtype=torch.bool)
    scaffold = torch.where(pinned)[0] + len(canvas.prompt_ids)
    expected_scaffold = torch.tensor(canvas.canvas_ids)[pinned]
    assert all(torch.equal(value[0, scaffold], expected_scaffold) for value in seen)
    torch.testing.assert_close(seen[-1][0, slots], torch.full_like(slots, 9))
    expected_aligned = torch.cat([raw_logits[-1][:, :1], raw_logits[-1][:, :-1]], dim=1)
    torch.testing.assert_close(selected_inputs[-1], expected_aligned)
    rows = batch["decision_label_rows"].long()
    positions = batch["decision_label_positions"].long()
    expected_selected = expected_aligned[rows.clamp_min(0), positions.clamp_min(0)]
    torch.testing.assert_close(selected_outputs[-1], expected_selected)
    decision = trainer._decision_config()
    expected_loss = decision_label_loss(
        selected_outputs[-1],
        batch["decision_examples"],
        batch["decision_supervision_mask"],
        label_softmax=decision.labels.label_softmax,
        brier_weight=decision.labels.brier_weight,
    ).loss
    torch.testing.assert_close(loss.detach(), expected_loss.detach())
    loss.backward()
    lora_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    assert lora_grads and all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in lora_grads
    )
    assert any(gradient.abs().sum() > 0 for gradient in lora_grads)
