"""Native encoder-canvas graph-carry coverage for decision unrolls."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from test_trainer import _MultistepTrainerHarness, _spec, _tiny_decision_gemma_config

import axolotl.model_support.diffusion_gemma.modeling as gemma_modeling
from axolotl.integrations.diffusion_decision.records import DecisionCanvas
from axolotl.integrations.diffusion_decision.training_collator import (
    DecisionTrainingCollator,
)
from axolotl.model_support import DiffusionLayout, LogitAlignment
from axolotl.model_support.diffusion_gemma.modeling import (
    AxolotlDiffusionGemmaForBlockDiffusion,
)


def _inputs():
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
    return DecisionTrainingCollator(
        _spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED)
    )([{"canvas": canvas, "source": "native"}])


@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("grad_through_steps", [False, True])
@pytest.mark.parametrize("evaluate", [False, True])
def test_native_gemma_peft_grad_through_steps_carries_intermediate_graph(
    monkeypatch, steps, grad_through_steps, evaluate
):
    torch.manual_seed(17)
    base = AxolotlDiffusionGemmaForBlockDiffusion(_tiny_decision_gemma_config())
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=r"^model\.decoder\.layers\.0\.self_attn\.q_proj$",
            task_type=None,
        ),
    ).train()
    seen = []
    states = []
    original = gemma_modeling.decode_packed_canvas

    def traced(*args, **kwargs):
        states.append(args[1].detach().clone())
        conditioning = args[5] if len(args) > 5 else None
        if conditioning is not None and conditioning.requires_grad:
            conditioning.retain_grad()
        seen.append(conditioning)
        return original(*args, **kwargs)

    monkeypatch.setattr(gemma_modeling, "decode_packed_canvas", traced)
    inputs = _inputs()
    trainer = _MultistepTrainerHarness(
        _spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        k_max=steps,
        sampled_steps=steps,
        grad_through_steps=grad_through_steps,
        decision={"latent": {"mode": "learned", "num_slots": 1}},
    )
    model.train(not evaluate)
    with torch.set_grad_enabled(not evaluate):
        loss = trainer.compute_loss(model, inputs)
    carried = [value for value in seen if value is not None]
    assert len(carried) == steps - 1
    assert len(states) == steps
    assert all(torch.equal(state, states[0]) for state in states[1:])
    assert not inputs["diffusion_batch"].canvas_loss_mask[0, 0]
    assert all(
        value.requires_grad is (grad_through_steps and not evaluate)
        for value in carried
    )
    if evaluate:
        assert not loss.requires_grad
        assert all(parameter.grad is None for parameter in model.parameters())
        return
    loss.backward()
    assert (
        any(
            value.grad is not None and torch.count_nonzero(value.grad)
            for value in carried
        )
        is grad_through_steps
    )
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for n, p in model.named_parameters()
        if ".lora_" in n
    )


def test_grad_through_steps_rejects_non_self_conditioning_layout():
    trainer = _MultistepTrainerHarness(
        _spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=2,
        sampled_steps=2,
        grad_through_steps=True,
    )
    with pytest.raises(NotImplementedError, match="encoder-canvas self-conditioning"):
        trainer.compute_loss(torch.nn.Linear(1, 1), _inputs())


def test_grad_through_steps_rejects_encoder_without_self_conditioning():
    spec = replace(
        _spec(DiffusionLayout.ENCODER_CANVAS, LogitAlignment.ALIGNED),
        self_conditioning=False,
    )
    trainer = _MultistepTrainerHarness(
        spec, k_max=2, sampled_steps=2, grad_through_steps=True
    )
    with pytest.raises(NotImplementedError, match="encoder-canvas self-conditioning"):
        trainer.compute_loss(torch.nn.Linear(1, 1), _inputs())
