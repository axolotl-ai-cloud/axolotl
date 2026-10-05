"""CUDA smoke coverage for native Nemotron free-slot decision recurrence."""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from axolotl.core.trainers.diffusion_lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.integrations.diffusion_decision.loss import (
    DecisionLabelExample,
    DecisionLabelQuestion,
    HardLabel,
)
from axolotl.model_support import DiffusionLayout, LogitAlignment
from axolotl.model_support.nemotron_diffusion.compat import (
    resolve_nemotron_model_class,
)

from tests.integrations.diffusion_decision.helpers import (
    MultistepTrainerHarness,
    trainer_spec,
)
from tests.native_source_fixtures import native_source_fixture_path


def _model(device: torch.device):
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
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
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
    base = resolve_nemotron_model_class(str(source))(config).to(
        device=device, dtype=torch.bfloat16
    )
    traces: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_forward = base.forward

    def traced_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        traces.append(
            (kwargs["input_ids"].detach().clone(), result.logits.detach().clone())
        )
        return result

    base.forward = traced_forward
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj"],
            r=2,
            lora_alpha=2,
        ),
    ).train()
    return model, traces, source


def _inputs(device: torch.device) -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[1, 7, 11, 8, 2, 3, 9, 12, 10, 4]], device=device),
        "document_ids": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], device=device),
        "semantic_validity": torch.ones((1, 10), dtype=torch.bool, device=device),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], device=device),
        "canvas_corruptible_mask": torch.tensor(
            [[False, False, True, False, False, False, False, True, False, False]],
            device=device,
        ),
        "canvas_input_pinned_mask": torch.tensor(
            [[False, False, False, True, True, True, False, False, True, True]],
            device=device,
        ),
        "canvas_update_mask": torch.zeros((1, 10), dtype=torch.bool, device=device),
        "decision_slot_mask": torch.tensor(
            [[False, True, False, True, False, False, True, False, False, False]],
            device=device,
        ),
        "canvas_loss_mask": torch.tensor(
            [[False, False, True, False, False, False, False, True, False, False]],
            device=device,
        ),
        "decision_examples": (
            DecisionLabelExample(
                questions=(DecisionLabelQuestion(0, (1, 2, 3), HardLabel(0)),)
            ),
            DecisionLabelExample(
                questions=(DecisionLabelQuestion(0, (4, 5, 6), HardLabel(1)),)
            ),
        ),
        "decision_question_mask": torch.tensor([[True], [True]], device=device),
        "decision_supervision_mask": torch.tensor([[True], [True]], device=device),
        "decision_label_rows": torch.tensor([[0], [0]], device=device),
        "decision_label_positions": torch.tensor([[2], [7]], device=device),
    }


@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_nemotron_varlen_free_slots_evolve_without_slot_loss(monkeypatch, steps):
    from axolotl.core.trainers.diffusion_lm import varlen

    device = torch.device("cuda")
    torch.manual_seed(314159)
    torch.cuda.reset_peak_memory_stats(device)
    kernel_calls = []
    original_kernel = varlen.varlen_attn

    def traced_kernel(*args, **kwargs):
        kernel_calls.append(args[3].detach().clone())
        return original_kernel(*args, **kwargs)

    monkeypatch.setattr(varlen, "varlen_attn", traced_kernel)
    model, traces, source = _model(device)
    trainer = MultistepTrainerHarness(
        trainer_spec(DiffusionLayout.FULL_SEQUENCE, LogitAlignment.ALIGNED),
        k_max=steps,
        sampled_steps=steps,
        decision={
            "latent": {
                "mode": "free",
                "num_slots": 1,
                "free_update_policy": "argmax",
            }
        },
    )
    trainer._full_sequence_backend = lambda: FullSequenceBackend(
        mask_token_id=100, attention_backend="varlen"
    )
    inputs = _inputs(device)
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    torch.cuda.synchronize(device)

    slots = torch.tensor([1, 6], device=device)
    pinned_slots = torch.tensor([3], device=device)
    held = torch.tensor([0, 2, 3, 4, 5, 7, 8, 9], device=device)
    metadata = FullSequenceBackend(mask_token_id=100, attention_backend="varlen").pack(
        inputs["input_ids"], inputs["document_ids"], inputs["semantic_validity"]
    )["diffusion_varlen"]
    assert metadata is not None
    assert metadata.cu_seqlens.tolist() == [0, 5, 10]
    assert len(kernel_calls) == steps
    assert all(cu_seqlens.tolist() == [0, 5, 10] for cu_seqlens in kernel_calls)
    assert len(traces) == steps
    assert all(
        torch.equal(state[:, held], traces[0][0][:, held]) for state, _ in traces
    )
    assert not torch.any(inputs["decision_slot_mask"] & inputs["canvas_loss_mask"])
    assert all(
        torch.equal(state[:, pinned_slots], traces[0][0][:, pinned_slots])
        for state, _ in traces
    )
    for (_state, logits), (next_state, _) in zip(traces[:-1], traces[1:], strict=True):
        torch.testing.assert_close(next_state[:, slots], logits.argmax(-1)[:, slots])
    assert torch.any(traces[1][0][:, slots] != traces[0][0][:, slots])
    lora_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if ".lora_" in name and parameter.requires_grad
    ]
    assert torch.isfinite(loss)
    assert lora_grads
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in lora_grads
    )
    assert any(torch.count_nonzero(gradient) for gradient in lora_grads)
    print(
        "native_nemotron_free_varlen"
        f" steps={steps} fixture={source} torch={torch.__version__}"
        f" peak_bytes={torch.cuda.max_memory_allocated(device)}"
    )
