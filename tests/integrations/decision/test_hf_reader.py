"""CPU coverage for the spec-driven in-process decision reader."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from axolotl.integrations.decision.readers import HFReader
from axolotl.integrations.decision.records import DecisionCanvas
from axolotl.integrations.diffusion.lm.unroll import run_unroll
from axolotl.model_support.diffusion import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    TimeWeighting,
)


def _spec(*, layout: DiffusionLayout, noise: DiffusionNoise, shifted: bool = False):
    return DiffusionSpec(
        noise=noise,
        layout=layout,
        logit_alignment=(LogitAlignment.SHIFTED if shifted else LogitAlignment.ALIGNED),
        first_position_alignment=(
            FirstPositionAlignment.DUPLICATE_FIRST
            if shifted
            else FirstPositionAlignment.REQUIRES_PREDECESSOR
        ),
        self_conditioning=layout is DiffusionLayout.ENCODER_CANVAS,
        max_canvas=256 if layout is DiffusionLayout.ENCODER_CANVAS else None,
        max_context=None,
        eos_handling=EosHandling.INDEPENDENT,
        mask_token_policy=(
            MaskTokenPolicy.NONE
            if noise is DiffusionNoise.UNIFORM
            else MaskTokenPolicy.MODEL
        ),
        default_time_weighting=TimeWeighting.NONE,
        objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
        generation_adapter=GenerationAdapter.ENCODER_CANVAS,
    )


def _canvas() -> DecisionCanvas:
    return DecisionCanvas(
        prompt_ids=(3, 4, 5),
        canvas_ids=(6, 7, 8, 9, 0, 0, 0, 0),
        label_positions=(1, 3),
        allowed_ids=((1, 2), (3, 4, 5)),
        question_ids=("q0", "q1"),
        targets=(0, 1),
        pinned_mask=(True, False, True, False, True, True, True, True),
        semantic_mask=(True, True, True, True, True, True, True, True),
        slot_mask=(False,) * 8,
        template_length=4,
    )


class _FullSequenceEcho(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=16, mask_token_id=2)
        self.calls = 0

    def forward(self, input_ids, attention_mask, position_ids, use_cache, **kwargs):
        del attention_mask, position_ids, use_cache, kwargs
        self.calls += 1
        logits = torch.nn.functional.one_hot(input_ids, num_classes=16).float() * 20
        return SimpleNamespace(logits=logits + self.anchor)


class _FullSequenceIncrement(_FullSequenceEcho):
    def forward(self, input_ids, attention_mask, position_ids, use_cache, **kwargs):
        del attention_mask, position_ids, use_cache, kwargs
        self.calls += 1
        next_ids = (input_ids + 1) % 16
        logits = torch.nn.functional.one_hot(next_ids, num_classes=16).float() * 20
        return SimpleNamespace(logits=logits + self.anchor)


class _FullSequence151(_FullSequenceEcho):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=151, mask_token_id=2)

    def forward(self, input_ids, attention_mask, position_ids, use_cache, **kwargs):
        del attention_mask, position_ids, use_cache, kwargs
        self.calls += 1
        logits = torch.zeros((*input_ids.shape, 151), device=input_ids.device)
        logits[..., 150] = 20
        return SimpleNamespace(logits=logits + self.anchor)


@pytest.mark.parametrize("steps", (1, 2))
def test_hf_reader_full_sequence_batch_matches_independent_reads_with_ragged_peers(
    steps: int,
):
    canvas = _canvas()
    other = replace(
        canvas,
        prompt_ids=(11, 12),
        canvas_ids=(10, 9, 8, 7, 0, 0),
        pinned_mask=(True, False, True, False, True, True),
        semantic_mask=(True,) * 6,
        slot_mask=(False,) * 6,
        question_ids=("q2", "q3"),
    )
    spec = _spec(layout=DiffusionLayout.FULL_SEQUENCE, noise=DiffusionNoise.ABSORBING)
    individual_model = _FullSequenceEcho()
    batched_model = _FullSequenceEcho()
    reader = HFReader(attention_backend="dense")
    expected = (
        reader.read(
            individual_model,
            spec,
            canvas,
            steps=steps,
            seed=11,
            diagnostics=True,
            hold_label_noise=steps == 2,
        ),
        reader.read(
            individual_model,
            spec,
            other,
            steps=steps,
            seed=12,
            diagnostics=True,
            hold_label_noise=steps == 2,
        ),
    )
    actual = reader.read_batch(
        batched_model,
        spec,
        (canvas, other),
        steps=steps,
        seeds=(11, 12),
        diagnostics=True,
        hold_label_noise=steps == 2,
    )

    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left.full_vocab_logprobs, right.full_vocab_logprobs)
        torch.testing.assert_close(left.restricted_probs, right.restricted_probs)
        assert left.diagnostics is not None and right.diagnostics is not None
        torch.testing.assert_close(
            left.diagnostics.initial_canvas_ids, right.diagnostics.initial_canvas_ids
        )
        torch.testing.assert_close(
            left.diagnostics.final_canvas_ids, right.diagnostics.final_canvas_ids
        )
    assert batched_model.calls == steps


def test_training_k1_pilot_default_is_preserved_while_serving_can_disable_it():
    state = torch.tensor([[3, 4]])
    update_mask = torch.zeros_like(state, dtype=torch.bool)

    def count_reads(pilot_for_single_step: bool) -> int:
        calls = []

        def forward_step(current, _conditioning, _conditioning_mask):
            calls.append(current)
            return current.float()[..., None]

        run_unroll(
            state=state,
            update_mask=update_mask,
            steps=1,
            grad_through_steps=False,
            supports_self_conditioning=True,
            k1_conditioning_mask=torch.ones_like(update_mask),
            recurrent_conditioning_mask=torch.ones_like(update_mask),
            forward_step=forward_step,
            logits_from_outputs=lambda output: output,
            update_state=lambda current, _logits, _mask: current,
            pilot_for_single_step=pilot_for_single_step,
        )
        return len(calls)

    assert count_reads(True) == 2
    assert count_reads(False) == 1


@pytest.mark.parametrize(
    ("shifted", "expected_token"),
    ((False, 2), (True, 6)),
    ids=("nemotron_aligned", "dream_shifted"),
)
def test_hf_reader_full_sequence_respects_noise_alignment_and_k(
    shifted: bool, expected_token: int
):
    model = _FullSequenceEcho()
    canvas = DecisionCanvas(
        prompt_ids=(3, 4),
        canvas_ids=(6, 7, 8, 9),
        label_positions=(1,),
        allowed_ids=((2, 6),),
        question_ids=("q",),
        targets=(0,),
        pinned_mask=(True, False, True, True),
        semantic_mask=(True, True, True, True),
        slot_mask=(False,) * 4,
        template_length=3,
    )
    result = HFReader(attention_backend="dense").read(
        model,
        _spec(
            layout=DiffusionLayout.FULL_SEQUENCE,
            noise=DiffusionNoise.ABSORBING,
            shifted=shifted,
        ),
        canvas,
        steps=2,
        seed=11,
        diagnostics=True,
    )

    assert model.calls == 2
    assert result.diagnostics is not None
    assert result.diagnostics.initial_canvas_ids[1].item() == 2
    assert result.full_vocab_logprobs.argmax(-1).tolist() == [expected_token]
    assert result.diagnostics.final_canvas_ids[1].item() == expected_token
    assert result.diagnostics.forward_count == 2


def test_hf_reader_normalizes_151_candidates_and_selects_final_index():
    canvas = DecisionCanvas(
        prompt_ids=(3, 4),
        canvas_ids=(6, 7, 8, 9),
        label_positions=(1,),
        allowed_ids=(tuple(range(151)),),
        question_ids=("q",),
        targets=(150,),
        pinned_mask=(True, False, True, True),
        semantic_mask=(True,) * 4,
        slot_mask=(False,) * 4,
        template_length=3,
    )
    result = HFReader(vocab_size=151, mask_token_id=2, attention_backend="dense").read(
        _FullSequence151(),
        _spec(layout=DiffusionLayout.FULL_SEQUENCE, noise=DiffusionNoise.ABSORBING),
        canvas,
        steps=1,
        seed=11,
    )

    assert result.allowed_ids.tolist() == [list(range(151))]
    torch.testing.assert_close(result.restricted_probs.sum(-1), torch.ones(1))
    assert result.restricted_probs.argmax(-1).item() == 150


def test_hf_reader_rejects_nonsemantic_label_position():
    canvas = _canvas()
    broken = DecisionCanvas(
        **{
            **canvas.__dict__,
            "semantic_mask": (True, False, True, True, True, True, True, True),
        }
    )
    with pytest.raises(ValueError, match="semantically valid"):
        HFReader(attention_backend="dense").read(
            _FullSequenceEcho(),
            _spec(layout=DiffusionLayout.FULL_SEQUENCE, noise=DiffusionNoise.ABSORBING),
            broken,
        )


def test_hf_reader_fixed_label_noise_is_an_explicit_read_only_control():
    canvas = DecisionCanvas(
        prompt_ids=(3, 4),
        canvas_ids=(6, 7, 8, 9),
        label_positions=(1,),
        allowed_ids=((2, 6),),
        question_ids=("q",),
        targets=(0,),
        pinned_mask=(True, False, True, True),
        semantic_mask=(True, True, True, True),
        slot_mask=(False,) * 4,
        template_length=3,
    )
    result = HFReader(attention_backend="dense").read(
        _FullSequenceEcho(),
        _spec(layout=DiffusionLayout.FULL_SEQUENCE, noise=DiffusionNoise.ABSORBING),
        canvas,
        steps=2,
        fixed_label_noise=True,
        diagnostics=True,
    )

    assert result.diagnostics is not None
    assert result.diagnostics.update_policy == "fixed_label_noise"
    assert torch.equal(
        result.diagnostics.initial_canvas_ids, result.diagnostics.final_canvas_ids
    )


def test_hf_reader_defaults_to_flex_attention():
    assert HFReader().attention_backend == "flex_attention"


def test_hf_reader_varlen_rejects_encoder_canvas_before_model_execution():
    with pytest.raises(ValueError, match="only full-sequence diffusion"):
        HFReader(attention_backend="varlen").read(
            _FullSequenceEcho(),
            _spec(
                layout=DiffusionLayout.ENCODER_CANVAS,
                noise=DiffusionNoise.UNIFORM,
            ),
            _canvas(),
        )


def test_hf_reader_matches_pinned_nemotron_bidirectional_forward():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.model_support.nemotron_diffusion import NemotronDiffusionSupport
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
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = resolve_nemotron_model_class(str(source))(config).eval()
    canvas = _canvas()
    result = HFReader(attention_backend="dense").read(
        model, NemotronDiffusionSupport.profile.diffusion, canvas, diagnostics=True
    )
    assert result.diagnostics is not None
    expected_canvas = torch.tensor(canvas.canvas_ids)
    expected_canvas[list(canvas.label_positions)] = 100
    torch.testing.assert_close(result.diagnostics.initial_canvas_ids, expected_canvas)
    ids = torch.cat((torch.tensor(canvas.prompt_ids), expected_canvas))[None]
    with torch.inference_mode():
        reference = model(input_ids=ids, use_cache=False, use_causal_mask=False).logits
    positions = len(canvas.prompt_ids) + torch.tensor(canvas.label_positions)
    expected = reference[0, positions].float().log_softmax(-1)
    torch.testing.assert_close(
        result.full_vocab_logprobs, expected, rtol=1e-5, atol=1e-6
    )
    assert result.diagnostics.forward_count == 1


def test_hf_reader_varlen_matches_pinned_nemotron_per_document_forward(monkeypatch):
    import importlib

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.integrations.diffusion.lm import varlen
    from axolotl.model_support.nemotron_diffusion import NemotronDiffusionSupport
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
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = resolve_nemotron_model_class(str(source))(config).eval()
    canvas = _canvas()
    expected_canvas = torch.tensor(canvas.canvas_ids)
    expected_canvas[list(canvas.label_positions)] = 100
    ids = torch.cat((torch.tensor(canvas.prompt_ids), expected_canvas))[None]
    with torch.inference_mode():
        reference = model(input_ids=ids, use_cache=False, use_causal_mask=False).logits

    calls = []

    def cpu_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        calls.append((cu_q.clone(), cu_k.clone(), max_q, max_k))
        outputs = []
        offsets = cu_q.tolist()
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            query = q[start:end]
            key = k[start:end].repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            value = v[start:end].repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            scores = torch.einsum("qhd,khd->hqk", query, key) * kwargs["scale"]
            outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
        return torch.cat(outputs)

    def forbidden(*args, **kwargs):
        pytest.fail("HFReader varlen must not build a native dense mask")

    monkeypatch.setattr(varlen, "varlen_attn", cpu_varlen)
    native_source = importlib.import_module(type(model.encoder).__module__)
    monkeypatch.setattr(native_source, "create_causal_mask", forbidden)
    monkeypatch.setattr(native_source, "create_sliding_window_causal_mask", forbidden)

    result = HFReader(attention_backend="varlen").read(
        model,
        NemotronDiffusionSupport.profile.diffusion,
        canvas,
        diagnostics=True,
    )

    positions = len(canvas.prompt_ids) + torch.tensor(canvas.label_positions)
    expected = reference[0, positions].float().log_softmax(-1)
    torch.testing.assert_close(
        result.full_vocab_logprobs, expected, rtol=1e-5, atol=1e-6
    )
    assert result.diagnostics is not None
    assert result.diagnostics.forward_count == 1
    assert len(calls) == config.num_hidden_layers
    assert calls[0][0].tolist() == calls[0][1].tolist() == [0, ids.shape[1]]
    assert calls[0][2:] == (ids.shape[1], ids.shape[1])


@pytest.mark.parametrize("steps", (1, 2))
def test_hf_reader_varlen_batch_matches_pinned_nemotron_independent_reads(
    monkeypatch, steps: int
):
    import importlib

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.integrations.diffusion.lm import varlen
    from axolotl.model_support.nemotron_diffusion import NemotronDiffusionSupport
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
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = resolve_nemotron_model_class(str(source))(config).eval()
    first = _canvas()
    second = replace(
        first,
        prompt_ids=(11, 12),
        canvas_ids=(10, 9, 8, 7, 0, 0),
        pinned_mask=(True, False, True, False, True, True),
        semantic_mask=(True,) * 6,
        slot_mask=(False,) * 6,
        question_ids=("peer-0", "peer-1"),
    )

    def cpu_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        del cu_k, max_q, max_k
        outputs = []
        offsets = cu_q.tolist()
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            query = q[start:end]
            key = k[start:end].repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            value = v[start:end].repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            scores = torch.einsum("qhd,khd->hqk", query, key) * kwargs["scale"]
            outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
        return torch.cat(outputs)

    def forbidden(*args, **kwargs):
        pytest.fail("HFReader varlen must not build a native dense mask")

    monkeypatch.setattr(varlen, "varlen_attn", cpu_varlen)
    native_source = importlib.import_module(type(model.encoder).__module__)
    monkeypatch.setattr(native_source, "create_causal_mask", forbidden)
    monkeypatch.setattr(native_source, "create_sliding_window_causal_mask", forbidden)
    reader = HFReader(attention_backend="varlen")
    spec = NemotronDiffusionSupport.profile.diffusion
    assert spec is not None
    expected = tuple(
        reader.read(
            model,
            spec,
            canvas,
            steps=steps,
            seed=31 + index,
            hold_label_noise=steps == 2,
            diagnostics=True,
        )
        for index, canvas in enumerate((first, second))
    )
    actual = reader.read_batch(
        model,
        spec,
        (first, second),
        steps=steps,
        seeds=(31, 32),
        hold_label_noise=steps == 2,
        diagnostics=True,
    )
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(
            left.full_vocab_logprobs, right.full_vocab_logprobs, rtol=1e-5, atol=1e-6
        )
        assert left.diagnostics is not None and right.diagnostics is not None
        torch.testing.assert_close(
            left.diagnostics.initial_canvas_ids, right.diagnostics.initial_canvas_ids
        )
        torch.testing.assert_close(
            left.diagnostics.final_canvas_ids, right.diagnostics.final_canvas_ids
        )


@pytest.mark.parametrize("fixed,held", [(False, False), (True, False), (False, True)])
def test_explicit_initial_state_and_model_mode_survive_shared_read(fixed, held):
    canvas = _canvas()
    initial = torch.tensor(canvas.canvas_ids)
    initial[list(canvas.label_positions)] = 1
    spec = _spec(layout=DiffusionLayout.FULL_SEQUENCE, noise=DiffusionNoise.UNIFORM)
    reader = HFReader(attention_backend="dense")
    model = _FullSequenceIncrement().train()
    result = reader.read(
        model,
        spec,
        canvas,
        steps=2,
        initial_canvas_ids=initial,
        fixed_label_noise=fixed,
        hold_label_noise=held,
        diagnostics=True,
    )
    assert model.training
    torch.testing.assert_close(result.diagnostics.initial_canvas_ids, initial)
    expected = initial.clone()
    if not (fixed or held):
        expected[list(canvas.label_positions)] += 2
    torch.testing.assert_close(result.diagnostics.final_canvas_ids, expected)


def test_shared_reader_restores_model_mode_after_forward_error():
    model = _FullSequenceEcho().train()

    def fail(*args, **kwargs):
        raise RuntimeError("forward failed")

    model.forward = fail
    with pytest.raises(RuntimeError, match="forward failed"):
        HFReader(attention_backend="dense").read(
            model,
            _spec(layout=DiffusionLayout.FULL_SEQUENCE, noise=DiffusionNoise.UNIFORM),
            _canvas(),
        )
    assert model.training


def test_multimodal_reader_forwards_media_on_every_denoising_step():
    class ImageEcho(_FullSequenceEcho):
        def __init__(self):
            super().__init__()
            self.media = []

        def forward(self, *args, **kwargs):
            self.media.append((kwargs.pop("pixel_values"), kwargs.pop("image_sizes")))
            return super().forward(*args, **kwargs)

    canvas = replace(
        _canvas(),
        model_inputs={"pixel_values": [torch.ones(3, 2, 4)], "image_sizes": [[2, 4]]},
    )
    model = ImageEcho()
    reader = HFReader(attention_backend="dense", mask_token_id=2)
    spec = _spec(layout=DiffusionLayout.FULL_SEQUENCE, noise=DiffusionNoise.ABSORBING)
    reads = reader.read_batch(model, spec, [canvas, _canvas()], steps=2)
    assert len(reads) == 2
    assert len(model.media) == 2
    for pixels, sizes in model.media:
        assert pixels.shape == (1, 3, 2, 4)
        assert sizes.tolist() == [[2, 4]]
