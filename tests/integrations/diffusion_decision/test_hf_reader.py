"""CPU coverage for the spec-driven in-process decision reader."""

from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import axolotl.model_support.diffusion_gemma.modeling as gemma_modeling
from axolotl.core.trainers.diffusion_lm import varlen
from axolotl.core.trainers.diffusion_lm.backends.encoder_canvas import (
    EncoderCanvasBackend,
)
from axolotl.core.trainers.diffusion_lm.unroll import run_unroll
from axolotl.integrations.diffusion_decision.readers import HFReader
from axolotl.integrations.diffusion_decision.records import DecisionCanvas
from axolotl.model_support.diffusion import (
    DiffusionLayout,
    DiffusionNoise,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
)
from axolotl.model_support.dream import _model_class as dream_model_class
from axolotl.model_support.nemotron_diffusion import NemotronDiffusionSupport
from axolotl.model_support.nemotron_diffusion.compat import (
    resolve_nemotron_model_class,
)

from tests.integrations.diffusion_decision.helpers import (
    EchoModel,
    make_canvas,
    make_spec,
    tiny_gemma_config,
)
from tests.native_source_fixtures import native_source_fixture_path

_READER_KEYS = frozenset(
    {"attention_backend", "vocab_size", "mask_token_id", "free_update_policy"}
)


def _spec(layout: DiffusionLayout, noise: DiffusionNoise, *, shifted: bool = False):
    return make_spec(
        noise=noise,
        layout=layout,
        logit_alignment=(LogitAlignment.SHIFTED if shifted else LogitAlignment.ALIGNED),
        first_position_alignment=(
            FirstPositionAlignment.DUPLICATE_FIRST
            if shifted
            else FirstPositionAlignment.REQUIRES_PREDECESSOR
        ),
        max_canvas=256 if layout is DiffusionLayout.ENCODER_CANVAS else None,
        max_context=None,
        generation_adapter=GenerationAdapter.ENCODER_CANVAS,
    )


FULL_ABSORBING = _spec(DiffusionLayout.FULL_SEQUENCE, DiffusionNoise.ABSORBING)
FULL_UNIFORM = _spec(DiffusionLayout.FULL_SEQUENCE, DiffusionNoise.UNIFORM)
ENCODER_UNIFORM = _spec(DiffusionLayout.ENCODER_CANVAS, DiffusionNoise.UNIFORM)


def _read(model, spec, canvas, **kwargs):
    reader = {"attention_backend": "dense"}
    reader.update(
        {key: kwargs.pop(key) for key in tuple(kwargs) if key in _READER_KEYS}
    )
    return HFReader(**reader).read(model, spec, canvas, diagnostics=True, **kwargs)


def _canvas() -> DecisionCanvas:
    return make_canvas(
        (3, 4, 5),
        (6, 7, 8, 9, 0, 0, 0, 0),
        (1, 3),
        allowed_ids=((1, 2), (3, 4, 5)),
        question_ids=("q0", "q1"),
        targets=(0, 1),
        pinned_mask=(True, False, True, False, True, True, True, True),
        template_length=4,
    )


def _free_canvas(canvas_ids=(9, 6, 7, 8, 0, 0, 0, 0), **fields) -> DecisionCanvas:
    values: dict[str, Any] = {
        "allowed_ids": ((1, 2), (3, 4)),
        "question_ids": ("q0", "q1"),
        "targets": (0, 0),
        "pinned_mask": (False, False, True, False, True, True, True, True),
        "slot_mask": (True, False, False, False, False, False, False, False),
        "template_length": 4,
    }
    values.update(fields)
    return make_canvas((3, 4), canvas_ids, (1, 3), **values)


def _held_slot_canvas() -> DecisionCanvas:
    return _free_canvas(pinned_mask=(True, False, False, False, True, True, True, True))


def _three_slot_canvas() -> DecisionCanvas:
    return _free_canvas(
        (9, 6, 7, 8, 4, 5, 0, 0),
        pinned_mask=(False, False, False, False, False, False, True, True),
        slot_mask=(True, False, True, False, True, False, False, False),
    )


def _single_label_canvas(allowed=(2, 6), target: int = 0) -> DecisionCanvas:
    return make_canvas(
        (3, 4),
        (6, 7, 8, 9),
        (1,),
        allowed_ids=(tuple(allowed),),
        question_ids=("q",),
        targets=(target,),
        pinned_mask=(True, False, True, True),
        template_length=3,
    )


def _ragged_peer(canvas: DecisionCanvas) -> DecisionCanvas:
    return replace(
        canvas,
        prompt_ids=(11, 12),
        canvas_ids=(10, 9, 8, 7, 0, 0),
        pinned_mask=(True, False, True, False, True, True),
        semantic_mask=(True,) * 6,
        slot_mask=(False,) * 6,
        question_ids=("q2", "q3"),
    )


def _assert_reads_match(expected, actual, **tolerance) -> None:
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(
            left.full_vocab_logprobs, right.full_vocab_logprobs, **tolerance
        )
        torch.testing.assert_close(
            left.restricted_probs, right.restricted_probs, **tolerance
        )
        assert left.diagnostics is not None and right.diagnostics is not None
        torch.testing.assert_close(
            left.diagnostics.initial_canvas_ids, right.diagnostics.initial_canvas_ids
        )
        torch.testing.assert_close(
            left.diagnostics.final_canvas_ids, right.diagnostics.final_canvas_ids
        )


class _FullSequenceEcho(EchoModel):
    def __init__(self) -> None:
        super().__init__(vocab_size=16, mask_token_id=2, scale=20)


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


class _EncoderStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            vocab_size=16,
            sliding_window=8,
            text_config=SimpleNamespace(vocab_size=16, sliding_window=8),
        )


def _patch_encoder_forward(monkeypatch, forward) -> _EncoderStub:
    monkeypatch.setattr(EncoderCanvasBackend, "forward", forward)
    return _EncoderStub()


def _capturing_encoder(monkeypatch) -> tuple[_EncoderStub, dict]:
    captured: dict = {}

    def forward(self, model, packed, input_ids, **kwargs):
        del self, model
        captured.update(kwargs)
        return SimpleNamespace(
            logits=torch.zeros(
                (*input_ids.shape, 16), device=input_ids.device, dtype=torch.float32
            ),
            denoised_input_ids=input_ids,
        )

    return _patch_encoder_forward(monkeypatch, forward), captured


@pytest.fixture
def gemma_model():
    torch.manual_seed(7)
    return gemma_modeling.AxolotlDiffusionGemmaForBlockDiffusion(
        tiny_gemma_config()
    ).eval()


def _nemotron_model():
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
    return resolve_nemotron_model_class(str(source))(config).eval()


def _nemotron_spec():
    spec = NemotronDiffusionSupport.profile.diffusion
    assert spec is not None
    return spec


def _nemotron_reference(model, canvas: DecisionCanvas):
    expected_canvas = torch.tensor(canvas.canvas_ids)
    expected_canvas[list(canvas.label_positions)] = 100
    ids = torch.cat((torch.tensor(canvas.prompt_ids), expected_canvas))[None]
    with torch.inference_mode():
        reference = model(input_ids=ids, use_cache=False, use_causal_mask=False).logits
    positions = len(canvas.prompt_ids) + torch.tensor(canvas.label_positions)
    return expected_canvas, ids, reference[0, positions].float().log_softmax(-1)


def _patch_varlen(monkeypatch, model) -> list:
    calls: list = []

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
    return calls


@pytest.mark.parametrize("steps", (1, 2))
@pytest.mark.parametrize(
    "backend", ("dense", "varlen"), ids=("full_sequence_echo", "pinned_nemotron")
)
def test_hf_reader_batch_matches_independent_reads_with_ragged_peers(
    monkeypatch, backend: str, steps: int
):
    if backend == "varlen":
        model = batched_model = _nemotron_model()
        _patch_varlen(monkeypatch, model)
        spec, tolerance = _nemotron_spec(), {"rtol": 1e-5, "atol": 1e-6}
    else:
        model, batched_model = _FullSequenceEcho(), _FullSequenceEcho()
        spec, tolerance = FULL_ABSORBING, {}
    canvas = _canvas()
    other = _ragged_peer(canvas)
    reader = HFReader(attention_backend=backend)
    hold = steps == 2
    expected = tuple(
        reader.read(
            model,
            spec,
            item,
            steps=steps,
            seed=seed,
            diagnostics=True,
            hold_label_noise=hold,
        )
        for item, seed in ((canvas, 11), (other, 12))
    )
    actual = reader.read_batch(
        batched_model,
        spec,
        (canvas, other),
        steps=steps,
        seeds=(11, 12),
        diagnostics=True,
        hold_label_noise=hold,
    )

    _assert_reads_match(expected, actual, **tolerance)
    if backend == "dense":
        assert batched_model.calls == steps


def test_hf_reader_uniform_encoder_canvas_is_seeded_and_pins_template_tokens(
    gemma_model,
):
    canvas = _canvas()
    first, second, changed = (
        _read(gemma_model, ENCODER_UNIFORM, canvas, steps=2, seed=seed)
        for seed in (19, 19, 23)
    )

    assert first.full_vocab_logprobs.shape == (2, 32)
    assert torch.equal(first.full_vocab_logprobs, second.full_vocab_logprobs)
    assert torch.equal(
        first.diagnostics.initial_canvas_ids, second.diagnostics.initial_canvas_ids
    )
    label_mask = torch.zeros(8, dtype=torch.bool)
    label_mask[torch.tensor(canvas.label_positions)] = True
    clean = torch.tensor(canvas.canvas_ids)
    assert torch.equal(
        first.diagnostics.initial_canvas_ids[~label_mask], clean[~label_mask]
    )
    assert not torch.equal(
        first.diagnostics.initial_canvas_ids[label_mask],
        changed.diagnostics.initial_canvas_ids[label_mask],
    )
    assert first.diagnostics.forward_count == 2
    assert torch.allclose(first.restricted_probs.sum(-1), torch.ones(2))
    assert torch.equal(
        first.candidate_mask, torch.tensor([[True, True, False], [True, True, True]])
    )


def test_hf_reader_k1_uses_one_actual_decoder_read(monkeypatch, gemma_model):
    calls = []
    original = gemma_modeling.decode_packed_canvas

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(gemma_modeling, "decode_packed_canvas", counted)
    result = _read(gemma_model, ENCODER_UNIFORM, _canvas(), steps=1, seed=19)

    assert len(calls) == 1
    assert result.diagnostics.forward_count == 1


def test_hf_reader_held_label_noise_keeps_slots_in_recurrent_sc_only(monkeypatch):
    model, captured = _capturing_encoder(monkeypatch)
    result = _read(
        model,
        ENCODER_UNIFORM,
        _held_slot_canvas(),
        steps=2,
        seed=7,
        hold_label_noise=True,
    )

    assert result.diagnostics is not None
    assert result.diagnostics.update_policy == "held_label_noise_with_sc"
    assert torch.equal(
        result.diagnostics.initial_canvas_ids, result.diagnostics.final_canvas_ids
    )
    assert not captured["k1_conditioning_mask"][0, 0]
    assert captured["recurrent_conditioning_mask"][0, 0]
    assert captured["recurrent_conditioning_mask"][0, 1]
    assert not captured["update_mask"].any()


def test_hf_reader_fixed_label_noise_remains_no_sc_ablation(monkeypatch):
    model, captured = _capturing_encoder(monkeypatch)
    _read(
        model,
        ENCODER_UNIFORM,
        _held_slot_canvas(),
        steps=2,
        seed=7,
        fixed_label_noise=True,
    )

    assert not captured["k1_conditioning_mask"].any()
    assert not captured["recurrent_conditioning_mask"].any()


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
    spec = _spec(
        DiffusionLayout.FULL_SEQUENCE, DiffusionNoise.ABSORBING, shifted=shifted
    )
    result = _read(model, spec, _single_label_canvas(), steps=2, seed=11)

    assert model.calls == 2
    assert result.diagnostics is not None
    assert result.diagnostics.initial_canvas_ids[1].item() == 2
    assert result.full_vocab_logprobs.argmax(-1).tolist() == [expected_token]
    assert result.diagnostics.final_canvas_ids[1].item() == expected_token
    assert result.diagnostics.forward_count == 2


def test_hf_reader_normalizes_151_candidates_and_selects_final_index():
    result = _read(
        _FullSequence151(),
        FULL_ABSORBING,
        _single_label_canvas(allowed=range(151), target=150),
        vocab_size=151,
        mask_token_id=2,
        steps=1,
        seed=11,
    )

    assert result.allowed_ids.tolist() == [list(range(151))]
    torch.testing.assert_close(result.restricted_probs.sum(-1), torch.ones(1))
    assert result.restricted_probs.argmax(-1).item() == 150


def test_hf_reader_rejects_nonsemantic_label_position():
    broken = replace(
        _canvas(), semantic_mask=(True, False, True, True, True, True, True, True)
    )
    with pytest.raises(ValueError, match="semantically valid"):
        _read(_FullSequenceEcho(), FULL_ABSORBING, broken)


def test_hf_reader_fixed_label_noise_is_an_explicit_read_only_control():
    result = _read(
        _FullSequenceEcho(),
        FULL_ABSORBING,
        _single_label_canvas(),
        steps=2,
        fixed_label_noise=True,
    )

    assert result.diagnostics is not None
    assert result.diagnostics.update_policy == "fixed_label_noise"
    assert torch.equal(
        result.diagnostics.initial_canvas_ids, result.diagnostics.final_canvas_ids
    )


@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize(
    "layout", [DiffusionLayout.FULL_SEQUENCE, DiffusionLayout.ENCODER_CANVAS]
)
def test_hf_reader_free_read_updates_only_slots_and_holds_labels(
    monkeypatch, layout, steps
):
    captured = {}

    def forward(self, model, packed, input_ids, *, unroll_steps, update_mask, **kwargs):
        del self, model, kwargs
        captured["update_mask"] = update_mask.detach().clone()
        final = input_ids.clone()
        final[update_mask] = (final[update_mask] + unroll_steps) % 16
        logits = torch.nn.functional.one_hot(final, num_classes=16).float() * 20
        return SimpleNamespace(logits=logits, denoised_input_ids=final)

    if layout is DiffusionLayout.ENCODER_CANVAS:
        model = _patch_encoder_forward(monkeypatch, forward)
    else:
        model = _FullSequenceIncrement()
    canvas = _free_canvas()
    result = _read(
        model,
        _spec(layout, DiffusionNoise.UNIFORM),
        canvas,
        free_update_policy="argmax",
        steps=steps,
        initial_canvas_ids=torch.tensor(canvas.canvas_ids),
    )

    labels = torch.tensor(canvas.label_positions)
    assert result.diagnostics is not None
    assert result.diagnostics.update_policy == "free_slot_argmax_held_labels"
    assert result.diagnostics.forward_count == steps
    assert torch.equal(
        result.diagnostics.initial_canvas_ids[labels],
        result.diagnostics.final_canvas_ids[labels],
    )
    assert result.diagnostics.final_canvas_ids[0].item() == (9 + steps) % 16
    assert torch.equal(
        result.diagnostics.initial_canvas_ids[2:],
        result.diagnostics.final_canvas_ids[2:],
    )
    if layout is DiffusionLayout.ENCODER_CANVAS:
        assert captured["update_mask"].tolist() == [[True] + [False] * 7]


def test_hf_reader_runs_tiny_pinned_dream_source():
    source = native_source_fixture_path("dream")
    if source is None:
        pytest.skip("native Dream source fixture unavailable")
    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    config.update(
        {
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "max_window_layers": 1,
            "bos_token_id": 1,
            "eos_token_id": 1,
            "pad_token_id": 1,
            "mask_token_id": 2,
            "use_cache": False,
        }
    )
    config._name_or_path = str(source)
    model = (
        dream_model_class()
        .from_config(config, trust_remote_code=True, torch_dtype=torch.float32)
        .eval()
    )
    canvas = make_canvas(
        (3, 4, 5),
        (6, 7, 8, 9, 1, 1, 1, 1),
        (1,),
        allowed_ids=((2, 6, 7),),
        question_ids=("dream",),
        targets=(0,),
        pinned_mask=(True, False, True, True, True, True, True, True),
        template_length=4,
    )
    spec = _spec(DiffusionLayout.FULL_SEQUENCE, DiffusionNoise.ABSORBING, shifted=True)
    result = _read(model, spec, canvas, steps=1)

    assert result.diagnostics is not None
    assert result.full_vocab_logprobs.shape == (1, 32)
    assert result.diagnostics.forward_count == 1
    assert torch.isfinite(result.full_vocab_logprobs).all()


def test_hf_reader_free_seeded_initialization_and_explicit_override():
    canvas = _free_canvas()
    seeded, override = (
        _read(
            _FullSequenceIncrement(),
            FULL_UNIFORM,
            canvas,
            free_update_policy="argmax",
            steps=2,
            seed=7,
            initial_canvas_ids=initial,
        )
        for initial in (None, torch.tensor(canvas.canvas_ids))
    )
    assert seeded.diagnostics is not None and override.diagnostics is not None
    assert seeded.diagnostics.slot_init_policy == "fresh_read_seed_v1"
    assert override.diagnostics.slot_init_policy == "explicit_canvas_v1"
    assert override.diagnostics.initial_canvas_ids[0].item() == canvas.canvas_ids[0]


@pytest.mark.parametrize(
    "layout", [DiffusionLayout.FULL_SEQUENCE, DiffusionLayout.ENCODER_CANVAS]
)
def test_hf_reader_free_seeded_slots_are_reproducible_in_each_layout(
    monkeypatch, layout
):
    canvas = _three_slot_canvas()
    if layout is DiffusionLayout.ENCODER_CANVAS:

        def forward(self, model, packed, input_ids, **kwargs):
            del self, model, packed, kwargs
            logits = torch.nn.functional.one_hot(input_ids, num_classes=16).float()
            return SimpleNamespace(logits=logits, denoised_input_ids=input_ids)

        model = _patch_encoder_forward(monkeypatch, forward)
    else:
        model = _FullSequenceEcho()
    spec = _spec(layout, DiffusionNoise.UNIFORM)
    first, second, changed = (
        _read(model, spec, canvas, free_update_policy="argmax", steps=2, seed=seed)
        for seed in (71, 71, 72)
    )

    slots = torch.tensor(canvas.slot_mask)
    assert all(read.diagnostics is not None for read in (first, second, changed))
    assert torch.equal(
        first.diagnostics.initial_canvas_ids, second.diagnostics.initial_canvas_ids
    )
    assert not torch.equal(
        first.diagnostics.initial_canvas_ids[slots],
        changed.diagnostics.initial_canvas_ids[slots],
    )


def test_hf_reader_free_unseeded_slots_draw_fresh_rng_and_absorbing_uses_override():
    canvas = _three_slot_canvas()
    first, second, absorbing = (
        _read(
            _FullSequenceEcho(),
            spec,
            canvas,
            free_update_policy="argmax",
            mask_token_id=11,
            steps=2,
        )
        for spec in (FULL_UNIFORM, FULL_UNIFORM, FULL_ABSORBING)
    )

    slots = torch.tensor(canvas.slot_mask)
    assert all(read.diagnostics is not None for read in (first, second, absorbing))
    assert not torch.equal(
        first.diagnostics.initial_canvas_ids[slots],
        second.diagnostics.initial_canvas_ids[slots],
    )
    assert torch.equal(
        absorbing.diagnostics.initial_canvas_ids[slots], torch.full((3,), 11)
    )


def test_hf_reader_fixed_mode_does_not_draw_free_slot_rng(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("fixed modes must not initialize free slots")

    monkeypatch.setattr(torch, "randint", fail)
    result = _read(_FullSequenceEcho(), FULL_UNIFORM, _canvas(), steps=2, seed=7)
    assert result.diagnostics is not None
    assert result.diagnostics.slot_init_policy == "prepared_canvas_v0"


def test_hf_reader_defaults_to_flex_attention():
    assert HFReader().attention_backend == "flex_attention"


def test_hf_reader_varlen_rejects_encoder_canvas_before_model_execution():
    with pytest.raises(ValueError, match="only full-sequence diffusion"):
        HFReader(attention_backend="varlen").read(
            _FullSequenceEcho(), ENCODER_UNIFORM, _canvas()
        )


@pytest.mark.parametrize("backend", ("dense", "varlen"))
def test_hf_reader_matches_pinned_nemotron_bidirectional_forward(
    monkeypatch, backend: str
):
    model = _nemotron_model()
    canvas = _canvas()
    expected_canvas, ids, expected = _nemotron_reference(model, canvas)
    calls = _patch_varlen(monkeypatch, model) if backend == "varlen" else None

    result = _read(model, _nemotron_spec(), canvas, attention_backend=backend)

    assert result.diagnostics is not None
    torch.testing.assert_close(result.diagnostics.initial_canvas_ids, expected_canvas)
    torch.testing.assert_close(
        result.full_vocab_logprobs, expected, rtol=1e-5, atol=1e-6
    )
    assert result.diagnostics.forward_count == 1
    if calls is not None:
        assert len(calls) == model.config.num_hidden_layers
        assert calls[0][0].tolist() == calls[0][1].tolist() == [0, ids.shape[1]]
        assert calls[0][2:] == (ids.shape[1], ids.shape[1])
