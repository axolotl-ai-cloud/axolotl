"""Parity tests for fused ``process_labels``.

Each strategy gets a legacy reference implementation (verbatim copy of the
pre-refactor code) and a fuzz batch generator that injects realistic
distributions of pad/image/etc tokens. We then assert byte-identical output
between the legacy ``process_labels`` and the new fused implementation.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import List

import pytest
import torch

from axolotl.processing_strategies import (
    Gemma3nProcessingStrategy,
    Gemma3ProcessingStrategy,
    Gemma4ProcessingStrategy,
    Glm4vProcessingStrategy,
    InternVLProcessingStrategy,
    Llama3_2VisionProcessingStrategy,
    Llama4ProcessingStrategy,
    Mistral3ProcessingStrategy,
    MistralV7TekkenProcessingStrategy,
    PixtralProcessingStrategy,
    ProcessingStrategy,
    Qwen2VLProcessingStrategy,
    Qwen3_5ProcessingStrategy,
    VoxtralProcessingStrategy,
    _apply_role_boundaries,
)

# --------------------------------------------------------------------------- #
# Tokenizer / processor stubs (mirrors test_processing_strategies.py)
# --------------------------------------------------------------------------- #


class _Tokenizer:
    def __init__(
        self,
        vocab: dict,
        pad_id: int = 0,
        unk_id: int = 3,
        eos_id: int | None = None,
        extras: dict | None = None,
    ):
        self.vocab = vocab
        self.pad_token_id = pad_id
        self.unk_token_id = unk_id
        if eos_id is not None:
            self.eos_token_id = eos_id
        if extras:
            for k, v in extras.items():
                setattr(self, k, v)

    def encode(self, text, add_special_tokens=False):
        return list(self.vocab.get(text, []))

    def convert_tokens_to_ids(self, token):
        v = self.vocab.get(token)
        if v is None:
            return self.unk_token_id
        return v[0] if len(v) == 1 else self.unk_token_id


class _Processor:
    def __init__(self, tokenizer, **extras):
        self.tokenizer = tokenizer
        for k, v in extras.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------- #
# Legacy reference implementations (verbatim pre-refactor body)
# --------------------------------------------------------------------------- #
#
# These functions implement process_labels exactly as it existed before the
# fusion refactor: 3-4 sequential masked writes against the labels tensor,
# with _mask_non_assistant materializing the role mask in-place.


def _legacy_mask_non_assistant(strategy, labels):
    """Pre-refactor ``_mask_non_assistant`` body."""
    if strategy.train_on_inputs:
        return labels
    if not strategy.role_boundaries:
        return labels
    return _apply_role_boundaries(
        labels,
        strategy.role_boundaries,
        roles_to_train=set(strategy.roles_to_train),
        train_on_eos=strategy.train_on_eos,
    )


def legacy_base_process_labels(strategy, input_ids):
    labels = input_ids.clone()
    labels = _legacy_mask_non_assistant(strategy, labels)
    pad_id = getattr(strategy.processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    if strategy.image_token_id is not None:
        labels[labels == strategy.image_token_id] = -100
    return labels


def legacy_qwen3_5_process_labels(strategy, input_ids):
    labels = legacy_base_process_labels(strategy, input_ids)
    if strategy.video_token_id is not None:
        labels[labels == strategy.video_token_id] = -100
    return labels


def legacy_gemma3_process_labels(strategy, input_ids):
    labels = legacy_base_process_labels(strategy, input_ids)
    tok = strategy.processor.tokenizer
    soft_id = tok.convert_tokens_to_ids("<image_soft_token>")
    unk_id = getattr(tok, "unk_token_id", None)
    if soft_id is not None and soft_id != unk_id:
        labels[labels == soft_id] = -100
    else:
        labels[labels == 262144] = -100
    return labels


def legacy_gemma3n_process_labels(strategy, input_ids):
    labels = legacy_base_process_labels(strategy, input_ids)
    tok = strategy.processor.tokenizer
    for attr in (
        "image_token_id",
        "audio_token_id",
        "boi_token_id",
        "eoi_token_id",
    ):
        tok_id = getattr(tok, attr, None)
        if tok_id is not None:
            labels[labels == tok_id] = -100
    return labels


def legacy_gemma4_process_labels(strategy, input_ids):
    labels = legacy_base_process_labels(strategy, input_ids)
    tokenizer = strategy.processor.tokenizer
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if getattr(tokenizer, "image_token_id", None) is not None:
        labels[labels == tokenizer.image_token_id] = -100
    if getattr(tokenizer, "audio_token_id", None) is not None:
        labels[labels == tokenizer.audio_token_id] = -100
    for attr in ("boi_token", "eoi_token", "boa_token", "eoa_token"):
        token_str = getattr(strategy.processor, attr, None)
        if token_str is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is None or token_id == unk_id:
            continue
        labels[labels == token_id] = -100
    video_token_id = getattr(strategy.processor, "video_token_id", None)
    if video_token_id is not None and video_token_id != unk_id:
        labels[labels == video_token_id] = -100
    return labels


def legacy_voxtral_process_labels(strategy, input_ids):
    labels = input_ids.clone()
    labels = _legacy_mask_non_assistant(strategy, labels)
    pad_id = getattr(strategy.processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    if strategy.audio_token is not None:
        labels[labels == strategy.audio_token] = -100
    if strategy.begin_audio_token is not None:
        labels[labels == strategy.begin_audio_token] = -100
    return labels


def legacy_mistral3_process_labels(strategy, input_ids):
    labels = input_ids.clone()
    labels = _legacy_mask_non_assistant(strategy, labels)
    pad_id = getattr(strategy.processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    for tok_id in (
        strategy.image_token,
        strategy.image_break_token,
        strategy.image_end_token,
    ):
        if tok_id is not None:
            labels[labels == tok_id] = -100
    return labels


def legacy_internvl_process_labels(strategy, input_ids):
    labels = input_ids.clone()
    labels = _legacy_mask_non_assistant(strategy, labels)
    pad_id = getattr(strategy.processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    for ids in strategy.image_token_ids:
        if ids is not None:
            labels[labels == ids] = -100
    return labels


def legacy_glm4v_process_labels(strategy, input_ids):
    labels = input_ids.clone()
    labels = _legacy_mask_non_assistant(strategy, labels)
    pad_id = getattr(strategy.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    for tok_id in (
        strategy.image_token_id,
        strategy.begin_image_token_id,
        strategy.end_image_token_id,
        strategy.video_token_id,
        strategy.begin_video_token_id,
        strategy.end_video_token_id,
    ):
        if tok_id is not None:
            keep = labels != tok_id  # noqa: F841 — keep var unused; legacy uses indexed write
            labels[labels == tok_id] = -100
    return labels


# --------------------------------------------------------------------------- #
# Batch construction with realistic distributions
# --------------------------------------------------------------------------- #


def _build_alternating_turns(
    rng: random.Random,
    boundaries,
    target_len: int,
    pad_id: int,
    extra_token_ids: List[int],
    pad_rate: float = 0.07,
    extra_rate: float = 0.07,
    filler_min: int = 6,
    filler_max: int = 24,
) -> List[int]:
    """Build a single row of length ``target_len`` from ``boundaries``.

    Alternates user/assistant turns (or whatever's first/second in the list),
    with random filler IDs that may be replaced by pad / extra (image, etc)
    tokens at ``pad_rate`` / ``extra_rate``. The injection happens *inside*
    assistant spans too, which is the realistic case.
    """
    role_b = list(boundaries)
    ids: List[int] = []
    if not role_b:
        # No boundaries (Voxtral/Mistral3/InternVL/Glm4v): still inject pad and
        # media tokens so the fused vs legacy parity check exercises the
        # pad/media masking path even with role-masking opted out.
        for _ in range(target_len):
            r = rng.random()
            if r < pad_rate:
                ids.append(pad_id)
            elif r < pad_rate + extra_rate and extra_token_ids:
                ids.append(rng.choice(extra_token_ids))
            else:
                ids.append(rng.randint(10000, 30000))
        return ids

    turn = 0
    while len(ids) < target_len:
        b = role_b[turn % len(role_b)]
        ids.extend(b.start_tokens)
        n_filler = rng.randint(filler_min, filler_max)
        for _ in range(n_filler):
            r = rng.random()
            if r < pad_rate:
                ids.append(pad_id)
            elif r < pad_rate + extra_rate and extra_token_ids:
                ids.append(rng.choice(extra_token_ids))
            else:
                # Use IDs well outside the boundary token space
                ids.append(rng.randint(10000, 30000))
        if b.end_tokens:
            ids.extend(b.end_tokens)
        turn += 1
    return ids[:target_len]


def _build_batch(rng, boundaries, batch_size, seq_len, pad_id, extra_ids):
    rows = []
    for _ in range(batch_size):
        # Vary length per row a bit so we get padding-like rows occasionally
        target = rng.randint(seq_len // 2, seq_len)
        row = _build_alternating_turns(rng, boundaries, target, pad_id, extra_ids)
        # Pad up to seq_len with pad_id (mimics collator behavior)
        if len(row) < seq_len:
            row.extend([pad_id] * (seq_len - len(row)))
        rows.append(row[:seq_len])
    return torch.tensor(rows, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Per-strategy fixtures
# --------------------------------------------------------------------------- #


def _make_qwen2vl():
    vocab = {
        "<|im_start|>system\n": [101, 102, 103],
        "<|im_start|>user\n": [101, 104, 103],
        "<|im_start|>assistant\n": [101, 105, 103],
        "<|im_end|>": [106],
        "<|image_pad|>": [200],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    return Qwen2VLProcessingStrategy(_Processor(tok))


def _make_qwen3_5():
    vocab = {
        "<|im_start|>system\n": [101, 102, 103],
        "<|im_start|>user\n": [101, 104, 103],
        "<|im_start|>assistant\n": [101, 105, 103],
        "<|im_end|>": [106],
        "<|image_pad|>": [200],
        "<|video_pad|>": [201],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    return Qwen3_5ProcessingStrategy(_Processor(tok))


def _make_gemma3():
    vocab = {
        "<start_of_turn>user\n": [110, 111, 112],
        "<start_of_turn>model\n": [110, 113, 112],
        "<end_of_turn>": [114],
        "<image_soft_token>": [262144],
        "<boi>": [120],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3, extras={"boi_token": "<boi>"})
    return Gemma3ProcessingStrategy(_Processor(tok))


def _make_gemma3n():
    vocab = {
        "<start_of_turn>user\n": [110, 111, 112],
        "<start_of_turn>model\n": [110, 113, 112],
        "<end_of_turn>": [114],
    }
    tok = _Tokenizer(
        vocab,
        pad_id=0,
        unk_id=3,
        extras={
            "image_token_id": 130,
            "audio_token_id": 131,
            "boi_token_id": 132,
            "eoi_token_id": 133,
        },
    )
    return Gemma3nProcessingStrategy(_Processor(tok))


def _make_gemma4():
    vocab = {
        "<|turn>user\n": [105, 4001],
        "<|turn>system\n": [105, 4002],
        "<|turn>model\n": [105, 4368],
        "<turn|>": [106],
        "<boi>": [140],
        "<eoi>": [141],
        "<boa>": [142],
        "<eoa>": [143],
    }
    tok = _Tokenizer(
        vocab,
        pad_id=0,
        unk_id=3,
        extras={"image_token_id": 150, "audio_token_id": 151},
    )
    proc = _Processor(
        tok,
        boi_token="<boi>",
        eoi_token="<eoi>",
        boa_token="<boa>",
        eoa_token="<eoa>",
        video_token_id=160,
    )
    return Gemma4ProcessingStrategy(proc)


def _make_llama32v():
    vocab = {
        "<|start_header_id|>system<|end_header_id|>\n\n": [1001, 2002, 1002, 1003],
        "<|start_header_id|>user<|end_header_id|>\n\n": [1001, 2001, 1002, 1003],
        "<|start_header_id|>assistant<|end_header_id|>\n\n": [1001, 2003, 1002, 1003],
        "<|eot_id|>": [1004],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    return Llama3_2VisionProcessingStrategy(_Processor(tok))


def _make_llama4():
    vocab = {
        "<|header_start|>system<|header_end|>\n\n": [1001, 2002, 1002, 1003],
        "<|header_start|>user<|header_end|>\n\n": [1001, 2001, 1002, 1003],
        "<|header_start|>assistant<|header_end|>\n\n": [1001, 2003, 1002, 1003],
        "<|eot|>": [1004],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    return Llama4ProcessingStrategy(_Processor(tok))


def _make_pixtral():
    vocab = {"[INST]": [50], "[/INST]": [51]}
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3, eos_id=99)
    return PixtralProcessingStrategy(_Processor(tok))


def _make_mistral_v7():
    vocab = {
        "[INST]": [50],
        "[/INST]": [51],
        "[SYSTEM_PROMPT]": [60],
        "[/SYSTEM_PROMPT]": [61],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3, eos_id=99)
    return MistralV7TekkenProcessingStrategy(_Processor(tok))


# --------------------------------------------------------------------------- #
# Strategies that opt out of role-masking (no role_boundaries declared)
# --------------------------------------------------------------------------- #
#
# These four fall back to pad+media masking only. We still want parity fuzz
# coverage of the fused vs legacy paths because the per-strategy media-token
# combinations differ (audio for Voxtral, three image markers for Mistral3,
# variable-length image_ids list for InternVL, six markers for Glm4v).


def _make_voxtral():
    """VoxtralProcessor stub.

    The strategy resolves audio token ids via the nested mistral-common path
    ``processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids``.
    Mirror just enough of that shape for __init__ to read the two ids.
    """
    tok = _Tokenizer({}, pad_id=0, unk_id=3)
    audio_encoder = SimpleNamespace(
        special_ids=SimpleNamespace(audio=300, begin_audio=301)
    )
    tok.tokenizer = SimpleNamespace(
        instruct_tokenizer=SimpleNamespace(audio_encoder=audio_encoder)
    )
    return VoxtralProcessingStrategy(_Processor(tok))


def _make_mistral3():
    """Mistral3 stub: image_encoder.special_ids carries (img, img_break, img_end)."""
    tok = _Tokenizer({}, pad_id=0, unk_id=3)
    image_encoder = SimpleNamespace(
        special_ids=SimpleNamespace(img=400, img_break=401, img_end=402)
    )
    tok.tokenizer = SimpleNamespace(
        instruct_tokenizer=SimpleNamespace(image_encoder=image_encoder)
    )
    return Mistral3ProcessingStrategy(_Processor(tok))


def _make_internvl():
    """InternVL stub: processor.image_ids is a list of media-token ids."""
    tok = _Tokenizer({}, pad_id=0, unk_id=3)
    proc = _Processor(tok)
    proc.image_ids = [500, 501, 502]
    return InternVLProcessingStrategy(proc)


def _make_glm4v():
    """Glm4v / Glm46V stub: convert_tokens_to_ids resolves six media markers."""
    vocab = {
        "<|image|>": [600],
        "<|begin_of_image|>": [601],
        "<|end_of_image|>": [602],
        "<|video|>": [610],
        "<|begin_of_video|>": [611],
        "<|end_of_video|>": [612],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    return Glm4vProcessingStrategy(_Processor(tok))


# --------------------------------------------------------------------------- #
# Parameterized parity fuzz
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "strategy_factory, legacy_fn, extra_ids, name",
    [
        (_make_qwen2vl, legacy_base_process_labels, [200], "qwen2vl"),
        (_make_qwen3_5, legacy_qwen3_5_process_labels, [200, 201], "qwen3_5"),
        (_make_gemma3, legacy_gemma3_process_labels, [120, 262144], "gemma3"),
        (_make_gemma3n, legacy_gemma3n_process_labels, [130, 131, 132, 133], "gemma3n"),
        (
            _make_gemma4,
            legacy_gemma4_process_labels,
            [140, 141, 142, 143, 150, 151, 160],
            "gemma4",
        ),
        (_make_llama32v, legacy_base_process_labels, [], "llama32v"),
        (_make_llama4, legacy_base_process_labels, [], "llama4"),
        (_make_pixtral, legacy_base_process_labels, [], "pixtral"),
        (_make_mistral_v7, legacy_base_process_labels, [], "mistral_v7"),
        (_make_voxtral, legacy_voxtral_process_labels, [300, 301], "voxtral"),
        (
            _make_mistral3,
            legacy_mistral3_process_labels,
            [400, 401, 402],
            "mistral3",
        ),
        (
            _make_internvl,
            legacy_internvl_process_labels,
            [500, 501, 502],
            "internvl",
        ),
        (
            _make_glm4v,
            legacy_glm4v_process_labels,
            [600, 601, 602, 610, 611, 612],
            "glm4v",
        ),
    ],
)
@pytest.mark.parametrize("seed", list(range(50)))
def test_process_labels_parity_fuzz(strategy_factory, legacy_fn, extra_ids, name, seed):
    """50 random batches per strategy: legacy vs fused must produce byte-identical labels."""
    rng = random.Random(seed * 7919 + sum(ord(c) for c in name) % 1000)

    strategy = strategy_factory()
    pad_id = strategy.processor.tokenizer.pad_token_id

    # Mix of small / medium / large, plus varied batch sizes
    batch_size = rng.choice([1, 2, 4, 8])
    seq_len = rng.choice([64, 128, 256, 512])

    batch = _build_batch(
        rng, strategy.role_boundaries, batch_size, seq_len, pad_id, extra_ids
    )

    expected = legacy_fn(strategy, batch.clone())
    got = strategy.process_labels(batch.clone())

    if not torch.equal(expected, got):
        # Find first divergence to make debug output useful
        diff = (expected != got).nonzero(as_tuple=False)
        first = diff[0].tolist()
        b_idx, t_idx = first
        lo = max(0, t_idx - 8)
        hi = min(seq_len, t_idx + 8)
        raise AssertionError(
            f"[{name} seed={seed}] mismatch at row={b_idx} col={t_idx}\n"
            f"  input_ids[{b_idx}, {lo}:{hi}] = {batch[b_idx, lo:hi].tolist()}\n"
            f"  legacy   [{b_idx}, {lo}:{hi}] = {expected[b_idx, lo:hi].tolist()}\n"
            f"  fused    [{b_idx}, {lo}:{hi}] = {got[b_idx, lo:hi].tolist()}\n"
            f"  boundaries: {strategy.role_boundaries}"
        )


def test_process_labels_train_on_inputs_passthrough():
    """When train_on_inputs=True the keep mask must be all-True (only pad/media masked)."""
    vocab = {
        "<|im_start|>user\n": [101, 104, 103],
        "<|im_start|>assistant\n": [101, 105, 103],
        "<|im_end|>": [106],
        "<|image_pad|>": [200],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    s = Qwen2VLProcessingStrategy(_Processor(tok), train_on_inputs=True)
    batch = torch.tensor([[101, 104, 103, 7, 8, 0, 200, 9, 106]])
    out = s.process_labels(batch.clone())
    expected = legacy_base_process_labels(s, batch.clone())
    assert torch.equal(out, expected)
    # And confirm the only masked positions are pad and image
    assert out.tolist() == [[101, 104, 103, 7, 8, -100, -100, 9, 106]]


def test_process_labels_no_boundaries_falls_through():
    """A boundary-less strategy + train_on_inputs=False keeps everything (warns once);
    pad / media tokens are still masked.
    """
    vocab = {"foo": [200]}
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    # Base strategy with no role_boundaries declared and train_on_inputs=False
    s = ProcessingStrategy(_Processor(tok))
    assert s.role_boundaries == []
    batch = torch.tensor([[5, 6, 7, 0, 8]])
    out = s.process_labels(batch.clone())
    expected = legacy_base_process_labels(s, batch.clone())
    assert torch.equal(out, expected)
    # Pad at index 3 must be masked, everything else kept (no boundaries → no role mask).
    assert out.tolist() == [[5, 6, 7, -100, 8]]


# --------------------------------------------------------------------------- #
# Targeted edge cases for fused path
# --------------------------------------------------------------------------- #


def test_fused_pad_inside_assistant_is_masked():
    """Pad inside assistant span must end up -100 even though role-mask kept it."""
    vocab = {
        "<|im_start|>user\n": [101, 104, 103],
        "<|im_start|>assistant\n": [101, 105, 103],
        "<|im_end|>": [106],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    s = Qwen2VLProcessingStrategy(_Processor(tok))
    seq = [101, 104, 103, 7, 106, 101, 105, 103, 8, 0, 9, 106]
    out = s.process_labels(torch.tensor([seq]))
    expected = legacy_base_process_labels(s, torch.tensor([seq]))
    assert torch.equal(out, expected)
    # Pad (0) inside assistant gets masked, content (8, 9) and end (106) kept.
    assert out.tolist() == [
        [-100, -100, -100, -100, -100, -100, -100, -100, 8, -100, 9, 106]
    ]


def test_fused_image_inside_assistant_is_masked():
    vocab = {
        "<|im_start|>user\n": [101, 104, 103],
        "<|im_start|>assistant\n": [101, 105, 103],
        "<|im_end|>": [106],
        "<|image_pad|>": [200],
    }
    tok = _Tokenizer(vocab, pad_id=0, unk_id=3)
    s = Qwen2VLProcessingStrategy(_Processor(tok))
    seq = [101, 105, 103, 8, 200, 9, 106]
    out = s.process_labels(torch.tensor([seq]))
    expected = legacy_base_process_labels(s, torch.tensor([seq]))
    assert torch.equal(out, expected)
    # image_token_id (200) gets masked even though it's inside the assistant span.
    assert out.tolist() == [[-100, -100, -100, 8, -100, 9, 106]]
