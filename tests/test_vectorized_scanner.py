"""Differential and targeted edge-case tests for the vectorized role-boundary scanner.

The vectorized scanner (:func:`axolotl.processing_strategies._apply_role_boundaries_vectorized`)
must be byte-identical to the reference implementation
(:func:`axolotl.processing_strategies._apply_role_boundaries`) for every valid input.

This file is structured in three layers:

1. Targeted edge-case tests covering each behavioral subtlety called out in
   the implementation comments (longest-prefix tie-break, Pixtral rewind,
   train_on_eos modes, empty end_tokens, include_end leak gate).

2. A boundary-shape catalog mimicking the real models: Gemma 4, Llama 3.2 V,
   Llama 4, Pixtral, Mistral V7.

3. A differential fuzzer that generates 2,000 random configurations and
   asserts vectorized output == reference output element-wise. On mismatch,
   dumps inputs + outputs + boundary spec.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Iterable

import pytest
import torch

from axolotl.processing_strategies import (
    RoleBoundary,
    _apply_role_boundaries,
    _apply_role_boundaries_vectorized,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _assert_equiv(
    boundaries: list[RoleBoundary],
    seq: list[list[int]],
    roles_to_train: Iterable[str],
    train_on_eos: str,
):
    """Assert vectorized output == reference output for the given input."""
    labels_a = torch.tensor(seq)
    labels_b = labels_a.clone()
    out_ref = _apply_role_boundaries(
        labels_a, boundaries, set(roles_to_train), train_on_eos
    )
    out_vec = _apply_role_boundaries_vectorized(
        labels_b, boundaries, set(roles_to_train), train_on_eos
    )
    if not torch.equal(out_ref, out_vec):
        # Build a focused failure dump.
        diff = (out_ref != out_vec).nonzero(as_tuple=False).tolist()
        msg = (
            "Vectorized scanner diverged from reference.\n"
            f"  boundaries: {[asdict(b) for b in boundaries]}\n"
            f"  roles_to_train: {sorted(roles_to_train)}\n"
            f"  train_on_eos: {train_on_eos}\n"
            f"  input seq: {seq}\n"
            f"  reference: {out_ref.tolist()}\n"
            f"  vectorized: {out_vec.tolist()}\n"
            f"  first 10 diff indices: {diff[:10]}\n"
        )
        raise AssertionError(msg)
    return out_ref


# --------------------------------------------------------------------------- #
# Layer 3: Targeted edge-case unit tests
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Hand-written cases that pin down each tricky behavior."""

    def test_longest_prefix_wins_at_same_position(self):
        """``<|im_start|>assistant`` (long) beats ``<|im_start|>`` (short)."""
        # token 100 = <|im_start|>, then 101 = "user", 102 = "assistant"
        boundaries = [
            RoleBoundary(role="user", start_tokens=[100], end_tokens=[200]),
            RoleBoundary(role="assistant", start_tokens=[100, 102], end_tokens=[200]),
        ]
        seq = [[100, 102, 1, 2, 3, 200, 100, 101, 4, 5, 6, 200]]
        # The longer "assistant" boundary should win at j=0; only positions
        # 1..5 are trainable (assistant content + end marker because
        # train_on_eos="turn" includes end on trainable turns by default).
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_pixtral_shared_end_marker_rewind(self):
        """[/INST] both ends user and starts assistant; rewind must re-match."""
        # Pixtral: user=[INST] ... [/INST], assistant content, then </s>.
        # The shared [/INST] must consume as user-end *and* assistant-start.
        INST_OPEN = 100  # [INST]
        INST_CLOSE = 200  # [/INST]
        EOS = 2  # </s>
        boundaries = [
            RoleBoundary(
                role="user",
                start_tokens=[INST_OPEN],
                end_tokens=[INST_CLOSE],
                include_start=False,
                include_end=False,  # critical: don't consume — let assistant re-match
            ),
            RoleBoundary(
                role="assistant",
                start_tokens=[INST_CLOSE],
                end_tokens=[EOS],
                include_start=False,
                include_end=True,
            ),
        ]
        # [INST] hello [/INST] reply </s>
        seq = [[INST_OPEN, 5, 6, INST_CLOSE, 7, 8, 9, EOS]]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_train_on_eos_turn(self):
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[200])
        ]
        seq = [[100, 1, 2, 3, 200, 9, 9, 9, 100, 4, 5, 6, 200]]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_train_on_eos_all(self):
        boundaries = [
            RoleBoundary(role="user", start_tokens=[100], end_tokens=[200]),
            RoleBoundary(role="assistant", start_tokens=[101], end_tokens=[200]),
        ]
        seq = [[100, 1, 2, 200, 101, 3, 4, 200, 9]]
        _assert_equiv(boundaries, seq, ["assistant"], "all")

    def test_train_on_eos_none(self):
        """train_on_eos=none disables the end-marker contribution entirely."""
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[200])
        ]
        seq = [[100, 1, 2, 3, 200, 9]]
        _assert_equiv(boundaries, seq, ["assistant"], "none")

    def test_train_on_eos_last_only_final_turn_unmasked(self):
        """Only the last trainable turn's end marker contributes."""
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[200])
        ]
        # Three assistant turns. Only the last 200 should be unmasked.
        seq = [[100, 1, 200, 100, 2, 200, 100, 3, 200, 9, 9]]
        _assert_equiv(boundaries, seq, ["assistant"], "last")

    def test_empty_end_tokens_runs_to_eos(self):
        """end_tokens=[] means the span runs to end-of-sequence."""
        boundaries = [RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[])]
        seq = [[100, 1, 2, 3, 4, 5]]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_include_start_true(self):
        boundaries = [
            RoleBoundary(
                role="assistant",
                start_tokens=[100, 101],
                end_tokens=[200],
                include_start=True,
                include_end=True,
            )
        ]
        seq = [[100, 101, 5, 6, 7, 200]]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_non_trainable_role_end_marker_leak_gate(self):
        """Non-trainable role with include_end=True, train_on_eos=all → end is unmasked."""
        boundaries = [
            RoleBoundary(
                role="user",
                start_tokens=[100],
                end_tokens=[200],
                include_start=False,
                include_end=True,
            ),
            RoleBoundary(
                role="assistant",
                start_tokens=[101],
                end_tokens=[201],
                include_start=False,
                include_end=True,
            ),
        ]
        seq = [[100, 1, 2, 200, 101, 3, 4, 201]]
        # roles_to_train=["assistant"] but train_on_eos="all" → user's end (200) leaks in
        _assert_equiv(boundaries, seq, ["assistant"], "all")

    def test_non_trainable_role_include_end_false_no_leak(self):
        """Non-trainable role with include_end=False, train_on_eos=all → no leak."""
        boundaries = [
            RoleBoundary(
                role="user",
                start_tokens=[100],
                end_tokens=[200],
                include_start=False,
                include_end=False,
            ),
            RoleBoundary(
                role="assistant",
                start_tokens=[200],  # shared with user-end
                end_tokens=[201],
            ),
        ]
        seq = [[100, 1, 2, 200, 3, 4, 201]]
        _assert_equiv(boundaries, seq, ["assistant"], "all")

    def test_truncated_final_turn_no_end(self):
        """Final assistant turn missing end marker — span runs to end."""
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[200])
        ]
        seq = [[100, 1, 2, 3, 4, 5, 6]]  # no 200 anywhere
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_all_pad_row(self):
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[200])
        ]
        seq = [[0, 0, 0, 0, 0, 0]]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_single_token_row(self):
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[200])
        ]
        seq = [[100]]  # start with no end & no content
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_empty_roles_to_train_masks_all(self):
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[200])
        ]
        seq = [[100, 1, 2, 200, 100, 3, 4, 200]]
        _assert_equiv(boundaries, seq, [], "turn")

    def test_adversarial_first_token_collision(self):
        """A bare token equal to start_tokens[0] inside filler must not trigger
        a partial match (multi-token start_tokens needed)."""
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100, 200], end_tokens=[201])
        ]
        # 100 appears in filler (alone, not followed by 200) → must NOT match.
        seq = [[100, 200, 5, 6, 100, 7, 8, 201, 100, 200, 9, 10, 201]]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_batch_of_mixed_rows(self):
        boundaries = [
            RoleBoundary(role="assistant", start_tokens=[100], end_tokens=[200])
        ]
        # Three rows with different shapes.
        seq = [
            [100, 1, 2, 200, 0, 0, 0, 0],
            [100, 3, 4, 5, 6, 200, 0, 0],
            [9, 9, 100, 7, 200, 9, 9, 9],
        ]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")


# --------------------------------------------------------------------------- #
# Layer 2-ish: Boundary-shape catalog (real-model-like configs)
# --------------------------------------------------------------------------- #


def _gemma4_like():
    """Gemma 4: <start_of_turn>role ... <end_of_turn>."""
    SOT, EOT = 50, 60
    return [
        RoleBoundary(role="user", start_tokens=[SOT, 70], end_tokens=[EOT]),
        RoleBoundary(role="assistant", start_tokens=[SOT, 71], end_tokens=[EOT]),
        RoleBoundary(role="system", start_tokens=[SOT, 72], end_tokens=[EOT]),
    ]


def _llama32v_like():
    """Llama 3.2 V: <|start_header_id|>role<|end_header_id|> ... <|eot_id|>."""
    SHID, EHID, EOT = 80, 81, 82
    return [
        RoleBoundary(role="user", start_tokens=[SHID, 90, EHID], end_tokens=[EOT]),
        RoleBoundary(role="assistant", start_tokens=[SHID, 91, EHID], end_tokens=[EOT]),
        RoleBoundary(role="system", start_tokens=[SHID, 92, EHID], end_tokens=[EOT]),
    ]


def _llama4_like():
    """Llama 4-style: similar to llama3 but 4 roles."""
    SHID, EHID, EOT = 80, 81, 82
    return [
        RoleBoundary(role="user", start_tokens=[SHID, 90, EHID], end_tokens=[EOT]),
        RoleBoundary(role="assistant", start_tokens=[SHID, 91, EHID], end_tokens=[EOT]),
        RoleBoundary(role="system", start_tokens=[SHID, 92, EHID], end_tokens=[EOT]),
        RoleBoundary(role="tool", start_tokens=[SHID, 93, EHID], end_tokens=[EOT]),
    ]


def _pixtral_like():
    """Pixtral: shared [/INST] between user-end and assistant-start, EOS terminates."""
    INST_O, INST_C, EOS = 100, 200, 2
    return [
        RoleBoundary(
            role="user",
            start_tokens=[INST_O],
            end_tokens=[INST_C],
            include_start=False,
            include_end=False,
        ),
        RoleBoundary(
            role="assistant",
            start_tokens=[INST_C],
            end_tokens=[EOS],
            include_start=False,
            include_end=True,
        ),
    ]


def _mistralv7_like():
    """Mistral V7 Tekken-ish: similar shared [/INST] pattern with system."""
    INST_O, INST_C, EOS, SYS = 100, 200, 2, 110
    return [
        RoleBoundary(
            role="system",
            start_tokens=[SYS],
            end_tokens=[INST_O],
            include_start=False,
            include_end=False,
        ),
        RoleBoundary(
            role="user",
            start_tokens=[INST_O],
            end_tokens=[INST_C],
            include_start=False,
            include_end=False,
        ),
        RoleBoundary(
            role="assistant",
            start_tokens=[INST_C],
            end_tokens=[EOS],
            include_start=False,
            include_end=True,
        ),
    ]


BOUNDARY_CATALOG = {
    "gemma4": _gemma4_like,
    "llama32v": _llama32v_like,
    "llama4": _llama4_like,
    "pixtral": _pixtral_like,
    "mistralv7": _mistralv7_like,
}


# --------------------------------------------------------------------------- #
# Layer 1: Differential fuzz test
# --------------------------------------------------------------------------- #


def _build_random_sequence(
    rng: random.Random,
    boundaries: list[RoleBoundary],
    seq_len: int,
    pad_id: int = 0,
) -> list[int]:
    """Build a plausible sequence by alternating turns until we hit seq_len.

    Uses random filler tokens that *don't* collide with any marker. ~5% of
    fillers are pathologically chosen as adversarial collisions: a single
    token equal to start_tokens[0] (so that multi-token starts don't match
    but the first byte does).
    """
    # Build a vocab of "safe" filler tokens that don't collide with any
    # start_tokens prefix.
    marker_tokens: set[int] = set()
    for b in boundaries:
        marker_tokens.update(b.start_tokens)
        marker_tokens.update(b.end_tokens)
    safe_filler = [t for t in range(300, 500) if t not in marker_tokens]

    seq: list[int] = []
    while len(seq) < seq_len:
        # Pick a boundary at random; emit start + filler + (sometimes) end.
        b = rng.choice(boundaries)
        seq.extend(b.start_tokens)
        filler_n = rng.randint(5, min(200, max(5, seq_len - len(seq))))
        for _ in range(filler_n):
            if rng.random() < 0.05 and safe_filler:
                # Adversarial: emit a token that equals a marker's first byte
                # but isn't followed by the rest. Picks from any boundary.
                bb = rng.choice(boundaries)
                if bb.start_tokens:
                    seq.append(bb.start_tokens[0])
                    continue
            seq.append(rng.choice(safe_filler))

        # 80% chance of a clean end. (Truncated turns intentional.)
        if rng.random() < 0.8 and b.end_tokens:
            seq.extend(b.end_tokens)

    seq = seq[:seq_len]
    # Pad up if we underran.
    while len(seq) < seq_len:
        seq.append(pad_id)
    return seq


@pytest.mark.parametrize(
    "shape_name",
    ["gemma4", "llama32v", "llama4", "pixtral", "mistralv7"],
)
def test_catalog_smoke_per_shape(shape_name):
    """Smoke test: each catalog shape works on a small batch."""
    boundaries = BOUNDARY_CATALOG[shape_name]()
    rng = random.Random(0xCAFE)
    rows = [_build_random_sequence(rng, boundaries, 256) for _ in range(4)]
    _assert_equiv(boundaries, rows, ["assistant"], "turn")
    _assert_equiv(boundaries, rows, ["assistant", "user"], "all")
    _assert_equiv(boundaries, rows, [], "none")
    _assert_equiv(boundaries, rows, ["assistant"], "last")


def test_differential_fuzz_2000_configs():
    """Run 2000 random configurations and verify byte-identical outputs.

    Uses fixed seeds 0..1999 so any failure is deterministically reproducible.
    """
    failures: list[tuple[int, str]] = []

    BATCH_SIZES = [1, 2, 4, 8]
    SEQ_LENS = [32, 256, 1024, 4096]
    EOS_MODES = ["turn", "all", "none", "last"]
    ROLES_OPTIONS = [
        ["assistant"],
        ["assistant", "user"],
        [],
        ["assistant", "system", "user"],
    ]
    SHAPES = list(BOUNDARY_CATALOG.keys())

    N_CONFIGS = 2000

    for seed in range(N_CONFIGS):
        rng = random.Random(seed)
        bs = rng.choice(BATCH_SIZES)
        sl = rng.choice(SEQ_LENS)
        eos = rng.choice(EOS_MODES)
        rtt = rng.choice(ROLES_OPTIONS)
        shape = rng.choice(SHAPES)
        boundaries = BOUNDARY_CATALOG[shape]()

        # Cap the largest pixtral-shape configs at 1024 to keep wall-time
        # under control; the small/medium configs already cover the rewind.
        if shape == "pixtral" and sl == 4096 and bs == 8:
            sl = 1024

        rows = [_build_random_sequence(rng, boundaries, sl) for _ in range(bs)]

        try:
            _assert_equiv(boundaries, rows, rtt, eos)
        except AssertionError as e:
            failures.append((seed, str(e)))
            if len(failures) >= 5:
                break

    if failures:
        joined = "\n\n---\n\n".join(f"seed={s}:\n{m}" for s, m in failures)
        pytest.fail(
            f"Differential fuzz: {len(failures)} mismatches in "
            f"{N_CONFIGS} configs.\n\n{joined}"
        )


# --------------------------------------------------------------------------- #
# Pathological inputs aimed at the rewind logic specifically
# --------------------------------------------------------------------------- #


class TestPixtralRewindAdversarial:
    def test_back_to_back_user_assistant_pairs(self):
        boundaries = _pixtral_like()
        # Five back-to-back turns.
        seq = [
            [
                100,
                1,
                2,
                200,
                3,
                4,
                5,
                2,  # turn 1
                100,
                6,
                7,
                200,
                8,
                9,
                2,  # turn 2
                100,
                10,
                11,
                12,
                200,
                13,
                14,
                2,
                100,
                15,
                200,
                16,
                2,
                100,
                17,
                18,
                19,
                20,
                200,
                21,
                22,
                2,
            ]
        ]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_user_with_no_end_then_assistant(self):
        """Truncated user — never closes — assistant never re-matches the [/INST]."""
        boundaries = _pixtral_like()
        seq = [[100, 1, 2, 3, 4, 5, 6]]  # no [/INST] anywhere
        _assert_equiv(boundaries, seq, ["assistant"], "turn")

    def test_double_end_marker(self):
        """Two [/INST] in a row — second one starts a (degenerate) assistant."""
        boundaries = _pixtral_like()
        seq = [[100, 1, 200, 200, 5, 6, 2]]
        _assert_equiv(boundaries, seq, ["assistant"], "turn")


# --------------------------------------------------------------------------- #
# Long-span / multi-end-marker inputs aimed at the bisect end-finder
# --------------------------------------------------------------------------- #


def _build_long_multi_end_sequence(
    rng: random.Random,
    boundaries: list[RoleBoundary],
    seq_len: int,
    pad_id: int = 0,
) -> list[int]:
    """Like ``_build_random_sequence`` but with long turns that embed *multiple*
    full end-marker copies inside one content region.

    The vectorized scanner finds turn ends by bisecting a sorted list of
    end-match positions for the next end >= start_of_content. Turns that carry
    several full end markers (plus partial first-byte collisions) exercise the
    "pick the first valid end, not the closest scanned" branch that a linear
    walk would otherwise mask.
    """
    marker_tokens: set[int] = set()
    for b in boundaries:
        marker_tokens.update(b.start_tokens)
        marker_tokens.update(b.end_tokens)
    safe_filler = [t for t in range(300, 500) if t not in marker_tokens]

    seq: list[int] = []
    while len(seq) < seq_len:
        b = rng.choice(boundaries)
        seq.extend(b.start_tokens)
        # Long filler so a turn can span well past the old 200-token cap.
        filler_n = rng.randint(200, max(200, min(1500, seq_len - len(seq) + 200)))
        for _ in range(filler_n):
            r = rng.random()
            if r < 0.04 and b.end_tokens:
                # Full extra end marker mid-content: with include_end this closes
                # the turn early; without it the rewind re-reads it as a start.
                seq.extend(b.end_tokens)
            elif r < 0.09 and safe_filler:
                bb = rng.choice(boundaries)
                if bb.start_tokens:
                    seq.append(bb.start_tokens[0])  # partial first-byte collision
                    continue
                seq.append(rng.choice(safe_filler))
            else:
                seq.append(rng.choice(safe_filler))
        if rng.random() < 0.8 and b.end_tokens:
            seq.extend(b.end_tokens)

    seq = seq[:seq_len]
    while len(seq) < seq_len:
        seq.append(pad_id)
    return seq


def test_bisect_first_end_in_span_explicit():
    """Two full end markers inside one assistant turn: the span must close on the
    first, leaving the second outside the trainable region."""
    boundaries = _pixtral_like()  # [/INST] == [200], rewind on include_end=False
    # assistant turn opens at [/INST] (200), content, end-of-turn (2) appears
    # twice; the first 2 closes the turn, everything after is a fresh scan.
    seq = [[100, 1, 2, 200, 5, 6, 7, 2, 9, 9, 9, 2, 100, 11, 200, 12, 2]]
    _assert_equiv(boundaries, seq, ["assistant"], "turn")
    _assert_equiv(boundaries, seq, ["assistant"], "all")
    _assert_equiv(boundaries, seq, ["assistant"], "last")
    _assert_equiv(boundaries, seq, ["assistant"], "none")


def test_differential_fuzz_long_spans():
    """500 configs with long, multi-end-marker turns over large sequences.

    Targets the bisect end-finder and bytearray slice-fills, which the original
    short-span fuzz (filler <= 200) under-exercises.
    """
    failures: list[tuple[int, str]] = []

    BATCH_SIZES = [1, 2, 4]
    SEQ_LENS = [1024, 2048, 4096]
    EOS_MODES = ["turn", "all", "none", "last"]
    ROLES_OPTIONS = [["assistant"], ["assistant", "user"], [], ["assistant", "system"]]
    SHAPES = list(BOUNDARY_CATALOG.keys())

    N_CONFIGS = 500

    for seed in range(N_CONFIGS):
        rng = random.Random(10_000 + seed)
        bs = rng.choice(BATCH_SIZES)
        sl = rng.choice(SEQ_LENS)
        eos = rng.choice(EOS_MODES)
        rtt = rng.choice(ROLES_OPTIONS)
        shape = rng.choice(SHAPES)
        boundaries = BOUNDARY_CATALOG[shape]()

        rows = [_build_long_multi_end_sequence(rng, boundaries, sl) for _ in range(bs)]

        try:
            _assert_equiv(boundaries, rows, rtt, eos)
        except AssertionError as e:
            failures.append((seed, str(e)))
            if len(failures) >= 5:
                break

    if failures:
        joined = "\n\n---\n\n".join(f"seed={s}:\n{m}" for s, m in failures)
        pytest.fail(
            f"Long-span differential fuzz: {len(failures)} mismatches in "
            f"{N_CONFIGS} configs.\n\n{joined}"
        )
