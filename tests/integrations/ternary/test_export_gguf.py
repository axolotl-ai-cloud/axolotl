"""CPU-only tests for the GGUF TQ2_0 / TQ1_0 and bitnet.cpp I2_S block packers.

Correctness is established three ways, none of which trusts the packer:
1. reference serializers written straight from the format text (below),
2. hand-computed micro-fixtures whose arithmetic is spelled out in the test,
3. round-trips through the inverse decoders, including — for TQ1_0 — the uint8
   trit extraction a llama.cpp runtime performs.
"""

import importlib.util
import math
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.export import parity
from axolotl.integrations.ternary.export.gguf_tq import (
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_SIZE,
    Q8_0_QMAX,
    QK_K,
    TQ1_0_BLOCK_BYTES,
    TQ1_0_QH_BYTES,
    TQ1_0_QS_BYTES,
    TQ2_0_BLOCK_BYTES,
    TQ2_0_QS_BYTES,
    decode_q8_0,
    dequantize_q8_0,
    export_gguf_tq,
    gguf_available,
    pack_q8_0,
    pack_tq1_0,
    pack_tq2_0,
    quantize_q8_0,
    unpack_tq1_0,
    unpack_tq2_0,
)
from axolotl.integrations.ternary.export.i2s import (
    I2S_BLOCK_SIZE,
    I2S_SCALE_TAIL_BYTES,
    export_i2s,
    pack_i2s,
    unpack_i2s,
)
from axolotl.integrations.ternary.swap import SwapEntry, SwapManifest

requires_gguf = pytest.mark.skipif(not gguf_available(), reason="gguf not installed")
without_gguf = pytest.mark.skipif(gguf_available(), reason="gguf is installed")

SCALE = torch.tensor(0.0123, dtype=torch.float32)


def _random_codes(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 3, (rows, cols), generator=generator) - 1).to(torch.int8)


def _one_hot_codes(cols: int) -> torch.Tensor:
    """One row per (position, code value): covers every slot of a block exhaustively."""
    codes = torch.zeros(cols * 3, cols, dtype=torch.int8)
    for position in range(cols):
        for value in (-1, 0, 1):
            codes[position * 3 + value + 1, position] = value
    return codes


def _tq2_0_reference(codes: list[int], scale: float) -> list[int]:
    """Serialize one 256-code TQ2_0 block from the format text."""
    out: list[int] = []
    for half in range(2):
        for m in range(32):
            byte = 0
            for lane in range(4):
                byte |= (codes[half * 128 + lane * 32 + m] + 1) << (2 * lane)
            out.append(byte)
    return out + list(struct.pack("<e", scale))


def _tq1_0_reference(codes: list[int], scale: float) -> list[int]:
    """Serialize one 256-code TQ1_0 block from the format text."""
    out = [0] * (TQ1_0_QS_BYTES + TQ1_0_QH_BYTES)
    # (first element, first byte, bytes in the group, trits per byte)
    for first_elem, first_byte, n_bytes, n_trits in (
        (0, 0, 32, 5),
        (160, 32, 16, 5),
        (240, 48, 4, 4),
    ):
        for m in range(n_bytes):
            value = 0
            for n in range(5):
                digit = codes[first_elem + n * n_bytes + m] + 1 if n < n_trits else 0
                value = value * 3 + digit
            out[first_byte + m] = math.ceil(value * 256 / 243)
    return out + list(struct.pack("<e", scale))


def _tq1_0_runtime_trits(byte: int, n_trits: int) -> list[int]:
    """The trit extraction a llama.cpp runtime performs, in uint8 arithmetic."""
    return [((((byte * 3**n) % 256) * 3) >> 8) for n in range(n_trits)]


def _i2s_reference(codes: list[int]) -> list[int]:
    """Serialize 128-code I2_S blocks from the format text."""
    out: list[int] = []
    for block in range(len(codes) // I2S_BLOCK_SIZE):
        for j in range(32):
            q = [codes[block * I2S_BLOCK_SIZE + lane * 32 + j] + 1 for lane in range(4)]
            out.append(q[0] << 6 | q[1] << 4 | q[2] << 2 | q[3])
    return out


# --------------------------------------------------------------------------- TQ2_0


def test_tq2_0_block_accounting():
    assert TQ2_0_QS_BYTES == 64
    assert TQ2_0_BLOCK_BYTES == 66
    assert TQ2_0_BLOCK_BYTES * 8 / QK_K == 2.0625

    packed = pack_tq2_0(_random_codes(3, 512), SCALE)

    assert packed.dtype == torch.uint8
    assert packed.numel() == 3 * 2 * TQ2_0_BLOCK_BYTES


def test_tq2_0_hand_computed_block():
    codes = torch.zeros(1, QK_K, dtype=torch.int8)
    # first half, byte 0: elements 0/32/64/96 at shifts 0/2/4/6
    codes[0, 0], codes[0, 32], codes[0, 64], codes[0, 96] = 1, 0, -1, 1
    # second half, byte 32: elements 128/160/192/224
    codes[0, 128], codes[0, 160], codes[0, 192], codes[0, 224] = -1, -1, 1, 0

    packed = pack_tq2_0(codes, SCALE)

    # byte 0 = (1+1) | (0+1)<<2 | (-1+1)<<4 | (1+1)<<6 = 2 + 4 + 0 + 128
    assert int(packed[0]) == 134
    # byte 32 = (-1+1) | (-1+1)<<2 | (1+1)<<4 | (0+1)<<6 = 0 + 0 + 32 + 64
    assert int(packed[32]) == 96
    # every untouched byte holds four zero codes: 1 | 1<<2 | 1<<4 | 1<<6
    untouched = [i for i in range(TQ2_0_QS_BYTES) if i not in (0, 32)]
    assert {int(packed[i]) for i in untouched} == {85}
    assert bytes(packed[64:66].tolist()) == struct.pack("<e", float(SCALE))


def test_tq2_0_matches_reference_serializer():
    codes = _random_codes(4, 768, seed=1)

    packed = pack_tq2_0(codes, SCALE).reshape(4, 3, TQ2_0_BLOCK_BYTES)

    for row in range(4):
        for block in range(3):
            expected = _tq2_0_reference(
                codes[row, block * QK_K : (block + 1) * QK_K].tolist(), float(SCALE)
            )
            assert packed[row, block].tolist() == expected


def test_tq2_0_roundtrip_every_slot():
    codes = _one_hot_codes(QK_K)

    packed = pack_tq2_0(codes, SCALE)

    assert torch.equal(unpack_tq2_0(packed, codes.shape), codes)


@pytest.mark.parametrize("shape", [(1, 256), (5, 512), (8, 1024)])
def test_tq2_0_roundtrip_random(shape):
    codes = _random_codes(*shape, seed=2)

    assert torch.equal(unpack_tq2_0(pack_tq2_0(codes, SCALE), shape), codes)


def test_tq2_0_block_scale_is_f16_little_endian():
    scale = torch.tensor(0.1, dtype=torch.float32)

    packed = pack_tq2_0(_random_codes(2, 512), scale).reshape(2, 2, TQ2_0_BLOCK_BYTES)

    for row in range(2):
        for block in range(2):
            raw = bytes(packed[row, block, TQ2_0_QS_BYTES:].tolist())
            assert struct.unpack("<e", raw)[0] == pytest.approx(0.0999755859375)


# --------------------------------------------------------------------------- TQ1_0


def test_tq1_0_block_accounting():
    assert (TQ1_0_QS_BYTES, TQ1_0_QH_BYTES) == (48, 4)
    assert TQ1_0_BLOCK_BYTES == 54
    assert TQ1_0_BLOCK_BYTES * 8 / QK_K == 1.6875
    # 48 qs bytes carry 5 trits each and 4 qh bytes carry 4: 240 + 16 = 256
    assert TQ1_0_QS_BYTES * 5 + TQ1_0_QH_BYTES * 4 == QK_K

    packed = pack_tq1_0(_random_codes(3, 512), SCALE)

    assert packed.dtype == torch.uint8
    assert packed.numel() == 3 * 2 * TQ1_0_BLOCK_BYTES


def test_tq1_0_hand_computed_block():
    codes = torch.zeros(1, QK_K, dtype=torch.int8)
    # qs byte 0 takes elements 0/32/64/96/128, most significant trit first
    for element, value in zip((0, 32, 64, 96, 128), (1, 0, -1, 1, 0), strict=True):
        codes[0, element] = value
    # qh byte 0 takes elements 240/244/248/252, four trits in the top four slots
    for element, value in zip((240, 244, 248, 252), (1, -1, 0, 1), strict=True):
        codes[0, element] = value

    packed = pack_tq1_0(codes, SCALE)

    # digits 2,1,0,2,1 -> 2*81 + 1*27 + 0*9 + 2*3 + 1*1 = 196; ceil(196*256/243) = 207
    assert 2 * 81 + 1 * 27 + 0 * 9 + 2 * 3 + 1 * 1 == 196
    assert math.ceil(196 * 256 / 243) == 207
    assert int(packed[0]) == 207
    # digits 2,0,1,2 in slots 3^4..3^1 -> 2*81 + 0*27 + 1*9 + 2*3 = 177; ceil -> 187
    assert 2 * 81 + 0 * 27 + 1 * 9 + 2 * 3 == 177
    assert math.ceil(177 * 256 / 243) == 187
    assert int(packed[TQ1_0_QS_BYTES]) == 187
    # untouched qs bytes hold five zero codes: 1*(81+27+9+3+1) = 121 -> 128
    assert {int(packed[i]) for i in range(1, TQ1_0_QS_BYTES)} == {
        math.ceil(121 * 256 / 243)
    }
    # untouched qh bytes hold four zero codes: 1*(81+27+9+3) = 120 -> 127
    assert {
        int(packed[i])
        for i in range(TQ1_0_QS_BYTES + 1, TQ1_0_QS_BYTES + TQ1_0_QH_BYTES)
    } == {math.ceil(120 * 256 / 243)}
    assert bytes(packed[52:54].tolist()) == struct.pack("<e", float(SCALE))


def test_tq1_0_exhaustive_qs_trit_groups():
    """All 243 five-trit combinations of one qs byte encode to ceil(v * 256 / 243)."""
    codes = torch.zeros(243, QK_K, dtype=torch.int8)
    for value in range(243):
        remainder = value
        for n in range(5):
            digit = remainder // 3 ** (4 - n)
            remainder -= digit * 3 ** (4 - n)
            codes[value, n * 32] = digit - 1

    packed = pack_tq1_0(codes, SCALE).reshape(243, TQ1_0_BLOCK_BYTES)

    for value in range(243):
        assert int(packed[value, 0]) == math.ceil(value * 256 / 243)
        assert _tq1_0_runtime_trits(int(packed[value, 0]), 5) == [
            (value // 3 ** (4 - n)) % 3 for n in range(5)
        ]


def test_tq1_0_exhaustive_qh_trit_groups():
    """All 81 four-trit combinations of one qh byte sit in the top four trit slots."""
    codes = torch.zeros(81, QK_K, dtype=torch.int8)
    for value in range(81):
        remainder = value
        for n in range(4):
            digit = remainder // 3 ** (3 - n)
            remainder -= digit * 3 ** (3 - n)
            codes[value, 240 + n * 4] = digit - 1

    packed = pack_tq1_0(codes, SCALE).reshape(81, TQ1_0_BLOCK_BYTES)

    for value in range(81):
        assert int(packed[value, TQ1_0_QS_BYTES]) == math.ceil(3 * value * 256 / 243)
        assert _tq1_0_runtime_trits(int(packed[value, TQ1_0_QS_BYTES]), 4) == [
            (value // 3 ** (3 - n)) % 3 for n in range(4)
        ]


def test_tq1_0_matches_reference_serializer():
    codes = _random_codes(4, 768, seed=3)

    packed = pack_tq1_0(codes, SCALE).reshape(4, 3, TQ1_0_BLOCK_BYTES)

    for row in range(4):
        for block in range(3):
            expected = _tq1_0_reference(
                codes[row, block * QK_K : (block + 1) * QK_K].tolist(), float(SCALE)
            )
            assert packed[row, block].tolist() == expected


def test_tq1_0_runtime_extraction_recovers_codes():
    """A byte-at-a-time decode with the runtime's uint8 trit math returns the codes."""
    codes = _random_codes(2, QK_K, seed=4)

    packed = pack_tq1_0(codes, SCALE).reshape(2, TQ1_0_BLOCK_BYTES)

    for row in range(2):
        decoded = [0] * QK_K
        for first_elem, first_byte, n_bytes, n_trits in (
            (0, 0, 32, 5),
            (160, 32, 16, 5),
            (240, 48, 4, 4),
        ):
            for m in range(n_bytes):
                trits = _tq1_0_runtime_trits(int(packed[row, first_byte + m]), n_trits)
                for n, trit in enumerate(trits):
                    decoded[first_elem + n * n_bytes + m] = trit - 1
        assert decoded == codes[row].tolist()


def test_tq1_0_roundtrip_every_slot():
    codes = _one_hot_codes(QK_K)

    assert torch.equal(unpack_tq1_0(pack_tq1_0(codes, SCALE), codes.shape), codes)


@pytest.mark.parametrize("shape", [(1, 256), (5, 512), (8, 1024)])
def test_tq1_0_roundtrip_random(shape):
    codes = _random_codes(*shape, seed=5)

    assert torch.equal(unpack_tq1_0(pack_tq1_0(codes, SCALE), shape), codes)


# ---------------------------------------------------------------------------- I2_S


def test_i2s_block_accounting():
    assert I2S_BLOCK_SIZE == 128
    assert I2S_SCALE_TAIL_BYTES == 32

    packed = pack_i2s(_random_codes(3, 256), SCALE)

    assert packed.dtype == torch.uint8
    # 2.0 bpw plus one 32-byte per-tensor tail
    assert packed.numel() == 3 * 256 // 4 + I2S_SCALE_TAIL_BYTES
    assert (packed.numel() - I2S_SCALE_TAIL_BYTES) * 8 / (3 * 256) == 2.0


def test_i2s_hand_computed_block():
    codes = torch.zeros(1, I2S_BLOCK_SIZE, dtype=torch.int8)
    # byte 0 packs elements 0/32/64/96 at shifts 6/4/2/0
    codes[0, 0], codes[0, 32], codes[0, 64], codes[0, 96] = 1, 0, -1, 1

    packed = pack_i2s(codes, SCALE)

    # (1+1)<<6 | (0+1)<<4 | (-1+1)<<2 | (1+1) = 128 + 16 + 0 + 2
    assert int(packed[0]) == 146
    # untouched bytes hold four zero codes: 1<<6 | 1<<4 | 1<<2 | 1
    assert {int(packed[i]) for i in range(1, 32)} == {85}


def test_i2s_scale_tail():
    packed = pack_i2s(_random_codes(2, 128), SCALE)

    tail = packed[-I2S_SCALE_TAIL_BYTES:]
    assert struct.unpack("<f", bytes(tail[:4].tolist()))[0] == pytest.approx(
        float(SCALE)
    )
    assert int(tail[4:].sum()) == 0


def test_i2s_matches_reference_serializer():
    codes = _random_codes(3, 384, seed=6)

    packed = pack_i2s(codes, SCALE)

    expected = _i2s_reference(codes.reshape(-1).tolist())
    assert packed[: len(expected)].tolist() == expected


def test_i2s_roundtrip_every_slot():
    codes = _one_hot_codes(I2S_BLOCK_SIZE)

    assert torch.equal(unpack_i2s(pack_i2s(codes, SCALE), codes.shape), codes)


@pytest.mark.parametrize("shape", [(1, 128), (5, 256), (8, 1024)])
def test_i2s_roundtrip_random(shape):
    codes = _random_codes(*shape, seed=7)

    assert torch.equal(unpack_i2s(pack_i2s(codes, SCALE), shape), codes)


# -------------------------------------------------------------------------- errors


@pytest.mark.parametrize("pack", [pack_tq2_0, pack_tq1_0, pack_i2s])
def test_pack_rejects_ragged_rows(pack):
    with pytest.raises(ValueError, match="not a multiple of"):
        pack(torch.zeros(2, 100, dtype=torch.int8), SCALE)


@pytest.mark.parametrize("pack", [pack_tq2_0, pack_tq1_0, pack_i2s])
def test_pack_rejects_non_ternary_codes(pack):
    codes = torch.zeros(1, QK_K, dtype=torch.int8)
    codes[0, 7] = 2

    with pytest.raises(ValueError, match="must all be -1, 0 or"):
        pack(codes, SCALE)


@pytest.mark.parametrize("pack", [pack_tq2_0, pack_tq1_0, pack_i2s])
def test_pack_rejects_non_2d_codes(pack):
    with pytest.raises(ValueError, match="out_features, in_features"):
        pack(torch.zeros(QK_K, dtype=torch.int8), SCALE)


@pytest.mark.parametrize("pack", [pack_tq2_0, pack_tq1_0, pack_i2s])
def test_pack_rejects_group_scale(pack):
    with pytest.raises(ValueError, match="per-tensor scale"):
        pack(torch.zeros(1, QK_K, dtype=torch.int8), torch.ones(2))


@pytest.mark.parametrize(
    "unpack,block_bytes",
    [(unpack_tq2_0, TQ2_0_BLOCK_BYTES), (unpack_tq1_0, TQ1_0_BLOCK_BYTES)],
)
def test_unpack_rejects_wrong_length(unpack, block_bytes):
    with pytest.raises(ValueError, match="expected"):
        unpack(torch.zeros(block_bytes - 1, dtype=torch.uint8), (1, QK_K))


def test_unpack_i2s_rejects_wrong_length():
    with pytest.raises(ValueError, match="expected"):
        unpack_i2s(torch.zeros(32, dtype=torch.uint8), (1, I2S_BLOCK_SIZE))


# ------------------------------------------------------------------- gguf container


def test_gguf_available_matches_the_environment():
    assert gguf_available() is (importlib.util.find_spec("gguf") is not None)


@without_gguf
def test_export_gguf_tq_reports_the_missing_dependency(tmp_path):
    with pytest.raises(ImportError, match="pip install gguf"):
        export_gguf_tq(tmp_path, tmp_path, _manifest())


@without_gguf
def test_export_i2s_reports_the_missing_dependency(tmp_path):
    with pytest.raises(ImportError, match="pip install gguf"):
        export_i2s(tmp_path, tmp_path, _manifest())


@requires_gguf
def test_export_gguf_tq_rejects_unknown_quant_type(tmp_path):
    with pytest.raises(ValueError, match="unknown gguf ternary type"):
        export_gguf_tq(tmp_path, tmp_path, _manifest(), quant_type="tq3_0")


@requires_gguf
@pytest.mark.parametrize("quant_type", ["tq2_0", "tq1_0"])
def test_export_gguf_tq_rejects_group_scales(tmp_path, quant_type):
    manifest = _manifest(weight_scale="group", group_size=128)

    with pytest.raises(ValueError, match="cannot be represented"):
        export_gguf_tq(tmp_path, tmp_path, manifest, quant_type=quant_type)


@requires_gguf
def test_export_i2s_rejects_group_scales(tmp_path):
    with pytest.raises(ValueError, match="cannot be represented"):
        export_i2s(tmp_path, tmp_path, _manifest(weight_scale="group", group_size=128))


@requires_gguf
def test_export_rejects_unsupported_architecture(tmp_path):
    with pytest.raises(ValueError, match="does not support model_type"):
        export_gguf_tq(tmp_path, tmp_path, _manifest(model_type="gpt_neox"))


def _manifest(
    model_type: str = "llama",
    weight_scale: str = "absmean",
    group_size: int | None = None,
) -> SwapManifest:
    return SwapManifest(
        model_type=model_type,
        entries=[
            SwapEntry(
                name="model.layers.0.self_attn.q_proj",
                in_features=QK_K,
                out_features=QK_K,
                family="q_proj",
                weight_scale=weight_scale,
                group_size=group_size,
            )
        ],
        weight_scale=weight_scale,
        group_size=group_size,
    )


# ------------------------------------------------------ block-level parity gate


def _baked(rows: int = 8, cols: int = QK_K, seed: int = 0) -> torch.Tensor:
    from axolotl.integrations.ternary.quant import fake_quant_weight

    torch.manual_seed(seed)
    return fake_quant_weight(torch.randn(rows, cols).to(torch.bfloat16) * 0.02, 1.0)


@pytest.mark.parametrize(
    "fmt,packer",
    [
        ("gguf_tq2_0", pack_tq2_0),
        ("gguf_tq1_0", pack_tq1_0),
        ("i2_s", pack_i2s),
    ],
)
def test_block_gate_accepts_a_faithful_pack(fmt, packer):
    """These formats have no in-CI container reader, so the bytes are gated as packed."""
    from axolotl.integrations.ternary.export import parity
    from axolotl.integrations.ternary.export.bake import derive_codes_and_scale

    master = _baked()
    codes, scale = derive_codes_and_scale(master)

    assert parity.gate_block_bytes(fmt, packer(codes, scale), master) == []


@pytest.mark.parametrize(
    "fmt,packer",
    [
        ("gguf_tq2_0", pack_tq2_0),
        ("gguf_tq1_0", pack_tq1_0),
        ("i2_s", pack_i2s),
    ],
)
def test_block_gate_catches_a_flipped_code_byte(fmt, packer):
    from axolotl.integrations.ternary.export import parity
    from axolotl.integrations.ternary.export.bake import derive_codes_and_scale

    master = _baked()
    codes, scale = derive_codes_and_scale(master)
    packed = packer(codes, scale)
    packed[0] ^= 0b11

    failures = parity.gate_block_bytes(fmt, packed, master)

    assert failures and "codes differ from the master" in failures[0]


@pytest.mark.parametrize("fmt,packer", [("gguf_tq2_0", pack_tq2_0), ("i2_s", pack_i2s)])
def test_block_gate_catches_a_wrong_block_scale(fmt, packer):
    from axolotl.integrations.ternary.export import parity
    from axolotl.integrations.ternary.export.bake import derive_codes_and_scale

    master = _baked()
    codes, scale = derive_codes_and_scale(master)
    packed = packer(codes, scale * 2)

    failures = parity.gate_block_bytes(fmt, packed, master)

    assert failures and "block scale drifted" in failures[0]


def test_block_gate_refuses_an_unregistered_format():
    from axolotl.integrations.ternary.export import parity

    master = _baked()

    assert parity.gate_block_bytes("nope", torch.zeros(4, dtype=torch.uint8), master)


def test_gate_packed_bytes_verifies_exactly_what_the_writer_serializes():
    """The i2_s packer hands the writer an int8 view; the gate must follow the bytes."""
    import hashlib

    import numpy as np

    from axolotl.integrations.ternary.export.bake import derive_codes_and_scale
    from axolotl.integrations.ternary.export.gguf_tq import gate_packed_bytes

    master = _baked()
    codes, scale = derive_codes_and_scale(master)
    data = pack_i2s(codes, scale).numpy().view(np.int8)

    digest = gate_packed_bytes("i2_s", data, master)

    assert digest == hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def test_gate_packed_bytes_raises_on_a_corrupted_pack():
    from axolotl.integrations.ternary.export.bake import derive_codes_and_scale
    from axolotl.integrations.ternary.export.gguf_tq import gate_packed_bytes

    master = _baked()
    codes, scale = derive_codes_and_scale(master)
    packed = pack_tq2_0(codes, scale)
    packed[3] ^= 0xFF

    with pytest.raises(RuntimeError, match="block gate rejected"):
        gate_packed_bytes("gguf_tq2_0", packed.numpy(), master)


def test_write_gate_record_lists_every_gated_tensor(tmp_path):
    import json

    from axolotl.integrations.ternary.export.gguf_tq import (
        GATE_RECORD_FILENAME,
        write_gate_record,
    )

    path = write_gate_record(
        tmp_path, "i2_s", tmp_path / "ggml-model-i2_s.gguf", {"q_proj": "abc"}
    )

    assert path.name == GATE_RECORD_FILENAME
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {
        "format": "i2_s",
        "artifact": "ggml-model-i2_s.gguf",
        "tensors": {"q_proj": "abc"},
    }


# ------------------------------------------------------------- Q8_0 embeddings


def test_q8_0_micro_fixture_matches_hand_computed_bytes():
    """One 32-weight block, arithmetic spelled out: d = 63.5/127 = 0.5, f16 0x3800."""
    weight = torch.zeros(1, Q8_0_BLOCK_SIZE)
    weight[0, 0] = 63.5  # amax, so q = +127
    weight[0, 1] = -63.5  # q = -127
    weight[0, 2] = 0.25  # 0.25/0.5 = 0.5 -> round-half-even -> 0
    weight[0, 3] = 0.75  # 0.75/0.5 = 1.5 -> round-half-even -> 2
    weight[0, 4] = 1.0  # 1.0/0.5  = 2.0 -> 2

    codes, scales = quantize_q8_0(weight)
    packed = pack_q8_0(codes, scales)

    assert float(scales[0, 0]) == 0.5
    assert codes[0, :5].tolist() == [127, -127, 0, 2, 2]
    expected = [0x00, 0x38, 0x7F, 0x81, 0x00, 0x02, 0x02] + [0x00] * 27
    assert packed.numel() == Q8_0_BLOCK_BYTES == 34
    assert packed.tolist() == expected


@pytest.mark.parametrize("shape", [(1, 32), (4, 64), (7, 256), (3, 32 * 11)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_q8_0_roundtrip(shape, dtype):
    torch.manual_seed(shape[0] * shape[1])
    weight = (torch.randn(*shape) * 0.05).to(dtype)

    codes, scales = quantize_q8_0(weight)
    packed = pack_q8_0(codes, scales)
    decoded_codes, decoded_scales = decode_q8_0(packed, shape)

    assert packed.numel() == shape[0] * (shape[1] // Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales, scales)
    # every element lands within half a quantization step of the master
    step = decoded_scales.repeat_interleave(Q8_0_BLOCK_SIZE, dim=1)
    error = (
        weight.to(torch.float32) - dequantize_q8_0(decoded_codes, decoded_scales)
    ).abs()
    assert bool((error <= 0.5 * step).all())


def test_q8_0_codes_stay_in_the_symmetric_int8_range():
    torch.manual_seed(0)
    weight = torch.randn(16, 128) * 3.0

    codes, _ = quantize_q8_0(weight)

    assert int(codes.amin()) >= -Q8_0_QMAX
    assert int(codes.amax()) <= Q8_0_QMAX


def test_q8_0_handles_an_all_zero_block():
    weight = torch.zeros(2, Q8_0_BLOCK_SIZE)

    codes, scales = quantize_q8_0(weight)

    assert not int(codes.abs().sum())
    assert not float(scales.abs().sum())
    assert torch.equal(decode_q8_0(pack_q8_0(codes, scales), (2, 32))[0], codes)


def test_q8_0_survives_a_scale_that_underflows_f16():
    """2e-6 / 127 = 1.6e-8 rounds to an f16 zero, which would kill the whole block."""
    weight = torch.full((1, Q8_0_BLOCK_SIZE), 2e-6)

    assert float(quant.f16_round_scale(torch.tensor(2e-6 / Q8_0_QMAX))) == 0.0
    codes, scales = quantize_q8_0(weight)

    assert float(scales[0, 0]) == pytest.approx(2.0**-24)
    assert int(codes[0, 0]) == 34
    step = scales.repeat_interleave(Q8_0_BLOCK_SIZE, dim=1)
    error = (weight - dequantize_q8_0(codes, scales)).abs()
    assert bool((error <= 0.5 * step).all())


@pytest.mark.parametrize("cols", [33, 31, 100])
def test_q8_0_rejects_a_row_length_that_is_not_a_multiple_of_32(cols):
    """llama.cpp cannot load a padded row, so this has to fail before the write."""
    weight = torch.randn(2, cols)

    with pytest.raises(ValueError, match="multiple of 32"):
        quantize_q8_0(weight)
    with pytest.raises(ValueError, match="multiple of 32"):
        decode_q8_0(torch.zeros(68, dtype=torch.uint8), (2, cols))


def test_q8_0_rejects_mismatched_scales():
    codes, scales = quantize_q8_0(torch.randn(4, 64))

    with pytest.raises(ValueError, match="block scales"):
        pack_q8_0(codes, scales[:, :1])


def test_q8_0_gate_accepts_a_faithful_pack_and_catches_a_flipped_byte():
    torch.manual_seed(0)
    weight = (torch.randn(8, 128) * 0.05).to(torch.bfloat16)
    packed = pack_q8_0(*quantize_q8_0(weight))

    assert parity.gate_q8_0_bytes(packed, weight) == []

    corrupted = packed.clone()
    corrupted[5] ^= 0xFF
    assert "codes differ" in parity.gate_q8_0_bytes(corrupted, weight)[0]

    rescaled = packed.clone()
    rescaled[0] ^= 0x0F
    assert "block scale drifted" in parity.gate_q8_0_bytes(rescaled, weight)[0]


# ------------------------------------------- GGUF writer path (stubbed container)


class _StubWriter:
    """Records what `write_gguf` hands the container, so the writer path is testable.

    The `gguf` package is optional and absent in CI; only the container belongs to
    it, and the tensor selection, naming and packing above it are ours.
    """

    def __init__(self, path, arch_name):
        self.path = Path(path)
        self.arch_name = arch_name
        self.tensors: dict[str, tuple] = {}
        self.closed = False

    def add_tensor(self, name, data, raw_shape=None, raw_dtype=None):
        self.tensors[name] = (data, raw_shape, raw_dtype)

    def write_header_to_file(self):
        self.path.write_bytes(b"GGUF")

    def write_kv_data_to_file(self):
        pass

    def write_tensors_to_file(self):
        pass

    def close(self):
        self.closed = True

    def __getattr__(self, name):  # add_name, add_block_count, ...
        if name.startswith("add_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


class _StubNameMap:
    _SUFFIXES = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
    }

    def get_name(self, key, try_suffixes=()):
        stem = key.removesuffix(".weight")
        if stem == "model.embed_tokens":
            return "token_embd.weight"
        if stem in ("lm_head", "model.norm"):
            return ("output" if stem == "lm_head" else "output_norm") + ".weight"
        if stem.startswith("model.layers."):
            _, _, index, rest = stem.split(".", 3)
            mapped = self._SUFFIXES.get(rest)
            return f"blk.{index}.{mapped}.weight" if mapped else None
        return None


def _stub_gguf(latest_writer: list):
    from types import SimpleNamespace

    def make_writer(path, arch_name):
        writer = _StubWriter(path, arch_name)
        latest_writer.append(writer)
        return writer

    return SimpleNamespace(
        MODEL_ARCH=SimpleNamespace(LLAMA="llama"),
        MODEL_ARCH_NAMES={"llama": "llama"},
        GGMLQuantizationType=SimpleNamespace(TQ2_0="TQ2_0", TQ1_0="TQ1_0", Q8_0="Q8_0"),
        LlamaFileType=SimpleNamespace(MOSTLY_TQ2_0="MOSTLY_TQ2_0"),
        get_tensor_name_map=lambda arch, layers: _StubNameMap(),
        GGUFWriter=make_writer,
    )


def _swapped_master(directory, tie_word_embeddings: bool):
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.integrations.ternary.export import bake
    from axolotl.integrations.ternary.modules import iter_ternary_modules
    from axolotl.integrations.ternary.swap import convert_model
    from axolotl.utils.dict import DictDefault

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=QK_K,
        intermediate_size=QK_K * 2,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    manifest = convert_model(
        model, DictDefault({"output_dir": str(directory), "ternary": {}})
    )
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    bake.write_quantizer_metadata(directory, manifest)
    return manifest


@pytest.fixture(name="stub_gguf")
def fixture_stub_gguf(monkeypatch):
    from axolotl.integrations.ternary.export import gguf_tq

    writers: list[_StubWriter] = []
    monkeypatch.setattr(gguf_tq, "require_gguf", lambda: _stub_gguf(writers))
    return writers


def test_int8_embeddings_are_written_as_q8_0(tmp_path, stub_gguf):
    manifest = _swapped_master(tmp_path / "master", tie_word_embeddings=False)

    export_gguf_tq(
        tmp_path / "master", tmp_path / "out", manifest, embedding_dtype="int8"
    )

    tensors = stub_gguf[-1].tensors
    for name in ("token_embd.weight", "output.weight"):
        data, raw_shape, raw_dtype = tensors[name]
        assert raw_dtype == "Q8_0"
        assert raw_shape == (64, QK_K)
        assert data.dtype == np.uint8
        assert data.size == 64 * (QK_K // Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES
    # the ternary tensors are untouched by the embedding dtype
    assert tensors["blk.0.attn_q.weight"][2] == "TQ2_0"
    assert tensors["blk.0.attn_norm.weight"][0].dtype == np.float32


def test_tied_embeddings_emit_one_quantized_tensor(tmp_path, stub_gguf):
    """llama.cpp derives `output.weight` from `token_embd.weight` when they are tied."""
    manifest = _swapped_master(tmp_path / "master", tie_word_embeddings=True)

    export_gguf_tq(
        tmp_path / "master", tmp_path / "out", manifest, embedding_dtype="int8"
    )

    tensors = stub_gguf[-1].tensors
    assert "token_embd.weight" in tensors
    assert "output.weight" not in tensors
    assert tensors["token_embd.weight"][2] == "Q8_0"


def test_default_embedding_dtype_keeps_f16_embeddings(tmp_path, stub_gguf):
    manifest = _swapped_master(tmp_path / "master", tie_word_embeddings=False)

    export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest)

    tensors = stub_gguf[-1].tensors
    for name in ("token_embd.weight", "output.weight"):
        data, _, raw_dtype = tensors[name]
        assert raw_dtype is None
        assert data.dtype == np.float16


def test_quantized_embeddings_are_gated_and_digested(tmp_path, stub_gguf):
    import json

    from axolotl.integrations.ternary.export.gguf_tq import GATE_RECORD_FILENAME

    manifest = _swapped_master(tmp_path / "master", tie_word_embeddings=False)

    export_gguf_tq(
        tmp_path / "master",
        tmp_path / "out",
        manifest,
        gate=True,
        embedding_dtype="int8",
    )

    record = json.loads((tmp_path / "out" / GATE_RECORD_FILENAME).read_text())
    assert {"token_embd.weight", "output.weight"} <= set(record["tensors"])
    assert len(record["tensors"]) == len(manifest.entries) + 2


def test_a_corrupted_embedding_block_fails_the_gate(tmp_path, stub_gguf, monkeypatch):
    from axolotl.integrations.ternary.export import gguf_tq

    manifest = _swapped_master(tmp_path / "master", tie_word_embeddings=False)
    original = gguf_tq.pack_q8_0

    def corrupt(codes, scales):
        packed = original(codes, scales)
        packed[7] ^= 0xFF
        return packed

    monkeypatch.setattr(gguf_tq, "pack_q8_0", corrupt)

    with pytest.raises(RuntimeError, match="q8_0 block gate rejected"):
        export_gguf_tq(
            tmp_path / "master",
            tmp_path / "out",
            manifest,
            gate=True,
            embedding_dtype="int8",
        )
