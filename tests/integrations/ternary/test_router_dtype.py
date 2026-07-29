"""CPU-only tests for `ternary.export.router_dtype`, the int8 (Q8_0) router path.

Routers are kept full precision by the swap, so `router_dtype` is the only knob that
puts them on a quantization grid. It is dormant on real MoE checkpoints — fused expert
stacks are still refused at swap time — so the masters below bolt a router-named Linear
onto a dense Llama, which is exactly the tensor shape the writer has to recognize.
"""

import json

import numpy as np
import pytest
import torch
from torch import nn

from axolotl.integrations.ternary import aux_modules
from axolotl.integrations.ternary.export import bake, run_export
from axolotl.integrations.ternary.export.gguf_tq import (
    GATE_RECORD_FILENAME,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_SIZE,
    QK_K,
    export_gguf_tq,
    is_router_tensor,
)
from axolotl.integrations.ternary.export.i2s import export_i2s
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault

from .test_export_gguf import _stub_gguf, _StubNameMap

ROUTER_KEY = "model.layers.0.mlp.gate.weight"
ROUTER_GGUF_NAME = "blk.0.ffn_gate_inp.weight"
N_EXPERTS = 8


class _RouterNameMap(_StubNameMap):
    """The llama name map plus the router entry gguf really carries for MoE llamas."""

    _SUFFIXES = {**_StubNameMap._SUFFIXES, "mlp.gate": "ffn_gate_inp"}

    def get_name(self, key, try_suffixes=()):
        if key.endswith(".bias"):
            weight_name = super().get_name(
                key.removesuffix(".bias") + ".weight", try_suffixes
            )
            if weight_name is None:
                return None
            return weight_name.removesuffix(".weight") + ".bias"
        return super().get_name(key, try_suffixes)


@pytest.fixture(name="stub_gguf")
def fixture_stub_gguf(monkeypatch):
    from axolotl.integrations.ternary.export import gguf_tq, i2s

    writers: list = []
    stub = _stub_gguf(writers)
    stub.get_tensor_name_map = lambda arch, layers: _RouterNameMap()
    for module in (gguf_tq, i2s):
        monkeypatch.setattr(module, "require_gguf", lambda: stub)
    return writers


def _master(
    directory,
    router_in_features: int | None = QK_K,
    router_bias: bool = False,
):
    """Write a baked master whose layer 0 carries a kept-FP `mlp.gate` router."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=QK_K,
        intermediate_size=QK_K * 2,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    if router_in_features is not None:
        model.model.layers[0].mlp.gate = nn.Linear(
            router_in_features, N_EXPERTS, bias=router_bias, dtype=torch.bfloat16
        )
    manifest = convert_model(
        model,
        DictDefault(
            {
                "output_dir": str(directory),
                "ternary": {
                    "keep_fp_modules": list(aux_modules.family_patterns("routers"))
                },
            }
        ),
    )
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    bake.write_quantizer_metadata(directory, manifest)
    return manifest


# --------------------------------------------------------------- registry lookup


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.mlp.gate.weight",
        "model.layers.3.block_sparse_moe.gate.weight",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.gate.wg.weight",
    ],
)
def test_router_family_names_are_recognized(key):
    assert is_router_tensor(key)


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    ],
)
def test_non_router_names_are_not_recognized(key):
    assert not is_router_tensor(key)


def test_recognition_is_the_registry_not_a_local_list():
    """Every canary the routers family declares has to reach the Q8_0 branch."""
    for name in aux_modules.AUX_MODULE_FAMILIES["routers"].canaries:
        assert is_router_tensor(f"{name}.weight")


# ------------------------------------------------------------------ default path


def test_default_router_dtype_keeps_f16_routers(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master")

    export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest)

    tensors = stub_gguf[-1].tensors
    data, raw_shape, raw_dtype = tensors[ROUTER_GGUF_NAME]
    assert raw_dtype is None
    assert raw_shape is None
    assert data.dtype == np.float16
    assert data.shape == (N_EXPERTS, QK_K)


def test_default_router_dtype_records_no_router_digest(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master")

    export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest, gate=True)

    record = json.loads((tmp_path / "out" / GATE_RECORD_FILENAME).read_text())
    assert ROUTER_GGUF_NAME not in record["tensors"]
    assert set(record["tensors"]) == {entry.name for entry in manifest.entries}


def test_bf16_is_the_default_and_changes_nothing(tmp_path, stub_gguf):
    """The knob's default has to be byte-identical to not passing it at all."""
    manifest = _master(tmp_path / "master")

    export_gguf_tq(tmp_path / "master", tmp_path / "implicit", manifest)
    export_gguf_tq(
        tmp_path / "master", tmp_path / "explicit", manifest, router_dtype="bf16"
    )

    implicit, explicit = stub_gguf[-2].tensors, stub_gguf[-1].tensors
    assert set(implicit) == set(explicit)
    for name, (data, raw_shape, raw_dtype) in implicit.items():
        other_data, other_shape, other_dtype = explicit[name]
        assert (raw_shape, raw_dtype) == (other_shape, other_dtype)
        assert data.tobytes() == other_data.tobytes()
    assert (tmp_path / "implicit" / GATE_RECORD_FILENAME).read_text() == (
        tmp_path / "explicit" / GATE_RECORD_FILENAME
    ).read_text()


# --------------------------------------------------------------------- int8 path


def test_int8_routers_are_written_as_q8_0(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master")

    export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest, router_dtype="int8")

    tensors = stub_gguf[-1].tensors
    data, raw_shape, raw_dtype = tensors[ROUTER_GGUF_NAME]
    assert raw_dtype == "Q8_0"
    assert raw_shape == (N_EXPERTS, QK_K)
    assert data.dtype == np.uint8
    assert data.size == N_EXPERTS * (QK_K // Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES
    # the router dtype touches nothing else
    assert tensors["blk.0.attn_q.weight"][2] == "TQ2_0"
    assert tensors["token_embd.weight"][0].dtype == np.float16
    assert tensors["blk.0.attn_norm.weight"][0].dtype == np.float32


def test_int8_routers_dequantize_back_to_the_master(tmp_path, stub_gguf):
    from axolotl.integrations.ternary.export.gguf_tq import decode_q8_0, dequantize_q8_0

    manifest = _master(tmp_path / "master")
    master, _ = bake.load_master(tmp_path / "master")

    export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest, router_dtype="int8")

    data = stub_gguf[-1].tensors[ROUTER_GGUF_NAME][0]
    codes, scales = decode_q8_0(torch.from_numpy(data).reshape(-1), (N_EXPERTS, QK_K))
    error = (
        master[ROUTER_KEY].to(torch.float32) - dequantize_q8_0(codes, scales)
    ).abs()
    step = scales.repeat_interleave(Q8_0_BLOCK_SIZE, dim=-1)
    assert bool((error <= step * 0.5 + 1e-9).all())


def test_int8_routers_are_gated_and_digested(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master")

    export_gguf_tq(
        tmp_path / "master",
        tmp_path / "out",
        manifest,
        gate=True,
        router_dtype="int8",
    )

    record = json.loads((tmp_path / "out" / GATE_RECORD_FILENAME).read_text())
    assert ROUTER_GGUF_NAME in record["tensors"]
    assert len(record["tensors"]) == len(manifest.entries) + 1


def test_a_corrupted_router_block_fails_the_gate(tmp_path, stub_gguf, monkeypatch):
    from axolotl.integrations.ternary.export import gguf_tq

    manifest = _master(tmp_path / "master")
    original = gguf_tq.pack_q8_0

    def corrupt(codes, scales):
        packed = original(codes, scales)
        packed[9] ^= 0xFF
        return packed

    monkeypatch.setattr(gguf_tq, "pack_q8_0", corrupt)

    with pytest.raises(RuntimeError, match="q8_0 block gate rejected"):
        export_gguf_tq(
            tmp_path / "master",
            tmp_path / "out",
            manifest,
            gate=True,
            router_dtype="int8",
        )


# --------------------------------------------------------- independence and shapes


@pytest.mark.parametrize(
    ("embedding_dtype", "router_dtype", "embedding_q8_0", "router_q8_0"),
    [
        ("bf16", "bf16", False, False),
        ("bf16", "int8", False, True),
        ("int8", "bf16", True, False),
        ("int8", "int8", True, True),
    ],
)
def test_embeddings_and_routers_are_independently_controllable(
    tmp_path, stub_gguf, embedding_dtype, router_dtype, embedding_q8_0, router_q8_0
):
    manifest = _master(tmp_path / "master")

    export_gguf_tq(
        tmp_path / "master",
        tmp_path / "out",
        manifest,
        embedding_dtype=embedding_dtype,
        router_dtype=router_dtype,
    )

    tensors = stub_gguf[-1].tensors
    assert (tensors["token_embd.weight"][2] == "Q8_0") is embedding_q8_0
    assert (tensors["output.weight"][2] == "Q8_0") is embedding_q8_0
    assert (tensors[ROUTER_GGUF_NAME][2] == "Q8_0") is router_q8_0


def test_a_router_row_length_q8_0_cannot_block_is_rejected(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master", router_in_features=100)

    with pytest.raises(ValueError, match="router_dtype") as excinfo:
        export_gguf_tq(
            tmp_path / "master", tmp_path / "out", manifest, router_dtype="int8"
        )

    assert "multiple of 32" in str(excinfo.value)
    assert ROUTER_KEY in str(excinfo.value)


def test_a_ragged_router_is_fine_at_the_default_dtype(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master", router_in_features=100)

    export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest)

    assert stub_gguf[-1].tensors[ROUTER_GGUF_NAME][0].dtype == np.float16


def test_a_router_bias_stays_f32(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master", router_bias=True)

    export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest, router_dtype="int8")

    tensors = stub_gguf[-1].tensors
    assert tensors[ROUTER_GGUF_NAME][2] == "Q8_0"
    bias, _, raw_dtype = tensors["blk.0.ffn_gate_inp.bias"]
    assert raw_dtype is None
    assert bias.dtype == np.float32


# ------------------------------------------------------------------- the warning


def test_int8_routers_warn_about_the_file_they_produced(tmp_path, stub_gguf, caplog):
    manifest = _master(tmp_path / "master")

    with caplog.at_level("WARNING"):
        export_gguf_tq(
            tmp_path / "master", tmp_path / "out", manifest, router_dtype="int8"
        )

    assert "packed 1 router tensors as Q8_0" in caplog.text
    assert ROUTER_KEY.removesuffix(".weight") in caplog.text
    assert "ffn_gate_inp" in caplog.text
    assert "expert utilization" in caplog.text


def test_int8_routers_warn_when_the_model_has_none(tmp_path, stub_gguf, caplog):
    manifest = _master(tmp_path / "master", router_in_features=None)

    with caplog.at_level("WARNING"):
        export_gguf_tq(
            tmp_path / "master", tmp_path / "out", manifest, router_dtype="int8"
        )

    assert "matched no router module" in caplog.text


def test_the_default_never_mentions_routers(tmp_path, stub_gguf, caplog):
    manifest = _master(tmp_path / "master")

    with caplog.at_level("WARNING"):
        export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest)

    assert "router" not in caplog.text


# ------------------------------------------------------------ i2_s and run_export


def test_i2s_honors_router_dtype(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master")

    export_i2s(tmp_path / "master", tmp_path / "out", manifest, router_dtype="int8")

    data, raw_shape, raw_dtype = stub_gguf[-1].tensors[ROUTER_GGUF_NAME]
    assert raw_dtype == "Q8_0"
    assert raw_shape == (N_EXPERTS, QK_K)
    assert data.size == N_EXPERTS * (QK_K // Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES


def test_i2s_default_keeps_f16_routers(tmp_path, stub_gguf):
    manifest = _master(tmp_path / "master")

    export_i2s(tmp_path / "master", tmp_path / "out", manifest)

    assert stub_gguf[-1].tensors[ROUTER_GGUF_NAME][0].dtype == np.float16


@pytest.mark.parametrize("fmt", ["gguf_tq2_0", "gguf_tq1_0", "i2_s"])
def test_run_export_threads_the_configured_router_dtype(tmp_path, stub_gguf, fmt):
    _master(tmp_path / "master")

    run_export(
        DictDefault(
            {
                "output_dir": str(tmp_path / "master"),
                "ternary": {
                    "keep_fp_modules": list(aux_modules.family_patterns("routers")),
                    "export": {"formats": [fmt], "router_dtype": "int8"},
                },
            }
        )
    )

    assert stub_gguf[-1].tensors[ROUTER_GGUF_NAME][2] == "Q8_0"


@pytest.mark.parametrize("fmt", ["gguf_tq2_0", "i2_s"])
def test_run_export_defaults_to_full_precision_routers(tmp_path, stub_gguf, fmt):
    _master(tmp_path / "master")

    run_export(
        DictDefault(
            {
                "output_dir": str(tmp_path / "master"),
                "ternary": {
                    "keep_fp_modules": list(aux_modules.family_patterns("routers")),
                    "export": {"formats": [fmt]},
                },
            }
        )
    )

    assert stub_gguf[-1].tensors[ROUTER_GGUF_NAME][0].dtype == np.float16
