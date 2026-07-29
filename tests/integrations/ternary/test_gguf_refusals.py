"""CPU-only tests for what the GGUF writers refuse to represent.

A GGUF tensor name map describes one text transformer stack. A master that carries a
vision tower, a projector or an MTP head has tensors with no name in it, and a writer
that skipped them would produce a file that loads, answers, and is quietly missing a
component the checkpoint was trained with. These pin the refusal.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from axolotl.integrations.ternary.export.gguf_tq import QK_K, export_gguf_tq
from axolotl.integrations.ternary.export.i2s import export_i2s
from axolotl.integrations.ternary.swap import SwapEntry, SwapManifest

from .test_export_gguf import _stub_gguf, _swapped_master


@pytest.fixture(name="stub_gguf")
def fixture_stub_gguf(monkeypatch):
    from axolotl.integrations.ternary.export import gguf_tq, i2s

    writers: list = []
    # i2s imported the loader by value, so both namespaces have to be replaced
    for module in (gguf_tq, i2s):
        monkeypatch.setattr(module, "require_gguf", lambda: _stub_gguf(writers))
    return writers


def _master_with(tmp_path: Path, **extra: torch.Tensor) -> tuple[Path, SwapManifest]:
    directory = tmp_path / "master"
    manifest = _swapped_master(directory, tie_word_embeddings=False)
    path = directory / "model.safetensors"
    tensors = load_file(path)
    tensors.update(extra)
    save_file(tensors, path, metadata={"format": "pt"})
    return directory, manifest


def _tower(rows: int = 8) -> torch.Tensor:
    return torch.zeros(rows, QK_K, dtype=torch.bfloat16)


# ------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "key,label",
    [
        ("model.visual.blocks.0.attn.qkv.weight", "vision tower"),
        ("model.vision_tower.encoder.layer.0.dense.weight", "vision tower"),
        ("model.multi_modal_projector.linear_1.weight", "modality projector"),
        ("model.layers.0.mtp.up_proj.weight", "multi-token-prediction head"),
        ("medusa_head.0.linear.weight", "draft head"),
    ],
)
def test_a_master_carrying_an_auxiliary_stack_is_refused(
    tmp_path, stub_gguf, key, label
):
    master, manifest = _master_with(tmp_path, **{key: _tower()})

    with pytest.raises(ValueError) as excinfo:
        export_gguf_tq(master, tmp_path / "out", manifest)

    message = str(excinfo.value)
    assert label in message
    assert key in message
    assert "no gguf tensor name" in message


def test_the_refusal_happens_before_anything_is_written(tmp_path, stub_gguf):
    master, manifest = _master_with(
        tmp_path, **{"model.visual.blocks.0.attn.qkv.weight": _tower()}
    )

    with pytest.raises(ValueError):
        export_gguf_tq(master, tmp_path / "out", manifest)

    assert stub_gguf == []
    assert not (tmp_path / "out").exists()


def test_the_refusal_points_at_the_formats_that_can_carry_the_tensor(
    tmp_path, stub_gguf
):
    master, manifest = _master_with(
        tmp_path, **{"model.visual.blocks.0.attn.qkv.weight": _tower()}
    )

    with pytest.raises(ValueError, match="master_bf16"):
        export_gguf_tq(master, tmp_path / "out", manifest)


def test_a_tensor_no_family_claims_is_refused_too(tmp_path, stub_gguf):
    master, manifest = _master_with(
        tmp_path, **{"model.side_car.proj.weight": _tower()}
    )

    with pytest.raises(ValueError, match="claimed by no known family") as excinfo:
        export_gguf_tq(master, tmp_path / "out", manifest)

    assert "model.side_car.proj.weight" in str(excinfo.value)


def test_the_refusal_truncates_a_whole_tower(tmp_path, stub_gguf):
    tower = {
        f"model.visual.blocks.{index}.attn.qkv.weight": _tower() for index in range(9)
    }
    master, manifest = _master_with(tmp_path, **tower)

    with pytest.raises(ValueError, match=r"\(\+4 more\)") as excinfo:
        export_gguf_tq(master, tmp_path / "out", manifest)

    assert "9 tensors of this master" in str(excinfo.value)


def test_the_i2s_writer_refuses_the_same_master(tmp_path, stub_gguf):
    master, manifest = _master_with(
        tmp_path, **{"model.visual.blocks.0.attn.qkv.weight": _tower()}
    )

    with pytest.raises(ValueError, match="vision tower"):
        export_i2s(master, tmp_path / "out", manifest)


# ------------------------------------------------------- what is still written


def test_a_text_only_master_still_exports(tmp_path, stub_gguf):
    master, manifest = _master_with(tmp_path)

    export_gguf_tq(master, tmp_path / "out", manifest)

    assert "blk.0.attn_q.weight" in stub_gguf[-1].tensors


def test_a_multimodal_model_type_is_refused_at_the_architecture_gate(
    tmp_path, stub_gguf
):
    """`qwen3_5` masters (vision tower included) never reach the tensor walk."""
    manifest = SwapManifest(
        model_type="qwen3_5",
        entries=[
            SwapEntry(
                name="model.language_model.layers.0.mlp.up_proj",
                in_features=QK_K,
                out_features=QK_K,
                family="up_proj",
                weight_scale="absmean",
            )
        ],
    )

    with pytest.raises(ValueError, match="does not support model_type 'qwen3_5'"):
        export_gguf_tq(tmp_path / "master", tmp_path / "out", manifest)
