"""Chain writer format, byte-exact reconstruction, limits, and commit semantics."""

import json
import zlib

import numpy as np
import pytest
import torch
import zstandard
from safetensors import safe_open
from safetensors.torch import load_file

from axolotl.core.publishing.chain import ChainWriter, combine_indexes
from axolotl.core.publishing.fingerprint import fingerprint_digest

DELTA_METADATA_KEYS = {
    "version",
    "base_version",
    "delta_encoding",
    "compression_format",
    "checksum_format",
}


def index_of(path):
    return json.loads((path / "model.safetensors.index.json").read_text())


def load_full(path):
    weights = {}
    for name in sorted(set(index_of(path)["weight_map"].values())):
        weights.update(load_file(path / name))
    return weights


def apply_delta(path, previous):
    """Reconstruct from a delta exactly as a hot-load server does."""
    result = {key: tensor.clone() for key, tensor in previous.items()}
    index = index_of(path)
    for name in set(index["weight_map"].values()):
        with safe_open(path / name, framework="numpy") as shard:
            for key in shard.keys():
                assert index["weight_map"][key] == name
                delta = np.frombuffer(
                    zstandard.ZstdDecompressor().decompress(shard.get_tensor(key)),
                    dtype=np.uint8,
                )
                target = result[key].reshape(-1).view(torch.uint8).numpy()
                target[:] ^= delta
                assert shard.metadata()[key] == f"{zlib.adler32(target):08x}"
    return result


def model_weights(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "attn.q.weight": torch.randn(32, 16, generator=generator).to(torch.bfloat16),
        "attn.k.weight": torch.randn(8, 16, generator=generator).to(torch.bfloat16),
        "experts.0.weight": torch.randint(
            0, 255, (16, 8), dtype=torch.uint8, generator=generator
        ),
        "experts.0.weight_scale": torch.rand(16, 1, generator=generator).to(
            torch.float8_e4m3fn
        ),
        "experts.0.weight_scale_2": torch.tensor(0.5 + seed * 1e-3),
        "experts.0.input_scale": torch.tensor(0.25),
        "norm.weight": torch.ones(16, dtype=torch.bfloat16),
    }


def assert_bytes_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].shape == expected[key].shape, key
        assert torch.equal(
            actual[key].contiguous().reshape(-1).view(torch.uint8),
            expected[key].contiguous().reshape(-1).view(torch.uint8),
        ), key


def commit(writer, weights, identity, **kwargs):
    staged = writer.write(weights, identity, config={"model_type": "test"}, **kwargs)
    writer.commit(staged)
    return staged


def test_full_checkpoint_format(tmp_path):
    writer = ChainWriter(tmp_path, max_shard_size=600)
    weights = model_weights(0)
    fingerprint = {"export": "merged", "base_identity": "base"}
    staged = commit(writer, weights, "full-0", fingerprint=fingerprint)
    assert staged.full and staged.base_identity is None
    index = index_of(staged.path)
    assert index["metadata"] == {
        "total_size": sum(t.numel() * t.element_size() for t in weights.values()),
        "merge_fingerprint": fingerprint_digest(fingerprint),
    }
    assert set(index["weight_map"]) == set(weights)
    assert len(set(index["weight_map"].values())) > 1
    assert json.loads((staged.path / "config.json").read_text()) == {
        "model_type": "test"
    }
    assert sorted(p.name for p in staged.path.iterdir()) == sorted(
        set(index["weight_map"].values()) | {"config.json", index_name()}
    )
    for name in set(index["weight_map"].values()):
        with safe_open(staged.path / name, framework="pt") as shard:
            assert shard.metadata() == {"format": "pt"}
    assert_bytes_equal(load_full(staged.path), weights)
    assert_bytes_equal(dict(writer.baseline), weights)
    writer.baseline.verify()


def index_name():
    return "model.safetensors.index.json"


def test_delta_format_one_frame_per_tensor(tmp_path):
    writer = ChainWriter(tmp_path)
    old = model_weights(0)
    commit(writer, old, "full-0")
    new = {key: tensor.clone() for key, tensor in old.items()}
    new["attn.q.weight"][0, 0] += 1
    new["experts.0.weight"][3, 3] ^= 0x0F
    new["experts.0.weight_scale_2"] = torch.tensor(0.75)
    staged = writer.write(new, "delta-1")
    index = index_of(staged.path)
    assert index["metadata"] == {
        "version": "delta-1",
        "base_version": "full-0",
        "delta_encoding": "xor",
        "compression_format": "zstd",
        "checksum_format": "adler32",
    }
    changed = {"attn.q.weight", "experts.0.weight", "experts.0.weight_scale_2"}
    assert set(index["weight_map"]) == changed
    assert not (staged.path / "config.json").exists()
    for name in set(index["weight_map"].values()):
        with safe_open(staged.path / name, framework="numpy") as shard:
            assert set(shard.metadata()) == set(shard.keys())
            for key in shard.keys():
                frame = shard.get_tensor(key)
                assert frame.dtype == np.uint8 and frame.ndim == 1
                params = zstandard.get_frame_parameters(frame.tobytes())
                assert params.content_size == new[key].numel() * new[key].element_size()
                raw = zstandard.ZstdDecompressor().decompress(frame.tobytes())
                assert len(raw) == params.content_size
                assert shard.metadata()[key] == (
                    f"{zlib.adler32(new[key].reshape(-1).view(torch.uint8).numpy()):08x}"
                )
    assert_bytes_equal(apply_delta(staged.path, old), new)


@pytest.mark.parametrize("workers,max_inflight", [(1, 1), (4, 1), (4, 16)])
def test_chain_reconstruction_is_byte_exact(tmp_path, workers, max_inflight):
    writer = ChainWriter(
        tmp_path,
        max_deltas=3,
        max_shard_size=700,
        workers=workers,
        max_inflight=max_inflight,
    )
    reconstructed = None
    kinds = []
    for step in range(9):
        weights = model_weights(step)
        if step % 4 == 2:
            weights["experts.0.weight"] = model_weights(step - 1)["experts.0.weight"]
        staged = commit(writer, weights, f"step-{step}")
        kinds.append(staged.full)
        if staged.full:
            reconstructed = load_full(staged.path)
        else:
            assert "experts.0.input_scale" not in staged.weight_map
            reconstructed = apply_delta(staged.path, reconstructed)
        assert_bytes_equal(reconstructed, weights)
        assert_bytes_equal(dict(writer.baseline), weights)
    assert kinds == [True, False, False, False, True, False, False, False, True]
    assert len(list(tmp_path.glob(".baseline-*"))) == 1
    writer.close()


def test_bf16_chain_matches_fresh_tensors_over_long_run(tmp_path):
    writer = ChainWriter(tmp_path, max_shard_size=1 << 12)
    torch.manual_seed(0)
    weight = torch.randn(64, 64).to(torch.bfloat16)
    reconstructed = None
    for step in range(12):
        weight = (weight.float() + 1e-2 * torch.randn(64, 64)).to(torch.bfloat16)
        current = {"w": weight.clone(), "fixed": torch.zeros(4, dtype=torch.bfloat16)}
        staged = commit(writer, current, f"s{step}")
        reconstructed = (
            load_full(staged.path)
            if staged.full
            else apply_delta(staged.path, reconstructed)
        )
        assert_bytes_equal(reconstructed, current)
    assert writer.delta_count == 11


def test_63_delta_limit(tmp_path):
    writer = ChainWriter(tmp_path, max_shard_size=1 << 12)
    fulls = []
    for step in range(66):
        staged = commit(writer, {"w": torch.tensor([float(step)])}, f"s{step}")
        fulls.append(staged.full)
    assert [i for i, full in enumerate(fulls) if full] == [0, 64]
    assert writer.delta_count == 1
    with pytest.raises(ValueError, match="max_deltas"):
        ChainWriter(tmp_path / "other", max_deltas=64)


def test_fingerprint_or_layout_change_forces_full(tmp_path):
    writer = ChainWriter(tmp_path)
    eager = {"merge_compile": "eager"}
    commit(writer, {"w": torch.ones(2)}, "a", fingerprint=eager)
    staged = commit(writer, {"w": torch.zeros(2)}, "b", fingerprint=eager)
    assert not staged.full
    staged = commit(
        writer, {"w": torch.ones(2)}, "c", fingerprint={"merge_compile": "strict"}
    )
    assert staged.full and staged.reason == "merge numerics changed"
    assert index_of(staged.path)["metadata"]["merge_fingerprint"] == (
        fingerprint_digest({"merge_compile": "strict"})
    )
    staged = commit(writer, {"w": torch.zeros(2)}, "d")
    assert staged.full
    staged = commit(
        writer,
        {"w": torch.ones(2)},
        "e",
        layout={"w": ("float32", [3])},
    )
    assert staged.full and staged.reason == "tensor layout changed"
    staged = commit(writer, {"w": torch.zeros(2)}, "f", layout={"w": ("float32", [2])})
    assert not staged.full


@pytest.mark.parametrize(
    "new",
    [
        {"w": torch.ones(3)},
        {"w": torch.ones(2, dtype=torch.float64)},
        {"w": torch.ones(2), "extra": torch.ones(1)},
        {},
    ],
)
def test_layout_change_mid_stream_leaves_no_residue(tmp_path, new):
    writer = ChainWriter(tmp_path)
    commit(writer, {"w": torch.ones(2), "v": torch.ones(1)}, "a")
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError):
        writer.write(new, "b")
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert writer.identity == "a"


def test_missing_tensor_in_delta_is_rejected(tmp_path):
    writer = ChainWriter(tmp_path)
    commit(writer, {"w": torch.ones(2), "v": torch.ones(1)}, "a")
    with pytest.raises(ValueError, match="missing"):
        writer.write({"w": torch.zeros(2)}, "b")


def test_uncommitted_checkpoint_leaves_baseline_unchanged(tmp_path):
    writer = ChainWriter(tmp_path)
    commit(writer, {"w": torch.ones(2)}, "a")
    committed = writer.baseline
    staged = writer.write({"w": torch.zeros(2)}, "b")
    assert writer.baseline is committed and torch.equal(
        writer.baseline["w"], torch.ones(2)
    )
    assert torch.equal(staged.baseline["w"], torch.zeros(2))
    writer.discard(staged)
    assert not staged.path.exists() and not staged.baseline.exists()
    assert committed.exists()
    staged = writer.write({"w": torch.full((2,), 3.0)}, "c")
    assert staged.base_identity == "a"
    rival = writer.write({"w": torch.full((2,), 4.0)}, "d")
    writer.commit(staged)
    assert not committed.exists()
    with pytest.raises(RuntimeError, match="committed"):
        writer.commit(rival)
    with pytest.raises(RuntimeError, match="committed baseline"):
        writer.discard(staged)


def test_staging_does_not_overwrite_existing_artifact(tmp_path):
    writer = ChainWriter(tmp_path)
    commit(writer, {"w": torch.ones(2)}, "a")
    with pytest.raises(ValueError, match="new identity"):
        writer.write({"w": torch.ones(2)}, "a")
    (tmp_path / "b").mkdir()
    with pytest.raises(FileExistsError):
        writer.write({"w": torch.ones(2)}, "b")


@pytest.mark.parametrize("identity", ["../bad", "a/b", "", ".", "..", "-x"])
def test_identity_validation(tmp_path, identity):
    with pytest.raises(ValueError, match="path segment"):
        ChainWriter(tmp_path).write({"w": torch.ones(1)}, identity, config={})


def test_unchanged_delta_is_empty(tmp_path):
    writer = ChainWriter(tmp_path)
    commit(writer, {"w": torch.ones(2)}, "a")
    staged = writer.write({"w": torch.ones(2)}, "b")
    assert staged.empty and index_of(staged.path)["weight_map"] == {}
    writer.discard(staged)
    assert writer.delta_count == 0


def test_resume_restores_committed_baseline(tmp_path):
    writer = ChainWriter(tmp_path)
    commit(writer, {"w": torch.ones(2)}, "a")
    commit(writer, {"w": torch.zeros(2)}, "b")
    writer.write({"w": torch.full((2,), 5.0)}, "c")
    resumed = ChainWriter(tmp_path, resume=True)
    assert resumed.identity == "b" and resumed.delta_count == 1
    assert torch.equal(resumed.baseline["w"], torch.zeros(2))
    assert [p.name for p in tmp_path.glob(".baseline-*")] == [".baseline-b"]
    staged = resumed.write({"w": torch.ones(2)}, "d")
    assert staged.base_identity == "b"


def test_resume_rejects_corrupt_baseline(tmp_path):
    writer = ChainWriter(tmp_path)
    commit(writer, {"w": torch.ones(2)}, "a")
    manifest = tmp_path / ".baseline-a" / "baseline.json"
    data = json.loads(manifest.read_text())
    data["checksums"]["w"] = "00000000"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        ChainWriter(tmp_path, resume=True)


def test_per_rank_shards_combine_into_one_index(tmp_path):
    ranks = [ChainWriter(tmp_path / f"rank{r}", rank=r) for r in range(2)]
    parts = [{"a": torch.ones(2)}, {"b": torch.ones(3)}]
    for writer, part in zip(ranks, parts, strict=True):
        commit(writer, part, "full")
    full = combine_indexes(index_of(tmp_path / f"rank{r}" / "full") for r in range(2))
    assert full["metadata"] == {"total_size": 20}
    assert full["weight_map"] == {
        "a": "model-rank000-00000.safetensors",
        "b": "model-rank001-00000.safetensors",
    }
    staged = [
        writer.write({k: v + 1 for k, v in part.items()}, "next")
        for writer, part in zip(ranks, parts, strict=True)
    ]
    delta = combine_indexes(index_of(s.path) for s in staged)
    assert set(delta["metadata"]) == DELTA_METADATA_KEYS
    assert set(delta["weight_map"]) == {"a", "b"}
    with pytest.raises(ValueError, match="more than one rank"):
        combine_indexes([index_of(staged[0].path)] * 2)
