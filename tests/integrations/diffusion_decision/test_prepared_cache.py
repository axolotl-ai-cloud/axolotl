import json
import os


def test_resolved_tokenizer_revision_falls_back_to_cached_snapshot(
    monkeypatch, tmp_path
):
    """transformers 5 tokenizers carry no _commit_hash; the Hub cache snapshot path still names it."""
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision import prepared_cache

    sha = "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc"
    snapshot = tmp_path / "models--org--name" / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer_config.json").write_text("{}")

    tag_sha = "0123456789abcdef0123456789abcdef01234567"
    tag_snapshot = tmp_path / "models--org--name" / "snapshots" / tag_sha
    tag_snapshot.mkdir(parents=True)
    (tag_snapshot / "tokenizer_config.json").write_text("{}")

    def fake_cache(repo_id, filename, revision=None, repo_type="model"):
        assert repo_id == "org/name"
        if filename != "tokenizer_config.json":
            return None
        root = tag_snapshot if revision == "v1" else snapshot
        return str(root / filename)

    monkeypatch.setattr(prepared_cache, "try_to_load_from_cache", fake_cache)
    tokenizer = SimpleNamespace(name_or_path="org/name", init_kwargs={})
    assert prepared_cache._resolved_tokenizer_revision({}, tokenizer) == sha
    # A branch/tag pin resolves through its own ref, not main's snapshot.
    assert (
        prepared_cache._resolved_tokenizer_revision(
            {"revision_of_model": "v1"}, tokenizer
        )
        == tag_sha
    )
    assert (
        prepared_cache._resolved_tokenizer_revision(
            {}, SimpleNamespace(name_or_path=None)
        )
        is None
    )


def _rows():
    from axolotl.integrations.diffusion_decision.loss import (
        decision_example_from_canvas,
    )
    from axolotl.integrations.diffusion_decision.records import OrdinalMetadata
    from axolotl.integrations.diffusion_decision.slots import SlotPlan

    from tests.integrations.diffusion_decision.helpers import make_canvas

    canvas = make_canvas(
        slot_mask=(True, False, False, False),
        ordinal_metadata=(OrdinalMetadata(("low",), ("0",), (0,)),),
    )
    train = [
        {
            "canvas": canvas,
            "decision_example": decision_example_from_canvas(canvas, source_weight=0.5),
            "record": {"id": "train-id", "grouped_record_ids": ("a", "b")},
            "source": "alpha",
            "length": 6,
            "slot_plan": SlotPlan((9,), "thought", (True,), (False,), (False,), (9,)),
        }
    ]
    eval_rows = [{"canvas": canvas, "record": {"id": "eval-id"}, "source": "beta"}]
    manifest = {
        "stratified_epoch_batches": ((0, 1), (2,)),
        "budget_drops": {"train": {"logical": 0, "physical": 0}},
    }
    return train, eval_rows, manifest


IDENTITY = {"tokenizer": {"vocab": [("a", 1)]}, "sources": [("/x", 3, 4)]}


def test_prepared_cache_round_trip(tmp_path):
    from axolotl.integrations.diffusion_decision import prepared_cache

    train, eval_rows, manifest = _rows()
    prepared_cache.store(tmp_path, "k", IDENTITY, {}, train, eval_rows, manifest)
    for verify in (False, True):
        loaded = prepared_cache.load(tmp_path, "k", IDENTITY, verify=verify)
        assert loaded is not None
        assert loaded[0] == train
        assert loaded[1] == eval_rows
        assert loaded[2] == manifest
        assert loaded[2]["stratified_epoch_batches"] == ((0, 1), (2,))
        assert isinstance(loaded[2]["stratified_epoch_batches"][0], tuple)
    assert not list((tmp_path / prepared_cache.CACHE_DIR).glob("*.tmp"))


def _tamper(path, old: bytes, new: bytes) -> None:
    data = path.read_bytes()
    assert data.count(old) == 1 and len(old) == len(new)
    path.write_bytes(data.replace(old, new))


def test_prepared_cache_verify_detects_tampered_rows(tmp_path):
    from axolotl.integrations.diffusion_decision import prepared_cache

    train, eval_rows, manifest = _rows()
    prepared_cache.store(tmp_path, "k", IDENTITY, {}, train, eval_rows, manifest)
    _tamper(prepared_cache._cache_path(tmp_path, "k"), b"train-id", b"TRAIN-ID")
    # Without verification a well-formed but altered pickle still loads...
    unverified = prepared_cache.load(tmp_path, "k", IDENTITY)
    assert unverified is not None
    assert unverified[0][0]["record"]["id"] == "TRAIN-ID"
    # ...and verification rejects it.
    assert prepared_cache.load(tmp_path, "k", IDENTITY, verify=True) is None


def test_prepared_cache_rejects_identity_mismatch_without_verify(tmp_path):
    from axolotl.integrations.diffusion_decision import prepared_cache

    train, eval_rows, manifest = _rows()
    prepared_cache.store(tmp_path, "k", IDENTITY, {}, train, eval_rows, manifest)
    other = {**IDENTITY, "sources": [("/x", 3, 5)]}
    assert prepared_cache.load(tmp_path, "k", other, verify=False) is None
    assert prepared_cache.load(tmp_path, "k", IDENTITY, verify=False) is not None


def test_prepared_cache_damaged_files_fall_back(tmp_path):
    from axolotl.integrations.diffusion_decision import prepared_cache

    train, eval_rows, manifest = _rows()
    prepared_cache.store(tmp_path, "k", IDENTITY, {}, train, eval_rows, manifest)
    path = prepared_cache._cache_path(tmp_path, "k")
    data = path.read_bytes()
    damaged = [
        data[: len(data) // 2],  # truncated
        data[:-1],  # trailer cut
        b"",
        b"garbage" * 50,
        data[:20] + bytes(64) + data[84:],  # pickle stream garbled, same size
    ]
    for payload in damaged:
        path.write_bytes(payload)
        for verify in (False, True):
            assert prepared_cache.load(tmp_path, "k", IDENTITY, verify=verify) is None
    # Old JSON caches are simply not looked at.
    path.unlink()
    (path.parent / "k.json").write_text('{"sha256": "x", "content": {}}')
    assert prepared_cache.load(tmp_path, "k", IDENTITY) is None


def test_prepared_cache_refuses_foreign_globals(tmp_path):
    import pickle

    from axolotl.integrations.diffusion_decision import prepared_cache

    prepared_cache.store(tmp_path, "k", IDENTITY, {}, [{"x": 1}], [], {})
    path = prepared_cache._cache_path(tmp_path, "k")
    data = path.read_bytes()
    header_end = len(data) - prepared_cache._TRAILER.size
    header_length, _ = prepared_cache._TRAILER.unpack(data[header_end:])
    header = json.loads(data[header_end - header_length : header_end])
    rows = pickle.dumps(
        {"train": [{"x": os.system}], "eval": [], "manifest": {}}, protocol=5
    )
    header["rows_length"] = len(rows)
    encoded = json.dumps(header).encode()
    path.write_bytes(
        prepared_cache.MAGIC
        + rows
        + encoded
        + prepared_cache._TRAILER.pack(len(encoded), prepared_cache.MAGIC)
    )
    assert prepared_cache.load(tmp_path, "k", IDENTITY) is None


def test_prepared_cache_source_stat_identity(tmp_path):
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision import prepared_cache

    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("tokenizer")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")

    class Tokenizer(SimpleNamespace):
        backend_tokenizer = SimpleNamespace(to_str=lambda: "backend")

        def get_vocab(self):
            return {"x": 0}

        def __len__(self):
            return 1

    tokenizer = Tokenizer(name_or_path=str(root))
    spec = SimpleNamespace(kind="x")

    def key(verify, cfg=None):
        return prepared_cache.identity(
            cfg or {}, tokenizer, spec, [source], verify=verify
        )[0]

    stat_key, sha_key = key(False), key(True)
    assert stat_key != sha_key
    stat = source.stat()
    # Same content, new mtime: the stat key misses, the content key does not.
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert key(False) != stat_key
    assert key(True) == sha_key
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert key(False) == stat_key
    # Size change misses.
    source.write_text("{} \n")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert key(False) != stat_key
    # The verify toggle itself is not part of the semantic config.
    assert key(False, {"diffusion_decision": {"prepared_cache_verify": True}}) == key(
        False, {"diffusion_decision": {"prepared_cache_verify": False}}
    )
