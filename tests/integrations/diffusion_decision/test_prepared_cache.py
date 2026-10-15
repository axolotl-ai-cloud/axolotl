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
