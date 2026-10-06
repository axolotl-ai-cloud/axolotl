

def test_resolved_tokenizer_revision_falls_back_to_cached_snapshot(monkeypatch, tmp_path):
    """transformers 5 tokenizers carry no _commit_hash; the Hub cache snapshot path still names it."""
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision import prepared_cache

    sha = "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc"
    snapshot = tmp_path / "models--org--name" / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer_config.json").write_text("{}")

    def fake_cache(repo_id, filename, repo_type="model"):
        assert repo_id == "org/name"
        return str(snapshot / filename) if filename == "tokenizer_config.json" else None

    monkeypatch.setattr(prepared_cache, "try_to_load_from_cache", fake_cache)
    tokenizer = SimpleNamespace(name_or_path="org/name", init_kwargs={})
    assert prepared_cache._resolved_tokenizer_revision({}, tokenizer) == sha
    assert prepared_cache._resolved_tokenizer_revision({}, SimpleNamespace(name_or_path=None)) is None
