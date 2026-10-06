"""CPU contract tests for native Nemotron source and revision forwarding."""

from types import SimpleNamespace

import pytest

from axolotl.model_support.nemotron_diffusion import _model_class, compat


class _MockNemotron:
    calls: list[tuple[object, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        return cls()

    @classmethod
    def _from_config(cls, config, **kwargs):
        cls.calls.append((config, kwargs))
        return cls()


@pytest.mark.parametrize(
    ("source", "revision"),
    [
        ("nvidia/Nemotron-Labs-Diffusion-3B", None),
        (
            "nvidia/Nemotron-Labs-Diffusion-8B",
            "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc",
        ),
        ("nvidia/Nemotron-Labs-Diffusion-8B", "arbitrary-tested-revision"),
    ],
)
def test_factory_forwards_hub_revision_untouched(monkeypatch, source, revision):
    resolved: list[tuple[object, object]] = []
    _MockNemotron.calls.clear()

    def resolve(model_source, *, revision=None):
        resolved.append((model_source, revision))
        return _MockNemotron

    monkeypatch.setattr(compat, "resolve_nemotron_model_class", resolve)
    assert isinstance(
        _model_class().from_pretrained(source, revision=revision), _MockNemotron
    )
    assert resolved == [(source, revision)]
    assert _MockNemotron.calls == [(source, {"revision": revision})]


def test_factory_preserves_config_commit_hash(monkeypatch):
    resolved: list[tuple[object, object]] = []
    _MockNemotron.calls.clear()

    def resolve(model_source, *, revision=None):
        resolved.append((model_source, revision))
        return _MockNemotron

    config = SimpleNamespace(
        _name_or_path="nvidia/Nemotron-Labs-Diffusion-8B", _commit_hash="custom-commit"
    )
    monkeypatch.setattr(compat, "resolve_nemotron_model_class", resolve)
    assert isinstance(
        _model_class().from_config(config, trust_remote_code=True), _MockNemotron
    )
    assert resolved == [(config._name_or_path, "custom-commit")]
    assert _MockNemotron.calls == [(config, {})]


def test_factory_records_remote_snapshot_revision(monkeypatch):
    _MockNemotron.calls.clear()
    monkeypatch.setattr(
        _MockNemotron, "_axolotl_resolved_revision", "resolved-snapshot", raising=False
    )
    config = SimpleNamespace(_commit_hash=None)

    monkeypatch.setattr(
        compat, "resolve_nemotron_model_class", lambda *args, **kwargs: _MockNemotron
    )
    assert isinstance(
        _model_class().from_pretrained(
            "nvidia/Nemotron-Labs-Diffusion-8B", config=config
        ),
        _MockNemotron,
    )

    assert config._commit_hash == "resolved-snapshot"
    assert _MockNemotron.calls == [
        (
            "nvidia/Nemotron-Labs-Diffusion-8B",
            {"config": config, "revision": "resolved-snapshot"},
        )
    ]


def test_factory_pins_weights_without_a_config(monkeypatch):
    _MockNemotron.calls.clear()
    revision = "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc"
    monkeypatch.setattr(
        _MockNemotron, "_axolotl_resolved_revision", revision, raising=False
    )
    monkeypatch.setattr(
        compat, "resolve_nemotron_model_class", lambda *args, **kwargs: _MockNemotron
    )

    assert isinstance(
        _model_class().from_pretrained("nvidia/Nemotron-Labs-Diffusion-8B"),
        _MockNemotron,
    )
    assert _MockNemotron.calls == [
        ("nvidia/Nemotron-Labs-Diffusion-8B", {"revision": revision})
    ]


def test_remote_source_download_receives_unmodified_revision(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in (
        "configuration_nemotron_labs_diffusion.py",
        "modeling_ministral.py",
        "modeling_nemotron_labs_diffusion.py",
    ):
        (snapshot / name).write_text("# fixture\n")
    download: list[dict[str, object]] = []

    def fake_download(**kwargs):
        download.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *args, **kwargs: _MockNemotron,
    )
    compat.resolve_nemotron_model_class(
        "nvidia/Nemotron-Labs-Diffusion-8B", revision="custom-revision"
    )
    assert download == [
        {
            "repo_id": "nvidia/Nemotron-Labs-Diffusion-8B",
            "revision": "custom-revision",
            "allow_patterns": [
                "config.json",
                "configuration_nemotron_labs_diffusion.py",
                "modeling_ministral.py",
                "modeling_nemotron_labs_diffusion.py",
            ],
        }
    ]


def test_remote_source_exposes_snapshot_revision(monkeypatch, tmp_path):
    revision = "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc"
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in (
        "configuration_nemotron_labs_diffusion.py",
        "modeling_ministral.py",
        "modeling_nemotron_labs_diffusion.py",
    ):
        (snapshot / name).write_text("# fixture\n")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda **kwargs: str(snapshot)
    )
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *args, **kwargs: _MockNemotron,
    )

    assert (
        compat.resolve_nemotron_model_class(
            "nvidia/Nemotron-Labs-Diffusion-8B"
        )._axolotl_resolved_revision
        == revision
    )


def test_remote_source_rejects_nonhex_snapshot_revision(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / ("z" * 40)
    snapshot.mkdir(parents=True)
    for name in (
        "configuration_nemotron_labs_diffusion.py",
        "modeling_ministral.py",
        "modeling_nemotron_labs_diffusion.py",
    ):
        (snapshot / name).write_text("# fixture\n")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda **kwargs: str(snapshot)
    )
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *args, **kwargs: _MockNemotron,
    )

    assert (
        compat.resolve_nemotron_model_class(
            "nvidia/Nemotron-Labs-Diffusion-8B"
        )._axolotl_resolved_revision
        is None
    )


def test_local_source_with_missing_native_files_fails_clearly(tmp_path):
    (tmp_path / "modeling_nemotron_labs_diffusion.py").write_text("# fixture\n")
    with pytest.raises(ValueError, match="lacks required native files"):
        compat.resolve_nemotron_model_class(tmp_path, revision="any-local-revision")


def test_incompatible_encoder_fails_before_attention_patching(monkeypatch, tmp_path):
    for name in (
        "configuration_nemotron_labs_diffusion.py",
        "modeling_ministral.py",
        "modeling_nemotron_labs_diffusion.py",
    ):
        (tmp_path / name).write_text("# fixture\n")

    class BadNemotron:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *args, **kwargs: BadNemotron,
    )
    model_class = compat.resolve_nemotron_model_class(tmp_path)
    with pytest.raises(ValueError, match="encoder.layers with self_attn"):
        model_class(SimpleNamespace(dlm_paradigm="bidirectional"))


def test_keep_rotary_fp32_disables_autocast_inside_rotary_forward():
    import torch

    from axolotl.model_support.nemotron_diffusion.compat import keep_rotary_fp32

    class Rotary(torch.nn.Module):
        def forward(self, x, position_ids):
            freqs = torch.ones(1, 4, 1) @ position_ids[:, None, :].float()
            return freqs.cos(), torch.tensor(torch.is_autocast_enabled("cpu"))

    rotary = Rotary()
    keep_rotary_fp32(rotary)
    x = torch.zeros(1, 8, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cos, autocast_inside = rotary(x, torch.arange(8)[None])

    assert cos.dtype == torch.float32
    assert not bool(autocast_inside)
