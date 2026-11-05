"""Config validation for the torch expert-parallel backend under gradient checkpointing."""

import importlib.util

import pytest

from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

EP_PLUGIN = "axolotl.integrations.expert_parallel.ExpertParallelPlugin"


def _validate(min_base_cfg, **kw):
    base = {
        "plugins": [EP_PLUGIN],
        "expert_parallel_size": 2,
        "activation_checkpointing": True,
        "fsdp_version": 2,
        "fsdp_config": {"state_dict_type": "FULL_STATE_DICT"},
    }
    cfg = DictDefault({**base, **kw}) | min_base_cfg
    prepare_plugins(cfg)
    return validate_config(cfg)


def _fake_deep_ep(monkeypatch, installed):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "deep_ep":
            return object() if installed else None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


class TestTorchExpertParallelCheckpointing:
    def test_rejects_reentrant(self, min_base_cfg):
        with pytest.raises(ValueError, match="non-reentrant"):
            _validate(
                min_base_cfg,
                expert_parallel_backend="torch",
                activation_checkpointing={"reentrant": True},
            )

    def test_allows_checkpointing_off(self, min_base_cfg):
        cfg = _validate(
            min_base_cfg,
            expert_parallel_backend="torch",
            activation_checkpointing=False,
        )
        assert cfg.activation_checkpointing.enabled is False

    def test_reentrant_without_checkpointing_rejected(self, min_base_cfg):
        with pytest.raises(ValueError, match="requires checkpointing"):
            _validate(
                min_base_cfg,
                expert_parallel_backend="torch",
                activation_checkpointing={"enabled": False, "reentrant": True},
            )

    @pytest.mark.parametrize(
        "placement",
        [
            True,
            "cpu",
            pytest.param(
                "disk",
                marks=pytest.mark.skip(reason="disk offload is disabled pending a fix"),
            ),
        ],
    )
    def test_accepts_trl_activation_offloading(self, min_base_cfg, placement):
        cfg = _validate(
            min_base_cfg,
            expert_parallel_backend="torch",
            activation_checkpointing={"offload": {"all": placement}},
        )
        expected = "cpu" if placement is True else placement
        assert cfg.activation_checkpointing.offload.all == expected

    @pytest.mark.parametrize("mode", [True, "legacy"])
    def test_migrates_legacy_activation_offloading(self, min_base_cfg, mode):
        cfg = _validate(
            min_base_cfg,
            expert_parallel_backend="torch",
            activation_checkpointing=None,
            gradient_checkpointing=True,
            activation_offloading=mode,
        )
        assert cfg.activation_checkpointing.enabled is True
        assert cfg.activation_checkpointing.offload.all == "cpu"

    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"activation_checkpointing": {"reentrant": False}},
            {"activation_checkpointing": {"offload": {"hidden_states": "cpu"}}},
            {"activation_checkpointing": {"selective": True}},
        ],
    )
    def test_accepts(self, min_base_cfg, extra):
        cfg = _validate(min_base_cfg, expert_parallel_backend="torch", **extra)
        assert cfg.expert_parallel_backend == "torch"
        assert cfg.expert_parallel_save_dispatch is False

    def test_deep_ep_allows_reentrant(self, min_base_cfg):
        _validate(
            min_base_cfg,
            expert_parallel_backend="deep_ep",
            activation_checkpointing={"reentrant": True},
        )

    def test_ep_disabled_allows_reentrant(self, min_base_cfg):
        cfg = (
            DictDefault(
                plugins=[EP_PLUGIN],
                expert_parallel_backend="torch",
                activation_checkpointing={"enabled": True, "reentrant": True},
            )
            | min_base_cfg
        )
        prepare_plugins(cfg)
        validate_config(cfg)

    @pytest.mark.parametrize("installed", [True, False])
    def test_auto_follows_deep_ep_availability(
        self, min_base_cfg, monkeypatch, installed
    ):
        _fake_deep_ep(monkeypatch, installed)
        kwargs = dict(
            expert_parallel_backend="auto",
            activation_checkpointing={"reentrant": True},
        )
        if installed:
            _validate(min_base_cfg, **kwargs)
        else:
            with pytest.raises(ValueError, match="non-reentrant"):
                _validate(min_base_cfg, **kwargs)


class TestExpertParallelDispatchChunks:
    def test_rejects_deep_ep(self, min_base_cfg):
        with pytest.raises(ValueError, match="requires expert_parallel_backend: torch"):
            _validate(
                min_base_cfg,
                expert_parallel_backend="deep_ep",
                expert_parallel_dispatch_chunks=2,
            )

    def test_accepts_torch(self, min_base_cfg):
        cfg = _validate(
            min_base_cfg,
            expert_parallel_backend="torch",
            expert_parallel_dispatch_chunks=2,
        )
        assert cfg.expert_parallel_dispatch_chunks == 2

    @pytest.mark.parametrize("installed", [True, False])
    def test_auto_backend_is_resolved(self, min_base_cfg, monkeypatch, installed):
        _fake_deep_ep(monkeypatch, installed)
        kwargs = dict(expert_parallel_backend="auto", expert_parallel_dispatch_chunks=2)
        if installed:
            with pytest.raises(
                ValueError, match="requires expert_parallel_backend: torch"
            ):
                _validate(min_base_cfg, **kwargs)
        else:
            assert (
                _validate(min_base_cfg, **kwargs).expert_parallel_dispatch_chunks == 2
            )

    def test_defaults_to_one(self, min_base_cfg):
        cfg = _validate(min_base_cfg, expert_parallel_backend="deep_ep")
        assert cfg.expert_parallel_dispatch_chunks == 1

    def test_rejects_zero(self, min_base_cfg):
        with pytest.raises(ValueError, match="expert_parallel_dispatch_chunks"):
            _validate(
                min_base_cfg,
                expert_parallel_backend="torch",
                expert_parallel_dispatch_chunks=0,
            )
