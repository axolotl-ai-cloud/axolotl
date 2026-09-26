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
        "gradient_checkpointing": True,
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
                gradient_checkpointing_kwargs={"use_reentrant": True},
            )

    @pytest.mark.parametrize("mode", [True, "legacy", "disk"])
    def test_rejects_trl_activation_offloading(self, min_base_cfg, mode):
        with pytest.raises(ValueError, match="incompatible with activation_offloading"):
            _validate(
                min_base_cfg,
                expert_parallel_backend="torch",
                activation_offloading=mode,
            )

    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
            {"activation_offloading": "hidden_states"},
            {"selective_checkpointing": True},
        ],
    )
    def test_accepts(self, min_base_cfg, extra):
        cfg = _validate(min_base_cfg, expert_parallel_backend="torch", **extra)
        assert cfg.expert_parallel_backend == "torch"
        assert cfg.expert_parallel_save_dispatch is True

    def test_deep_ep_allows_reentrant(self, min_base_cfg):
        _validate(
            min_base_cfg,
            expert_parallel_backend="deep_ep",
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

    def test_ep_disabled_allows_reentrant(self, min_base_cfg):
        cfg = (
            DictDefault(
                plugins=[EP_PLUGIN],
                expert_parallel_backend="torch",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
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
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
        if installed:
            _validate(min_base_cfg, **kwargs)
        else:
            with pytest.raises(ValueError, match="non-reentrant"):
                _validate(min_base_cfg, **kwargs)
