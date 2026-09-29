"""ddp_timeout must reach Ray's TorchConfig, which builds the process group first."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from axolotl.cli import train as train_cli
from axolotl.utils.dict import DictDefault


def _fake_ray(monkeypatch):
    ray = types.ModuleType("ray")
    ray_train = types.ModuleType("ray.train")
    ray_train_torch = types.ModuleType("ray.train.torch")
    ray_train.RunConfig = MagicMock(name="RunConfig")
    ray_train.ScalingConfig = MagicMock(name="ScalingConfig")
    ray_train_torch.TorchConfig = MagicMock(name="TorchConfig")
    ray_train_torch.TorchTrainer = MagicMock(name="TorchTrainer")
    ray.train = ray_train
    ray_train.torch = ray_train_torch
    for name, module in (
        ("ray", ray),
        ("ray.train", ray_train),
        ("ray.train.torch", ray_train_torch),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return ray_train_torch


@pytest.mark.parametrize("ddp_timeout", [21600, None])
def test_ray_trainer_receives_ddp_timeout(monkeypatch, tmp_path, ddp_timeout):
    ray_torch = _fake_ray(monkeypatch)
    cfg = DictDefault(
        use_ray=True,
        ddp_timeout=ddp_timeout,
        ray_num_workers=2,
        resources_per_worker=DictDefault(GPU=1),
        ray_run_name="run",
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(train_cli, "load_cfg", lambda *a, **k: cfg, raising=False)
    parser = MagicMock()
    parser.return_value.parse_args_into_dataclasses.return_value = (None, [])
    monkeypatch.setattr(train_cli, "HfArgumentParser", parser, raising=False)

    train_cli.do_cli(config=str(tmp_path / "config.yaml"))

    trainer_kwargs = ray_torch.TorchTrainer.call_args.kwargs
    if ddp_timeout is None:
        ray_torch.TorchConfig.assert_not_called()
        assert trainer_kwargs["torch_config"] is None
    else:
        ray_torch.TorchConfig.assert_called_once_with(timeout_s=ddp_timeout)
        assert trainer_kwargs["torch_config"] is ray_torch.TorchConfig.return_value
    ray_torch.TorchTrainer.return_value.fit.assert_called_once()
