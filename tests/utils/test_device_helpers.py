"""XPU takes the same device-selection paths as CUDA."""

import os

import torch

from axolotl.utils import distributed
from axolotl.utils.config import choose_device
from axolotl.utils.dict import DictDefault


def _only_available(monkeypatch, *, cuda=False, xpu=False, mps=False, npu=False):
    monkeypatch.setattr(distributed, "is_torch_cuda_available", lambda: cuda)
    monkeypatch.setattr(distributed, "is_torch_xpu_available", lambda: xpu)
    monkeypatch.setattr(distributed, "is_torch_mps_available", lambda: mps)
    monkeypatch.setattr(distributed, "is_torch_npu_available", lambda: npu)


class TestGetDeviceType:
    def test_xpu_is_selected_when_it_is_the_only_accelerator(self, monkeypatch):
        _only_available(monkeypatch, xpu=True)
        assert distributed.get_device_type() == torch.device("xpu")

    def test_cuda_wins_over_xpu(self, monkeypatch):
        _only_available(monkeypatch, cuda=True, xpu=True)
        assert distributed.get_device_type() == torch.device("cuda")

    def test_cpu_when_nothing_is_available(self, monkeypatch):
        _only_available(monkeypatch)
        assert distributed.get_device_type() == torch.device("cpu")
        assert distributed.get_device_count() == 1
        assert distributed.get_current_device() == 0

    def test_xpu_count_and_current_device_come_from_torch_xpu(self, monkeypatch):
        _only_available(monkeypatch, xpu=True)
        monkeypatch.setattr(torch.xpu, "device_count", lambda: 4)
        monkeypatch.setattr(torch.xpu, "current_device", lambda: 2)
        assert distributed.get_device_count() == 4
        assert distributed.get_current_device() == 2


class TestChooseDevice:
    @staticmethod
    def _cfg(world_size, local_rank=0):
        return DictDefault({"world_size": world_size, "local_rank": local_rank})

    @staticmethod
    def _only_xpu(monkeypatch):
        for key in [k for k in os.environ if k.startswith("ACCELERATE_USE_")]:
            monkeypatch.delenv(key)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.xpu, "is_available", lambda: True)

    def test_single_process_selects_xpu(self, monkeypatch):
        self._only_xpu(monkeypatch)
        cfg = self._cfg(world_size=1)
        choose_device(cfg)
        assert cfg.device == "xpu:0"
        assert cfg.device_map == "auto"

    def test_cuda_still_wins_over_xpu(self, monkeypatch):
        self._only_xpu(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        cfg = self._cfg(world_size=1)
        choose_device(cfg)
        assert cfg.device == "cuda:0"
