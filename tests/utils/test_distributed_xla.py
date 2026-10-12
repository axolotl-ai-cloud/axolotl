"""
CPU-only tests for XLA device helpers in axolotl.utils.distributed.

torch_xla is not installed in CI; these tests stub the necessary modules so
every XLA branch can be exercised without real TPU hardware.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal torch_xla stubs
# ---------------------------------------------------------------------------

def _make_xla_stubs(device_count: int = 4, local_ordinal: int = 0):
    """Return (xr_mod, xm_mod) stubs for torch_xla.runtime and xla_model."""
    xr = types.ModuleType("torch_xla.runtime")
    xr.addressable_device_count = MagicMock(return_value=device_count)
    xr.local_ordinal = MagicMock(return_value=local_ordinal)

    xm = types.ModuleType("torch_xla.core.xla_model")
    xm.get_local_ordinal = MagicMock(return_value=local_ordinal)
    xm.xla_device = MagicMock(return_value="xla:0")
    xm.rendezvous = MagicMock()
    xm.wait_device_ops = MagicMock()
    xm.set_rng_state = MagicMock()
    xm.get_memory_info = MagicMock(
        return_value={"bytes_used": 1024 ** 3, "bytes_limit": 16 * 1024 ** 3}
    )

    return xr, xm


@pytest.fixture()
def xla_env(monkeypatch):
    """Patch is_xla() → True and inject minimal torch_xla stubs."""
    xr, xm = _make_xla_stubs()

    torch_xla = types.ModuleType("torch_xla")
    torch_xla.runtime = xr
    torch_xla.core = types.ModuleType("torch_xla.core")
    torch_xla.core.xla_model = xm

    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", xr)
    monkeypatch.setitem(sys.modules, "torch_xla.core", torch_xla.core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xm)

    with patch("axolotl.utils.distributed.is_xla", return_value=True):
        # Re-import so get_device_type etc. pick up the patched is_xla
        import importlib
        import axolotl.utils.distributed as dist_mod
        importlib.reload(dist_mod)
        yield dist_mod, xr, xm

    # Reload once more to restore the real module state for subsequent tests
    import importlib
    import axolotl.utils.distributed as dist_mod
    importlib.reload(dist_mod)


# ---------------------------------------------------------------------------
# get_device_type
# ---------------------------------------------------------------------------

class TestGetDeviceTypeXLA:
    def test_returns_xla_device(self, xla_env):
        import torch
        dist_mod, _, _ = xla_env
        with patch.object(dist_mod, "is_xla", return_value=True):
            device = dist_mod.get_device_type()
        assert device == torch.device("xla")

    def test_non_xla_falls_through(self):
        import axolotl.utils.distributed as dist_mod
        with patch.object(dist_mod, "is_xla", return_value=False):
            device = dist_mod.get_device_type()
        # On a CPU-only CI box this will be "cpu"
        assert str(device) in ("cpu", "cuda", "mps", "npu")


# ---------------------------------------------------------------------------
# get_device_count
# ---------------------------------------------------------------------------

class TestGetDeviceCountXLA:
    def test_returns_xla_device_count(self, xla_env):
        dist_mod, xr, _ = xla_env
        with patch.object(dist_mod, "is_xla", return_value=True):
            count = dist_mod.get_device_count()
        assert count == 4
        xr.addressable_device_count.assert_called()

    def test_non_xla_returns_int(self):
        import axolotl.utils.distributed as dist_mod
        with patch.object(dist_mod, "is_xla", return_value=False):
            count = dist_mod.get_device_count()
        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# get_current_device
# ---------------------------------------------------------------------------

class TestGetCurrentDeviceXLA:
    def test_returns_local_ordinal(self, xla_env):
        dist_mod, xr, _ = xla_env
        with patch.object(dist_mod, "is_xla", return_value=True):
            ordinal = dist_mod.get_current_device()
        assert ordinal == 0
        xr.local_ordinal.assert_called()


# ---------------------------------------------------------------------------
# get_device_str
# ---------------------------------------------------------------------------

class TestGetDeviceStrXLA:
    def test_returns_xla_string(self, xla_env):
        dist_mod, _, _ = xla_env
        with patch.object(dist_mod, "is_xla", return_value=True):
            s = dist_mod.get_device_str()
        assert s == "xla"

    def test_non_xla_returns_typed_string(self):
        import axolotl.utils.distributed as dist_mod
        with patch.object(dist_mod, "is_xla", return_value=False):
            s = dist_mod.get_device_str()
        assert ":" in s or s == "cpu"


# ---------------------------------------------------------------------------
# empty_device_cache
# ---------------------------------------------------------------------------

class TestEmptyDeviceCacheXLA:
    def test_noop_on_xla(self, xla_env):
        """Should return without touching torch.cuda or torch.npu."""
        dist_mod, _, _ = xla_env
        with patch.object(dist_mod, "is_xla", return_value=True), \
             patch("torch.cuda.empty_cache") as mock_cuda:
            dist_mod.empty_device_cache()
        mock_cuda.assert_not_called()


# ---------------------------------------------------------------------------
# build_parallelism_config
# ---------------------------------------------------------------------------

class TestBuildParallelismConfigXLA:
    def test_returns_none_none_on_xla(self, xla_env):
        dist_mod, _, _ = xla_env
        from axolotl.utils.dict import DictDefault
        cfg = DictDefault(
            tensor_parallel_size=1,
            context_parallel_size=1,
            dp_shard_size=None,
            dp_replicate_size=None,
            fsdp=None,
            fsdp_config=None,
            deepspeed=None,
            expert_parallel_size=None,
        )
        with patch.object(dist_mod, "is_xla", return_value=True):
            pc, mesh = dist_mod.build_parallelism_config(cfg)
        assert pc is None
        assert mesh is None
