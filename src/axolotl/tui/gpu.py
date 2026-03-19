"""GPU polling wrapper around pynvml with graceful fallback."""

from __future__ import annotations

import logging

from axolotl.tui.state import GPUStats

LOG = logging.getLogger(__name__)

_nvml_available = False
try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    LOG.debug("pynvml unavailable — GPU stats will not be shown")


class GPUPoller:
    """Polls local GPU stats via pynvml. Falls back gracefully if unavailable."""

    def __init__(self):
        self._device_count = 0
        if _nvml_available:
            try:
                self._device_count = pynvml.nvmlDeviceGetCount()
            except Exception:
                self._device_count = 0

    @property
    def available(self) -> bool:
        return _nvml_available and self._device_count > 0

    def poll(self) -> list[GPUStats]:
        if not self.available:
            return []

        stats = []
        for i in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power = None

                stats.append(
                    GPUStats(
                        id=i,
                        name=name,
                        util_pct=util.gpu,
                        vram_used_gb=mem.used / (1024**3),
                        vram_total_gb=mem.total / (1024**3),
                        temp_c=temp,
                        power_w=power,
                    )
                )
            except Exception:
                LOG.debug("Error polling GPU device %d", i, exc_info=True)
        return stats
