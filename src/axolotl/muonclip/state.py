"""State management helpers for MuonClip momentum/RMS buffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class MuonParameterState:
    """Holds momentum/RMS buffers for a single parameter."""

    momentum: torch.Tensor
    rms: Optional[torch.Tensor] = None


class MuonStateStore:
    """
    Lightweight registry that tracks optimizer-style buffers per parameter.

    Fallbacks to CPU when requested so unit tests can exercise logic without GPUs.
    """

    def __init__(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        pin_memory: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.pin_memory = pin_memory
        self._states: Dict[int, MuonParameterState] = {}

    def get_or_create(
        self,
        param: torch.nn.Parameter,
        *,
        with_rms: bool = False,
    ) -> MuonParameterState:
        key = id(param)
        if key in self._states:
            state = self._states[key]
            if with_rms and state.rms is None:
                state.rms = self._allocate_like(param)
            self._ensure_device(state, param)
            return state

        momentum = self._allocate_like(param)
        rms = self._allocate_like(param) if with_rms else None
        state = MuonParameterState(momentum=momentum, rms=rms)
        self._states[key] = state
        self._ensure_device(state, param)
        return state

    def peek(self, param: torch.nn.Parameter) -> MuonParameterState | None:
        """
        Return the existing state entry for `param` without creating new buffers.
        """

        return self._states.get(id(param))

    def _ensure_device(self, state: MuonParameterState, param: torch.nn.Parameter):
        target_device = param.device
        if state.momentum.device != target_device:
            state.momentum = state.momentum.to(target_device)
        if state.rms is not None and state.rms.device != target_device:
            state.rms = state.rms.to(target_device)

    def _allocate_like(self, param: torch.nn.Parameter) -> torch.Tensor:
        data = param.data
        device = self.device if self.device is not None else data.device
        dtype = self.dtype if self.dtype is not None else data.dtype
        buffer = torch.zeros_like(data, device=device, dtype=dtype)
        if self.pin_memory and buffer.device.type == "cpu":
            buffer = buffer.pin_memory()
        return buffer

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Serialize buffers into a dict keyed by tensor id - checkpoint integration will rehydrate.
        """

        result: Dict[str, torch.Tensor] = {}
        for key, state in self._states.items():
            result[f"{key}:momentum"] = state.momentum
            if state.rms is not None:
                result[f"{key}:rms"] = state.rms
        return result

    def load_state_dict(self, buffers: Dict[str, torch.Tensor]) -> None:
        for key, tensor in buffers.items():
            ptr_str, kind = key.split(":")
            ptr = int(ptr_str)
            state = self._states.get(ptr)
            if state is None:
                continue
            if kind == "momentum":
                state.momentum.copy_(tensor)
            elif kind == "rms":
                if state.rms is None:
                    state.rms = torch.zeros_like(tensor)
                state.rms.copy_(tensor)
