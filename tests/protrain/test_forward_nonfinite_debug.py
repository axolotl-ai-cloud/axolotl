"""Forward-side non-finite diagnostics for ProTrain block hooks."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from axolotl.integrations.protrain.runtime.hooks import _make_forward_post_hook
from axolotl.integrations.protrain.types import BlockId


class _Scheduler:
    def __init__(self) -> None:
        self.calls: list[BlockId] = []

    def post_block_forward(self, block_id: BlockId) -> None:
        self.calls.append(block_id)


def test_forward_nonfinite_debug_hook_raises_with_block_context(monkeypatch) -> None:
    monkeypatch.setenv("PROTRAIN_DEBUG_FORWARD_NONFINITE", "1")
    scheduler = _Scheduler()
    hook = _make_forward_post_hook(scheduler, BlockId(7))

    with pytest.raises(RuntimeError, match="non-finite forward output") as exc_info:
        hook(nn.Linear(2, 2), (), {"hidden": torch.tensor([1.0, float("nan")])})

    msg = str(exc_info.value)
    assert "block=7" in msg
    assert "path=output.hidden" in msg
    assert "nonfinite=1/2" in msg
    assert scheduler.calls == []


def test_forward_nonfinite_debug_hook_ignores_finite_outputs(monkeypatch) -> None:
    monkeypatch.setenv("PROTRAIN_DEBUG_FORWARD_NONFINITE", "1")
    scheduler = _Scheduler()
    hook = _make_forward_post_hook(scheduler, BlockId(3))

    hook(nn.Linear(2, 2), (), (torch.tensor([1.0, 2.0]),))

    assert scheduler.calls == [BlockId(3)]


def test_forward_nonfinite_debug_hook_is_env_gated(monkeypatch) -> None:
    monkeypatch.delenv("PROTRAIN_DEBUG_FORWARD_NONFINITE", raising=False)
    scheduler = _Scheduler()
    hook = _make_forward_post_hook(scheduler, BlockId(5))

    hook(nn.Linear(2, 2), (), cast(Any, torch.tensor([float("nan")])))

    assert scheduler.calls == [BlockId(5)]
