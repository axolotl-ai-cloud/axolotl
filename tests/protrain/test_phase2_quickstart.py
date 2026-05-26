"""Item 3 + Item 4 unit tests.

Item 3: Phase-2 re-pick quickstart predicate (opt-in via
``protrain_phase2_quickstart``). Exercises the pure-function helper
``_phase2_quickstart_should_skip`` so we don't have to spin up the full
GPU Phase-2 measurement path.

Item 4: Searcher's "all configs rejected" RuntimeError appends concrete
fix steps when the root cause is ``cpu_adam_bytes_per_sec=0``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from axolotl.integrations.protrain.api.model_wrapper import (
    _phase2_quickstart_should_skip,
    _teardown_phase2_bootstrap_runtime,
)
from axolotl.integrations.protrain.args import ProTrainArgs
from axolotl.integrations.protrain.search import search

# Reuse the synthetic builders + fixtures from test_cost_search; pytest
# only auto-injects fixtures defined in conftest.py or the importing
# module, so we redeclare local thin wrappers around the shared builders.
from tests.protrain.test_cost_search import (  # noqa: E402
    _make_hw,
    _make_layout,
    _make_trace,
)


@pytest.fixture
def toy_trace():
    return _make_trace()


@pytest.fixture
def toy_layout():
    return _make_layout()


@pytest.fixture
def toy_hw():
    return _make_hw()


# ---------------------------------------------------------------------------
# Item 3: Phase-2 quickstart predicate
# ---------------------------------------------------------------------------


def test_quickstart_disabled_never_skips_even_when_measurement_is_close():
    """Default (quickstart=False): re-pick always fires, even with tight match."""
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.00,
            predicted_iter_s=1.00,
            quickstart=False,
            envelope=0.30,
        )
        is False
    )


def test_quickstart_enabled_skips_when_within_envelope():
    """quickstart=True + measurement within envelope -> skip re-pick."""
    # 10% error, envelope 30% -> skip.
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.10,
            predicted_iter_s=1.00,
            quickstart=True,
            envelope=0.30,
        )
        is True
    )
    # Symmetric: measurement under prediction.
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=0.85,
            predicted_iter_s=1.00,
            quickstart=True,
            envelope=0.30,
        )
        is True
    )


def test_quickstart_enabled_does_not_skip_when_outside_envelope():
    """quickstart=True + measurement outside envelope -> re-pick still fires."""
    # 50% over prediction, envelope 30% -> no skip.
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.50,
            predicted_iter_s=1.00,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )
    # At envelope boundary -> no skip (strict <).
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.30,
            predicted_iter_s=1.00,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )


def test_quickstart_refuses_to_skip_on_nonpositive_predictions():
    """Guard against div-by-zero / garbage measurements."""
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.0,
            predicted_iter_s=0.0,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=0.0,
            predicted_iter_s=1.0,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )


def test_phase2_teardown_closes_bootstrap_when_restore_raises():
    """The rebuild teardown closes wrapper resources even when restore fails."""
    pytest.importorskip("torch")
    from torch import nn

    calls: list[str] = []

    class _Handle:
        def remove(self) -> None:
            calls.append("remove")

    class _ChunkManager:
        def restore_to_gpu(self) -> None:
            calls.append("restore")
            raise RuntimeError("restore boom")

    class _BootWrapped:
        _hook_handles = [_Handle()]

        def close(self) -> None:
            calls.append("close")

    with pytest.raises(RuntimeError, match="restore boom"):
        _teardown_phase2_bootstrap_runtime(
            model=nn.Sequential(),
            blocks=[],
            handles=[_Handle()],
            chunk_manager=_ChunkManager(),
            boot_wrapped=_BootWrapped(),
            context="test phase-2 teardown",
        )

    assert calls == ["remove", "restore", "close"]


# ---------------------------------------------------------------------------
# Item 3: Pydantic args surface the new flags with the expected defaults
# ---------------------------------------------------------------------------


def test_protrain_args_quickstart_defaults():
    """The two new flags default to False / 0.30 and accept overrides."""
    args = ProTrainArgs.model_validate(
        {
            "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
            "protrain_auto_memory": True,
            "base_model": "HuggingFaceTB/SmolLM2-135M",
        }
    )
    assert args.protrain_phase2_quickstart is False
    assert args.protrain_phase2_quickstart_envelope == pytest.approx(0.30)

    args_opt_in = ProTrainArgs.model_validate(
        {
            "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
            "protrain_auto_memory": True,
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "protrain_phase2_quickstart": True,
            "protrain_phase2_quickstart_envelope": 0.15,
        }
    )
    assert args_opt_in.protrain_phase2_quickstart is True
    assert args_opt_in.protrain_phase2_quickstart_envelope == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Item 4: searcher rejection message includes concrete fix steps
# ---------------------------------------------------------------------------


def test_search_rejection_message_includes_fix_steps_when_cpu_adam_zero(
    toy_trace, toy_layout, toy_hw
):
    """When ``cpu_adam_bytes_per_sec=0`` causes every offloaded config to be
    runtime-rejected, the searcher's RuntimeError must include the
    DS_SKIP_CUDA_CHECK / Mode A escape-hatch fix steps."""
    # Force the cpu_adam=0 + all-offload runtime-rejection path:
    #  - cpu_adam_bytes_per_sec=0 makes every n_persist<N_chunk config
    #    return inf from estimate_runtime (round-3 R15 contract).
    #  - capacity_bytes tight enough that all-persistent configs are
    #    capacity-rejected, leaving only offloaded (runtime-rejected) configs.
    hw_no_adam = replace(
        toy_hw,
        cpu_adam_bytes_per_sec=0.0,
        gpu_adam_bytes_per_sec=0.0,
    )

    # toy_layout: N_chunk=12, S_chunk=64MB. All-persistent footprint alone
    # exceeds ~768MB, so capacity=500MB rules out persistent configs while
    # leaving offloaded configs as the "feasible by capacity, infeasible by
    # runtime" set.
    MB = 1 << 20
    capacity_bytes = 500 * MB

    with pytest.raises(RuntimeError) as excinfo:
        search(toy_trace, toy_layout, capacity_bytes, hw_no_adam)

    msg = str(excinfo.value)
    # Diagnostic preserved.
    assert "cpu_adam_bytes_per_sec=0" in msg, msg
    # Concrete remediation steps must appear.
    assert "DS_SKIP_CUDA_CHECK" in msg, msg
    assert "Mode A" in msg, msg
    assert "pip install deepspeed" in msg, msg
    assert "CUDA_HOME" in msg, msg
    assert "protrain_force_all_persistent" in msg, msg


def test_search_rejection_message_unchanged_when_cpu_adam_positive(
    toy_trace, toy_layout, toy_hw
):
    """Non-zero cpu_adam keeps the original concise rejection message."""
    # capacity_bytes=0 still triggers the no-feasible-config branch, which
    # is a *different* failure mode (capacity gate, not runtime gate). The
    # new fix-step message must NOT leak into unrelated failure paths.
    with pytest.raises(RuntimeError) as excinfo:
        search(toy_trace, toy_layout, 0, toy_hw)
    msg = str(excinfo.value)
    assert "DS_SKIP_CUDA_CHECK" not in msg, msg
