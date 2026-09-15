"""CPU tests for the merge-aware NVFP4 schedule callback + metadata stamping."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torchao")

from axolotl.integrations.kernels.libs.sonicmoe import (
    merge_aware_enabled,
    set_merge_aware_enabled,
)
from axolotl.integrations.kernels.merge_aware_callback import (
    MergeAwareScheduleCallback,
    write_merge_aware_metadata,
)


def _state(step, max_steps=1000):
    return SimpleNamespace(
        global_step=step, max_steps=max_steps, is_world_process_zero=True
    )


def test_write_metadata_roundtrip(tmp_path):
    import torchao

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 4, "lora_alpha": 8, "peft_type": "LORA"})
    )
    assert write_merge_aware_metadata(tmp_path, start_step=100)
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["r"] == 4
    meta = cfg["nvfp4_merge_aware"]
    assert meta["scale_mode"] == "fresh"
    assert meta["pts_policy"] == "base_fused_max"
    assert meta["encoder"] == f"torchao-{torchao.__version__}"
    assert meta["start_step"] == 100
    # NOTE: no lora_merge import here; that module configures axolotl logging on
    # import (propagate=False) and breaks caplog-based tests later in the session.
    # The metadata -> scale-mode resolution is covered in tests/utils/lora/.


def test_write_metadata_missing_config(tmp_path):
    assert not write_merge_aware_metadata(tmp_path)


def test_schedule_absolute_and_fractional_start():
    try:
        cb = MergeAwareScheduleCallback(100)
        cb.on_train_begin(None, _state(0), None)
        assert not merge_aware_enabled()
        cb.on_step_begin(None, _state(99), None)
        assert not merge_aware_enabled()
        cb.on_step_begin(None, _state(100), None)
        assert merge_aware_enabled()
    finally:
        set_merge_aware_enabled(False)

    try:
        cb = MergeAwareScheduleCallback(0.1)  # 10% of 1000 steps
        cb.on_step_begin(None, _state(99), None)
        assert not merge_aware_enabled()
        cb.on_step_begin(None, _state(100), None)
        assert merge_aware_enabled()
    finally:
        set_merge_aware_enabled(False)

    try:
        cb = MergeAwareScheduleCallback(None)
        cb.on_train_begin(None, _state(0), None)
        assert merge_aware_enabled()
    finally:
        set_merge_aware_enabled(False)


def test_on_save_stamps_only_after_start(tmp_path):
    ckpt = tmp_path / "checkpoint-50"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(json.dumps({"r": 4}))
    args = SimpleNamespace(output_dir=str(tmp_path))
    try:
        cb = MergeAwareScheduleCallback(100)
        cb.on_step_begin(args, _state(50), None)
        cb.on_save(args, _state(50), None)
        # pre-warm-up checkpoint = unprepared adapter: must stay unstamped
        assert "nvfp4_merge_aware" not in json.loads(
            (ckpt / "adapter_config.json").read_text()
        )

        ckpt2 = tmp_path / "checkpoint-150"
        ckpt2.mkdir()
        (ckpt2 / "adapter_config.json").write_text(json.dumps({"r": 4}))
        cb.on_step_begin(args, _state(150), None)
        cb.on_save(args, _state(150), None)
        assert "nvfp4_merge_aware" in json.loads(
            (ckpt2 / "adapter_config.json").read_text()
        )
    finally:
        set_merge_aware_enabled(False)
