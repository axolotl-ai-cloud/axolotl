"""Distributed regressions for complete EP model and ordinary AdamW checkpoints."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

NUM_EXPERTS = 8
LORA_RANK = 2  # the worker's
WORKER = Path(__file__).with_name("_ep_checkpoint_resume_worker.py")
LAYOUTS = [pytest.param(2, 1, id="ep2"), pytest.param(2, 2, id="ep2xdp_shard2")]
TRAINING = [pytest.param(False, id="full"), pytest.param(True, id="lora")]


def _run(tmp_path, ep, dp_shard, mode, lora):
    env = os.environ | {"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    out = tmp_path / mode
    out.mkdir()
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ep * dp_shard}",
            str(WORKER),
            f"--ep={ep}",
            f"--dp-shard={dp_shard}",
            f"--mode={mode}",
            f"--out={out}",
            *(["--lora"] if lora else []),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    assert process.returncode == 0, process.stdout
    lines = [
        line for line in process.stdout.splitlines() if "EP_CHECKPOINT_REPORT " in line
    ]
    assert len(lines) == 1, process.stdout
    reports = json.loads(lines[0].split("EP_CHECKPOINT_REPORT ", 1)[1])
    assert len(reports) == ep * dp_shard
    return reports


def _experts_held(name, shape):
    """How many experts a saved tensor holds along its experts axis (PEFT's expert LoRA
    packs ``r`` entries per expert: ``lora_A`` ``[E*r, in]``, ``lora_B`` ``[out, r*E]``)."""
    if "lora_B" in name:
        return shape[1] // LORA_RANK
    if "lora_A" in name:
        return shape[0] // LORA_RANK
    return shape[0]


def _expert_counts(reports):
    return {_experts_held(name, shape) for name, _key, shape in reports[0]["file"]}


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("lora", TRAINING)
@pytest.mark.parametrize("ep,dp_shard", LAYOUTS)
def test_ep_checkpoint_round_trips_every_ep_ranks_experts(tmp_path, ep, dp_shard, lora):
    reports = _run(tmp_path, ep, dp_shard, "fixed", lora)

    # 2 layers x (gate_up_proj, down_proj) [x (lora_A, lora_B)] x (weight, 2 moments)
    assert len(reports[0]["file"]) == (24 if lora else 12)
    assert _expert_counts(reports) == {NUM_EXPERTS}
    if lora:  # the gathered adapter export the checkpoint also carries
        assert len(reports[0]["adapter"]) == 8
        assert {_experts_held(k, shape) for k, shape in reports[0]["adapter"]} == {
            NUM_EXPERTS
        }
    for report in reports:
        assert "load_error" not in report, report
        assert report["mismatch"] == [], report
        assert report["best_model_matches"], report


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("lora", TRAINING)
@pytest.mark.parametrize("ep,dp_shard", LAYOUTS)
def test_accelerate_checkpoint_keeps_only_ep_group_zero(tmp_path, ep, dp_shard, lora):
    """The unpatched accelerate path loses the experts the fix restores (guards the
    round-trip test's ability to see the bug)."""
    reports = _run(tmp_path, ep, dp_shard, "accelerate", lora)

    assert _expert_counts(reports) == {NUM_EXPERTS // ep}
    for report in reports:
        reloaded = {(kind, key) for kind, name, key in report["mismatch"]}
        wrong_names = {name for kind, name, _ in report["mismatch"] if kind == "param"}
        assert all(".experts." in name for name in wrong_names), report
        if report["ep_rank"] == 0:
            assert wrong_names == set(), report
        else:
            assert {"weight", "exp_avg", "exp_avg_sq"} <= {
                key for kind, key in reloaded if kind == "param"
            }, report


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("lora", TRAINING)
@pytest.mark.parametrize("ep,dp_shard", LAYOUTS[:1])
def test_ep_checkpoint_refuses_a_checkpoint_holding_one_ep_group(
    tmp_path, ep, dp_shard, lora
):
    reports = _run(tmp_path, ep, dp_shard, "legacy", lora)

    for report in reports:
        assert "EP group 0's experts" in report.get("load_error", ""), report
        assert "lora_model_dir" in report["load_error"], report
        assert report["unchanged"], report


@pytest.mark.parametrize("invalid", ["shape", "missing"])
def test_model_preflight_rejects_before_copying(monkeypatch, invalid):
    import torch
    from torch import nn

    from axolotl.monkeypatch.accelerate import fsdp2_checkpoint as checkpoint

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    saved = {name: value + 10 for name, value in before.items()}
    if invalid == "shape":
        saved["1.bias"] = torch.ones(3)
    else:
        del saved["1.bias"]
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        checkpoint.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(checkpoint.dist, "broadcast_object_list", lambda *a, **kw: None)

    def unexpected_restore(*args, **kwargs):
        raise AssertionError("Restore started before validating the complete model")

    monkeypatch.setattr(checkpoint, "_restore_tensor", unexpected_restore)
    with pytest.raises(ValueError, match="1.bias"):
        checkpoint.restore_model_state(model, saved)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_model_preflight_accepts_scalar_buffers(monkeypatch):
    import torch
    from torch import nn

    from axolotl.monkeypatch.accelerate import fsdp2_checkpoint as checkpoint

    model = nn.Linear(2, 2)
    model.register_buffer("counter", torch.tensor(0))
    saved = {name: value.clone() + 1 for name, value in model.state_dict().items()}
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        checkpoint.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(checkpoint.dist, "broadcast_object_list", lambda *a, **kw: None)
    monkeypatch.setattr(checkpoint, "_restore_tensor", lambda value, *args: value)
    checkpoint.restore_model_state(model, saved)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
