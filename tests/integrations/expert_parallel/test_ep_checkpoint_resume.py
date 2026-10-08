"""EP x FSDP2 FULL_STATE_DICT checkpoints must round-trip every EP rank's experts (CPU, gloo).

Each EP rank holds a different block of experts (and of a ``target_parameters`` expert
LoRA) under the same parameter name. accelerate's FSDP2 checkpoint functions gather them
over their own (non-``ep``) mesh only and rank 0 writes the file, so it holds EP group 0's
experts and on load every EP rank gets group 0's expert weights / expert LoRA and Adam
moments. The trainer routes its checkpoint save/load through
``ep_fsdp_checkpoint_functions``, which gathers across ``ep`` on save and hands each rank
its own experts back on load.
"""

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
        assert "EP group 0's experts only" in report.get("load_error", ""), report


def test_trainer_routes_fsdp_checkpoints_through_ep_functions(monkeypatch):
    """The trainer's checkpoint save/load see the EP-aware functions under the trainer's
    ``transformers.trainer`` names, and only while EP full-parameter experts are active."""
    from types import SimpleNamespace

    import transformers.trainer as hf_trainer

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )
    from axolotl.integrations.expert_parallel import checkpoint
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

    group = object()
    monkeypatch.setattr(
        ExpertParallelPlugin, "_resolve_ep_group", staticmethod(lambda cfg: group)
    )
    names = (
        "save_fsdp_model",
        "save_fsdp_optimizer",
        "load_fsdp_model",
        "load_fsdp_optimizer",
    )
    originals = {name: getattr(hf_trainer, name) for name in names}

    for ep_active in (False, True):
        trainer = SimpleNamespace(
            axolotl_cfg=SimpleNamespace(), _ep_sharded_checkpoint=lambda a=ep_active: a
        )
        with DistributedParallelMixin._ep_checkpoint_functions(trainer):
            seen = {name: getattr(hf_trainer, name) for name in names}
        assert {name: getattr(hf_trainer, name) for name in names} == originals
        if not ep_active:
            assert seen == originals
            continue
        assert seen["load_fsdp_model"] is checkpoint.load_fsdp_model
        assert seen["load_fsdp_optimizer"] is checkpoint.load_fsdp_optimizer
        assert seen["save_fsdp_model"].func is checkpoint.save_fsdp_model
        assert seen["save_fsdp_optimizer"].func is checkpoint.save_fsdp_optimizer
        assert seen["save_fsdp_model"].keywords == {"ep_group": group}
