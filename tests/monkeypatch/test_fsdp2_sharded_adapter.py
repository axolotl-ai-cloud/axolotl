"""Sharded FSDP2 adapter checkpoints: byte-identical consolidation, and nothing treats a
checkpoint as complete before its consolidation finishes."""

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.monkeypatch.accelerate import fsdp2_sharded_adapter
from axolotl.monkeypatch.accelerate.fsdp2_sharded_adapter import (
    SHARD_DIRNAME,
    ConsolidationQueue,
    _header,
    safetensors_order,
)

WORKER = Path(__file__).with_name("_fsdp2_sharded_adapter_worker.py")
CASES = (
    "fsdp-only",
    "ep",
    "ep-remote-ranks",
    "regular-checkpoint-resume",
    "torch-dcp-consolidation",
    "interrupted-consolidation",
    "ep-fsdp",
    "modules-to-save",
    "modules-to-save-tied",
    "replicated-wrapper",
    "final-fsdp-only",
    "final-ep",
    "final-ep-fsdp",
)


def run_worker(tmp_path, env=None, timeout=300):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=4",
            f"--master-port={get_torch_dist_unique_port()}",
            str(WORKER),
            str(tmp_path),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("Sharded adapter worker timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    return stdout


@pytest.mark.distributed_cpu
def test_sharded_adapter_matches_gathered_adapter(tmp_path):
    stdout = run_worker(tmp_path)
    for case in CASES:
        assert f"PASS sharded-{case}" in stdout, stdout


def test_header_and_layout_match_safetensors(tmp_path):
    from safetensors.torch import save_file

    tensors = {
        "b.lora_B.weight": torch.randn(3, 5),
        "a.lora_A.weight": torch.randn(2, 3).to(torch.bfloat16),
        "c.scale": torch.tensor(2.0, dtype=torch.float64),
        "empty": torch.zeros(0, 4),
        "z.idx": torch.arange(6, dtype=torch.int8),
    }
    path = tmp_path / "reference.safetensors"
    save_file(tensors, path, metadata={"format": "pt"})
    order = safetensors_order({k: v.dtype for k, v in tensors.items()})
    entries = [
        (
            name,
            dtype,
            list(tensors[name].shape),
            tensors[name].numel() * tensors[name].element_size(),
        )
        for name, dtype in order
    ]
    prefix, _, _ = _header(entries, {"format": "pt"})
    data = b"".join(
        tensors[name].reshape(-1).view(torch.uint8).numpy().tobytes()
        for name, _ in order
    )
    assert prefix + data == path.read_bytes()


class Mirror:
    """A mirror-like on_save consumer: snapshots the checkpoint the way the mirror queues it."""

    def __init__(self, root):
        self.root = root
        self.events = []

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda args, state, control, **kwargs: control
        raise AttributeError(name)

    def on_save(self, args, state, control, **kwargs):
        from axolotl.core.checkpoint_publisher import retain_checkpoint

        source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        retained = retain_checkpoint(source, self.root)
        self.events.append(
            (
                "save",
                state.global_step,
                sorted(p.name for p in retained.rglob("*") if p.is_file()),
            )
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.events.append(("train_end",))
        return control


def _checkpoint(run_dir, step, *, consolidating=False, mtime=0):
    directory = run_dir / f"checkpoint-{step}"
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text("{}")
    if consolidating:
        (directory / SHARD_DIRNAME).mkdir()
    else:
        (directory / "adapter_model.safetensors").write_bytes(b"done")
    os.utime(directory, (1000 + mtime, 1000 + mtime))
    return directory


@pytest.fixture
def gated(tmp_path):
    from transformers.trainer_callback import (
        CallbackHandler,
        TrainerControl,
        TrainerState,
    )

    from axolotl.core.trainers.mixins.sharded_adapter_save import (
        ShardedAdapterSaveMixin,
    )

    class Gated(ShardedAdapterSaveMixin):
        pass

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gates: dict = {}

    def slow_consolidate(shard_dir, *args, **kwargs):
        checkpoint = Path(shard_dir).parent
        gate = gates[checkpoint.name]
        assert gate["release"].wait(30)
        if gate.get("fail"):
            raise OSError("disk full")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"consolidated")
        os.rmdir(shard_dir)
        return checkpoint / "adapter_model.safetensors", 0.0, 12

    mirror = Mirror(tmp_path / "mirror-queue")
    trainer = object.__new__(Gated)
    trainer.axolotl_cfg = SimpleNamespace(adapter_checkpoint_format="sharded")
    trainer.args = SimpleNamespace(output_dir=str(run_dir))
    trainer.state = TrainerState()
    trainer.control = TrainerControl()
    trainer.callback_handler = CallbackHandler([mirror], None, None, None, None)
    queue = ConsolidationQueue()
    with (
        patch.object(fsdp2_sharded_adapter, "CONSOLIDATIONS", queue),
        patch.object(fsdp2_sharded_adapter, "consolidate", slow_consolidate),
    ):
        trainer._consolidation_state()
        trainer._install_consolidation_gate()

        def save(step, *, sharded=True, limit=1):
            """What ``Trainer._save_checkpoint`` does around a sharded save."""
            directory = run_dir / f"checkpoint-{step}"
            if sharded:
                gates[directory.name] = {"release": threading.Event()}
                _checkpoint(run_dir, step, consolidating=True, mtime=step)
                queue.submit(directory / SHARD_DIRNAME)
                trainer._axolotl_unconsolidated.add(os.path.abspath(directory))
            else:
                _checkpoint(run_dir, step, mtime=step)
            trainer._axolotl_last_checkpoint = os.path.abspath(directory)
            trainer.state.global_step = step
            import transformers.trainer as hf_trainer

            with trainer._gated_rotation():
                hf_trainer.rotate_checkpoints(
                    output_dir=str(run_dir),
                    save_total_limit=limit,
                    best_model_checkpoint=None,
                    use_mtime=True,
                )
            trainer.control = trainer.callback_handler.on_save(
                trainer.args, trainer.state, trainer.control
            )

        def step_end():
            trainer.control = trainer.callback_handler.on_step_end(
                trainer.args, trainer.state, trainer.control
            )

        def release(step, fail=False):
            gates[f"checkpoint-{step}"]["fail"] = fail
            gates[f"checkpoint-{step}"]["release"].set()
            queue.wait(run_dir / f"checkpoint-{step}")

        def checkpoints():
            return sorted(p.name for p in run_dir.glob("checkpoint-*"))

        yield SimpleNamespace(
            trainer=trainer,
            mirror=mirror,
            save=save,
            step_end=step_end,
            release=release,
            checkpoints=checkpoints,
            queue=queue,
        )
        for gate in gates.values():
            gate["release"].set()
        queue.wait()


def test_mirror_and_rotation_wait_for_consolidation(gated):
    _checkpoint(Path(gated.trainer.args.output_dir), 1, mtime=1)
    _checkpoint(Path(gated.trainer.args.output_dir), 2, mtime=2)
    gated.save(3)
    assert gated.trainer.control.should_save is False
    gated.step_end()
    # neither the mirror nor save_total_limit acts on a half-consolidated checkpoint
    assert gated.mirror.events == []
    assert gated.checkpoints() == ["checkpoint-1", "checkpoint-2", "checkpoint-3"]
    gated.save(4)
    gated.release(3)
    gated.step_end()
    assert gated.mirror.events == [
        ("save", 3, ["adapter_config.json", "adapter_model.safetensors"])
    ]
    # checkpoint-4 is still consolidating: it is neither counted nor deleted
    assert gated.checkpoints() == ["checkpoint-3", "checkpoint-4"]
    released = threading.Timer(0.2, gated.release, args=(4,))
    released.start()
    gated.trainer.control = gated.trainer.callback_handler.on_train_end(
        gated.trainer.args, gated.trainer.state, gated.trainer.control
    )
    released.join()
    assert gated.mirror.events[1:] == [
        ("save", 4, ["adapter_config.json", "adapter_model.safetensors"]),
        ("train_end",),
    ]
    assert gated.checkpoints() == ["checkpoint-4"]
    assert gated.queue.pending() == []


def test_failed_consolidation_is_never_published_or_counted(gated):
    _checkpoint(Path(gated.trainer.args.output_dir), 1, mtime=1)
    gated.save(2)
    gated.release(2, fail=True)
    gated.step_end()
    assert gated.mirror.events == []
    assert gated.checkpoints() == ["checkpoint-1", "checkpoint-2"]
    # a later gathered save still rotates, around the failed checkpoint
    gated.save(3, sharded=False)
    assert [event[1] for event in gated.mirror.events] == [3]
    assert gated.checkpoints() == ["checkpoint-2", "checkpoint-3"]


def test_unsharded_saves_pass_straight_through(gated):
    gated.save(1, sharded=False)
    gated.save(2, sharded=False)
    assert [event[1] for event in gated.mirror.events] == [1, 2]
    assert gated.checkpoints() == ["checkpoint-2"]


def test_default_format_keeps_the_gather_path():
    from transformers import Trainer

    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.utils.dict import DictDefault
    from axolotl.utils.schemas.config import AxolotlInputConfig

    assert AxolotlInputConfig.model_fields["adapter_checkpoint_format"].default == (
        "gather"
    )
    trainer = object.__new__(AxolotlTrainer)
    trainer._axolotl_cfg = DictDefault({"adapter": "lora"})
    trainer._axolotl_saving_checkpoint = True
    assert not trainer._sharded_adapter_save_applies()
    calls = []
    with patch.object(
        Trainer, "_save_checkpoint", lambda self, *a, **k: calls.append(1)
    ):
        from axolotl.core.trainers.mixins.sharded_adapter_save import (
            ShardedAdapterSaveMixin,
        )

        ShardedAdapterSaveMixin._save_checkpoint(trainer, None, None)
    assert calls == [1] and not hasattr(trainer, "_axolotl_unconsolidated")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({}, None),
        ({"adapter": None}, "an adapter"),
        ({"fsdp_config": None, "fsdp_version": None}, "fsdp_version: 2"),
        (
            {"fsdp_config": {"state_dict_type": "SHARDED_STATE_DICT"}},
            "FULL_STATE_DICT",
        ),
        ({"load_best_model_at_end": True}, "load_best_model_at_end"),
        ({"hub_model_id": "org/model"}, "hub_strategy: end"),
        ({"hub_model_id": "org/model", "hub_strategy": "checkpoint"}, "hub_strategy"),
        ({"hub_model_id": "org/model", "hub_strategy": "end"}, None),
        ({"adapter_final_save_format": "gather"}, None),
        (
            {
                "adapter_checkpoint_format": "gather",
                "adapter_final_save_format": "sharded",
            },
            "adapter_final_save_format: sharded requires",
        ),
    ],
)
def test_sharded_format_validation(min_base_cfg, overrides, message):
    from axolotl.utils.config import validate_config
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            **min_base_cfg,
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_target_linear": True,
            "fsdp_version": 2,
            "fsdp_config": {"state_dict_type": "FULL_STATE_DICT"},
            "adapter_checkpoint_format": "sharded",
        }
    )
    for key, value in overrides.items():
        cfg[key] = value
    if message is None:
        assert validate_config(cfg).adapter_checkpoint_format == "sharded"
        return
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def _final_save_trainer(tmp_path, **cfg):
    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )
    from axolotl.core.trainers.mixins.sharded_adapter_save import (
        ShardedAdapterSaveMixin,
    )
    from axolotl.utils.dict import DictDefault

    class Final(ShardedAdapterSaveMixin, DistributedParallelMixin):
        axolotl_cfg = property(lambda self: self._axolotl_cfg)

    trainer = object.__new__(Final)
    trainer._axolotl_cfg = DictDefault(
        {"adapter": "lora", "adapter_checkpoint_format": "sharded", **cfg}
    )
    trainer.model = SimpleNamespace(peft_config={"default": None})
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2, state_dict_type="StateDictType.FULL_STATE_DICT"
            )
        )
    )
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), push_to_hub=False)
    return trainer


@pytest.fixture
def final_save(monkeypatch):
    """Record which writer the save paths reach, without collectives."""
    from axolotl.core.trainers.mixins import distributed_parallel, sharded_adapter_save

    calls = []
    sharded_result = {"value": True}

    def sharded(self, output_dir, final=False):
        calls.append(("sharded", output_dir, final))
        return sharded_result["value"]

    def gather(self, output_dir=None, _internal_call=False):
        calls.append(("gather", output_dir))

    monkeypatch.setattr(sharded_adapter_save, "_distributed", lambda: True)
    monkeypatch.setattr("accelerate.utils.modeling.is_peft_model", lambda model: True)
    monkeypatch.setattr(
        sharded_adapter_save.ShardedAdapterSaveMixin, "_save_sharded_adapter", sharded
    )
    monkeypatch.setattr(
        distributed_parallel.DistributedParallelMixin, "_save_model_native", gather
    )
    return SimpleNamespace(calls=calls, sharded_result=sharded_result)


def test_final_save_goes_through_the_sharded_writer(tmp_path, final_save):
    trainer = _final_save_trainer(tmp_path)
    assert trainer.save_final_adapter(str(tmp_path / "out"))
    assert final_save.calls == [("sharded", str(tmp_path / "out"), True)]
    # a later plain save_model (merge, export, callbacks) keeps the gather
    trainer.save_model(str(tmp_path / "other"))
    assert final_save.calls[1:] == [("gather", str(tmp_path / "other"))]
    assert not trainer._final_adapter_save()


def test_final_save_fallback_leaves_the_save_to_the_caller(tmp_path, final_save):
    final_save.sharded_result["value"] = False
    trainer = _final_save_trainer(tmp_path)
    assert not trainer.save_final_adapter(str(tmp_path / "out"))
    # nothing was gathered into output_dir: save_trained_model's own path saves it
    assert final_save.calls == [("sharded", str(tmp_path / "out"), True)]


@pytest.mark.parametrize(
    "cfg",
    [
        {"adapter_final_save_format": "gather"},
        {"adapter_checkpoint_format": "gather"},
    ],
)
def test_final_save_follows_the_final_format(tmp_path, final_save, cfg):
    trainer = _final_save_trainer(tmp_path, **cfg)
    assert not trainer.save_final_adapter(str(tmp_path / "out"))
    assert final_save.calls == []


def test_save_trained_model_routes_the_final_export(tmp_path):
    from axolotl.train import save_trained_model
    from axolotl.utils.dict import DictDefault

    calls = []
    trainer = SimpleNamespace(
        save_final_adapter=lambda output_dir: calls.append(output_dir) or True,
        save_model=lambda *a, **k: pytest.fail("saved through the gather"),
    )
    model = SimpleNamespace(named_modules=lambda: [])
    cfg = DictDefault(
        {
            "adapter": "lora",
            "adapter_checkpoint_format": "sharded",
            "expert_parallel_size": 4,
            "output_dir": str(tmp_path),
        }
    )
    save_trained_model(cfg, trainer, model)
    assert calls == [str(tmp_path)]
