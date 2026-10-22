"""Recovery saves from the staged LoRA snapshot: shards, rotation, consolidation, resume."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import load_file

from axolotl.core.publishing.callback import StepPublishCallback
from axolotl.core.publishing.coalesce import FREE, HELD, WAITING, SlotTable
from axolotl.core.publishing.comm import GlooComm, LocalComm
from axolotl.core.publishing.lora_shards import Piece, full_box
from axolotl.core.publishing.recovery import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_NAME,
    OptimizerStager,
    RecoveryJob,
    RecoverySchedule,
    RecoveryWriter,
    complete_steps,
    load_optimizer_state,
    load_recovery_checkpoint,
    manifest_name,
    optimizer_name,
    step_dir,
    write_safetensors,
)
from axolotl.utils.schemas.checkpoint_publish import (
    CheckpointPublishConfig,
    RecoveryCheckpointConfig,
)


def _tiny_peft_model(seed=0):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    return get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        LoraConfig(r=2, target_modules=["q_proj", "v_proj"], init_lora_weights=False),
    )


def _whole(name, tensor):
    shape = tuple(tensor.shape)
    return Piece(name, shape, shape, full_box(shape), str(tensor.dtype)[6:])


def _job(step, tensors, released=None, **kwargs):
    def release():
        if released is not None:
            released.append(step)

    return RecoveryJob(step, lambda: tensors, release, **kwargs)


def test_slot_table_holds_keep_a_slot_until_released():
    table = SlotTable(2)
    slot = table.acquire(1)
    table.hold(slot)
    table.written(slot)
    other = table.acquire(2)
    assert other != slot and table.coalesced == 0
    table.written(other)
    assert table.acquire(3) == other and table.coalesced == 1
    table.written(other)
    table.take(3)
    assert table.states[slot] == HELD and table.coalesced == 2
    assert table.acquire(4) is None
    table.unhold(slot)
    assert table.states[slot] == FREE and table.steps[slot] is None


def test_slot_table_hold_outlives_the_publish():
    table = SlotTable(2)
    slot = table.acquire(5)
    table.hold(slot)
    table.written(slot)
    assert table.states[slot] == WAITING
    table.take(5)
    table.release(slot)
    assert table.states[slot] == HELD and table.steps[slot] == 5
    table.unhold(slot)
    assert table.states[slot] == FREE


def test_streaming_safetensors_round_trips_dtypes(tmp_path):
    tensors = {
        "a": torch.randn(3, 5, dtype=torch.bfloat16),
        "b": torch.randn(7),
        "c": torch.randint(0, 255, (4, 4), dtype=torch.uint8),
        "d": torch.randn(2, 8).to(torch.float8_e4m3fn),
    }
    names = sorted(tensors)
    specs = {name: (t.dtype, tuple(t.shape)) for name, t in tensors.items()}
    path = tmp_path / "x.safetensors"
    size = write_safetensors(path, names, specs, (tensors[n] for n in names))
    loaded = load_file(path)
    assert size == sum(t.numel() * t.element_size() for t in tensors.values())
    for name, tensor in tensors.items():
        assert torch.equal(loaded[name].view(torch.uint8), tensor.view(torch.uint8)), (
            name
        )
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata() == {"format": "pt"}
    with pytest.raises(ValueError):
        write_safetensors(path, ["a"], {"a": (torch.float32, (3, 5))}, [tensors["a"]])
    assert torch.equal(load_file(path)["b"], tensors["b"])


def test_writer_rotates_and_keeps_the_newest_optimizer_state(tmp_path):
    tensors = [torch.full((2, 3), 0.0), torch.zeros(4)]
    pieces = [
        _whole("x.lora_A.weight", tensors[0]),
        _whole("x.lora_B.weight", tensors[1]),
    ]
    writer = RecoveryWriter(tmp_path / "rec", pieces, LocalComm(), keep=2)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    stager = OptimizerStager()
    released = []
    for step in (1, 2, 3, 4, 5):
        for tensor in tensors:
            tensor.fill_(step)
        optimizer_state = None
        if step == 2:
            stager.stage(optimizer)
            optimizer_state = stager
        writer.submit(_job(step, tensors, released, optimizer=optimizer_state))
        writer.wait()
    writer.close()
    assert released == [1, 2, 3, 4, 5]
    assert complete_steps(tmp_path / "rec", 0) == [2, 4, 5]
    assert (step_dir(tmp_path / "rec", 2) / optimizer_name(0)).exists()
    assert not step_dir(tmp_path / "rec", 1).exists()
    shard = load_file(step_dir(tmp_path / "rec", 5) / "rank-00000.safetensors")
    assert torch.all(shard["x.lora_A.weight"] == 5)
    manifest = json.loads(
        (step_dir(tmp_path / "rec", 5) / manifest_name(0)).read_text()
    )
    assert manifest["world"] == 1 and not manifest["optimizer"]
    metrics = writer.drain_metrics()
    assert [m["recovery_step"] for m in metrics] == [1, 2, 3, 4, 5]
    assert [m["recovery_rotated"] for m in metrics] == [0, 0, 1, 0, 1]


def test_writer_discards_a_failed_write(tmp_path):
    tensor = torch.zeros(2)
    writer = RecoveryWriter(tmp_path, [_whole("k", tensor)], LocalComm())
    released = []

    def failing():
        raise OSError("disk full")

    writer.submit(RecoveryJob(3, failing, lambda: released.append(3)))
    writer.wait()
    writer.submit(_job(4, [tensor]))
    writer.close()
    assert released == [3]
    assert complete_steps(tmp_path, 0) == [4]
    assert not step_dir(tmp_path, 3).exists()
    failed, ok = writer.drain_metrics()
    assert failed["recovery_failed"] == 1 and "recovery_failed" not in ok
    assert writer.last_written == 4


def test_writer_is_busy_while_writing(tmp_path):
    tensor = torch.zeros(2)
    writer = RecoveryWriter(tmp_path, [_whole("k", tensor)], LocalComm())
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return [tensor]

    writer.submit(RecoveryJob(1, slow, lambda: None))
    assert writer.busy()
    with pytest.raises(RuntimeError, match="still being written"):
        writer.submit(_job(2, [tensor]))
    gate.set()
    writer.wait()
    assert not writer.busy()
    writer.close()


def test_optimizer_state_round_trips(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    for _ in range(2):
        model(torch.randn(4, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    stager = OptimizerStager()
    stager.stage(optimizer)
    expected = {k: v.clone() for k, v in optimizer.state[model.weight].items()}
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    skeleton, tensors = stager.ready()
    torch.save({"skeleton": skeleton, "tensors": tensors}, tmp_path / "o.pt")

    fresh = torch.nn.Linear(3, 2)
    restored = torch.optim.AdamW(fresh.parameters(), lr=0.5)
    load_optimizer_state(tmp_path / "o.pt", restored)
    state = restored.state[fresh.weight]
    for key, value in expected.items():
        assert torch.equal(state[key], value), key
    assert restored.param_groups[0]["lr"] == 0.1
    assert restored.param_groups[0]["betas"] == optimizer.param_groups[0]["betas"]


def _train(model, optimizer, callback, state, steps):
    for step in range(state.global_step + 1, steps + 1):
        callback.on_pre_optimizer_step(None, state, None)
        model(input_ids=torch.tensor([[1, 2, 3]])).logits.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        state.global_step = step
        callback.on_step_end(None, state, None, optimizer=optimizer)


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, buffers, step):
        self.published.append(step)
        return {"identity": f"id-{step}", "action": "commit"}

    def restore(self, buffers, step):
        return True

    def close(self):
        pass


def test_callback_saves_publishes_and_consolidates(tmp_path):
    from peft import get_peft_model_state_dict

    model = _tiny_peft_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    recovery = RecoverySchedule(
        every=3,
        directory=tmp_path / "scratch",
        keep=2,
        optimizer_every=6,
        consolidate_every=6,
        consolidate_dir=tmp_path / "adapters",
    )
    fake = FakePublisher()
    trainer = MagicMock()
    callback = StepPublishCallback(
        lambda **kwargs: fake, every=2, trainer=trainer, recovery=recovery
    )
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None, model=model)
    assert len(callback.scheduler.stager.buffers) == 3
    _train(model, optimizer, callback, state, 8)
    live = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    callback.on_train_end(None, state, None, optimizer=optimizer)

    assert fake.published and set(fake.published) <= {2, 4, 6, 8}
    assert complete_steps(recovery.directory, 0) == [6, 8]
    assert (step_dir(recovery.directory, 6) / optimizer_name(0)).exists()
    assert sorted(p.name for p in recovery.consolidate_dir.iterdir()) == [
        "checkpoint-6",
        "checkpoint-8",
    ]
    final = recovery.consolidate_dir / "checkpoint-8"
    consolidated = load_file(final / ADAPTER_NAME)
    assert consolidated.keys() == live.keys()
    for key, value in live.items():
        assert torch.equal(consolidated[key], value), key
    config = json.loads((final / ADAPTER_CONFIG_NAME).read_text())
    assert config["r"] == 2 and config["peft_type"] == "LORA"
    logged = [call.args[0] for call in trainer.log.call_args_list]
    assert any("checkpoint_publish/recovery_write_seconds" in m for m in logged)

    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1)
    base = LlamaForCausalLM(model.base_model.model.config)
    reloaded = PeftModel.from_pretrained(base, final)
    for key, value in get_peft_model_state_dict(reloaded).items():
        assert torch.equal(value, live[key]), key
    assert isinstance(base.config, LlamaConfig)


def test_resume_restores_lora_and_optimizer(tmp_path):
    from peft import get_peft_model_state_dict

    model = _tiny_peft_model()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    recovery = RecoverySchedule(
        every=2, directory=tmp_path / "rec", optimizer_every=4, final=False
    )
    callback = StepPublishCallback(None, 0, recovery=recovery)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None, model=model)
    _train(model, optimizer, callback, state, 4)
    callback.writer.wait()
    expected_lora = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    expected_state = {
        i: {k: v.clone() for k, v in optimizer.state[p].items()}
        for i, p in enumerate(params)
    }
    _train(model, optimizer, callback, state, 5)
    callback.on_train_end(None, state, None, optimizer=optimizer)
    assert complete_steps(recovery.directory, 0) == [2, 4]
    assert recovery.consolidate_dir is None

    fresh = _tiny_peft_model(seed=3)
    fresh_params = [p for p in fresh.parameters() if p.requires_grad]
    fresh_optimizer = torch.optim.AdamW(fresh_params, lr=1e-2)
    step = load_recovery_checkpoint(
        recovery.directory, fresh, optimizer=fresh_optimizer
    )
    assert step == 4
    for key, value in get_peft_model_state_dict(fresh).items():
        assert torch.equal(value, expected_lora[key]), key
    for i, param in enumerate(fresh_params):
        for key, value in expected_state[i].items():
            assert torch.equal(fresh_optimizer.state[param][key], value)
    with pytest.raises(ValueError, match="no optimizer state"):
        load_recovery_checkpoint(
            recovery.directory, fresh, step=2, optimizer=fresh_optimizer
        )


def test_callback_skips_a_recovery_step_while_writing(tmp_path):
    model = _tiny_peft_model()
    recovery = RecoverySchedule(every=1, directory=tmp_path, final=False)
    callback = StepPublishCallback(None, 0, recovery=recovery)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None, model=model)
    gate = threading.Event()
    original = callback.writer._write

    def slow(job):
        gate.wait(5)
        return original(job)

    callback.writer._write = slow
    state.global_step = 1
    callback.on_step_end(None, state, None)
    state.global_step = 2
    callback.on_step_end(None, state, None)
    assert callback.recovery_skipped == 1
    gate.set()
    callback.on_train_end(None, state, None)
    assert complete_steps(tmp_path, 0) == [1]


def test_final_save_consolidates_an_already_written_step(tmp_path):
    model = _tiny_peft_model()
    recovery = RecoverySchedule(
        every=2, directory=tmp_path / "rec", consolidate_dir=tmp_path / "out"
    )
    callback = StepPublishCallback(None, 0, recovery=recovery)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None, model=model)
    state.global_step = 2
    callback.on_step_end(None, state, None)
    callback.on_train_end(None, state, None)
    assert complete_steps(recovery.directory, 0) == [2]
    assert (tmp_path / "out" / "checkpoint-2" / ADAPTER_NAME).exists()


def _writer_worker(rank, world, init_file, out):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        comm = GlooComm.create()
        torch.manual_seed(0)
        full_a = torch.randn(6, 4, dtype=torch.bfloat16)
        full_b = torch.randn(3, 2 * 4)
        rows = full_a.shape[0] // world
        a_box = ((rank * rows, (rank + 1) * rows), (0, 4))
        experts = 4 // world
        b_view = (3, 2, 4)
        b_box = ((0, 3), (0, 2), (rank * experts, (rank + 1) * experts))
        pieces = [
            Piece("m.lora_A.weight", (6, 4), (6, 4), a_box, "bfloat16"),
            Piece("e.lora_B.weight", (3, 8), b_view, b_box, "float32"),
        ]
        local = [
            full_a[a_box[0][0] : a_box[0][1]].clone(),
            full_b.reshape(b_view)[:, :, b_box[2][0] : b_box[2][1]].contiguous(),
        ]
        writer = RecoveryWriter(
            Path(out) / f"node-{rank}",
            pieces,
            comm,
            consolidate_dir=Path(out) / "adapters",
            adapter_config={"r": 2},
        )
        writer.submit(RecoveryJob(7, lambda: local, lambda: None, consolidate=True))
        writer.close()
        if rank == 0:
            merged = load_file(Path(out) / "adapters" / "checkpoint-7" / ADAPTER_NAME)
            assert torch.equal(merged["m.lora_A.weight"], full_a)
            assert torch.equal(merged["e.lora_B.weight"], full_b)
        Path(out, f"{rank}.json").write_text(
            json.dumps(complete_steps(Path(out) / f"node-{rank}", rank))
        )
    finally:
        dist.destroy_process_group()


def test_writer_consolidates_slices_from_every_rank(tmp_path):
    world = 2
    mp.spawn(
        _writer_worker,
        args=(world, str(tmp_path / "init"), str(tmp_path)),
        nprocs=world,
    )
    for rank in range(world):
        assert json.loads((tmp_path / f"{rank}.json").read_text()) == [7]
    assert (tmp_path / "adapters" / "checkpoint-7" / ADAPTER_CONFIG_NAME).exists()


def test_recovery_config_validation():
    config = CheckpointPublishConfig.model_validate(
        {"recovery": {"every": 10, "optimizer_every": 50, "consolidate_every": 100}}
    )
    assert config.recovery.keep == 2 and config.recovery.final
    with pytest.raises(ValueError, match="multiple of every"):
        RecoveryCheckpointConfig(every=10, optimizer_every=15)
    with pytest.raises(ValueError):
        RecoveryCheckpointConfig(every=0)
    schedule = RecoverySchedule.from_config(
        config.recovery, output_dir="/out", workdir="/scratch/publish"
    )
    assert schedule.directory == Path("/scratch/publish/recovery")
    assert schedule.consolidate_dir == Path("/out/recovery-adapters")


def test_callback_requires_publishing_or_recovery():
    with pytest.raises(ValueError):
        StepPublishCallback(None, 0)
    with pytest.raises(ValueError):
        StepPublishCallback(None, 2)


def _coreweave_cfg(settings, adapter="lora"):
    from axolotl.integrations.coreweave.args import CoreWeaveConfig
    from axolotl.utils.dict import DictDefault

    config = CoreWeaveConfig(
        deployment_id="d",
        bucket="bucket",
        path_prefix="weights",
        gateway_url="https://gateway.example",
        model="m",
    )
    return DictDefault(
        {
            "coreweave": config.model_dump(),
            "adapter": adapter,
            "base_model": "test/model",
            "output_dir": "/out",
            "checkpoint_publish": settings,
        }
    )


def test_coreweave_plugin_adds_a_recovery_callback():
    from axolotl.integrations.coreweave.plugin import (
        CoreWeaveCheckpointCallback,
        CoreWeavePlugin,
    )

    cfg = _coreweave_cfg({"recovery": {"every": 5, "dir": "/scratch/rec"}})
    CoreWeavePlugin().register(cfg)
    trainer = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
    publish, recovery = CoreWeavePlugin().add_callbacks_post_trainer(cfg, trainer)
    assert isinstance(publish, CoreWeaveCheckpointCallback)
    assert isinstance(recovery, StepPublishCallback) and recovery.every == 0
    assert recovery.recovery.directory == Path("/scratch/rec")

    cfg = _coreweave_cfg({"every": 2, "recovery": {"every": 4}})
    (callback,) = CoreWeavePlugin().add_callbacks_post_trainer(cfg, trainer)
    assert callback.every == 2 and callback.recovery.every == 4
    assert callback.recovery.directory == Path("/out/recovery")

    with pytest.raises(ValueError, match="LoRA adapters only"):
        CoreWeavePlugin().register(_coreweave_cfg({"recovery": {"every": 5}}, None))
