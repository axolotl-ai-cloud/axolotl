"""Coalescing, staging, slice planning and the trainer callback, without a merge."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from axolotl.core.publishing.callback import (
    METRIC_PREFIX,
    StepPublishCallback,
    live_adapter_config,
)
from axolotl.core.publishing.chain import ChainWriter
from axolotl.core.publishing.coalesce import (
    BUILDING,
    WAITING,
    CoalescingScheduler,
    SlotTable,
)
from axolotl.core.publishing.comm import GlooComm, LocalComm
from axolotl.core.publishing.destinations import LocalDestination, S3Destination
from axolotl.core.publishing.lora_shards import (
    Need,
    Piece,
    assemble,
    canonical_adapter_key,
    collect_model_pieces,
    expert_need,
    full_box,
    need_shape,
    plan_transfers,
)
from axolotl.core.publishing.staging import LoRAStager
from axolotl.core.publishing.step_publisher import tensor_class


def test_slot_table_keeps_only_the_newest_waiting_snapshot():
    table = SlotTable()
    first = table.acquire(1)
    table.written(first)
    assert table.newest() == 1
    second = table.acquire(2)
    assert second == first and table.coalesced == 1
    table.written(second)
    slot = table.take(2)
    assert table.states[slot] == BUILDING
    for step in (3, 4, 5):
        table.written(table.acquire(step))
    assert table.newest() == 5 and table.coalesced == 3
    assert table.states.count(WAITING) == 1


def test_slot_table_protects_the_agreed_target():
    table = SlotTable()
    table.written(table.acquire(7))
    table.target = 7
    other = table.acquire(8)
    table.written(other)
    assert table.has(7) and table.has(8)
    assert table.acquire(9) == other and table.coalesced == 1
    table.written(other)
    assert table.acquire(10) == other
    table.written(other)
    slot = table.take(7)
    assert table.steps[slot] == 7 and table.has(10) and table.target is None


def test_slot_table_skips_when_every_slot_is_busy():
    table = SlotTable()
    table.written(table.acquire(1))
    table.take(1)
    table.written(table.acquire(2))
    table.target = 2
    assert table.acquire(3) is None and table.coalesced == 1


class FakeStager:
    def __init__(self, source):
        self.source = source
        self.buffers = [None, None]
        self.waits = 0

    def stage(self, slot):
        self.buffers[slot] = [self.source.clone()]

    def wait_training(self):
        self.waits += 1

    def ready(self, slot):
        return self.buffers[slot]


def test_scheduler_publishes_the_latest_snapshot_and_counts_skips():
    source = torch.zeros(1)
    published = []
    gate = threading.Event()

    def publish(buffers, step):
        gate.wait(5)
        assert buffers[0].item() == step
        published.append(step)
        return {"published": True}

    scheduler = CoalescingScheduler(FakeStager(source), publish, LocalComm())
    scheduler.start()
    for step in range(1, 11):
        source.fill_(step)
        assert scheduler.offer(step)
        if step == 1:
            time.sleep(0.2)
    gate.set()
    scheduler.close(flush=True)
    assert published[0] == 1 and published[-1] == 10
    assert published == sorted(published) and len(published) < 10
    metrics = scheduler.drain_metrics()
    assert [m["step"] for m in metrics] == published
    assert sum(m["coalesced_steps"] for m in metrics) == 10 - len(published)
    assert scheduler.coalesced == 10 - len(published)


def test_scheduler_close_without_flush_drops_waiting():
    gate = threading.Event()
    published = []

    def publish(buffers, step):
        gate.wait(5)
        published.append(step)
        return {}

    source = torch.zeros(1)
    scheduler = CoalescingScheduler(FakeStager(source), publish, LocalComm())
    scheduler.start()
    scheduler.offer(1)
    time.sleep(0.2)
    scheduler.offer(2)
    closer = threading.Thread(target=scheduler.close, kwargs={"flush": False})
    closer.start()
    time.sleep(0.1)
    gate.set()
    closer.join(5)
    assert published == [1]


def test_scheduler_surfaces_publisher_thread_errors():
    def publish(buffers, step):
        raise OSError("disk full")

    scheduler = CoalescingScheduler(FakeStager(torch.zeros(1)), publish, LocalComm())
    scheduler.start()
    scheduler.offer(1)
    scheduler.thread.join(5)
    with pytest.raises(RuntimeError, match="publishing failed"):
        scheduler.offer(2)


def test_scheduler_restores_before_publishing():
    calls = []
    source = torch.full((1,), 4.0)
    scheduler = CoalescingScheduler(
        FakeStager(source),
        lambda buffers, step: calls.append(("publish", step)) or {},
        LocalComm(),
        restore=lambda buffers, step: calls.append(
            ("restore", step, buffers[0].item())
        ),
    )
    scheduler.start(restore_step=4)
    source.fill_(5)
    scheduler.offer(5)
    scheduler.close()
    assert calls == [("restore", 4, 4.0), ("publish", 5)]


def _scheduler_worker(rank, world, init_file, out):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        comm = GlooComm.create()
        source = torch.zeros(1)
        published = []

        def publish(buffers, step):
            steps = comm.all_gather_object(step)
            assert len(set(steps)) == 1, steps
            assert buffers[0].item() == step
            time.sleep(0.05 * (rank + 1))
            published.append(step)
            return {}

        scheduler = CoalescingScheduler(FakeStager(source), publish, comm)
        scheduler.start()
        for step in range(1, 30):
            time.sleep(0.003 * ((step * (rank + 3)) % 7))
            source.fill_(step)
            scheduler.offer(step)
            dist.barrier()
        scheduler.close(flush=True)
        Path(out, f"{rank}.json").write_text(json.dumps(published))
    finally:
        dist.destroy_process_group()


def test_scheduler_ranks_agree_on_coalesced_steps(tmp_path):
    world = 3
    mp.spawn(
        _scheduler_worker,
        args=(world, str(tmp_path / "init"), str(tmp_path)),
        nprocs=world,
    )
    published = [json.loads((tmp_path / f"{r}.json").read_text()) for r in range(world)]
    assert published[0] == published[1] == published[2]
    assert published[0][-1] == 29 and len(published[0]) < 29


def test_stager_reuses_buffers_and_copies_values():
    source = torch.arange(6.0).reshape(2, 3)
    stager = LoRAStager([source])
    stager.stage(0)
    first = stager.ready(0)[0]
    pointer = first.data_ptr()
    source.add_(10)
    assert torch.equal(first, torch.arange(6.0).reshape(2, 3))
    stager.stage(0)
    assert stager.ready(0)[0].data_ptr() == pointer
    assert torch.equal(stager.ready(0)[0], source)
    assert stager.nbytes == 24
    stager.wait_training()
    with pytest.raises(RuntimeError, match="never staged"):
        stager.ready(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stager_copies_from_gpu_on_a_side_stream():
    source = torch.randn(1024, device="cuda")
    stager = LoRAStager([source])
    stager.stage(0)
    stager.wait_training()
    expected = source.cpu()
    source.add_(1)
    assert stager.ready(0)[0].is_pinned()
    assert torch.equal(stager.ready(0)[0], expected)


def test_plan_transfers_covers_needs_and_prefers_local_slices():
    shape = (4, 2)
    pieces = [
        [Piece("a", shape, shape, ((0, 2), (0, 2)), "float32")],
        [Piece("a", shape, shape, ((2, 4), (0, 2)), "float32")],
    ]
    needs = [[Need("a", shape, full_box(shape))], [Need("a", shape, ((1, 3), (0, 2)))]]
    transfers = plan_transfers(pieces, needs)
    assert [(t.src, t.dst, t.box) for t in transfers] == [
        (0, 0, ((0, 2), (0, 2))),
        (1, 0, ((2, 4), (0, 2))),
        (0, 1, ((1, 2), (0, 2))),
        (1, 1, ((2, 3), (0, 2))),
    ]
    replicated = [[Piece("a", shape, shape, full_box(shape), "float32")]] * 2
    transfers = plan_transfers(replicated, needs)
    assert [(t.src, t.dst) for t in transfers] == [(0, 0), (1, 1)]


def test_plan_transfers_rejects_overlaps_and_gaps():
    shape = (4, 2)
    overlapping = [
        [Piece("a", shape, shape, ((0, 3), (0, 2)), "float32")],
        [Piece("a", shape, shape, ((2, 4), (0, 2)), "float32")],
    ]
    with pytest.raises(ValueError, match="Overlapping"):
        plan_transfers(overlapping, [[], []])
    gap = [[Piece("a", shape, shape, ((0, 2), (0, 2)), "float32")]]
    with pytest.raises(ValueError, match="do not cover"):
        plan_transfers(gap, [[Need("a", shape, full_box(shape))]])


def test_expert_needs_slice_lora_b_by_expert():
    experts, rank, out = 4, 2, 3
    b = torch.arange(out * rank * experts, dtype=torch.float32).reshape(
        out, rank * experts
    )
    need = expert_need("b", "B", b.shape, experts, 1, 3)
    piece = Piece("b", tuple(b.shape), need.view, full_box(need.view), "float32")
    transfers = plan_transfers([[piece]], [[need]])
    (assembled,) = assemble(LocalComm(), 0, [b], [need], transfers, [[piece]])
    expected = b.reshape(out, rank, experts)[:, :, 1:3].reshape(out, rank * 2)
    assert torch.equal(assembled.reshape(need_shape(need)), expected)


def tiny_moe():
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        head_dim=8,
    )
    model = AutoModelForCausalLM.from_config(config)
    lora = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj"],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        init_lora_weights=False,
    )
    return get_peft_model(model, lora)


def test_model_pieces_reassemble_the_saved_adapter(tmp_path):
    from peft.utils import get_peft_model_state_dict

    model = tiny_moe()
    saved = get_peft_model_state_dict(model)
    collected = collect_model_pieces(model)
    pieces = [piece for piece, _ in collected]
    assert {piece.key for piece in pieces} == set(saved)
    needs = [Need(p.key, p.view, full_box(p.view)) for p in pieces]
    out = assemble(
        LocalComm(),
        0,
        [tensor for _, tensor in collected],
        needs,
        plan_transfers([pieces], [needs]),
        [pieces],
    )
    for piece, tensor in zip(pieces, out, strict=True):
        assert torch.equal(tensor.reshape(piece.shape), saved[piece.key])
    config = live_adapter_config(model)
    model.save_pretrained(tmp_path)
    saved_config = json.loads((tmp_path / "adapter_config.json").read_text())
    for key in ("r", "lora_alpha"):
        assert config[key] == saved_config[key]
    for key in ("target_modules", "target_parameters"):
        assert sorted(config[key]) == sorted(saved_config[key])


def test_model_pieces_place_ep_local_experts():
    from peft.utils import get_peft_model_state_dict

    from axolotl.integrations.expert_parallel.shard import shard_expert_lora

    model = tiny_moe()
    saved = get_peft_model_state_dict(model)
    for module in model.modules():
        if type(module).__name__.endswith("Experts"):
            module.num_local_experts, module.num_experts_global = 2, 4
            module.local_expert_offset = 2
    assert shard_expert_lora(model, 2) > 0
    pieces = dict(collect_model_pieces(model))
    key = "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight"
    piece = next(p for p in pieces if p.key == key)
    assert piece.box[2] == (2, 4) and piece.shape == tuple(saved[key].shape)
    need = expert_need(key, "B", piece.shape, 4, 2, 4)
    (assembled,) = assemble(
        LocalComm(),
        0,
        [pieces[piece]],
        [need],
        plan_transfers([[piece]], [[need]]),
        [[piece]],
    )
    expected = saved[key].reshape(32, 2, 4)[:, :, 2:4].reshape(32, 4)
    assert torch.equal(assembled.reshape(need_shape(need)), expected)
    key_a = key.replace("lora_B", "lora_A")
    piece_a = next(p for p in pieces if p.key == key_a)
    assert piece_a.box[0] == (4, 8)
    assert torch.equal(pieces[piece_a], saved[key_a][4:8])


def test_canonical_adapter_keys():
    assert (
        canonical_adapter_key(
            "base_model.model.model.layers.0._checkpoint_wrapped_module.q.lora_A.default.weight"
        )
        == "base_model.model.model.layers.0.q.lora_A.weight"
    )
    assert canonical_adapter_key("model.q.lora_B.default.weight") == (
        "base_model.model.model.q.lora_B.weight"
    )


def test_tensor_classes():
    stem = "model.layers.3.mlp.experts.7.gate_proj"
    assert tensor_class(f"{stem}.weight") == "expert_weight"
    assert tensor_class(f"{stem}.weight_scale") == "block_scales"
    assert tensor_class(f"{stem}.weight_scale_2") == "global_scales"
    assert tensor_class("model.layers.3.self_attn.q_proj.weight") == "attention"
    assert tensor_class("model.embed_tokens.weight") == "other"


def test_session_keep_carries_unchanged_tensors(tmp_path):
    writer = ChainWriter(tmp_path)
    weights = {"a": torch.ones(2), "b": torch.zeros(2)}
    writer.commit(writer.write(weights, "one", config={}))
    with writer.begin("two") as session:
        session.add("a", torch.full((2,), 2.0))
        session.keep("b")
        with pytest.raises(ValueError, match="Duplicate"):
            session.keep("b")
        with pytest.raises(KeyError):
            session.keep("c")
        staged = session.finish()
    assert set(staged.weight_map) == {"a"}
    writer.commit(staged)
    assert torch.equal(writer.baseline["b"], torch.zeros(2))
    assert writer.baseline.delta_count == 1
    with writer.begin("three", force_full=True) as session:
        with pytest.raises(ValueError, match="full checkpoint"):
            session.keep("a")


def test_local_destination_assembles_per_rank_files(tmp_path):
    ranks = []
    for rank in range(2):
        writer = ChainWriter(tmp_path / f"rank-{rank}", rank=rank)
        staged = writer.write({f"w{rank}": torch.ones(2)}, "snap", config={})
        ranks.append(staged)
    destination = LocalDestination(tmp_path / "out")
    for staged in ranks:
        destination.publish_files(
            staged.path, "snap", sorted(set(staged.weight_map.values()))
        )
    assert not (tmp_path / "out" / "snap").exists()
    index = tmp_path / "index"
    index.mkdir()
    from axolotl.core.publishing.chain import INDEX_NAME, combine_indexes

    combined = combine_indexes(
        json.loads((s.path / INDEX_NAME).read_text()) for s in ranks
    )
    (index / INDEX_NAME).write_text(json.dumps(combined))
    with pytest.raises(ValueError, match="config.json"):
        destination.publish_index(index, "snap", base_identity=None, full=True)
    (index / "config.json").write_text("{}")
    destination.publish_index(index, "snap", base_identity=None, full=True)
    assert sorted(p.name for p in (tmp_path / "out" / "snap").iterdir()) == [
        "config.json",
        "model-rank000-00000.safetensors",
        "model-rank001-00000.safetensors",
        INDEX_NAME,
    ]
    destination.publish_files(
        ranks[0].path, "other", ["model-rank000-00000.safetensors"]
    )
    destination.discard_files("other")
    assert not list((tmp_path / "out").glob(".incoming-*"))


def test_s3_per_rank_upload_puts_the_index_last(tmp_path):
    client = MagicMock()
    destination = S3Destination("bucket", "prefix", client=client)
    directory = tmp_path / "files"
    directory.mkdir()
    for name in ("a.safetensors", "config.json", "model.safetensors.index.json"):
        (directory / name).write_text(name)
    destination.publish_files(directory, "snap", ["a.safetensors"])
    destination.publish_index(directory, "snap", base_identity=None, full=True)
    keys = [call.args[2] for call in client.upload_file.call_args_list]
    assert keys == [
        "prefix/snap/a.safetensors",
        "prefix/snap/config.json",
        "prefix/snap/model.safetensors.index.json",
    ]
    client.reset_mock()
    destination.publish_index(directory, "delta", base_identity="snap", full=False)
    assert [c.args[2] for c in client.upload_file.call_args_list] == [
        "prefix/delta/model.safetensors.index.json"
    ]


def test_coreweave_destination_hot_loads_after_the_index(tmp_path):
    from axolotl.integrations.coreweave.args import CoreWeaveConfig
    from axolotl.integrations.coreweave.destination import CoreWeaveDestination

    config = CoreWeaveConfig(
        deployment_id="d",
        bucket="bucket",
        path_prefix="weights",
        gateway_url="https://gateway.example",
        model="m",
    )
    calls = []
    s3 = MagicMock()
    s3.upload_file.side_effect = lambda path, bucket, key: calls.append(key)
    client = MagicMock()
    client.hot_load.side_effect = lambda identity, base_identity: calls.append(
        ("hot_load", identity, base_identity)
    )
    destination = CoreWeaveDestination(config, client, s3)
    directory = tmp_path / "index"
    directory.mkdir()
    (directory / "model.safetensors.index.json").write_text("{}")
    destination.publish_index(directory, "two", base_identity="one", full=False)
    assert calls == [
        "weights/two/model.safetensors.index.json",
        ("hot_load", "two", "one"),
    ]


class FakePublisher:
    def __init__(self):
        self.published, self.restored, self.closed = [], [], False

    def publish(self, buffers, step):
        self.published.append((step, buffers[0].clone()))
        return {"identity": f"id-{step}", "action": "commit", "full": step == 2}

    def restore(self, buffers, step):
        self.restored.append(step)
        return True

    def close(self):
        self.closed = True


def test_callback_stages_every_n_steps_and_logs(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        LoraConfig(r=2, target_modules=["q_proj"], init_lora_weights=False),
    )
    fake = FakePublisher()
    received = {}

    def factory(*, model, comm, pieces):
        received.update(comm=comm, pieces=pieces)
        return fake

    trainer = MagicMock()
    callback = StepPublishCallback(factory, every=2, trainer=trainer)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None, model=model)
    assert isinstance(received["comm"], LocalComm)
    assert {p.key.split(".lora_")[1] for p in received["pieces"]} == {
        "A.weight",
        "B.weight",
    }
    lora_a = next(p for n, p in model.named_parameters() if "lora_A" in n)
    for step in range(1, 5):
        callback.on_pre_optimizer_step(None, state, None)
        with torch.no_grad():
            lora_a.fill_(step)
        state.global_step = step
        callback.on_step_end(None, state, None)
    callback.on_train_end(None, state, None)
    steps = [step for step, _ in fake.published]
    assert steps[-1] == 4 and set(steps) <= {2, 4}
    assert all(
        torch.all(buffer == step)
        for step, buffer in fake.published
        if received["pieces"][0].key.endswith("lora_A.weight")
    )
    assert fake.restored == [] and fake.closed
    logged = [call.args[0] for call in trainer.log.call_args_list]
    assert logged and all(
        key.startswith(METRIC_PREFIX) for values in logged for key in values
    )
    assert logged[-1][METRIC_PREFIX + "step"] == 4.0


def test_callback_restores_on_resume():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        LoraConfig(r=2, target_modules=["q_proj"]),
    )
    fake = FakePublisher()
    callback = StepPublishCallback(lambda **kwargs: fake, every=1, trainer=None)
    state = SimpleNamespace(global_step=7, is_world_process_zero=True)
    callback.on_train_begin(None, state, None, model=model)
    callback.on_train_end(None, state, None)
    assert fake.restored == [7] and fake.published == []


def test_coreweave_plugin_registers_the_step_callback():
    from axolotl.integrations.coreweave.args import CoreWeaveConfig
    from axolotl.integrations.coreweave.plugin import CoreWeavePlugin
    from axolotl.utils.dict import DictDefault

    config = CoreWeaveConfig(
        deployment_id="d",
        bucket="bucket",
        path_prefix="weights",
        gateway_url="https://gateway.example",
        model="m",
    )
    cfg = DictDefault(
        {
            "coreweave": config.model_dump(),
            "adapter": "lora",
            "base_model": "test/model",
            "checkpoint_publish": {"every": 3},
        }
    )
    CoreWeavePlugin().register(cfg)
    trainer = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
    (callback,) = CoreWeavePlugin().add_callbacks_post_trainer(cfg, trainer)
    assert isinstance(callback, StepPublishCallback) and callback.every == 3
