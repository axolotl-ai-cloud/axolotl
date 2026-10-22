"""In-trainer per-step publishing: per-rank builds over gloo match the offline merge."""

import json
import shutil
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
import zstandard
from safetensors import safe_open
from safetensors.torch import load_file

from axolotl.cli.utils import lora_merge
from axolotl.core.publishing import (
    ChainPublisher,
    ChainWriter,
    LocalDestination,
    MergedExporter,
    pin_base,
)
from axolotl.core.publishing.adapters import load_adapter_config
from axolotl.core.publishing.comm import GlooComm
from axolotl.core.publishing.lora_shards import (
    Piece,
    expert_view,
)
from axolotl.core.publishing.step_publisher import StepPublisher

from tests.core.publishing.nvfp4_moe import (
    INDEX,
    E,
    apply_delta,
    assert_bytes_equal,
    build_base,
    load_checkpoint,
    save_adapter,
)

pytest.importorskip("torchao")


def _ranges(size: int, parts: int) -> list[tuple[int, int]]:
    chunk = -(-size // parts)
    return [(min(i * chunk, size), min((i + 1) * chunk, size)) for i in range(parts)]


def split_adapter(state, rank, world, *, ep=None):
    """This rank's pieces, sharded the way EP plus FSDP2 Shard(0) would shard them.

    Experts are split contiguously over ``ep`` groups (default: every rank its
    own group); inside a group, rows are FSDP-chunked. Linear LoRA is chunked by
    rows over all ranks.
    """
    ep = ep or world
    dp = world // ep
    group, dp_rank = rank // dp, rank % dp
    pieces, data = [], []
    for key in sorted(state):
        tensor = state[key]
        shape = tuple(tensor.shape)
        if ".mlp.experts" not in key:
            start, stop = _ranges(shape[0], world)[rank]
            if start == stop:
                continue
            box = ((start, stop), (0, shape[1]))
            pieces.append(Piece(key, shape, shape, box, "bfloat16"))
            data.append(tensor[start:stop].clone())
            continue
        first, last = _ranges(E, ep)[group]
        kind = "A" if ".lora_A." in key else "B"
        view = expert_view(kind, shape, E)
        if kind == "A":
            r = shape[0] // E
            rows = _ranges((last - first) * r, dp)[dp_rank]
            box = ((first * r + rows[0], first * r + rows[1]), (0, shape[1]))
            local = tensor[box[0][0] : box[0][1]]
        else:
            rows = _ranges(shape[0], dp)[dp_rank]
            box = (rows, (0, view[1]), (first, last))
            local = tensor.reshape(view)[rows[0] : rows[1], :, first:last]
        pieces.append(Piece(key, shape, view, box, "bfloat16"))
        data.append(local.contiguous().clone())
    return pieces, data


class FailingDestination(LocalDestination):
    def __init__(self, root, fail_index_at=None):
        super().__init__(root)
        self.fail_index_at, self.calls = fail_index_at, 0

    def publish_index(self, directory, identity, *, base_identity, full):
        self.calls += 1
        if self.calls == self.fail_index_at:
            raise RuntimeError("hot-load failed")
        super().publish_index(
            directory, identity, base_identity=base_identity, full=full
        )


def _worker(rank, world, init_file, spec_path):
    spec = json.loads(Path(spec_path).read_text())
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        comm = GlooComm.create()
        work = Path(spec["work"])
        base = pin_base(spec["base"], work / f"pin-{rank}")
        config = load_adapter_config(Path(spec["adapters"][0]))

        def pieces_of(adapter):
            return split_adapter(
                load_file(Path(adapter) / "adapter_model.safetensors"),
                rank,
                world,
                ep=spec.get("ep"),
            )

        pieces, _ = pieces_of(spec["adapters"][0])
        destination = FailingDestination(spec["published"], spec.get("fail_index_at"))
        publisher = StepPublisher(
            base=base,
            adapter_config=config,
            pieces=pieces,
            destination=destination,
            workdir=work / "chain",
            state_dir=work / "state",
            comm=comm,
            max_shard_size=64 << 10,
            max_deltas=spec.get("max_deltas", 63),
            expert_chunk=spec.get("expert_chunk", 8),
        )
        if rank == spec.get("fail_build_rank"):
            build = publisher.builder.build
            calls = iter(range(len(spec["steps"])))

            def failing(assembled, sink, *, full):
                if next(calls) == spec["fail_build_call"]:
                    raise RuntimeError("build failed")
                return build(assembled, sink, full=full)

            publisher.builder.build = failing
        results = []
        if "restore_step" in spec:
            _, data = pieces_of(spec["restore_adapter"])
            results.append({"restored": publisher.restore(data, spec["restore_step"])})
        for step, adapter in zip(spec["steps"], spec["adapters"], strict=True):
            _, data = pieces_of(adapter)
            metrics = publisher.publish(data, step)
            metrics["chain_identity"] = publisher.identity
            metrics["delta_count"] = publisher.chain.delta_count
            results.append(metrics)
        publisher.close()
        Path(spec["out"]).joinpath(f"rank-{rank}.json").write_text(
            json.dumps(results, default=str)
        )
    finally:
        dist.destroy_process_group()


def run_ranks(tmp_path, world, **spec):
    run = tmp_path / f"run-{len(list(tmp_path.glob('run-*')))}"
    run.mkdir()
    spec = {
        "out": str(run),
        **{k: str(v) if isinstance(v, Path) else v for k, v in spec.items()},
    }
    spec["adapters"] = [str(a) for a in spec["adapters"]]
    path = run / "spec.json"
    path.write_text(json.dumps(spec))
    mp.spawn(_worker, args=(world, str(run / "init"), str(path)), nprocs=world)
    return [json.loads((run / f"rank-{r}.json").read_text()) for r in range(world)]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("step")
    base = build_base(root / "base")
    adapters = [save_adapter(root / f"adapter-{step}", step) for step in range(4)]
    return root, base, adapters


def offline_chain(tmp_path, base_dir, adapters, *, merge_aware=False):
    """The offline path's artifacts, identity per step, to compare against."""
    work = tmp_path / "offline"
    base = pin_base(base_dir, work)
    publisher = ChainPublisher(
        ChainWriter(work, max_shard_size=64 << 10),
        LocalDestination(tmp_path / "offline-published"),
    )
    exporter = MergedExporter(base, workdir=work)
    identities = []
    for step, adapter in enumerate(adapters):
        identities.append(
            publisher.publish_export(exporter, adapter, f"offline-{step:04d}")
        )
    return [tmp_path / "offline-published" / i for i in identities]


def frames(directory: Path) -> dict[str, tuple[bytes, str]]:
    index = json.loads((directory / INDEX).read_text())
    out = {}
    for name in set(index["weight_map"].values()):
        with safe_open(directory / name, framework="numpy") as shard:
            for key in shard.keys():
                out[key] = (
                    zstandard.ZstdDecompressor().decompress(shard.get_tensor(key)),
                    shard.metadata()[key],
                )
    return out


def reconstruct(paths):
    state = load_checkpoint(paths[0])
    states = [state]
    for path in paths[1:]:
        state = apply_delta(path, state)
        states.append(state)
    return states


@pytest.mark.parametrize(("world", "ep"), [(2, None), (3, None), (4, 2)])
def test_rank_builds_match_the_offline_merge(tmp_path, setup, world, ep):
    root, base, adapters = setup
    results = run_ranks(
        tmp_path,
        world,
        base=base,
        adapters=adapters,
        steps=list(range(len(adapters))),
        work=tmp_path / "work",
        published=tmp_path / "published",
        ep=ep,
    )
    identities = [m["identity"] for m in results[0]]
    for rank_results in results:
        assert [m["identity"] for m in rank_results] == identities
        assert [m["delta_count"] for m in rank_results] == [0, 1, 2, 3]
    assert [m["full"] for m in results[0]] == [True, False, False, False]
    published = [tmp_path / "published" / identity for identity in identities]
    offline = offline_chain(tmp_path, base, adapters)

    full = json.loads((published[0] / INDEX).read_text())
    assert {Path(f).name.split("-")[1] for f in full["weight_map"].values()} == {
        f"rank{r:03d}" for r in range(world)
    }
    assert_bytes_equal(load_checkpoint(published[0]), load_checkpoint(offline[0]))
    for ours, theirs in zip(published[1:], offline[1:], strict=True):
        ours_index = json.loads((ours / INDEX).read_text())["metadata"]
        assert ours_index["delta_encoding"] == "xor"
        assert frames(ours) == frames(theirs)
    for ours, theirs in zip(reconstruct(published), reconstruct(offline), strict=True):
        assert_bytes_equal(ours, theirs)
    assert not list((tmp_path / "published").glob(".incoming-*"))

    metrics = results[0][1]
    assert metrics["changed_attention"] > 0 and metrics["changed_expert_weight"] > 0
    assert metrics["compressed_attention"] > 0
    assert metrics["tensors_other"] == 0
    for stage in ("gather", "build", "encode", "upload", "publish_index"):
        assert metrics[f"seconds_{stage}"] >= 0


def test_merge_aware_rank_builds_match(tmp_path, setup):
    root, base, _ = setup
    adapters = [
        save_adapter(tmp_path / f"aware-{step}", step, merge_aware=True)
        for step in range(2)
    ]
    results = run_ranks(
        tmp_path,
        2,
        base=base,
        adapters=adapters,
        steps=[0, 1],
        work=tmp_path / "work",
        published=tmp_path / "published",
        expert_chunk=1,
    )
    published = [tmp_path / "published" / m["identity"] for m in results[0]]
    offline = offline_chain(tmp_path, base, adapters)
    for ours, theirs in zip(reconstruct(published), reconstruct(offline), strict=True):
        assert_bytes_equal(ours, theirs)


def test_failed_publish_rolls_back_every_rank(tmp_path, setup):
    root, base, adapters = setup
    results = run_ranks(
        tmp_path,
        2,
        base=base,
        adapters=adapters,
        steps=[0, 1, 2, 3],
        work=tmp_path / "work",
        published=tmp_path / "published",
        fail_index_at=2,
        fail_build_rank=1,
        fail_build_call=2,
    )
    actions = [m["action"] for m in results[0]]
    assert actions == ["commit", "discard", "discard", "commit"]
    assert "hot-load failed" in results[0][1]["errors"][0]
    assert "rank 1" in results[0][2]["errors"][0]
    first = results[0][0]["identity"]
    for rank_results in results:
        assert [m["chain_identity"] for m in rank_results][:3] == [first] * 3
        assert [m["delta_count"] for m in rank_results] == [0, 0, 0, 1]
    published = sorted(p.name for p in (tmp_path / "published").iterdir())
    assert published == sorted([first, results[0][3]["identity"]])
    last = tmp_path / "published" / results[0][3]["identity"]
    index = json.loads((last / INDEX).read_text())["metadata"]
    assert index["base_version"] == first
    reference = tmp_path / "reference"
    lora_merge.merge_lora_sharded_efficient(base, adapters[3], reference)
    assert_bytes_equal(
        apply_delta(last, load_checkpoint(tmp_path / "published" / first)),
        load_checkpoint(reference),
    )
    for rank in range(2):
        work = tmp_path / "work" / "chain" / f"rank-{rank:05d}"
        assert not list(work.glob(".staging-*"))
        assert len(list(work.glob(".baseline-*"))) == 1


def _lose_scratch(work: Path, world: int):
    for rank in range(world):
        shutil.rmtree(work / "chain" / f"rank-{rank:05d}")


@pytest.fixture
def committed_chain(tmp_path, setup):
    root, base, adapters = setup
    first = run_ranks(
        tmp_path,
        2,
        base=base,
        adapters=adapters[:2],
        steps=[0, 1],
        work=tmp_path / "work",
        published=tmp_path / "published",
    )
    return base, adapters, [m["identity"] for m in first[0]]


def test_resume_rebuilds_the_baseline_and_continues_the_chain(
    tmp_path, committed_chain
):
    base, adapters, identities = committed_chain
    _lose_scratch(tmp_path / "work", 2)
    results = run_ranks(
        tmp_path,
        2,
        base=base,
        adapters=adapters[2:3],
        steps=[2],
        work=tmp_path / "work",
        published=tmp_path / "published",
        restore_step=1,
        restore_adapter=adapters[1],
    )
    for rank_results in results:
        assert rank_results[0] == {"restored": True}
        assert rank_results[1]["action"] == "commit"
        assert rank_results[1]["delta_count"] == 2
    assert not results[0][1]["full"]
    delta = tmp_path / "published" / results[0][1]["identity"]
    assert (
        json.loads((delta / INDEX).read_text())["metadata"]["base_version"]
        == (identities[1])
    )
    paths = [tmp_path / "published" / i for i in identities] + [delta]
    reference = tmp_path / "reference"
    lora_merge.merge_lora_sharded_efficient(base, adapters[2], reference)
    assert_bytes_equal(reconstruct(paths)[-1], load_checkpoint(reference))


def test_resume_keeps_a_surviving_baseline(tmp_path, committed_chain):
    base, adapters, identities = committed_chain
    results = run_ranks(
        tmp_path,
        2,
        base=base,
        adapters=adapters[2:3],
        steps=[2],
        work=tmp_path / "work",
        published=tmp_path / "published",
        restore_step=1,
        restore_adapter=adapters[1],
    )
    assert all(r[0] == {"restored": True} for r in results)
    assert not results[0][1]["full"] and results[0][1]["delta_count"] == 2


@pytest.mark.parametrize("case", ["wrong adapter", "other step", "no record"])
def test_resume_falls_back_to_a_full_checkpoint(tmp_path, committed_chain, case):
    base, adapters, identities = committed_chain
    _lose_scratch(tmp_path / "work", 2)
    restore = {"restore_step": 1, "restore_adapter": adapters[0]}
    if case == "other step":
        restore = {"restore_step": 5, "restore_adapter": adapters[1]}
    elif case == "no record":
        shutil.rmtree(tmp_path / "work" / "state")
        restore["restore_adapter"] = adapters[1]
    results = run_ranks(
        tmp_path,
        2,
        base=base,
        adapters=adapters[2:3],
        steps=[2],
        work=tmp_path / "work",
        published=tmp_path / "published",
        **restore,
    )
    for rank_results in results:
        assert rank_results[0] == {"restored": False}
        assert rank_results[1]["full"] and rank_results[1]["delta_count"] == 0
        work = tmp_path / "work" / "chain"
        assert len(list(work.glob("rank-*/.baseline-*"))) == 2
