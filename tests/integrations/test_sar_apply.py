"""E2E tests for SAR checkpoint projection on a tiny Llama."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.sar.core import DEFAULT_TARGET_MODULES, _rank_for, run_sar

RATIO = 0.25


def _is_targeted(name: str, shape: tuple[int, ...]) -> bool:
    return (
        name.endswith(".weight")
        and len(shape) == 2
        and any(module in name for module in DEFAULT_TARGET_MODULES)
    )


def _rank(ratio: float, shape: tuple[int, ...]) -> int:
    return _rank_for(ratio, min(shape))


def test_rank_for_matches_exact_ceil():
    # 0.07 * 300 = 21.000000000000004 in float64; exact ceil is 21
    assert _rank_for(0.07, 300) == 21
    assert _rank_for(0.14, 300) == 42
    assert _rank_for(0.25, 64) == 16
    assert _rank_for(1.0, 300) == 300
    assert _rank_for(0.001, 64) == 1


def _project(
    w_base: torch.Tensor, w_trained: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    base = w_base.to(torch.float32)
    delta = w_trained.to(torch.float32) - base
    delta_u, delta_s, delta_vh = torch.linalg.svd(delta, full_matrices=False)
    delta_k = (delta_u[:, :rank] * delta_s[:rank]) @ delta_vh[:rank, :]
    base_u, _, base_vh = torch.linalg.svd(base, full_matrices=False)
    u_k = base_u[:, :rank]
    vh_k = base_vh[:rank, :]
    m = u_k.T @ delta_k @ vh_k.T
    return u_k @ m @ vh_k, m


@dataclass
class TinyFamily:
    base: Path
    trained: Path
    merge: Path
    inspace: Path
    base_sd: dict[str, torch.Tensor]
    trained_sd: dict[str, torch.Tensor]
    merge_sd: dict[str, torch.Tensor]
    inspace_sd: dict[str, torch.Tensor]
    targeted: set[str]


@pytest.fixture(scope="module", name="tiny_family")
def fixture_tiny_family(tmp_path_factory) -> TinyFamily:
    root = tmp_path_factory.mktemp("sar_models")
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    base_dir = root / "base"
    model.save_pretrained(base_dir)
    (base_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    del model
    base_sd = load_file(str(base_dir / "model.safetensors"))
    targeted = {
        name for name, tensor in base_sd.items() if _is_targeted(name, tensor.shape)
    }

    def save_variant(dirname: str, state_dict: dict[str, torch.Tensor]) -> Path:
        out = root / dirname
        out.mkdir()
        save_file(state_dict, str(out / "model.safetensors"))
        return out

    torch.manual_seed(7)
    trained_sd = {
        name: tensor + 0.02 * torch.randn_like(tensor)
        for name, tensor in base_sd.items()
    }
    trained_dir = save_variant("trained", trained_sd)

    torch.manual_seed(11)
    merge_sd = {
        name: tensor + 0.02 * torch.randn_like(tensor)
        for name, tensor in base_sd.items()
    }
    merge_dir = save_variant("merge", merge_sd)
    for filename in ("config.json", "generation_config.json", "tokenizer_config.json"):
        shutil.copy2(base_dir / filename, merge_dir / filename)

    inspace_sd = {
        name: 1.1 * tensor if name in targeted else tensor.clone()
        for name, tensor in base_sd.items()
    }
    inspace_dir = save_variant("inspace", inspace_sd)

    return TinyFamily(
        base=base_dir,
        trained=trained_dir,
        merge=merge_dir,
        inspace=inspace_dir,
        base_sd=base_sd,
        trained_sd=trained_sd,
        merge_sd=merge_sd,
        inspace_sd=inspace_sd,
        targeted=targeted,
    )


def test_extraction_e2e(tiny_family, tmp_path):
    out = tmp_path / "sar_out"
    result = run_sar(
        str(tiny_family.base),
        str(tiny_family.trained),
        str(out),
        rank_ratios=[RATIO],
        svd_device="cpu",
    )

    assert result.outputs == {RATIO: str(out)}
    assert result.num_projected == len(tiny_family.targeted) == 14
    assert result.total_params == sum(
        tensor.numel() for tensor in tiny_family.base_sd.values()
    )

    out_sd = load_file(str(out / "model.safetensors"))
    assert set(out_sd) == set(tiny_family.base_sd)
    for name, base_w in tiny_family.base_sd.items():
        got = out_sd[name]
        assert got.dtype == torch.float16
        if name in tiny_family.targeted:
            rank = _rank(RATIO, base_w.shape)
            delta_star, _ = _project(base_w, tiny_family.trained_sd[name], rank)
            expected = base_w.to(torch.float32) + delta_star
            torch.testing.assert_close(
                got.to(torch.float32), expected, atol=1e-3, rtol=0
            )
            assert not torch.equal(got, base_w.to(torch.float16))
        else:
            assert torch.equal(got, base_w.to(torch.float16))

    for filename in ("config.json", "generation_config.json", "tokenizer_config.json"):
        assert (out / filename).is_file()

    with open(out / "config.json", encoding="utf-8") as fp:
        model_config = json.load(fp)
    assert model_config.get("dtype", model_config.get("torch_dtype")) == "float16"

    with open(out / "sar_config.json", encoding="utf-8") as fp:
        sar_config = json.load(fp)
    assert sar_config["rank_ratio"] == RATIO
    assert sar_config["projection"] == "spectral"
    assert sar_config["num_projected"] == 14
    assert sar_config["total_params"] == result.total_params
    assert set(sar_config["layer_ranks"]) == tiny_family.targeted
    for name, ranks in sar_config["layer_ranks"].items():
        expected_rank = _rank(RATIO, tiny_family.base_sd[name].shape)
        assert ranks == {"rank": expected_rank, "delta_rank": expected_rank}
    assert result.m_params == sum(
        ranks["rank"] ** 2 for ranks in sar_config["layer_ranks"].values()
    )


def test_merge_mode(tiny_family, tmp_path):
    out = tmp_path / "merge_out"
    run_sar(
        str(tiny_family.base),
        str(tiny_family.trained),
        str(out),
        merge_target=str(tiny_family.merge),
        rank_ratios=[RATIO],
        svd_device="cpu",
    )

    out_sd = load_file(str(out / "model.safetensors"))
    assert set(out_sd) == set(tiny_family.merge_sd)
    for name, base_w in tiny_family.base_sd.items():
        got = out_sd[name]
        if name in tiny_family.targeted:
            rank = _rank(RATIO, base_w.shape)
            delta_star, _ = _project(base_w, tiny_family.trained_sd[name], rank)
            expected = tiny_family.merge_sd[name].to(torch.float32) + delta_star
            torch.testing.assert_close(
                got.to(torch.float32), expected, atol=1e-3, rtol=0
            )
            base_anchored = base_w.to(torch.float32) + delta_star
            assert not torch.allclose(got.to(torch.float32), base_anchored, atol=1e-3)
        else:
            assert torch.equal(got, tiny_family.merge_sd[name].to(torch.float16))
            assert not torch.equal(got, base_w.to(torch.float16))

    with open(out / "sar_config.json", encoding="utf-8") as fp:
        sar_config = json.load(fp)
    assert sar_config["merge_target"] == str(tiny_family.merge)


def test_rank_sweep_full_ratio_recovers_trained(tiny_family, tmp_path):
    out = tmp_path / "sweep"
    result = run_sar(
        str(tiny_family.base),
        str(tiny_family.inspace),
        str(out),
        rank_ratios=[RATIO, 1.0],
        svd_device="cpu",
    )

    assert set(result.outputs) == {RATIO, 1.0}
    quarter_dir = Path(result.outputs[RATIO])
    full_dir = Path(result.outputs[1.0])
    assert quarter_dir == out / f"rank_{RATIO}"
    assert full_dir == out / "rank_1.0"

    full_sd = load_file(str(full_dir / "model.safetensors"))
    quarter_sd = load_file(str(quarter_dir / "model.safetensors"))
    for name in tiny_family.targeted:
        want = tiny_family.inspace_sd[name].to(torch.float32)
        torch.testing.assert_close(
            full_sd[name].to(torch.float32), want, atol=1e-3, rtol=0
        )
    probe = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.allclose(
        quarter_sd[probe].to(torch.float32),
        tiny_family.inspace_sd[probe].to(torch.float32),
        atol=1e-3,
    )

    for ratio, ratio_dir in ((RATIO, quarter_dir), (1.0, full_dir)):
        with open(ratio_dir / "sar_config.json", encoding="utf-8") as fp:
            sar_config = json.load(fp)
        assert sar_config["rank_ratio"] == ratio
        assert sar_config["rank_ratios"] == [RATIO, 1.0]
        for name, ranks in sar_config["layer_ranks"].items():
            expected_rank = _rank(ratio, tiny_family.base_sd[name].shape)
            assert ranks == {"rank": expected_rank, "delta_rank": expected_rank}


def test_streamed_sharded_output(tiny_family, tmp_path, monkeypatch):
    import axolotl.integrations.sar.core as sar_core

    monkeypatch.setattr(sar_core, "MAX_SHARD_BYTES", 20_000)
    out = tmp_path / "sharded"
    run_sar(
        str(tiny_family.base),
        str(tiny_family.trained),
        str(out),
        rank_ratios=[RATIO],
        svd_device="cpu",
    )

    index_path = out / "model.safetensors.index.json"
    assert index_path.is_file()
    with open(index_path, encoding="utf-8") as fp:
        index = json.load(fp)
    shard_files = sorted(set(index["weight_map"].values()))
    assert len(shard_files) > 1
    assert all((out / shard).is_file() for shard in shard_files)
    out_sd: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        out_sd.update(load_file(str(out / shard)))
    assert set(out_sd) == set(index["weight_map"]) == set(tiny_family.base_sd)
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in out_sd.values()
    )
    for name in tiny_family.targeted:
        rank = _rank(RATIO, tiny_family.base_sd[name].shape)
        delta_star, _ = _project(
            tiny_family.base_sd[name], tiny_family.trained_sd[name], rank
        )
        expected = tiny_family.base_sd[name].to(torch.float32) + delta_star
        torch.testing.assert_close(
            out_sd[name].to(torch.float32), expected, atol=1e-3, rtol=0
        )


def test_projection_none_reports_no_m_params(tiny_family, tmp_path):
    out = tmp_path / "none_out"
    result = run_sar(
        str(tiny_family.base),
        str(tiny_family.trained),
        str(out),
        rank_ratios=[RATIO],
        projection="none",
        svd_device="cpu",
    )

    assert result.m_params == 0
    with open(out / "sar_config.json", encoding="utf-8") as fp:
        sar_config = json.load(fp)
    assert sar_config["m_params"] == 0
    assert sar_config["projection"] == "none"


def test_rewiring_artifact(tiny_family, tmp_path):
    out = tmp_path / "artifact"
    run_sar(
        str(tiny_family.base),
        str(tiny_family.trained),
        str(out),
        rank_ratios=[RATIO],
        svd_device="cpu",
        save_rewiring_matrix=True,
    )

    rewiring_dir = out / "rewiring"
    shards = sorted(rewiring_dir.glob("rewiring*.safetensors"))
    assert shards
    m_sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        m_sd.update(load_file(str(shard)))
    assert set(m_sd) == tiny_family.targeted
    for name, m in m_sd.items():
        rank = _rank(RATIO, tiny_family.base_sd[name].shape)
        assert m.dtype == torch.float32
        assert m.shape == (rank, rank)
        _, expected_m = _project(
            tiny_family.base_sd[name], tiny_family.trained_sd[name], rank
        )
        torch.testing.assert_close(m, expected_m, atol=1e-5, rtol=0)

    with open(rewiring_dir / "metadata.json", encoding="utf-8") as fp:
        metadata = json.load(fp)
    assert isinstance(metadata["formula_version"], int)
    assert metadata["formula_version"] >= 1
    assert metadata["base_model"] == str(tiny_family.base)
    assert metadata["trained_model"] == str(tiny_family.trained)
    assert metadata["rank_ratio"] == RATIO
    assert set(metadata["layer_ranks"]) == tiny_family.targeted
    for name, ranks in metadata["layer_ranks"].items():
        assert ranks["rank"] == m_sd[name].shape[0]


def test_validation_errors(tiny_family, tmp_path):
    bad_dir = tmp_path / "bad_trained"
    bad_dir.mkdir()
    bad_sd = {name: tensor.clone() for name, tensor in tiny_family.trained_sd.items()}
    bad_sd["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(16, 64)
    save_file(bad_sd, str(bad_dir / "model.safetensors"))

    with pytest.raises(ValueError, match=r"shape mismatch(es)?.*q_proj"):
        run_sar(
            str(tiny_family.base),
            str(bad_dir),
            str(tmp_path / "out_mismatch"),
            rank_ratios=[RATIO],
            svd_device="cpu",
        )

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="safetensors"):
        run_sar(
            str(tiny_family.base),
            str(empty_dir),
            str(tmp_path / "out_missing"),
            rank_ratios=[RATIO],
            svd_device="cpu",
        )


def _write_config(path: Path, sar_block: dict, base_model: str, output_dir: str):
    cfg = {
        "base_model": base_model,
        "learning_rate": 1e-4,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_len": 512,
        "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
        "output_dir": output_dir,
        "plugins": ["axolotl.integrations.sar.SARPlugin"],
        "sar": sar_block,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_config_schema_roundtrip(tiny_family, tmp_path):
    from axolotl.cli.config import load_cfg

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "rank_ratio": [0.5, 0.25],
            "projection": "spectral",
            "rewiring": "diagonal",
            "scale": 0.5,
            "target_modules": ["q_proj", "v_proj"],
            "exclude_modules": ["layers.0"],
            "svd_device": "cpu",
            "save_dtype": "bfloat16",
            "save_rewiring_matrix": True,
            "run_after_training": False,
        },
        str(tiny_family.base),
        str(tmp_path / "train_out"),
    )
    cfg = load_cfg(str(config_path))

    assert cfg.plugins == ["axolotl.integrations.sar.SARPlugin"]
    assert cfg.sar["rank_ratio"] == [0.25, 0.5]
    assert cfg.sar["projection"] == "spectral"
    assert cfg.sar["rewiring"] == "diagonal"
    assert cfg.sar["scale"] == 0.5
    assert cfg.sar["target_modules"] == ["q_proj", "v_proj"]
    assert cfg.sar["exclude_modules"] == ["layers.0"]
    assert cfg.sar["svd_device"] == "cpu"
    assert cfg.sar["save_dtype"] == "bfloat16"
    assert cfg.sar["save_rewiring_matrix"] is True
    assert cfg.sar["run_after_training"] is False

    bad_blocks = [
        ({"rank_ratio": 0}, r"rank_ratio values must be in"),
        ({"rank_ratio": 1.5}, r"rank_ratio values must be in"),
        (
            {"projection": "none", "rewiring": "diagonal"},
            r"rewiring ablations require projection",
        ),
    ]
    for idx, (sar_block, message) in enumerate(bad_blocks):
        bad_path = tmp_path / f"bad_{idx}.yaml"
        _write_config(
            bad_path,
            sar_block,
            str(tiny_family.base),
            str(tmp_path / f"bad_out_{idx}"),
        )
        with pytest.raises(ValueError, match=message):
            load_cfg(str(bad_path))
