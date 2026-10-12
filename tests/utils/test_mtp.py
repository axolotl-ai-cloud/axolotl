import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from axolotl.utils.mtp import INDEX_NAME, MTP_SHARD, copy_mtp_weights


def _write_model(path, tensors, config):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    save_file(tensors, path / "model.safetensors")


def test_copies_mtp_layers_and_shards_single_file(tmp_path):
    config = {"num_hidden_layers": 1, "num_nextn_predict_layers": 1}
    trunk = {"model.layers.0.w": torch.ones(2), "lm_head.weight": torch.ones(2)}
    mtp = {
        "model.layers.1.eh_proj.weight": torch.full((2,), 3.0),
        "mtp.fc.weight": torch.zeros(2),
    }
    _write_model(tmp_path / "base", {**trunk, **mtp}, config)
    _write_model(tmp_path / "out", {k: v * 2 for k, v in trunk.items()}, config)

    assert copy_mtp_weights(str(tmp_path / "base"), tmp_path / "out") == 2

    index = json.loads((tmp_path / "out" / INDEX_NAME).read_text())
    assert not (tmp_path / "out" / "model.safetensors").exists()
    assert {k for k, f in index["weight_map"].items() if f == MTP_SHARD} == set(mtp)
    with safe_open(tmp_path / "out" / MTP_SHARD, framework="pt") as f:
        assert torch.equal(
            f.get_tensor("model.layers.1.eh_proj.weight"),
            mtp["model.layers.1.eh_proj.weight"],
        )

    assert copy_mtp_weights(str(tmp_path / "base"), tmp_path / "out") == 0


def test_noop_without_mtp(tmp_path):
    config = {"num_hidden_layers": 2}
    tensors = {"model.layers.1.w": torch.ones(2)}
    _write_model(tmp_path / "base", tensors, config)
    _write_model(tmp_path / "out", {}, config)

    assert copy_mtp_weights(str(tmp_path / "base"), tmp_path / "out") == 0
    assert not (tmp_path / "out" / INDEX_NAME).exists()
