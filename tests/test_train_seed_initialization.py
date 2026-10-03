"""Model and adapter initialization must use the configured training seed."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from axolotl.train import seed_model_initialization, setup_model_and_trainer
from axolotl.utils.dict import DictDefault


def _native_source() -> Path:
    from tests.native_source_fixtures import native_source_fixture_path

    source = native_source_fixture_path("nemotron")
    if source is None:
        pytest.skip("native Nemotron source fixture unavailable")
    assert source is not None
    return source


def _subprocess_fingerprint(
    source: Path, ambient_seed: int, configured_seed: int, slot_mode: str
) -> str:
    program = r"""
import hashlib
import random
import sys
import torch
from peft import LoraConfig, get_peft_model
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from axolotl.model_support.nemotron_diffusion.compat import resolve_nemotron_model_class
from axolotl.train import seed_model_initialization
from axolotl.utils.dict import DictDefault

source, ambient, configured, slot_mode = sys.argv[1:]
random.seed(int(ambient))
torch.manual_seed(int(ambient))
seed_model_initialization(DictDefault({"seed": int(configured), "full_determinism": False}))
config_cls = get_class_from_dynamic_module(
    "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
    source,
    local_files_only=True,
)
config = config_cls(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
    mask_token_id=100,
    dlm_paradigm="bidirectional",
    use_cache=False,
    rope_parameters={"llama_4_scaling_beta": 1.0, "original_max_position_embeddings": 1},
)
config._name_or_path = source
model = resolve_nemotron_model_class(source)(config)
model = get_peft_model(
    model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], task_type=None)
)
if slot_mode == "learned":
    model.register_parameter(
        "trainable_token_delta", torch.nn.Parameter(torch.randn(8, 32))
    )
digest = hashlib.sha256()
for name, parameter in sorted(model.named_parameters()):
    if ".lora_A." in name or ".lora_B." in name:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
print(digest.hexdigest())
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            program,
            str(source),
            str(ambient_seed),
            str(configured_seed),
            slot_mode,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
    )
    return output.strip().splitlines()[-1]


def test_configured_seed_determines_tiny_native_adapter_initialization():
    source = _native_source()

    none = _subprocess_fingerprint(
        source, ambient_seed=3, configured_seed=42, slot_mode="none"
    )
    pad = _subprocess_fingerprint(
        source, ambient_seed=999, configured_seed=42, slot_mode="pad"
    )
    learned = _subprocess_fingerprint(
        source, ambient_seed=77, configured_seed=42, slot_mode="learned"
    )
    different = _subprocess_fingerprint(
        source, ambient_seed=3, configured_seed=43, slot_mode="none"
    )

    assert none == pad == learned
    assert none != different


def test_setup_seeds_before_model_loader(monkeypatch):
    calls: list[str] = []
    cfg = DictDefault({"seed": 42, "full_determinism": False, "use_ray": False})
    dataset_meta = SimpleNamespace(
        train_dataset=object(), eval_dataset=object(), total_num_steps=1
    )

    def seed(config):
        assert config is cfg
        calls.append("seed")

    def load(config):
        assert config is cfg
        assert calls == ["seed"]
        return object(), object(), None, None

    monkeypatch.setattr("axolotl.train.seed_model_initialization", seed)
    monkeypatch.setattr("axolotl.train.setup_model_and_tokenizer", load)
    monkeypatch.setattr("axolotl.train.setup_reference_model", lambda *_args: None)
    monkeypatch.setattr("axolotl.train.setup_trainer", lambda **_kwargs: object())

    setup_model_and_trainer(cfg, dataset_meta)

    assert calls == ["seed"]


def test_seed_model_initialization_resets_torch_rng():
    torch.manual_seed(1)
    seed_model_initialization(DictDefault({"seed": 42, "full_determinism": False}))
    first = torch.rand(4)
    torch.manual_seed(99)
    seed_model_initialization(DictDefault({"seed": 42, "full_determinism": False}))

    torch.testing.assert_close(torch.rand(4), first)
