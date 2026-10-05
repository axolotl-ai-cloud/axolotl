"""Model and adapter initialization must use the configured training seed."""

from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.train import seed_model_initialization, setup_model_and_trainer
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def _lora_fingerprint(ambient_seed: int, cfg: DictDefault) -> str:
    random.seed(ambient_seed)
    torch.manual_seed(ambient_seed)
    seed_model_initialization(cfg)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model = get_peft_model(
        model,
        LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"]),
    )
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if ".lora_A." in name or ".lora_B." in name:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_configured_seed_determines_lora_initialization():
    seeded = DictDefault({"seed": 42})

    first = _lora_fingerprint(3, seeded)
    other_ambient = _lora_fingerprint(999, seeded)
    other_seed = _lora_fingerprint(3, DictDefault({"seed": 43}))

    assert first == other_ambient
    assert first != other_seed


def test_without_seed_initialization_follows_ambient_rng():
    unseeded = DictDefault({})

    assert _lora_fingerprint(3, unseeded) == _lora_fingerprint(3, unseeded)
    assert _lora_fingerprint(3, unseeded) != _lora_fingerprint(4, unseeded)


def test_setup_seeds_before_model_loader(monkeypatch):
    calls: list[str] = []
    cfg = DictDefault({"seed": 42, "use_ray": False})
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


def test_setup_without_seed_does_not_crash(monkeypatch):
    cfg = DictDefault({"use_ray": False})
    dataset_meta = SimpleNamespace(
        train_dataset=object(), eval_dataset=object(), total_num_steps=1
    )
    monkeypatch.setattr(
        "axolotl.train.setup_model_and_tokenizer",
        lambda _cfg: (object(), object(), None, None),
    )
    monkeypatch.setattr("axolotl.train.setup_reference_model", lambda *_args: None)
    monkeypatch.setattr("axolotl.train.setup_trainer", lambda **_kwargs: object())

    setup_model_and_trainer(cfg, dataset_meta)


def test_seed_model_initialization_resets_torch_rng():
    torch.manual_seed(1)
    seed_model_initialization(DictDefault({"seed": 42}))
    first = torch.rand(4)
    torch.manual_seed(99)
    seed_model_initialization(DictDefault({"seed": 42}))

    torch.testing.assert_close(torch.rand(4), first)


@pytest.mark.parametrize("full_determinism", [False, True])
def test_full_determinism_selects_seeding_function(monkeypatch, full_determinism):
    calls = []
    monkeypatch.setattr(
        "axolotl.train.set_seed", lambda seed: calls.append(("set_seed", seed))
    )
    monkeypatch.setattr(
        "axolotl.train.enable_full_determinism",
        lambda seed: calls.append(("enable_full_determinism", seed)),
    )

    seed_model_initialization(
        DictDefault({"seed": 7, "full_determinism": full_determinism})
    )

    expected = "enable_full_determinism" if full_determinism else "set_seed"
    assert calls == [(expected, 7)]


def test_full_determinism_is_a_schema_field(min_base_cfg):
    validated = validate_config(
        min_base_cfg | DictDefault(seed=7, full_determinism=True)
    )

    assert validated.full_determinism is True
