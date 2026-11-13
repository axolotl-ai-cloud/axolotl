"""TorchTPU device handling, run on CPU with the ``mock_tpu`` fixture."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast

from axolotl.core.builders.causal import HFCausalTrainerBuilder
from axolotl.utils.config import choose_device, resolve_dtype, validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import (
    build_parallelism_config,
    get_current_device,
    get_device_count,
    get_device_type,
    is_tpu,
)

EXAMPLES = sorted((Path(__file__).parents[2] / "examples" / "tpu").glob("*.yml"))
SEQ_LEN = 32


class TestIsTpu:
    def test_false_without_torch_tpu(self, monkeypatch):
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **kw: (
                None if name == "torch_tpu" else real_find_spec(name, *a, **kw)
            ),
        )
        assert is_tpu() is False

    @pytest.mark.parametrize("accelerator", [None, torch.device("cpu")])
    def test_false_when_tpu_is_not_the_accelerator(self, mock_tpu, accelerator):
        with patch.object(
            torch.accelerator, "current_accelerator", return_value=accelerator
        ):
            is_tpu.cache_clear()
            assert is_tpu() is False

    def test_true_without_probing_the_runtime(self, mock_tpu):
        with patch.object(
            torch.accelerator,
            "is_available",
            side_effect=AssertionError("initializes the TPU runtime"),
        ):
            assert is_tpu() is True


class TestDeviceHelpers:
    def test_device_type(self, mock_tpu):
        assert get_device_type().type == "tpu"

    def test_device_count_and_index(self, mock_tpu):
        assert get_device_count() == 8
        assert get_current_device() == 0

    def test_no_device_mesh_for_one_process(self, mock_tpu, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "1")
        cfg = DictDefault(tensor_parallel_size=1, context_parallel_size=1)
        assert build_parallelism_config(cfg) == (None, None)


class TestChooseDevice:
    def test_single_chip(self, mock_tpu):
        cfg = DictDefault(world_size=1, device_map="auto")
        choose_device(cfg)
        assert cfg.device == "tpu:0"
        assert cfg.device_map is None

    def test_multi_process_rejected(self, mock_tpu):
        with pytest.raises(NotImplementedError, match="Multi-chip"):
            choose_device(DictDefault(world_size=8))


class TestResolveDtype:
    def test_bf16_auto_enables_bf16(self, mock_tpu):
        cfg = DictDefault(device="tpu:0", bf16="auto", tf32="auto")
        resolve_dtype(cfg)
        assert cfg.bf16 is True
        assert not cfg.fp16
        assert cfg.tf32 is False
        assert cfg.torch_dtype == torch.bfloat16

    def test_fp16_switches_to_bf16(self, mock_tpu):
        cfg = DictDefault(device="tpu:0", fp16=True, bf16="auto")
        resolve_dtype(cfg)
        assert cfg.bf16 is True
        assert cfg.fp16 is False
        assert cfg.torch_dtype == torch.bfloat16

    def test_mps_unchanged(self):
        cfg = DictDefault(device="mps", bf16=True, load_in_8bit=True)
        resolve_dtype(cfg)
        assert cfg.fp16 is True
        assert cfg.bf16 is False
        assert cfg.load_in_8bit is False


@pytest.fixture(name="tokenizer")
def fixture_tokenizer():
    vocab = {"<pad>": 0, "<unk>": 1, **{f"w{i}": i + 2 for i in range(50)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>"
    )


def _padding_kwargs(**cfg):
    builder: Any = SimpleNamespace(cfg=DictDefault(sequence_len=SEQ_LEN, **cfg))
    return HFCausalTrainerBuilder._collator_padding_kwargs(builder)


class TestCollatorPadding:
    @pytest.mark.parametrize("lengths", [[3, 7], [1], [SEQ_LEN, 5], [17, 2, 30]])
    def test_tpu_batches_are_sequence_len_wide(self, mock_tpu, tokenizer, lengths):
        collator = DataCollatorForSeq2Seq(tokenizer, **_padding_kwargs())
        batch = collator([{"input_ids": [2] * n, "labels": [2] * n} for n in lengths])
        for key in ("input_ids", "labels", "attention_mask"):
            assert batch[key].shape == (len(lengths), SEQ_LEN)

    @pytest.mark.parametrize(
        ("pad_to_sequence_len", "expected"),
        [
            (None, {"padding": True, "pad_to_multiple_of": 64}),
            (True, {"padding": True, "pad_to_multiple_of": 64}),
            (False, {"padding": True}),
        ],
    )
    def test_padding_unchanged_off_tpu(self, pad_to_sequence_len, expected):
        assert _padding_kwargs(pad_to_sequence_len=pad_to_sequence_len) == expected


def _post_lora_setup(monkeypatch):
    from axolotl.loaders.model import ModelLoader

    monkeypatch.setenv("WORLD_SIZE", "1")
    model = torch.nn.Linear(2, 2)
    model.is_parallelizable = False
    model.model_parallel = False
    loader: Any = SimpleNamespace(cfg=DictDefault(ddp=False), model=model)
    ModelLoader._apply_post_lora_load_setup(loader, skip_move_to_device=True)
    return model


def test_single_tpu_process_is_not_naive_model_parallel(mock_tpu, monkeypatch):
    model = _post_lora_setup(monkeypatch)
    assert model.is_parallelizable is False
    assert model.model_parallel is False


def test_multi_gpu_single_process_is_still_naive_model_parallel(monkeypatch):
    monkeypatch.setattr("axolotl.loaders.model.get_device_count", lambda: 2)
    model = _post_lora_setup(monkeypatch)
    assert model.is_parallelizable is True
    assert model.model_parallel is True


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_tpu_examples_validate(mock_tpu, example):
    cfg = validate_config(DictDefault(yaml.safe_load(example.read_text())))
    assert cfg.attn_implementation == "sdpa"
    assert cfg.pad_to_sequence_len is True
    assert not cfg.sample_packing
    for flag in ("lora_mlp_kernel", "lora_qkv_kernel", "lora_o_kernel"):
        assert not cfg.get(flag)


def test_examples_exist():
    assert {path.name for path in EXAMPLES} == {
        "qwen3-0.6b-fft.yml",
        "qwen3-0.6b-lora.yml",
    }
