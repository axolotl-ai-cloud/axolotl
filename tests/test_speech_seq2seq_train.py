"""CPU smoke test: LoRA-train a tiny random Cohere Transcribe model end to end."""

import io
import math

import numpy as np
import pytest

pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")

from datasets import Audio, Dataset  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from axolotl.common.datasets import load_datasets  # noqa: E402
from axolotl.train import train  # noqa: E402
from axolotl.utils.config import normalize_config, validate_config  # noqa: E402
from axolotl.utils.dict import DictDefault  # noqa: E402

from tests.hf_offline_utils import enable_hf_offline  # noqa: E402

MIRROR = "onnx-community/cohere-transcribe-03-2026-ONNX"
SR = 16000


@pytest.fixture(name="tiny_model_dir")
@enable_hf_offline
def fixture_tiny_model_dir(tmp_path, download_cohere_transcribe_processor_fixture):
    from transformers import (
        AutoProcessor,
        CohereAsrConfig,
        CohereAsrForConditionalGeneration,
    )

    model_dir = tmp_path / "tiny"
    config = CohereAsrConfig(
        encoder_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "subsampling_conv_channels": 8,
            # the default 0.02 init flattens the subsampled audio over time, zeroing cross-attn k grads
            "initializer_range": 0.3,
        },
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    CohereAsrForConditionalGeneration(config).save_pretrained(model_dir)
    AutoProcessor.from_pretrained(MIRROR).save_pretrained(model_dir)
    return model_dir


@pytest.fixture(name="audio_parquet")
def fixture_audio_parquet(tmp_path):
    rng = np.random.default_rng(0)
    clips = []
    for i in range(8):
        buf = io.BytesIO()
        tone = np.sin(2 * np.pi * (220 + 40 * i) * np.arange(SR) / SR)
        soundfile.write(
            buf, 0.3 * tone + 0.01 * rng.standard_normal(SR), SR, format="WAV"
        )
        clips.append({"bytes": buf.getvalue(), "path": None})
    texts = [f"Tone number {i}." for i in range(8)]
    path = tmp_path / "train.parquet"
    Dataset.from_dict({"audio": clips, "text": texts}).cast_column(
        "audio", Audio()
    ).to_parquet(path)
    return path


@enable_hf_offline
def test_cohere_asr_lora_train(tiny_model_dir, audio_parquet, tmp_path):
    output_dir = tmp_path / "out"
    cfg = DictDefault(
        {
            "base_model": str(tiny_model_dir),
            "skip_prepare_dataset": True,
            "remove_unused_columns": False,
            "datasets": [{"path": str(audio_parquet), "ds_type": "parquet"}],
            "asr": {"language": "en"},
            "sequence_len": 64,
            "adapter": "lora",
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_target_linear": True,
            "activation_checkpointing": True,
            "val_set_size": 0,
            "max_steps": 2,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-3,
            "optimizer": "adamw_torch",
            "lr_scheduler": "constant",
            "save_strategy": "no",
            "output_dir": str(output_dir),
            "seed": 42,
        }
    )
    cfg = validate_config(cfg)
    normalize_config(cfg)

    _, _, trainer = train(cfg=cfg, dataset_meta=load_datasets(cfg=cfg))

    assert math.isfinite(trainer.state.log_history[-1]["train_loss"])
    adapter = load_file(output_dir / "adapter_model.safetensors")
    # any, not all: on a random tiny model some cross-attn LoRA can see ~0 gradient in 2 steps
    assert any(v.any() for k, v in adapter.items() if "lora_B" in k)
    assert (output_dir / "processor_config.json").exists()
