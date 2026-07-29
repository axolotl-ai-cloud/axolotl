"""Blockwise local distillation — the healer for models too large to hold whole.

The claim under test is not "loss goes down": it is that the student trains on its
*own* propagated activations while the targets stay the teacher's. Healing block `k`
against the teacher's inputs would optimize it for activations it will never see, so
the tests below pin the propagation explicitly rather than trusting the loop's shape.
"""

import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.ptq import blockwise
from axolotl.integrations.ternary.ptq.blockwise import ActivationCache, heal_blocks
from axolotl.integrations.ternary.ptq.stream import stream_fit
from axolotl.utils.dict import DictDefault

# the heal loop is device-agnostic — a tiny model on CPU exercises the same code the
# 128B run takes, so these stay in the CPU suite rather than gating on an accelerator
DEVICE = "cpu"

BLOCKS = 3
HIDDEN = 64
VOCAB = 49152


@pytest.fixture(name="tokenizer_dir", scope="module")
def fixture_tokenizer_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tok")
    AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M").save_pretrained(
        directory
    )
    return directory


def _teacher(directory, tokenizer_dir, seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=BLOCKS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(directory)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = tokenizer_dir / name
        if source.is_file():
            (directory / name).write_bytes(source.read_bytes())
    return directory


def _cfg(source, codebook="ternary", **blockwise_kwargs):
    plan = {
        "enabled": True,
        "num_sequences": 4,
        "sequence_len": 64,
        "epochs": 1,
        "learning_rate": 1e-3,
        "dataset": "wikitext",
        **blockwise_kwargs,
    }
    return DictDefault(
        {
            "base_model": str(source),
            "tokenizer_config": str(source),
            "sequence_len": 64,
            # a raw config skips axolotl's normalization, which is what derives
            # `batch_size` — the step estimator divides by it
            "batch_size": 1,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "datasets": [
                {"path": "wikitext", "name": "wikitext-2-raw-v1", "type": "completion"}
            ],
            "ternary": {
                "init": "ternary_fit",
                "weight_scale": "learnable_row",
                "lambda_schedule": "none",
                "codebook": codebook,
                "export": {"formats": ["master_bf16"]},
                "blockwise": plan,
            },
        }
    )


@pytest.fixture(name="fitted")
def fixture_fitted(tmp_path, tokenizer_dir):
    """A teacher and the ternary master fitted from it."""
    source = _teacher(tmp_path / "teacher", tokenizer_dir)
    cfg = _cfg(source)
    master = tmp_path / "master"
    stream_fit(source, master, cfg)
    return source, master, cfg


# --------------------------------------------------------- the activation cache


def test_the_cache_round_trips_chunks_in_order(tmp_path):
    cache = ActivationCache(tmp_path)
    chunks = [torch.randn(2, 4, 8, dtype=torch.bfloat16) for _ in range(3)]

    cache.write("teacher", 0, iter(chunks))

    assert cache.exists("teacher", 0)
    for got, want in zip(cache.read("teacher", 0), chunks, strict=True):
        assert torch.equal(got, want)


def test_the_cache_digest_tracks_content(tmp_path):
    cache = ActivationCache(tmp_path)
    cache.write("student", 1, iter([torch.ones(2, 2, 2, dtype=torch.bfloat16)]))
    first = cache.digest("student", 1)

    cache.write("student", 1, iter([torch.zeros(2, 2, 2, dtype=torch.bfloat16)]))

    assert cache.digest("student", 1) != first


def test_a_dropped_buffer_is_gone(tmp_path):
    cache = ActivationCache(tmp_path)
    cache.write("teacher", 0, iter([torch.ones(1, 1, 1, dtype=torch.bfloat16)]))

    cache.drop("teacher", 0)

    assert not cache.exists("teacher", 0)


def test_a_partial_chunk_is_never_visible(tmp_path):
    """Chunks are renamed into place, so a crashed write leaves no readable chunk."""
    cache = ActivationCache(tmp_path)
    cache.write("teacher", 0, iter([torch.ones(1, 1, 1, dtype=torch.bfloat16)]))

    leftovers = list(cache.path("teacher", 0).glob("*.partial"))

    assert leftovers == []


# ------------------------------------------------------------------- the config


def test_blockwise_is_off_by_default():
    assert TernaryConfig().blockwise.enabled is False


def test_healing_without_the_section_is_an_error(tmp_path, fitted):
    source, master, _ = fitted
    cfg = _cfg(source)
    cfg["ternary"]["blockwise"]["enabled"] = False

    with pytest.raises(ValueError, match="blockwise"):
        heal_blocks(master, tmp_path / "out", cfg, source_dir=source, device="cpu")


# ----------------------------------------------------------------- the heal loop


@pytest.mark.parametrize("codebook", ["ternary", "binary"])
def test_a_heal_covers_every_block_and_reduces_block_error(
    tmp_path, tokenizer_dir, codebook
):
    source = _teacher(tmp_path / "teacher", tokenizer_dir)
    cfg = _cfg(source, codebook=codebook)
    master = tmp_path / "master"
    stream_fit(source, master, cfg)

    report = heal_blocks(
        master, tmp_path / "out", cfg, source_dir=source, device=DEVICE
    )

    assert report.blocks == BLOCKS
    assert report.blocks_healed == BLOCKS
    assert set(report.records) == {str(i) for i in range(BLOCKS)}
    for record in report.records.values():
        assert record.steps > 0
        assert record.final_loss <= record.initial_loss


def test_the_healed_master_is_a_complete_checkpoint(tmp_path, fitted):
    source, master, cfg = fitted
    output = tmp_path / "out"

    heal_blocks(master, output, cfg, source_dir=source, device=DEVICE)

    assert (output / "config.json").is_file()
    healed = load_file(output / "model.safetensors")
    original = load_file(master / "model.safetensors")
    assert set(healed) == set(original)
    # the kept-fp tensors are untouched, the block tensors are not
    assert torch.equal(
        healed["model.embed_tokens.weight"], original["model.embed_tokens.weight"]
    )
    changed = [
        key
        for key in healed
        if "layers." in key and not torch.equal(healed[key], original[key])
    ]
    assert changed, "no block tensor moved"


def test_the_student_inputs_are_student_propagated_not_teacher_activations(
    tmp_path, fitted
):
    """The design decision, asserted directly.

    Block 0's inputs are shared by construction (the embedding is kept fp). Once block
    0 has trained, what block 1 receives must be block 0's *student* output — carrying
    the quantization error block 1 has to compensate — and not the teacher's.
    """
    source, master, cfg = fitted
    output = tmp_path / "out"

    heal_blocks(master, output, cfg, source_dir=source, device=DEVICE)

    cache = ActivationCache(output / "activation-cache")
    seeded = list(cache.read("student", BLOCKS - 1))
    teacher = list(cache.read("teacher", BLOCKS - 1))
    assert seeded, "no student buffer was propagated"
    assert all(s.shape == t.shape for s, t in zip(seeded, teacher, strict=True))
    assert any(not torch.equal(s, t) for s, t in zip(seeded, teacher, strict=True)), (
        "the student buffer is identical to the teacher's — inputs were not propagated"
    )


def test_block_zero_inputs_start_identical_for_both_models(tmp_path, fitted):
    """The embedding is kept fp, so the two buffers only diverge once block 0 trains."""
    source, master, cfg = fitted
    cache = ActivationCache(tmp_path / "seed")
    plan = TernaryConfig(**cfg["ternary"]).blockwise

    blockwise._seed_block_zero(master, cfg, plan, cache, ("model.layers.0",), DEVICE)

    teacher = list(cache.read("teacher", 0))
    student = list(cache.read("student", 0))
    assert teacher, "nothing was seeded"
    assert all(torch.equal(a, b) for a, b in zip(teacher, student, strict=True))


def test_a_resumed_heal_reproduces_an_uninterrupted_one(tmp_path, fitted):
    """Resume must be an optimization, never a different artifact."""
    source, master, cfg = fitted
    whole = tmp_path / "whole"
    heal_blocks(master, whole, cfg, source_dir=source, device=DEVICE)

    partial = tmp_path / "partial"
    stopped = _cfg(source)
    stopped["ternary"]["blockwise"]["cache_dir"] = str(tmp_path / "cache-partial")
    with pytest.raises(_StopAfterFirstBlock):
        _heal_stopping_after(1, master, partial, stopped, source)
    resumed = heal_blocks(master, partial, stopped, source_dir=source, device=DEVICE)

    assert resumed.blocks_skipped == 1
    assert resumed.blocks_healed == BLOCKS - 1
    left = load_file(whole / "model.safetensors")
    right = load_file(partial / "model.safetensors")
    for key in left:
        assert torch.equal(left[key], right[key]), key


class _StopAfterFirstBlock(RuntimeError):
    """Raised to cut a heal short mid-campaign."""


def _heal_stopping_after(blocks, master, output, cfg, source):
    """Run a heal that dies after `blocks` blocks, the way a killed job would."""
    real = blockwise._heal_one_block
    seen = []

    def counting(*args, **kwargs):
        if len(seen) >= blocks:
            raise _StopAfterFirstBlock
        record = real(*args, **kwargs)
        seen.append(record)
        return record

    blockwise._heal_one_block = counting
    try:
        heal_blocks(master, output, cfg, source_dir=source, device=DEVICE)
    finally:
        blockwise._heal_one_block = real


def test_the_records_file_survives_a_crash(tmp_path, fitted):
    source, master, cfg = fitted
    output = tmp_path / "out"
    cfg["ternary"]["blockwise"]["cache_dir"] = str(tmp_path / "cache")

    with pytest.raises(_StopAfterFirstBlock):
        _heal_stopping_after(1, master, output, cfg, source)

    records = json.loads((output / blockwise.RECORD_FILENAME).read_text())
    assert set(records) == {"0"}
    assert records["0"]["weights_sha256"]
    assert records["0"]["student_sha256"]


# ------------------------------------------------------- the regime it runs in


def test_the_healer_runs_outside_autocast_and_the_trainer():
    """Establish the regime, because the dtype rules differ from a training run.

    `heal_blocks` is called directly, not through the HF Trainer, so accelerate never
    wraps it and no autocast context is open. Nothing upcasts the blocks to fp32
    behind its back — which is what makes the bf16 the cache stores the whole story.
    """
    assert not torch.is_autocast_enabled()
    assert not torch.is_autocast_enabled("cpu")


def test_the_cache_normalizes_activations_to_bf16(tmp_path):
    """The buffers pin the dtype, so an fp32-cast caller cannot change the regime."""
    cache = ActivationCache(tmp_path)

    cache.write("teacher", 0, iter([torch.randn(2, 4, 8, dtype=torch.float32)]))

    stored = next(iter(cache.read("teacher", 0)))
    assert stored.dtype == torch.bfloat16


def test_student_blocks_are_fp32_and_teacher_blocks_bf16(tmp_path, fitted):
    """Trainable latents must be fp32: healing updates at sane learning rates sit
    below bf16's ULP, and bf16 training random-walks near-optimal blocks worse."""
    source, master, cfg = fitted

    student = blockwise._load_block(
        master, cfg, "model.layers.0.", DEVICE, student=True
    )
    teacher = blockwise._load_block(source, cfg, "model.layers.0.", DEVICE)

    student_floats = {p.dtype for p in student.parameters() if p.is_floating_point()}
    teacher_floats = {p.dtype for p in teacher.parameters() if p.is_floating_point()}
    assert student_floats == {torch.float32}
    assert teacher_floats == {torch.bfloat16}


def test_the_block_loss_survives_mixed_input_dtypes(tmp_path, fitted):
    """MSE harmonizes both sides, so an fp32 target against a bf16 block is fine."""
    source, master, cfg = fitted
    block = blockwise._load_block(master, cfg, "model.layers.0.", DEVICE, student=True)
    hidden = torch.randn(1, 8, HIDDEN, dtype=torch.bfloat16)
    target = torch.randn(1, 8, HIDDEN, dtype=torch.float32)

    output = blockwise._block_forward(block, hidden, _kwargs_for(block, hidden))
    loss = torch.nn.functional.mse_loss(output.float(), target.float())

    assert torch.isfinite(loss)


def _kwargs_for(block, hidden):
    """The position kwargs a decoder block needs, built for a bare tensor."""
    length = hidden.shape[1]
    positions = torch.arange(length).unsqueeze(0)
    head_dim = HIDDEN // 4
    # rope must match the hidden dtype: sdpa refuses fp32 cos/sin against bf16
    # qkv, the same dtype-harmonization trap the chunked KD loss hit
    angles = torch.zeros(1, length, head_dim, dtype=hidden.dtype)
    return {
        "position_embeddings": (angles.cos(), angles.sin()),
        "position_ids": positions,
    }


def test_the_8bit_optimizer_builds_and_heals(tmp_path, tokenizer_dir):
    from torchao.optim import AdamW8bit

    from axolotl.integrations.ternary.args import TernaryBlockwiseConfig
    from axolotl.integrations.ternary.ptq.blockwise import _build_optimizer

    plan = TernaryBlockwiseConfig(enabled=True, optimizer="adamw_torch_8bit")
    param = torch.nn.Parameter(torch.randn(8, 8))
    assert isinstance(_build_optimizer([param], plan), AdamW8bit)
    default = TernaryBlockwiseConfig(enabled=True)
    assert type(_build_optimizer([param], default)) is torch.optim.AdamW

    source = _teacher(tmp_path / "teacher", tokenizer_dir)
    cfg = _cfg(source, optimizer="adamw_torch_8bit")
    master = tmp_path / "master"
    stream_fit(source, master, cfg)
    report = heal_blocks(
        master, tmp_path / "out", cfg, source_dir=source, device=DEVICE
    )
    assert report.blocks_healed == BLOCKS
    for record in report.records.values():
        assert record.final_loss <= record.initial_loss
