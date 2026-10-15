"""Focused tests for the full-sequence diffusion backend."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from torch import nn
from transformers import DiffusionGemmaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from axolotl.core.trainers.diffusion_lm.backends.encoder_canvas import (
    EncoderCanvasBackend,
)
from axolotl.core.trainers.diffusion_lm.backends.full_sequence import (
    FullSequenceBackend,
    corrupt_packed_absorbing,
    create_bidirectional_attention_mask,
    create_packed_document_attention_mask,
    reset_position_ids,
    shift_logits_to_input_positions,
    shift_logits_within_documents,
)
from axolotl.core.trainers.diffusion_lm.batch import CorruptedCanvas, DiffusionBatch
from axolotl.core.trainers.diffusion_lm.tokens import resolve_mask_token_id
from axolotl.core.trainers.diffusion_lm.trainer import (
    AxolotlDiffusionTrainer,
    normalized_sft_loss,
)
from axolotl.core.trainers.diffusion_lm.unroll import run_unroll
from axolotl.core.training_args import AxolotlTrainingArguments
from axolotl.model_support.diffusion_gemma.modeling import (
    AxolotlDiffusionGemmaForBlockDiffusion,
)
from axolotl.utils.dict import DictDefault


class _DDPTinyNemotron(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)
        self.config = SimpleNamespace(
            model_type="nemotron_labs_diffusion", mask_token_id=15
        )

    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        del attention_mask, position_ids, use_cache
        return CausalLMOutputWithPast(logits=self.head(self.embedding(input_ids)))


def _ddp_gemma_config() -> DiffusionGemmaConfig:
    text = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "layer_types": ["sliding_attention", "full_attention"],
        "per_layer_config": {"0": {"head_dim": 8}, "1": {"head_dim": 8}},
        "sliding_window": 2,
        "use_bidirectional_attention": "vision",
        "num_experts": 2,
        "top_k_experts": 1,
        "moe_intermediate_size": 32,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
    }
    vision = {
        "model_type": "gemma4_vision",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "patch_size": 16,
        "position_embedding_size": 16,
    }
    return DiffusionGemmaConfig(
        text_config=text,
        vision_config=vision,
        canvas_length=8,
        boi_token_id=31,
        eoi_token_id=30,
        image_token_id=29,
    )


def _ddp_runner(model, *, world_size: int, gemma: bool):
    runner = object.__new__(AxolotlDiffusionTrainer)
    runner.model = model
    runner.args = SimpleNamespace(world_size=world_size)
    runner.processing_class = None
    runner._special_token_ids = set()
    cast(Any, runner).store_metrics = lambda *_args, **_kwargs: None
    diffusion: dict[str, object] = {"from_causal_lm": False, "t_eps": 1.0}
    if gemma:
        diffusion.update({"self_conditioning": {"p": 0.0}, "encoder_ar_weight": 1.0})
    runner.axolotl_cfg = DictDefault({"diffusion_lm": diffusion})
    return runner


def _ddp_full_sequence_inputs(rank: int) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[2, 3, 4]])
    support = (
        torch.tensor([[True, True, False]])
        if rank == 0
        else torch.zeros_like(input_ids, dtype=torch.bool)
    )
    return {
        "input_ids": input_ids,
        "document_ids": torch.zeros_like(input_ids),
        "semantic_validity": torch.ones_like(input_ids, dtype=torch.bool),
        "canvas_loss_mask": support,
        "canvas_corruptible_mask": support,
    }


def _ddp_gemma_batch(rank: int) -> DiffusionBatch:
    canvas_support = (
        torch.tensor([[True, True]])
        if rank == 0
        else torch.zeros((1, 2), dtype=torch.bool)
    )
    return DiffusionBatch(
        encoder_input_ids=torch.tensor([[2, 3, 4]]),
        encoder_validity=torch.ones((1, 3), dtype=torch.bool),
        encoder_ar_valid_mask=(
            torch.tensor([[False, True, True]])
            if rank == 0
            else torch.zeros((1, 3), dtype=torch.bool)
        ),
        encoder_document_ids=torch.zeros((1, 3), dtype=torch.long),
        encoder_position_ids=torch.tensor([[0, 1, 2]]),
        canvas_clean_ids=torch.tensor([[5, 6]]),
        canvas_semantic_validity=torch.ones((1, 2), dtype=torch.bool),
        canvas_loss_mask=canvas_support,
        canvas_corruptible_mask=torch.zeros((1, 2), dtype=torch.bool),
        canvas_input_pinned_mask=torch.zeros((1, 2), dtype=torch.bool),
        canvas_sc_eligible_mask=torch.ones((1, 2), dtype=torch.bool),
        canvas_read_only_mask=torch.zeros((1, 2), dtype=torch.bool),
        canvas_update_mask=torch.zeros((1, 2), dtype=torch.bool),
        logical_ids=torch.tensor([0]),
        encoder_lengths=torch.tensor([3]),
        canvas_lengths=torch.tensor([2]),
        decoder_prefix_lengths=torch.tensor([3]),
        selected_block_ids=torch.tensor([0]),
    )


def _ddp_global_normalization_worker(rank, init_path, kind):
    world_size = 2
    dist.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(123)
        if kind == "full_sequence":
            reference = _DDPTinyNemotron()
            initial = copy.deepcopy(reference.state_dict())
            if rank == 0:
                torch.manual_seed(246)
                reference_runner = _ddp_runner(reference, world_size=1, gemma=False)
                reference_loss, _ = reference_runner._compute_native_full_sequence_loss(
                    reference,
                    _ddp_full_sequence_inputs(0),
                    reference_runner._native_spec,
                    num_items_in_batch=torch.tensor(2),
                )
                reference_loss.backward()
                torch.optim.SGD(reference.parameters(), lr=1e-2).step()
            model = _DDPTinyNemotron()
            model.load_state_dict(initial)
            batch = _ddp_full_sequence_inputs(rank)
            denominator = torch.tensor(2)
        else:
            rhine = kind == "gemma_rhine"
            reference = AxolotlDiffusionGemmaForBlockDiffusion(_ddp_gemma_config())
            reference.eval()
            initial = copy.deepcopy(reference.state_dict())
            if rank == 0:
                torch.manual_seed(246)
                reference_runner = _ddp_runner(reference, world_size=1, gemma=True)
                if rhine:
                    reference_runner.axolotl_cfg = DictDefault(
                        {
                            "diffusion_lm": {
                                "from_causal_lm": False,
                                "self_conditioning": {"p": 0.0},
                                "encoder_ar_weight": 0.0,
                                "time_weighting": "loo",
                                "objective_reduction": "example_mean",
                            }
                        }
                    )
                reference_loss, _ = (
                    reference_runner._compute_native_encoder_canvas_loss(
                        reference,
                        {"diffusion_batch": _ddp_gemma_batch(0)},
                        reference_runner._native_spec,
                        num_items_in_batch={
                            "canvas": torch.tensor(2),
                            "encoder_ar": torch.tensor(0 if rhine else 1),
                            "examples": torch.tensor(1),
                        },
                    )
                )
                reference_loss.backward()
                torch.optim.SGD(reference.parameters(), lr=1e-2).step()
            model = AxolotlDiffusionGemmaForBlockDiffusion(_ddp_gemma_config())
            model.load_state_dict(initial)
            model.eval()
            batch = {"diffusion_batch": _ddp_gemma_batch(rank)}
            denominator = {
                "canvas": torch.tensor(2),
                "encoder_ar": torch.tensor(0 if rhine else 1),
                "examples": torch.tensor(1),
            }
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )
        runner = _ddp_runner(ddp, world_size=world_size, gemma=kind != "full_sequence")
        if kind == "gemma_rhine":
            runner.axolotl_cfg = DictDefault(
                {
                    "diffusion_lm": {
                        "from_causal_lm": False,
                        "self_conditioning": {"p": 0.0},
                        "encoder_ar_weight": 0.0,
                        "time_weighting": "loo",
                        "objective_reduction": "example_mean",
                    }
                }
            )
        torch.manual_seed(246)
        if kind == "full_sequence":
            loss, _ = runner._compute_native_full_sequence_loss(
                ddp, batch, runner._native_spec, num_items_in_batch=denominator
            )
        else:
            loss, _ = runner._compute_native_encoder_canvas_loss(
                ddp, batch, runner._native_spec, num_items_in_batch=denominator
            )
        loss.backward()
        torch.optim.SGD(ddp.parameters(), lr=1e-2).step()
        dist.barrier()
        if rank == 0:
            for (name, actual), expected in zip(
                ddp.module.named_parameters(), reference.parameters(), strict=True
            ):
                if not torch.allclose(actual, expected, rtol=2e-5, atol=2e-6):
                    initial_value = initial[name]
                    raise AssertionError(
                        f"{name}: actual update "
                        f"{(actual - initial_value).abs().max().item():.6g}, "
                        f"reference update "
                        f"{(expected - initial_value).abs().max().item():.6g}"
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("kind", ["full_sequence", "gemma", "gemma_rhine"])
def test_ddp_global_normalization_matches_single_rank_reference(tmp_path, kind):
    init_path = tmp_path / f"{kind}-gloo"
    mp.spawn(
        _ddp_global_normalization_worker,
        args=(str(init_path), kind),
        nprocs=2,
        join=True,
    )


def test_encoder_canvas_rejects_unimplemented_objective_overrides():
    model = SimpleNamespace(config=SimpleNamespace(model_type="diffusion_gemma"))
    runner = object.__new__(AxolotlDiffusionTrainer)
    runner.model = model
    runner.axolotl_cfg = DictDefault(
        {"diffusion_lm": {"from_causal_lm": False, "time_weighting": "inv_t"}}
    )

    with pytest.raises(ValueError, match="time_weighting: none, loo"):
        runner._compute_native_diffusion_loss(model, {})

    runner.axolotl_cfg = DictDefault(
        {"diffusion_lm": {"from_causal_lm": False, "token_reweighting": True}}
    )
    with pytest.raises(ValueError, match="token_reweighting"):
        runner._compute_native_diffusion_loss(model, {})


def test_gemma_rhine_loo_trainer_route_matches_pinned_fixture(monkeypatch):
    fixture_path = Path(__file__).with_name("fixtures") / "rhine_loo_ce_26e764.json"
    fixture = json.loads(fixture_path.read_text())

    class FixtureGemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(
                torch.tensor(fixture["logits"], dtype=torch.float).reshape(1, 4, 3)
            )
            self.config = SimpleNamespace(
                model_type="diffusion_gemma",
                text_config=SimpleNamespace(vocab_size=3, sliding_window=2),
            )

        def forward(self, **_kwargs):
            return SimpleNamespace(logits=self.logits, encoder_last_hidden_state=None)

    batch = DiffusionBatch(
        encoder_input_ids=torch.tensor([[0], [0]]),
        encoder_validity=torch.ones((2, 1), dtype=torch.bool),
        encoder_ar_valid_mask=torch.zeros((2, 1), dtype=torch.bool),
        encoder_document_ids=torch.tensor([[0], [1]]),
        encoder_position_ids=torch.zeros((2, 1), dtype=torch.long),
        canvas_clean_ids=torch.tensor(fixture["targets"]),
        canvas_semantic_validity=torch.ones((2, 2), dtype=torch.bool),
        canvas_loss_mask=torch.tensor(fixture["loss_mask"], dtype=torch.bool),
        canvas_corruptible_mask=torch.ones((2, 2), dtype=torch.bool),
        canvas_input_pinned_mask=torch.zeros((2, 2), dtype=torch.bool),
        canvas_sc_eligible_mask=torch.ones((2, 2), dtype=torch.bool),
        canvas_read_only_mask=torch.zeros((2, 2), dtype=torch.bool),
        canvas_update_mask=torch.zeros((2, 2), dtype=torch.bool),
        logical_ids=torch.tensor([0, 1]),
        encoder_lengths=torch.tensor([1, 1]),
        canvas_lengths=torch.tensor([2, 2]),
        decoder_prefix_lengths=torch.tensor([1, 1]),
        selected_block_ids=torch.tensor([0, 0]),
    )
    noisy_ids = torch.tensor(fixture["noisy_ids"]).reshape(1, 4)

    def fixed_corrupt(_backend, packed, times, generator=None):
        del generator
        return CorruptedCanvas(
            noisy_ids,
            torch.ones_like(noisy_ids, dtype=torch.bool),
            times[packed.canvas_logical_row_indices],
        )

    monkeypatch.setattr(EncoderCanvasBackend, "corrupt", fixed_corrupt)
    model = FixtureGemma()
    runner = _ddp_runner(model, world_size=1, gemma=True)
    runner.axolotl_cfg = DictDefault(
        {
            "diffusion_lm": {
                "from_causal_lm": False,
                "self_conditioning": {"p": 0.0},
                "encoder_ar_weight": 0.0,
                "time_weighting": "loo",
                "objective_reduction": "example_mean",
            }
        }
    )
    times = torch.tensor(fixture["times"])
    runner._sample_native_times = lambda *_args: times
    loss, _ = runner._compute_native_encoder_canvas_loss(
        model, {"diffusion_batch": batch}, runner._native_spec
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(fixture["expected_loss"]))
    torch.testing.assert_close(
        model.logits.grad.reshape(2, 2, 3),
        torch.tensor(fixture["expected_logit_gradients"]),
    )

    weighted_model = FixtureGemma()
    weighted_runner = _ddp_runner(weighted_model, world_size=1, gemma=True)
    weighted_runner.axolotl_cfg = DictDefault(
        {
            "diffusion_lm": {
                "from_causal_lm": False,
                "self_conditioning": {"p": 0.0},
                "encoder_ar_weight": 0.0,
                "time_weighting": "inv_one_minus_t",
                "objective_reduction": "example_mean",
                "rhine_weight_clip": 2.0,
            }
        }
    )
    weighted_runner._sample_native_times = lambda *_args: times
    weighted_loss, _ = weighted_runner._compute_native_encoder_canvas_loss(
        weighted_model, {"diffusion_batch": batch}, weighted_runner._native_spec
    )
    weighted_loss.backward()
    manual_logits = (
        torch.tensor(fixture["logits"]).reshape(1, 4, 3).detach().requires_grad_()
    )
    manual_targets = torch.tensor(fixture["targets"]).reshape(1, 4)
    manual_noisy = torch.tensor(fixture["noisy_ids"]).reshape(1, 4)
    expanded_times = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
    correction = torch.log1p(
        3 * (1 - expanded_times).clamp_min(0) / expanded_times.clamp_min(1e-12)
    )
    adjusted = manual_logits.clone()
    adjusted.scatter_add_(-1, manual_noisy[..., None], correction[..., None])
    manual_nll = torch.nn.functional.cross_entropy(
        adjusted.flatten(0, -2), manual_targets.flatten(), reduction="none"
    ).reshape_as(manual_targets)
    manual_weights = (1 - expanded_times).clamp_min(1e-12).reciprocal().clamp_max(2)
    manual_token_loss = manual_nll * manual_weights
    manual_loss = torch.stack(
        (manual_token_loss[0, :2].mean(), manual_token_loss[0, 2:3].mean())
    ).mean()
    manual_loss.backward()
    torch.testing.assert_close(weighted_loss, manual_loss)
    torch.testing.assert_close(weighted_model.logits.grad, manual_logits.grad)


def test_dream_focal_configuration_changes_the_native_token_objective():
    class TinyDream(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)
            self.config = SimpleNamespace(model_type="Dream", mask_token_id=15)

        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            del attention_mask, position_ids, use_cache
            return CausalLMOutputWithPast(logits=self.head(self.embedding(input_ids)))

    torch.manual_seed(29)
    model = TinyDream()
    reference = TinyDream()
    reference.load_state_dict(model.state_dict())
    inputs = {
        "input_ids": torch.tensor([[2, 3, 4]]),
        "document_ids": torch.zeros((1, 3), dtype=torch.long),
        "semantic_validity": torch.ones((1, 3), dtype=torch.bool),
        "canvas_loss_mask": torch.ones((1, 3), dtype=torch.bool),
        "canvas_corruptible_mask": torch.ones((1, 3), dtype=torch.bool),
    }
    runner = _ddp_runner(model, world_size=1, gemma=False)
    runner.axolotl_cfg = DictDefault(
        {
            "diffusion_lm": {
                "from_causal_lm": False,
                "t_eps": 1.0,
                "token_reweighting": True,
                "alpha": 0.3,
                "gamma": 1.7,
            }
        }
    )
    loss, _ = runner._compute_native_full_sequence_loss(
        model, inputs, runner._native_spec
    )

    backend = FullSequenceBackend(mask_token_id=15)
    packed = backend.pack(
        inputs["input_ids"], inputs["document_ids"], inputs["semantic_validity"]
    )
    noisy, events, _ = backend.corrupt_native(
        packed,
        inputs["canvas_corruptible_mask"],
        torch.ones(1),
        document_time_indices=torch.zeros((1, 3), dtype=torch.long),
    )
    logits = backend.canvas_logits(
        backend.forward(reference, packed, noisy), packed, aligned=False
    )
    nll = torch.nn.functional.cross_entropy(
        logits.flatten(0, -2), inputs["input_ids"].flatten(), reduction="none"
    ).view_as(inputs["input_ids"])
    expected = (0.3 * (1 - torch.exp(-nll)).pow(1.7) * nll)[events].mean()
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for actual, manual in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, manual.grad)


def test_corruption_preserves_legacy_random_draw_order_and_exclusions():
    input_ids = torch.tensor([[1, 3, 4, 2], [1, 5, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = torch.tensor([[-100, 3, 4, -100], [-100, 5, 6, -100]])
    backend = FullSequenceBackend(mask_token_id=99, special_token_ids={0, 1, 2})

    torch.manual_seed(7)
    noisy, masked, p_mask = backend.corrupt(input_ids, attention_mask, labels, 0.1)
    torch.manual_seed(7)
    expected_p = ((1 - 0.1) * torch.rand(2) + 0.1)[:, None].repeat(1, 4)
    expected_p = expected_p * attention_mask.bool().float()
    expected_mask = torch.rand((2, 4)) < expected_p
    special = (input_ids == 0) | (input_ids == 1) | (input_ids == 2)
    expected_mask &= ~special & attention_mask.bool() & (labels != -100)

    assert torch.equal(p_mask, expected_p)
    assert torch.equal(masked, expected_mask)
    assert torch.equal(
        noisy, torch.where(masked, torch.full_like(input_ids, 99), input_ids)
    )


def test_full_sequence_mask_preserves_packed_boundaries_and_shift_is_legacy():
    input_ids = torch.tensor([[1, 3, 2, 1, 4, 2]])
    packed = torch.tensor([[1, 1, 1, 2, 2, 2]])
    mask = create_bidirectional_attention_mask(input_ids, packed, sample_packing=True)
    assert mask[0, 0, 0, 2]
    assert not mask[0, 0, 0, 3]
    logits = torch.tensor([[[1.0], [2.0], [3.0]]])
    assert torch.equal(
        shift_logits_to_input_positions(logits), torch.tensor([[[1.0], [1.0], [2.0]]])
    )


def test_mask_token_resolution_preserves_top_level_legacy_fallback():
    class Tokenizer:
        vocab_size = 10
        unk_token_id = 0
        all_special_tokens = ["<legacy>"]
        additional_special_tokens = []

        def convert_tokens_to_ids(self, token):
            return {"<legacy>": 8, "<|diffusion_mask|>": 0}[token]

    explicit = DictDefault({"diffusion_mask_token_id": 7})
    assert resolve_mask_token_id(Tokenizer(), explicit, allow_add=False) == 7

    configured = DictDefault({"diffusion_mask_token_str": "<legacy>"})
    assert resolve_mask_token_id(Tokenizer(), configured, allow_add=False) == 8
    assert configured.diffusion_mask_token_id == 8

    canonical = DictDefault(
        {
            "diffusion_lm": {"mask_token_str": "<legacy>"},
            "diffusion": {"mask_token_id": 1},
        }
    )
    assert resolve_mask_token_id(Tokenizer(), canonical, allow_add=False) == 8
    assert canonical.diffusion_lm.mask_token_id == 8
    assert canonical.diffusion.mask_token_id == 8


def test_sft_scatter_add_matches_legacy_loop_and_gradients():
    labels = torch.tensor([[-100, 4, 5], [-100, -100, 7], [-100, -100, -100]])
    batch_indices = torch.tensor([0, 0, 1])
    vectorized_values = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
    vectorized = normalized_sft_loss(vectorized_values, batch_indices, labels)
    vectorized.backward()

    loop_values = vectorized_values.detach().clone().requires_grad_()
    expected_per_sample = torch.zeros(3)
    answer_lengths = (labels != -100).sum(dim=1).float()
    for index in range(3):
        selected = batch_indices == index
        if selected.any():
            expected_per_sample[index] = loop_values[selected].sum() / answer_lengths[
                index
            ].clamp(min=1.0)
    expected = expected_per_sample.mean()
    expected.backward()

    assert torch.equal(vectorized, expected)
    assert torch.equal(vectorized_values.grad, loop_values.grad)


def test_native_packed_full_sequence_resets_attention_shift_and_eos_groups():
    ids = torch.tensor([[7, 8, 1, 1, 4, 5, 1, 1, 0]])
    docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]])
    valid = docs >= 0
    mask = create_packed_document_attention_mask(docs, valid)
    assert mask[0, 0, 0, 3] and not mask[0, 0, 0, 4]
    assert not mask[0, 0, 0, 8]
    logits = torch.arange(9, dtype=torch.float)[None, :, None].requires_grad_()
    shifted = shift_logits_within_documents(logits, docs, valid)
    assert torch.equal(
        shifted.squeeze(-1), torch.tensor([[0, 0, 1, 2, 4, 4, 5, 6, 8.0]])
    )
    noisy, events, probs = corrupt_packed_absorbing(
        ids,
        document_ids=docs,
        semantic_validity=valid,
        corruptible_mask=valid,
        logical_times=torch.tensor([1.0, 1.0]),
        mask_token_id=99,
        eos_token_id=1,
        treat_eos_as_one=True,
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal(
        probs, torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    )
    assert events[0, 2] == events[0, 3]
    assert events[0, 6] == events[0, 7]
    assert not events[0, 8] and noisy[0, 8] == 0
    assert torch.equal(
        reset_position_ids(docs, valid), torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0]])
    )


def _legacy_eos_tail_reference(
    input_ids,
    document_ids,
    semantic_validity,
    corruptible_mask,
    logical_times,
    document_time_indices,
    seed,
):
    probabilities = torch.zeros_like(input_ids, dtype=torch.float)
    valid_docs = (document_ids >= 0) & (document_time_indices >= 0)
    probabilities[valid_docs] = logical_times[document_time_indices[valid_docs]]
    events = torch.rand(input_ids.shape, generator=torch.Generator().manual_seed(seed))
    events = events < probabilities
    events &= semantic_validity & corruptible_mask
    for row in range(input_ids.shape[0]):
        for document in document_ids[row][semantic_validity[row]].unique().tolist():
            positions = torch.where(
                (document_ids[row] == document) & semantic_validity[row]
            )[0]
            tail_start = positions[-1] + 1
            while (
                tail_start > positions[0]
                and input_ids[row, tail_start - 1] == 1
                and corruptible_mask[row, tail_start - 1]
            ):
                tail_start -= 1
            if tail_start <= positions[-1]:
                events[row, tail_start : positions[-1] + 1] = events[
                    row, tail_start
                ].clone()
    events &= semantic_validity & corruptible_mask
    return (
        torch.where(events, torch.full_like(input_ids, 99), input_ids),
        events,
        probabilities,
    )


@pytest.mark.parametrize(
    (
        "input_ids",
        "document_ids",
        "semantic_validity",
        "corruptible_mask",
        "time_indices",
    ),
    [
        (
            [[7, 1, 1, 4, 5, 1, 1, 0]],
            [[0, 0, 0, 1, 1, 1, 1, -1]],
            [[True, True, True, True, True, True, True, False]],
            [[True, True, True, True, True, True, True, False]],
            [[0, 0, 0, 1, 1, 1, 1, -1]],
        ),
        (
            [[1, 1, 0, 9, 1, 1, 0], [5, 1, 1, 0, 1, 1, 0]],
            [[0, 0, -1, 1, 1, 1, -1], [2, 2, 2, -1, 3, 3, -1]],
            [
                [True, True, False, True, True, True, False],
                [True, True, True, False, True, True, False],
            ],
            [
                [True, True, False, True, True, True, False],
                [True, True, False, False, True, True, False],
            ],
            [[0, 0, -1, 1, 1, 1, -1], [2, 2, 2, -1, 3, 3, -1]],
        ),
        (
            [[1, 1, 1, 1], [1, 1, 0, 0]],
            [[0, 0, 0, 0], [1, 1, -1, -1]],
            [[True, True, True, True], [True, True, False, False]],
            [[True, True, True, True], [True, False, False, False]],
            [[0, 0, 0, 0], [1, 1, -1, -1]],
        ),
        (
            [[1, 1, 1, 1]],
            [[0, 0, 1, 1]],
            [[True, True, True, True]],
            [[True, True, True, True]],
            [[0, 0, 1, 1]],
        ),
        (
            [[7, 8, 9, 0]],
            [[0, 0, 0, -1]],
            [[True, True, True, False]],
            [[True, True, True, False]],
            [[0, 0, 0, -1]],
        ),
    ],
)
def test_native_eos_tail_vectorized_matches_legacy_reference(
    input_ids,
    document_ids,
    semantic_validity,
    corruptible_mask,
    time_indices,
):
    input_ids = torch.tensor(input_ids)
    document_ids = torch.tensor(document_ids)
    semantic_validity = torch.tensor(semantic_validity)
    corruptible_mask = torch.tensor(corruptible_mask)
    time_indices = torch.tensor(time_indices)
    times = torch.tensor([0.2, 0.8, 0.4, 0.6])
    expected = _legacy_eos_tail_reference(
        input_ids,
        document_ids,
        semantic_validity,
        corruptible_mask,
        times,
        time_indices,
        17,
    )
    actual = corrupt_packed_absorbing(
        input_ids,
        document_ids=document_ids,
        semantic_validity=semantic_validity,
        corruptible_mask=corruptible_mask,
        logical_times=times,
        document_time_indices=time_indices,
        mask_token_id=99,
        eos_token_id=1,
        treat_eos_as_one=True,
        generator=torch.Generator().manual_seed(17),
    )
    assert all(
        torch.equal(got, want) for got, want in zip(actual, expected, strict=True)
    )


def test_native_eos_tail_keeps_adjacent_document_events_distinct():
    _, events, _ = corrupt_packed_absorbing(
        torch.tensor([[1, 1, 1, 1]]),
        document_ids=torch.tensor([[0, 0, 1, 1]]),
        semantic_validity=torch.tensor([[True, True, True, True]]),
        corruptible_mask=torch.tensor([[True, True, True, True]]),
        logical_times=torch.tensor([0.5, 0.5]),
        mask_token_id=99,
        eos_token_id=1,
        treat_eos_as_one=True,
        generator=torch.Generator().manual_seed(1),
    )
    assert events.tolist() == [[False, False, True, True]]


@pytest.mark.parametrize(
    ("document_ids", "semantic_validity"),
    [
        ([[0, 0, -1, 0, 0]], [[True, True, False, True, True]]),
        ([[0, 0, 0]], [[True, False, True]]),
    ],
)
def test_native_eos_tail_rejects_noncontiguous_document_runs(
    document_ids, semantic_validity
):
    document_ids = torch.tensor(document_ids)
    semantic_validity = torch.tensor(semantic_validity)
    with pytest.raises(RuntimeError, match="contiguous valid run"):
        corrupt_packed_absorbing(
            torch.ones_like(document_ids),
            document_ids=document_ids,
            semantic_validity=semantic_validity,
            corruptible_mask=semantic_validity,
            logical_times=torch.tensor([1.0]),
            mask_token_id=99,
            eos_token_id=1,
            treat_eos_as_one=True,
        )


def test_native_eos_tail_never_extracts_tensor_scalars(monkeypatch):
    def fail_scalar_extraction(*args, **kwargs):
        pytest.fail("EOS-tail grouping extracted a tensor scalar")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", fail_scalar_extraction)
        context.setattr(torch.Tensor, "__int__", fail_scalar_extraction)
        context.setattr(torch.Tensor, "__bool__", fail_scalar_extraction)
        _, events, _ = corrupt_packed_absorbing(
            torch.tensor([[1, 1, 1, 1]]),
            document_ids=torch.tensor([[0, 0, 1, 1]]),
            semantic_validity=torch.tensor([[True, True, True, True]]),
            corruptible_mask=torch.tensor([[True, True, True, True]]),
            logical_times=torch.tensor([0.5, 0.5]),
            mask_token_id=99,
            eos_token_id=1,
            treat_eos_as_one=True,
            generator=torch.Generator().manual_seed(1),
        )

    assert events.tolist() == [[False, False, True, True]]


def test_native_full_sequence_flex_uses_fixed_physical_bucket():
    backend = FullSequenceBackend(mask_token_id=99, attention_backend="flex_attention")
    short = backend.pack(
        torch.tensor([[2, 3, 4]]),
        torch.tensor([[17, 17, 17]]),
        torch.tensor([[True, True, True]]),
    )
    longer = backend.pack(
        torch.tensor([[2, 3, 4, 5, 6]]),
        torch.tensor([[17, 17, 17, 17, 17]]),
        torch.tensor([[True, True, True, True, True]]),
    )
    assert short["input_ids"].shape == longer["input_ids"].shape == (1, 128)
    assert short["document_ids"][0, 3:].eq(-1).all()
    assert not short["semantic_validity"][0, 3:].any()


def test_native_eos_tail_never_corrupts_or_links_through_a_pinned_token():
    ids = torch.tensor([[7, 1, 1]])
    docs = torch.zeros_like(ids)
    valid = torch.ones_like(ids, dtype=torch.bool)
    _, events, _ = corrupt_packed_absorbing(
        ids,
        document_ids=docs,
        semantic_validity=valid,
        corruptible_mask=torch.tensor([[True, True, False]]),
        logical_times=torch.tensor([1.0]),
        mask_token_id=99,
        eos_token_id=1,
        treat_eos_as_one=True,
    )
    assert events.tolist() == [[True, True, False]]


def test_native_full_sequence_maps_opaque_document_ids_to_logical_rows():
    documents = torch.tensor([[17, 17, 91, 91, -1]])
    validity = documents >= 0
    rows, count = AxolotlDiffusionTrainer._logical_row_indices(documents, validity)
    assert count == 2
    assert rows.tolist() == [[0, 0, 1, 1, -1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dense_backend", ["sdpa", "eager"])
def test_native_nemotron_flex_packed_logits_loss_and_gradients_match_dense(
    dense_backend,
):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

    from tests.native_source_fixtures import native_source_fixture_path

    source = native_source_fixture_path("nemotron")
    if source is None:
        pytest.skip("native Nemotron source-only fixture is unavailable")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    dense_model = resolve_nemotron_model_class(source)(config).to(device).train()
    flex_model = copy.deepcopy(dense_model).train()
    dense_model.config._attn_implementation = dense_backend
    dense_model.encoder.config._attn_implementation = dense_backend
    flex_model.config._attn_implementation = "flex_attention"
    flex_model.encoder.config._attn_implementation = "flex_attention"
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]], device=device)
    documents = torch.tensor([[17, 17, 17, 91, 91, 91]], device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    dense = FullSequenceBackend(mask_token_id=100).pack(ids, documents, valid)
    flex = FullSequenceBackend(
        mask_token_id=100, attention_backend="flex_attention"
    ).pack(ids, documents, valid)
    assert type(flex["attention_mask"]).__name__ == "BlockMask"
    dense_logits = dense_model(
        input_ids=ids,
        attention_mask=dense["attention_mask"],
        position_ids=dense["position_ids"],
        use_cache=False,
    ).logits
    flex_logits = flex_model(
        input_ids=flex["input_ids"],
        attention_mask=flex["attention_mask"],
        position_ids=flex["position_ids"],
        use_cache=False,
    ).logits
    logical_length = dense_logits.shape[1]
    flex_logits = flex_logits[:, :logical_length]
    changed_ids = ids.clone()
    changed_ids[:, 3:] = torch.tensor([[21, 22, 23]], device=device)
    isolated_logits = flex_model(
        input_ids=torch.cat((changed_ids, flex["input_ids"][:, 6:]), dim=1),
        attention_mask=flex["attention_mask"],
        position_ids=flex["position_ids"],
        use_cache=False,
    ).logits
    torch.testing.assert_close(isolated_logits[:, :3], flex_logits[:, :3])
    dense_loss = dense_logits.square().mean()
    flex_loss = flex_logits.square().mean()
    dense_loss.backward()
    flex_loss.backward()
    torch.testing.assert_close(flex_logits, dense_logits, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(flex_loss, dense_loss, rtol=3e-4, atol=3e-5)
    for (_, dense_parameter), (_, flex_parameter) in zip(
        dense_model.named_parameters(), flex_model.named_parameters(), strict=True
    ):
        if dense_parameter.grad is None or flex_parameter.grad is None:
            assert dense_parameter.grad is flex_parameter.grad is None
        else:
            torch.testing.assert_close(
                flex_parameter.grad, dense_parameter.grad, rtol=5e-4, atol=5e-5
            )
    varied_ids = torch.tensor([[3, 4, 5, 6, 7]], device=device)
    varied_documents = torch.tensor([[42, 42, 73, 73, 73]], device=device)
    varied = FullSequenceBackend(
        mask_token_id=100, attention_backend="flex_attention"
    ).pack(varied_ids, varied_documents, torch.ones_like(varied_ids, dtype=torch.bool))
    counters = torch._dynamo.utils.counters
    before = counters["stats"].get("unique_graphs", 0)
    assert before > 0
    flex_model(
        input_ids=varied["input_ids"],
        attention_mask=varied["attention_mask"],
        position_ids=varied["position_ids"],
        use_cache=False,
    )
    assert counters["stats"].get("unique_graphs", 0) == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dense_backend", ["sdpa", "eager"])
def test_native_dream_flex_packed_logits_loss_and_gradients_match_dense(dense_backend):
    from transformers import AutoConfig

    from axolotl.model_support.dream import _model_class

    from tests.native_source_fixtures import native_source_fixture_path

    source = native_source_fixture_path("dream")
    if source is None:
        pytest.skip("native Dream source-only fixture is unavailable")
    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    for name, value in {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "max_window_layers": 1,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "mask_token_id": 100,
        "use_cache": False,
    }.items():
        setattr(config, name, value)
    config._name_or_path = str(source)
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    dense_model = (
        _model_class()
        .from_config(config, trust_remote_code=True, torch_dtype=torch.float32)
        .to(device)
        .train()
    )
    flex_model = copy.deepcopy(dense_model).train()
    dense_model.config._attn_implementation = dense_backend
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]], device=device)
    documents = torch.tensor([[17, 17, 17, 91, 91, 91]], device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    dense = FullSequenceBackend(mask_token_id=100).pack(ids, documents, valid)
    flex = FullSequenceBackend(
        mask_token_id=100, attention_backend="flex_attention"
    ).pack(ids, documents, valid)
    assert type(flex["attention_mask"]).__name__ == "BlockMask"
    dense_logits = dense_model(
        input_ids=ids,
        attention_mask=dense["attention_mask"],
        position_ids=dense["position_ids"],
        use_cache=False,
    ).logits
    flex_logits = flex_model(
        input_ids=flex["input_ids"],
        attention_mask=flex["attention_mask"],
        position_ids=flex["position_ids"],
        use_cache=False,
    ).logits
    logical_length = dense_logits.shape[1]
    flex_logits = flex_logits[:, :logical_length]
    changed_ids = ids.clone()
    changed_ids[:, 3:] = torch.tensor([[21, 22, 23]], device=device)
    isolated_logits = flex_model(
        input_ids=torch.cat((changed_ids, flex["input_ids"][:, 6:]), dim=1),
        attention_mask=flex["attention_mask"],
        position_ids=flex["position_ids"],
        use_cache=False,
    ).logits
    torch.testing.assert_close(isolated_logits[:, :3], flex_logits[:, :3])
    dense_loss = dense_logits.square().mean()
    flex_loss = flex_logits.square().mean()
    dense_loss.backward()
    flex_loss.backward()
    torch.testing.assert_close(flex_logits, dense_logits, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(flex_loss, dense_loss, rtol=3e-4, atol=3e-5)
    for (_, dense_parameter), (_, flex_parameter) in zip(
        dense_model.named_parameters(), flex_model.named_parameters(), strict=True
    ):
        if dense_parameter.grad is None or flex_parameter.grad is None:
            assert dense_parameter.grad is flex_parameter.grad is None
        else:
            torch.testing.assert_close(
                flex_parameter.grad, dense_parameter.grad, rtol=5e-4, atol=5e-5
            )
    varied_ids = torch.tensor([[3, 4, 5, 6, 7]], device=device)
    varied_documents = torch.tensor([[42, 42, 73, 73, 73]], device=device)
    varied = FullSequenceBackend(
        mask_token_id=100, attention_backend="flex_attention"
    ).pack(varied_ids, varied_documents, torch.ones_like(varied_ids, dtype=torch.bool))
    counters = torch._dynamo.utils.counters
    before = counters["stats"].get("unique_graphs", 0)
    assert before > 0
    flex_model(
        input_ids=varied["input_ids"],
        attention_mask=varied["attention_mask"],
        position_ids=varied["position_ids"],
        use_cache=False,
    )
    assert counters["stats"].get("unique_graphs", 0) == before


def test_native_full_sequence_loss_runs_a_packed_training_step_with_reset_rope(
    tmp_path,
):
    class TinyDream(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)
            self.config = SimpleNamespace(model_type="Dream", mask_token_id=15)
            self.position_ids = None

        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            assert attention_mask.shape == (1, 1, 6, 6)
            assert not use_cache
            self.position_ids = position_ids
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    model = TinyDream()
    trainer = AxolotlDiffusionTrainer(
        model=model,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=1,
            report_to=[],
            remove_unused_columns=False,
            use_cpu=True,
        ),
        train_dataset=Dataset.from_list([{"example": 0}]),
        data_collator=lambda _: inputs,
    )
    trainer.axolotl_cfg = DictDefault(
        {"diffusion_lm": {"from_causal_lm": False, "t_eps": 1.0}}
    )
    trainer._special_token_ids = set()
    initial_embedding = model.embedding.weight.detach().clone()
    inputs = {
        "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7]]),
        "document_ids": torch.tensor([[0, 0, 0, 1, 1, 1]]),
        "semantic_validity": torch.ones((1, 6), dtype=torch.bool),
        "canvas_loss_mask": torch.ones((1, 6), dtype=torch.bool),
        "canvas_corruptible_mask": torch.ones((1, 6), dtype=torch.bool),
    }
    trainer.train()
    assert not torch.equal(model.embedding.weight.detach(), initial_embedding)
    assert trainer.state.global_step == 1
    assert torch.equal(model.position_ids, torch.tensor([[0, 1, 2, 0, 1, 2]]))


def test_global_masked_window_matches_packed_reference_with_unequal_mask_counts(
    tmp_path, monkeypatch
):
    class TinyNemotron(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)
            self.config = SimpleNamespace(
                model_type="nemotron_labs_diffusion", mask_token_id=15
            )

        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            del attention_mask, position_ids
            assert not use_cache
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    samples = [
        (torch.tensor([2, 3, 4]), torch.tensor([True, True, True])),
        (torch.tensor([5, 6, 7]), torch.tensor([True, True, True])),
    ]

    def fixed_corruption(self, packed, corruptible_mask, logical_times, **kwargs):
        del logical_times, kwargs
        events = torch.zeros_like(corruptible_mask)
        for document in (
            packed["document_ids"][packed["semantic_validity"]].unique().tolist()
        ):
            positions = torch.where(packed["document_ids"][0] == document)[0]
            count = 1 if packed["input_ids"][0, positions[0]] == 2 else 2
            events[0, positions[:count]] = True
        events &= corruptible_mask
        return (
            torch.where(
                events,
                torch.full_like(packed["input_ids"], self.mask_token_id),
                packed["input_ids"],
            ),
            events,
            torch.zeros_like(packed["input_ids"], dtype=torch.float),
        )

    monkeypatch.setattr(FullSequenceBackend, "corrupt_native", fixed_corruption)

    def collate(features):
        ids = torch.cat([samples[item["example"]][0] for item in features])[None]
        corruptible = torch.cat([samples[item["example"]][1] for item in features])[
            None
        ]
        docs = torch.cat(
            [
                torch.full_like(samples[item["example"]][0], index)
                for index, item in enumerate(features)
            ]
        )[None]
        return {
            "input_ids": ids,
            "document_ids": docs,
            "semantic_validity": torch.ones_like(ids, dtype=torch.bool),
            "canvas_loss_mask": corruptible.clone(),
            "canvas_corruptible_mask": corruptible,
        }

    dataset = Dataset.from_list([{"example": 0}, {"example": 1}])
    initial = TinyNemotron().state_dict()

    def train(batch_size, accumulation, output_dir):
        model = TinyNemotron()
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = AxolotlDiffusionTrainer(
            model=model,
            args=AxolotlTrainingArguments(
                output_dir=str(output_dir),
                max_steps=1,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                learning_rate=1e-3,
                max_grad_norm=0.0,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                report_to=[],
                remove_unused_columns=False,
                save_strategy="no",
                use_cpu=True,
            ),
            train_dataset=dataset,
            data_collator=collate,
            optimizers=(optimizer, None),
        )
        trainer.axolotl_cfg = DictDefault(
            {"diffusion_lm": {"from_causal_lm": False, "t_eps": 1.0}}
        )
        trainer._special_token_ids = set()
        trainer.post_set_axolotl_cfg()
        torch.manual_seed(9)
        trainer.train()
        return model

    accumulated = train(1, 2, tmp_path / "accumulated")
    packed = train(2, 1, tmp_path / "packed")
    for accumulated_parameter, packed_parameter in zip(
        accumulated.parameters(), packed.parameters(), strict=True
    ):
        assert torch.allclose(accumulated_parameter, packed_parameter, atol=2e-7)

    expected = TinyNemotron()
    expected.load_state_dict(initial)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=1e-3)
    backend = FullSequenceBackend(mask_token_id=15, special_token_ids=set())
    numerator = torch.zeros(())
    denominator = 0
    for index in range(len(samples)):
        inputs = collate([{"example": index}])
        packed_inputs = backend.pack(
            inputs["input_ids"],
            inputs["document_ids"],
            inputs["semantic_validity"],
        )
        noisy, events, _ = fixed_corruption(
            backend,
            packed_inputs,
            inputs["canvas_corruptible_mask"],
            torch.ones(1),
        )
        logits = backend.canvas_logits(
            backend.forward(expected, packed_inputs, noisy), packed_inputs, aligned=True
        )
        token_loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, -2),
            inputs["input_ids"].flatten(),
            reduction="none",
        ).view_as(inputs["input_ids"])
        support = events & inputs["canvas_loss_mask"]
        numerator = numerator + token_loss[support].sum()
        denominator += int(support.sum())
    assert denominator == 3
    (numerator / denominator).backward()
    expected_optimizer.step()
    for accumulated_parameter, expected_parameter in zip(
        accumulated.parameters(), expected.parameters(), strict=True
    ):
        torch.testing.assert_close(
            accumulated_parameter, expected_parameter, atol=2e-7, rtol=0
        )


def test_dream_accumulation_matches_arithmetic_mean_of_microbatch_masked_means(
    tmp_path, monkeypatch
):
    class TinyDream(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)
            self.config = SimpleNamespace(model_type="Dream", mask_token_id=15)

        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            del attention_mask, position_ids
            assert not use_cache
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    samples = [torch.tensor([2, 3, 4]), torch.tensor([5, 6, 7])]

    def fixed_corruption(self, packed, corruptible_mask, logical_times, **kwargs):
        del logical_times, kwargs
        count = 1 if packed["input_ids"][0, 0] == 2 else 2
        events = torch.zeros_like(corruptible_mask)
        events[:, :count] = True
        events &= corruptible_mask
        return (
            torch.where(
                events,
                torch.full_like(packed["input_ids"], self.mask_token_id),
                packed["input_ids"],
            ),
            events,
            torch.zeros_like(packed["input_ids"], dtype=torch.float),
        )

    monkeypatch.setattr(FullSequenceBackend, "corrupt_native", fixed_corruption)

    def collate(features):
        ids = torch.stack([samples[item["example"]] for item in features])
        return {
            "input_ids": ids,
            "document_ids": torch.zeros_like(ids),
            "semantic_validity": torch.ones_like(ids, dtype=torch.bool),
            "canvas_loss_mask": torch.ones_like(ids, dtype=torch.bool),
            "canvas_corruptible_mask": torch.ones_like(ids, dtype=torch.bool),
        }

    torch.manual_seed(17)
    initial = TinyDream().state_dict()

    actual = TinyDream()
    actual.load_state_dict(initial)
    optimizer = torch.optim.SGD(actual.parameters(), lr=1e-3)
    trainer = AxolotlDiffusionTrainer(
        model=actual,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path / "actual"),
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            train_sampling_strategy="sequential",
            learning_rate=1e-3,
            max_grad_norm=0.0,
            weight_decay=0.0,
            lr_scheduler_type="constant",
            report_to=[],
            remove_unused_columns=False,
            save_strategy="no",
            use_cpu=True,
        ),
        train_dataset=Dataset.from_list([{"example": 0}, {"example": 1}]),
        data_collator=collate,
        optimizers=(optimizer, None),
    )
    trainer.axolotl_cfg = DictDefault(
        {"diffusion_lm": {"from_causal_lm": False, "t_eps": 1.0}}
    )
    trainer._special_token_ids = set()
    trainer.post_set_axolotl_cfg()

    expected = TinyDream()
    expected.load_state_dict(initial)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=1e-3)
    backend = FullSequenceBackend(mask_token_id=15, special_token_ids=set())
    microbatch_losses = []
    masked_counts = []
    for index, _sample in enumerate(samples):
        inputs = collate([{"example": index}])
        packed = backend.pack(
            inputs["input_ids"], inputs["document_ids"], inputs["semantic_validity"]
        )
        noisy, events, _ = fixed_corruption(
            backend, packed, inputs["canvas_corruptible_mask"], torch.ones(1)
        )
        logits = backend.canvas_logits(
            backend.forward(expected, packed, noisy), packed, aligned=False
        )
        token_loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, -2),
            inputs["input_ids"].flatten(),
            reduction="none",
        ).view_as(inputs["input_ids"])
        support = events & inputs["canvas_loss_mask"]
        masked_counts.append(int(support.sum()))
        microbatch_losses.append(token_loss[support].mean())
    assert masked_counts == [1, 2]
    expected_loss = torch.stack(microbatch_losses).mean()
    expected_loss.backward()
    expected_optimizer.step()

    torch.manual_seed(43)
    trainer.train()
    for actual_parameter, expected_parameter in zip(
        actual.parameters(), expected.parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_parameter, expected_parameter, atol=2e-7, rtol=0
        )


@pytest.mark.parametrize("grad_through_steps", [False, True])
def test_shared_unroll_retains_only_final_step_by_default_and_updates_selected_tokens(
    grad_through_steps,
):
    class Runner:
        _run_native_unroll = AxolotlDiffusionTrainer._run_native_unroll

        @staticmethod
        def _native_unroll_settings():
            return 2, grad_through_steps

        @staticmethod
        def _sample_native_unroll_steps(k_max, device):
            assert k_max == 2
            del device
            return 2

    weight = torch.tensor(1.0, requires_grad=True)
    calls = []

    def forward_step(state, conditioning, conditioning_mask):
        calls.append(
            (
                state.detach().clone(),
                conditioning,
                conditioning_mask,
                torch.is_grad_enabled(),
            )
        )
        logits = state.float()[..., None].expand(-1, -1, 2) * weight
        if conditioning is not None:
            logits = logits + conditioning
        return SimpleNamespace(logits=logits)

    state = torch.tensor([[2, 3]])
    update_mask = torch.tensor([[True, False]])
    outputs, logits, final_state, steps = Runner()._run_native_unroll(
        state=state,
        update_mask=update_mask,
        supports_self_conditioning=True,
        k1_conditioning_mask=torch.tensor([[True, True]]),
        recurrent_conditioning_mask=torch.tensor([[True, True]]),
        forward_step=forward_step,
        logits_from_outputs=lambda output: output.logits,
        update_state=lambda current, _logits, mask: torch.where(
            mask, torch.full_like(current, 9), current
        ),
    )

    assert outputs.logits is logits
    assert steps == 2
    assert torch.equal(final_state, torch.tensor([[9, 3]]))
    assert torch.equal(calls[0][0], state)
    assert torch.equal(calls[1][0], final_state)
    assert calls[0][3] is grad_through_steps
    assert calls[1][3]
    assert calls[1][1] is not None
    assert calls[1][1].requires_grad is grad_through_steps
    logits.sum().backward()
    assert weight.grad is not None


def test_full_sequence_unroll_preserves_clean_targets_and_pinned_state(
    tmp_path, monkeypatch
):
    class TinyDream(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)
            self.config = SimpleNamespace(model_type="Dream", mask_token_id=15)
            self.calls = []

        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            del attention_mask, position_ids
            assert not use_cache
            self.calls.append((input_ids.detach().clone(), torch.is_grad_enabled()))
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    def fixed_corruption(self, packed, corruptible_mask, logical_times, **kwargs):
        del logical_times, kwargs
        events = torch.zeros_like(corruptible_mask)
        events[:, :2] = True
        events &= corruptible_mask
        return (
            torch.where(
                events,
                torch.full_like(packed["input_ids"], self.mask_token_id),
                packed["input_ids"],
            ),
            events,
            torch.zeros_like(packed["input_ids"], dtype=torch.float),
        )

    monkeypatch.setattr(FullSequenceBackend, "corrupt_native", fixed_corruption)
    monkeypatch.setattr(
        AxolotlDiffusionTrainer,
        "_sample_native_unroll_steps",
        staticmethod(lambda k_max, device: 2),
    )
    model = TinyDream()
    trainer = AxolotlDiffusionTrainer(
        model=model,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            report_to=[],
            remove_unused_columns=False,
            use_cpu=True,
        ),
        train_dataset=Dataset.from_list([{"example": 0}]),
        data_collator=lambda _: {},
    )
    trainer.axolotl_cfg = DictDefault(
        {
            "diffusion_lm": {
                "from_causal_lm": False,
                "t_eps": 1.0,
                "unroll": {"k_max": 2, "grad_through_steps": False},
            }
        }
    )
    trainer._special_token_ids = set()
    trainer.post_set_axolotl_cfg()
    clean = torch.tensor([[2, 3, 4]])
    loss = trainer.compute_loss(
        model,
        {
            "input_ids": clean,
            "document_ids": torch.zeros_like(clean),
            "semantic_validity": torch.ones_like(clean, dtype=torch.bool),
            "canvas_loss_mask": torch.ones_like(clean, dtype=torch.bool),
            "canvas_corruptible_mask": torch.ones_like(clean, dtype=torch.bool),
            "canvas_input_pinned_mask": torch.tensor([[False, False, True]]),
            "canvas_update_mask": torch.tensor([[True, False, True]]),
        },
    )
    loss.backward()

    assert torch.equal(clean, torch.tensor([[2, 3, 4]]))
    assert len(model.calls) == 2
    assert model.calls[0][1] is False and model.calls[1][1]
    assert model.calls[0][0].tolist() == [[15, 15, 4]]
    assert model.calls[1][0][0, 1:].tolist() == [15, 4]


@pytest.mark.parametrize(
    "k1_gate",
    [None, torch.tensor([[True, False, True]])],
    ids=["p_zero", "p_one"],
)
def test_k2_self_conditioning_uses_recurrent_eligibility_not_k1_gate(k1_gate):
    class Runner:
        _run_native_unroll = AxolotlDiffusionTrainer._run_native_unroll

        @staticmethod
        def _native_unroll_settings():
            return 2, True

        @staticmethod
        def _sample_native_unroll_steps(k_max, device):
            del k_max, device
            return 2

    recurrent_mask = torch.tensor([[True, False, False]])
    first_logits = []
    calls = []

    def forward_step(state, conditioning, conditioning_mask):
        calls.append((conditioning, conditioning_mask))
        if conditioning is None:
            logits = state.float()[..., None].expand(-1, -1, 2).clone()
            logits.requires_grad_()
            logits.retain_grad()
            first_logits.append(logits)
        else:
            logits = conditioning * conditioning_mask[..., None].to(conditioning.dtype)
        return SimpleNamespace(logits=logits)

    _, logits, _, _ = Runner()._run_native_unroll(
        state=torch.tensor([[2, 3, 4]]),
        update_mask=torch.zeros((1, 3), dtype=torch.bool),
        supports_self_conditioning=True,
        k1_conditioning_mask=k1_gate,
        recurrent_conditioning_mask=recurrent_mask,
        forward_step=forward_step,
        logits_from_outputs=lambda output: output.logits,
        update_state=lambda state, _logits, _mask: state,
    )
    logits.sum().backward()

    assert calls[1][0] is first_logits[0]
    assert torch.equal(calls[1][1], recurrent_mask)
    assert torch.equal(first_logits[0].grad[..., 0], torch.tensor([[1.0, 0.0, 0.0]]))


@pytest.mark.parametrize(("train_only", "training"), [(False, True), (True, False)])
def test_native_cce_final_loss_matches_dense_per_token_objective(train_only, training):
    class TinyCCENemotron(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)
            self.config = SimpleNamespace(
                model_type="nemotron_labs_diffusion", mask_token_id=15
            )
            self.cce_calls = 0
            type(self)._axolotl_cce_options = SimpleNamespace(train_only=train_only)

        def forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            use_cache,
            cce_targets=None,
        ):
            del attention_mask, position_ids, use_cache
            logits = self.head(self.embedding(input_ids))
            if cce_targets is None:
                return SimpleNamespace(logits=logits)
            self.cce_calls += 1
            return SimpleNamespace(
                logits=None,
                loss=torch.nn.functional.cross_entropy(
                    logits.flatten(0, -2),
                    cce_targets.flatten(),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(cce_targets),
            )

    torch.manual_seed(123)
    dense = TinyCCENemotron()
    cce = TinyCCENemotron()
    cce.load_state_dict(copy.deepcopy(dense.state_dict()))
    dense.train(training)
    cce.train(training)
    dense_runner = _ddp_runner(dense, world_size=1, gemma=False)
    cce_runner = _ddp_runner(cce, world_size=1, gemma=False)
    cce_runner.axolotl_cfg = DictDefault(
        {
            "cut_cross_entropy": True,
            "diffusion_lm": {"from_causal_lm": False, "t_eps": 1.0},
        }
    )
    inputs = _ddp_full_sequence_inputs(0)
    torch.manual_seed(456)
    dense_loss, _ = dense_runner._compute_native_full_sequence_loss(
        dense, inputs, dense_runner._native_spec
    )
    dense_loss.backward()
    torch.manual_seed(456)
    cce_loss, _ = cce_runner._compute_native_full_sequence_loss(
        cce, inputs, cce_runner._native_spec
    )
    cce_loss.backward()
    torch.testing.assert_close(cce_loss, dense_loss)
    assert cce.cce_calls == int(training or not train_only)
    for actual, expected in zip(cce.parameters(), dense.parameters(), strict=True):
        assert actual.grad is not None and expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad)


def test_native_unroll_uses_cce_final_forward_only_after_dense_updates():
    dense_calls, cce_calls = [], []

    def dense(state, *_):
        dense_calls.append(state.clone())
        return SimpleNamespace(logits=state.float().unsqueeze(-1))

    def final(state, *_):
        cce_calls.append(state.clone())
        return SimpleNamespace(loss=torch.ones_like(state, dtype=torch.float))

    outputs, logits, final_state = run_unroll(
        state=torch.tensor([[1, 2]]),
        update_mask=torch.tensor([[True, False]]),
        steps=3,
        grad_through_steps=False,
        supports_self_conditioning=False,
        k1_conditioning_mask=None,
        recurrent_conditioning_mask=None,
        forward_step=dense,
        logits_from_outputs=lambda output: output.logits,
        update_state=lambda state, _logits, mask: torch.where(mask, state + 1, state),
        forward_final=final,
        final_logits_from_outputs=lambda _output: None,
    )
    assert logits is None
    assert outputs.loss.shape == final_state.shape
    assert len(dense_calls) == 2
    assert len(cce_calls) == 1
    assert torch.equal(final_state, torch.tensor([[3, 2]]))
