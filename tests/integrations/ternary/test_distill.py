"""CPU-only tests for the ternary in-process distillation trainer."""

import copy
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

import pydantic
import pytest
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import distill as distill_mod
from axolotl.integrations.ternary.distill import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    TernaryDistillTrainer,
    TernaryDistillTrainingArgsMixin,
    _chunk_size,
    _chunked_ce_kd,
)
from axolotl.integrations.ternary.modules import TernaryLinear
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault


@dataclass
class _StubArgs(TernaryDistillTrainingArgsMixin):
    """The distill knobs plus the `TrainingArguments` fields `compute_loss` reads."""

    include_tkps: bool = False
    sample_packing: bool = False
    orpo_alpha: float | None = None


def _training_args(**knobs) -> _StubArgs:
    return _StubArgs(
        **{f"ternary_distill_{key}": value for key, value in knobs.items()}
    )


def _bare_trainer(teacher=None, **knobs) -> TernaryDistillTrainer:
    """Build a trainer without `Trainer.__init__` — only the distill state is used."""
    trainer = TernaryDistillTrainer.__new__(TernaryDistillTrainer)
    trainer.args = _training_args(**knobs)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    trainer.state = SimpleNamespace()
    trainer._stored_metrics = defaultdict(
        lambda: defaultdict(lambda: {"values": [], "reduction": "mean"})
    )
    trainer._init_distill_state(teacher)
    return trainer


def _tiny_llama(seed: int = 0, vocab: int = 48, hidden: int = 32) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _batch(vocab: int = 48, batch: int = 2, seq: int = 7, seed: int = 3) -> dict:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab, (batch, seq), generator=generator)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _naive_terms(student_hidden, teacher_hidden, head, teacher_head, targets, tau):
    """Full-logits CE and KL — the oracle the chunked path must match."""
    student_logits = head(student_hidden).float()
    teacher_logits = teacher_head(teacher_hidden).float()
    ce = F.cross_entropy(student_logits, targets, reduction="mean")
    kd = (
        F.kl_div(
            F.log_softmax(student_logits / tau, dim=-1),
            F.log_softmax(teacher_logits / tau, dim=-1),
            reduction="sum",
            log_target=True,
        )
        / targets.shape[0]
    )
    return ce, kd


def _reference_ce(student, batch):
    """Plain next-token CE from full `(B, T, V)` logits."""
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    logits = student(**inputs).logits[:, :-1].float()
    targets = batch["labels"][:, 1:]
    keep = targets != -100
    return F.cross_entropy(logits[keep], targets[keep])


def _reference_loss(student, teacher, batch, alpha, tau):
    """Reference `(1-α)·CE + α·τ²·KL` computed from full `(B, T, V)` logits."""
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    student_logits = student(**inputs).logits[:, :-1].float()
    with torch.no_grad():
        teacher_logits = teacher(**inputs).logits[:, :-1].float()
    targets = batch["labels"][:, 1:]
    keep = targets != -100
    student_logits = student_logits[keep]
    teacher_logits = teacher_logits[keep]
    targets = targets[keep]
    ce = F.cross_entropy(student_logits, targets)
    kd = (
        F.kl_div(
            F.log_softmax(student_logits / tau, dim=-1),
            F.log_softmax(teacher_logits / tau, dim=-1),
            reduction="sum",
            log_target=True,
        )
        / targets.shape[0]
    )
    return (1.0 - alpha) * ce + alpha * tau**2 * kd


# ---------------------------------------------------------------- chunked logits KD


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 37, 512])
def test_chunked_matches_naive_full_logits(chunk_size):
    torch.manual_seed(0)
    tokens, hidden, vocab = 37, 16, 53
    student_hidden = torch.randn(tokens, hidden)
    teacher_hidden = torch.randn(tokens, hidden)
    head = nn.Linear(hidden, vocab, bias=False)
    teacher_head = nn.Linear(hidden, vocab, bias=False)
    targets = torch.randint(0, vocab, (tokens,))

    ce, kd = _chunked_ce_kd(
        student_hidden,
        teacher_hidden,
        head.weight,
        None,
        teacher_head.weight,
        None,
        targets,
        2.0,
        chunk_size,
    )
    ref_ce, ref_kd = _naive_terms(
        student_hidden, teacher_hidden, head, teacher_head, targets, 2.0
    )
    assert ce.item() == pytest.approx(ref_ce.item(), rel=1e-5, abs=1e-6)
    assert kd.item() == pytest.approx(ref_kd.item(), rel=1e-5, abs=1e-6)


def test_chunked_gradients_match_naive():
    torch.manual_seed(1)
    tokens, hidden, vocab = 20, 12, 31
    base_hidden = torch.randn(tokens, hidden)
    teacher_hidden = torch.randn(tokens, hidden)
    weight = torch.randn(vocab, hidden) * 0.1
    teacher_head = nn.Linear(hidden, vocab, bias=False)
    targets = torch.randint(0, vocab, (tokens,))

    chunked_hidden = base_hidden.clone().requires_grad_(True)
    chunked_weight = weight.clone().requires_grad_(True)
    ce, kd = _chunked_ce_kd(
        chunked_hidden,
        teacher_hidden,
        chunked_weight,
        None,
        teacher_head.weight,
        None,
        targets,
        2.0,
        7,
    )
    (0.5 * ce + 0.5 * 4.0 * kd).backward()

    naive_hidden = base_hidden.clone().requires_grad_(True)
    naive_weight = weight.clone().requires_grad_(True)
    head = nn.Linear(hidden, vocab, bias=False)
    head.weight = nn.Parameter(naive_weight)
    ref_ce, ref_kd = _naive_terms(
        naive_hidden, teacher_hidden, head, teacher_head, targets, 2.0
    )
    (0.5 * ref_ce + 0.5 * 4.0 * ref_kd).backward()

    torch.testing.assert_close(
        chunked_hidden.grad, naive_hidden.grad, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        chunked_weight.grad, head.weight.grad, rtol=1e-5, atol=1e-6
    )


def test_chunked_kd_is_zero_for_identical_logits():
    torch.manual_seed(2)
    hidden_states = torch.randn(9, 8)
    head = nn.Linear(8, 17, bias=False)
    _, kd = _chunked_ce_kd(
        hidden_states,
        hidden_states,
        head.weight,
        None,
        head.weight,
        None,
        None,
        3.0,
        4,
    )
    assert kd.item() == pytest.approx(0.0, abs=1e-6)


def test_chunked_ce_skipped_without_targets():
    torch.manual_seed(3)
    hidden_states = torch.randn(5, 8)
    head = nn.Linear(8, 11, bias=False)
    ce, _ = _chunked_ce_kd(
        hidden_states, hidden_states, head.weight, None, head.weight, None, None, 1.0, 2
    )
    assert ce.item() == 0.0


def test_chunked_empty_selection_is_zero_and_differentiable():
    hidden_states = torch.zeros(0, 8, requires_grad=True)
    head = nn.Linear(8, 11, bias=False)
    ce, kd = _chunked_ce_kd(
        hidden_states,
        torch.zeros(0, 8),
        head.weight,
        None,
        head.weight,
        None,
        torch.zeros(0, dtype=torch.long),
        2.0,
        4,
    )
    (ce + kd).backward()
    assert ce.item() == 0.0 and kd.item() == 0.0
    assert head.weight.grad is not None


def test_chunked_logits_kd_method_uses_teacher_head():
    torch.manual_seed(4)
    trainer = _bare_trainer()
    student_hidden = torch.randn(2, 6, 10)
    teacher_hidden = torch.randn(2, 6, 10)
    head = nn.Linear(10, 23, bias=False)
    teacher_head = nn.Linear(10, 23, bias=False)

    value = trainer.chunked_logits_kd(
        student_hidden,
        teacher_hidden,
        head,
        2.0,
        chunk_size=5,
        teacher_lm_head=teacher_head,
    )
    _, reference = _naive_terms(
        student_hidden.reshape(-1, 10),
        teacher_hidden.reshape(-1, 10),
        head,
        teacher_head,
        torch.zeros(12, dtype=torch.long),
        2.0,
    )
    assert value.item() == pytest.approx(reference.item(), rel=1e-5)

    shared = trainer.chunked_logits_kd(student_hidden, teacher_hidden, head, 2.0)
    assert shared.item() != pytest.approx(value.item(), rel=1e-3)


@pytest.mark.parametrize(
    "vocab, expected",
    [(128, MAX_CHUNK_TOKENS), (8192, 1024), (128256, 65), (262144, MIN_CHUNK_TOKENS)],
)
def test_chunk_size_bounds(vocab, expected):
    assert _chunk_size(vocab) == expected


def test_chunks_are_bounded_and_recomputed_in_backward(monkeypatch):
    shapes = []
    original = distill_mod._chunk_terms

    def recording(student_hidden, *args):
        shapes.append(student_hidden.shape[0])
        return original(student_hidden, *args)

    monkeypatch.setattr(distill_mod, "_chunk_terms", recording)

    tokens, hidden, vocab, chunk = 25, 8, 13, 10
    student_hidden = torch.randn(tokens, hidden, requires_grad=True)
    head = nn.Linear(hidden, vocab, bias=False)
    ce, kd = _chunked_ce_kd(
        student_hidden,
        torch.randn(tokens, hidden),
        head.weight,
        None,
        head.weight.detach(),
        None,
        torch.randint(0, vocab, (tokens,)),
        2.0,
        chunk,
    )
    assert shapes == [10, 10, 5]

    # every chunk is recomputed in the backward pass, so only one lives at a time
    (ce + kd).backward()
    assert sorted(shapes) == sorted([10, 10, 5] * 2)


# ------------------------------------------------------------------------ knobs


def test_knobs_default_and_clamp(caplog):
    trainer = _bare_trainer()
    assert (trainer.logits_weight, trainer.logits_temperature) == (1.0, 2.0)
    assert trainer.hidden_weight == 0.0
    assert trainer.attn_relation_layer is None

    clamped = _bare_trainer(logits_weight=4.0)
    assert clamped.logits_weight == 1.0

    configured = _bare_trainer(
        logits_weight=0.25, logits_temperature=5.0, hidden_weight=0.5
    )
    assert configured.logits_weight == 0.25
    assert configured.logits_temperature == 5.0
    assert configured.hidden_weight == 0.5


# ----------------------------------------------------------------- compute_loss


def test_compute_loss_matches_full_logits_reference():
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    trainer = _bare_trainer(teacher, logits_weight=0.5, logits_temperature=2.0)
    batch = _batch()

    loss = trainer.compute_loss(student, batch)
    reference = _reference_loss(student, copy.deepcopy(teacher), batch, 0.5, 2.0)
    assert loss.item() == pytest.approx(reference.item(), rel=1e-5, abs=1e-6)


def test_compute_loss_never_materializes_full_logits():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(copy.deepcopy(student))
    batch = _batch()

    _, outputs = trainer.compute_loss(student, batch, return_outputs=True)
    assert outputs.logits.shape[1] == 1
    assert outputs.hidden_states[-1].shape[1] == batch["input_ids"].shape[1]


def test_compute_loss_is_zero_when_teacher_is_the_student_copy():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(copy.deepcopy(student), logits_weight=1.0)
    assert trainer.compute_loss(student, _batch()).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_compute_loss_requires_labels():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(copy.deepcopy(student))
    batch = _batch()
    batch.pop("labels")
    with pytest.raises(KeyError, match="labels"):
        trainer.compute_loss(student, batch)


def test_masked_positions_do_not_reach_the_loss():
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    trainer = _bare_trainer(teacher, logits_weight=0.5)

    batch = _batch(seq=5)
    padded = {
        "input_ids": F.pad(batch["input_ids"], (0, 3), value=0),
        "attention_mask": F.pad(batch["attention_mask"], (0, 3), value=0),
        "labels": F.pad(batch["labels"], (0, 3), value=-100),
    }
    other = dict(padded)
    other["input_ids"] = padded["input_ids"].clone()
    other["input_ids"][:, 5:] = 7

    with torch.no_grad():
        first = trainer.compute_loss(student, padded).item()
        second = trainer.compute_loss(student, other).item()
        trimmed = trainer.compute_loss(student, batch).item()

    assert first == pytest.approx(second, rel=1e-6)
    assert first == pytest.approx(trimmed, rel=1e-4, abs=1e-5)


def test_fully_masked_row_contributes_nothing():
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    trainer = _bare_trainer(teacher, logits_weight=0.5)

    batch = _batch(batch=2)
    masked = dict(batch)
    masked["labels"] = batch["labels"].clone()
    masked["labels"][1] = -100
    single = {key: value[:1] for key, value in batch.items()}

    with torch.no_grad():
        assert trainer.compute_loss(student, masked).item() == pytest.approx(
            trainer.compute_loss(student, single).item(), rel=1e-5
        )


def test_hidden_weight_adds_a_positive_term():
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    batch = _batch()

    without = _bare_trainer(copy.deepcopy(teacher), logits_weight=0.5)
    with_hidden = _bare_trainer(
        copy.deepcopy(teacher), logits_weight=0.5, hidden_weight=2.0
    )
    with torch.no_grad():
        base = without.compute_loss(student, batch).item()
        boosted = with_hidden.compute_loss(student, batch).item()
    assert boosted > base


def test_metrics_are_stored_per_term():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(
        copy.deepcopy(_tiny_llama(seed=1)),
        logits_weight=0.5,
        hidden_weight=1.0,
        attn_relation_layer=-1,
    )
    trainer.compute_loss(student, _batch())
    assert set(trainer._stored_metrics["train"]) == {
        "ternary/ce",
        "ternary/kd_logits",
        "ternary/kd_hidden",
        "ternary/kd_attn",
    }

    student.eval()
    trainer.compute_loss(student, _batch())
    assert set(trainer._stored_metrics["eval"]) == set(trainer._stored_metrics["train"])


# ------------------------------------------------------------------- extra terms


def test_hidden_cosine_kd_endpoints_and_masking():
    trainer = _bare_trainer()
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert trainer.hidden_cosine_kd(student, student.clone()).item() == pytest.approx(
        0.0, abs=1e-6
    )
    assert trainer.hidden_cosine_kd(student, -student).item() == pytest.approx(2.0)

    orthogonal = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    mask = torch.tensor([[1, 0]])
    assert trainer.hidden_cosine_kd(student, orthogonal).item() == pytest.approx(0.5)
    assert trainer.hidden_cosine_kd(student, orthogonal, mask=mask).item() == (
        pytest.approx(1.0)
    )


def test_attn_relation_kd_endpoints_and_masking():
    torch.manual_seed(5)
    trainer = _bare_trainer()
    student = torch.randn(2, 3, 5, 4)
    assert trainer.attn_relation_kd(student, student.clone()).item() == pytest.approx(
        0.0, abs=1e-6
    )

    teacher = torch.randn(2, 3, 5, 4)
    assert trainer.attn_relation_kd(student, teacher).item() > 0.0

    mask = torch.ones(2, 5, dtype=torch.long)
    mask[:, 3:] = 0
    masked = trainer.attn_relation_kd(student, teacher, mask=mask)
    assert torch.isfinite(masked)

    poisoned_student = student.clone()
    poisoned_teacher = teacher.clone()
    poisoned_student[:, :, 3:] = 50.0
    poisoned_teacher[:, :, 3:] = -50.0
    assert trainer.attn_relation_kd(
        poisoned_student, poisoned_teacher, mask=mask
    ).item() == pytest.approx(masked.item(), rel=1e-4)


def test_attn_relation_kd_is_chunked_over_heads(monkeypatch):
    torch.manual_seed(6)
    trainer = _bare_trainer()
    student = torch.randn(2, 4, 5, 4, requires_grad=True)
    teacher = torch.randn(2, 4, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.long)
    mask[:, 4:] = 0

    reference = trainer.attn_relation_kd(student, teacher, mask=mask)
    monkeypatch.setattr(distill_mod, "CHUNK_ELEMENT_BUDGET", 1)
    assert distill_mod._head_chunk(5) == 1
    chunked = trainer.attn_relation_kd(student, teacher, mask=mask)
    assert chunked.item() == pytest.approx(reference.item(), rel=1e-6)

    chunked.backward()
    reference_grad = student.grad.clone()
    student.grad = None
    monkeypatch.undo()
    trainer.attn_relation_kd(student, teacher, mask=mask).backward()
    torch.testing.assert_close(student.grad, reference_grad, rtol=1e-5, atol=1e-6)


def test_attn_relation_hooks_capture_both_models():
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    trainer = _bare_trainer(teacher, logits_weight=0.5, attn_relation_layer=-1)
    seen: list[set[str]] = []
    original = trainer._attn_relation_term
    trainer._attn_relation_term = lambda mask: (
        seen.append(set(trainer._captured)) or original(mask)
    )

    loss = trainer.compute_loss(student, _batch())

    assert seen == [
        {
            f"{tag}.{projection}"
            for tag in ("student", "teacher")
            for projection in ("q_proj", "k_proj", "v_proj")
        }
    ]
    assert torch.isfinite(loss)
    assert trainer._stored_metrics["train"]["ternary/kd_attn"]["values"]


def test_attn_relation_hooks_and_captures_are_released_after_the_step():
    """Hooks that outlive the step keep the previous q/k/v projections alive."""
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(_tiny_llama(seed=1), attn_relation_layer=-1)

    for _ in range(2):
        trainer.compute_loss(student, _batch())
        assert trainer._hook_handles == []
        assert trainer._captured == {}

    attention = student.model.layers[-1].self_attn
    assert not attention.q_proj._forward_hooks

    identical = _bare_trainer(
        copy.deepcopy(student), logits_weight=1.0, attn_relation_layer=-1
    )
    assert identical.compute_loss(student, _batch()).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_attn_relation_layer_out_of_range():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(copy.deepcopy(student), attn_relation_layer=9)
    with pytest.raises(ValueError, match="out of range"):
        trainer.compute_loss(student, _batch())


# ----------------------------------------------------------------- teacher state


def test_teacher_is_frozen_and_stays_out_of_the_student_graph():
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    teacher.train()
    trainer = _bare_trainer(teacher, logits_weight=0.5)

    trainer.compute_loss(student, _batch()).backward()

    assert not teacher.training
    assert all(not param.requires_grad for param in teacher.parameters())
    assert all(param.grad is None for param in teacher.parameters())
    assert any(param.grad is not None for param in student.parameters())

    student_params = {id(param) for param in student.parameters()}
    assert not any(id(param) in student_params for param in teacher.parameters())


def test_teacher_is_loaded_lazily_once(monkeypatch):
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    calls = []

    def fake_from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return teacher

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    trainer = _bare_trainer(teacher_model="dummy/teacher", logits_weight=1.0)
    assert not calls

    trainer.compute_loss(student, _batch())
    trainer.compute_loss(student, _batch())
    assert len(calls) == 1
    assert calls[0][0] == "dummy/teacher"
    assert calls[0][1]["dtype"] == torch.float32
    assert "device_map" not in calls[0][1]


def test_teacher_device_map_is_passed_through(monkeypatch):
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    calls = []

    def fake_from_pretrained(name, **kwargs):
        calls.append(kwargs)
        return teacher

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    trainer = _bare_trainer(teacher_model="dummy/teacher", teacher_device_map="cpu")
    trainer.compute_loss(student, _batch())
    assert calls[0]["device_map"] == "cpu"


def test_missing_teacher_name_raises():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer()
    with pytest.raises(ValueError, match="teacher"):
        trainer.compute_loss(student, _batch())


# ------------------------------------------------------------------- micro-loop


def test_real_trainer_construction_wires_the_knobs(tmp_path):
    from axolotl.core.training_args import AxolotlTrainingArguments

    student = _tiny_llama(seed=0)
    teacher = copy.deepcopy(student)
    args = AxolotlTrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=1,
        report_to=[],
        save_strategy="no",
    )
    trainer = TernaryDistillTrainer(
        model=student,
        args=args,
        train_dataset=None,
        teacher_model=teacher,
    )
    assert trainer.model_accepts_loss_kwargs is False
    assert (trainer.logits_weight, trainer.logits_temperature) == (1.0, 2.0)

    # the teacher stays where the caller put it, so this also covers a split placement
    batch = {key: value.to(args.device) for key, value in _batch().items()}
    loss = trainer.compute_loss(trainer.model, batch)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_distillation_loss_decreases_for_a_swapped_student():
    student = _tiny_llama(seed=0)
    teacher = copy.deepcopy(student)
    convert_model(student, DictDefault({}))
    assert any(isinstance(module, TernaryLinear) for module in student.modules())

    trainer = _bare_trainer(teacher, logits_weight=1.0, logits_temperature=2.0)
    batch = _batch(seq=8)
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3)

    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = trainer.compute_loss(student, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(loss > 0 for loss in losses)
    assert losses[-1] < losses[0]
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3


# ------------------------------------------------- teacher placement and fixups


def test_the_teacher_head_is_never_copied_to_the_student(monkeypatch):
    """A per-step `(vocab, hidden)` copy defeats `teacher_device_map: cpu` entirely."""
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    trainer = _bare_trainer(teacher, logits_weight=0.5)
    seen: list = []
    original = distill_mod._chunked_ce_kd

    def spy(*args, **kwargs):
        seen.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(distill_mod, "_chunked_ce_kd", spy)

    trainer.compute_loss(student, _batch())

    assert seen and seen[0] is teacher.lm_head.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_the_chunked_kd_runs_the_teacher_head_on_the_teachers_device():
    """`teacher_device_map: cpu` means only per-chunk log-probs cross the bus."""
    torch.manual_seed(0)
    student_hidden = torch.randn(6, 8, device="cuda", requires_grad=True)
    student_weight = torch.randn(12, 8, device="cuda", requires_grad=True)
    teacher_hidden = torch.randn(6, 8)
    teacher_weight = torch.randn(12, 8)
    targets = torch.randint(0, 12, (6,), device="cuda")

    ce, kd = _chunked_ce_kd(
        student_hidden,
        teacher_hidden,
        student_weight,
        None,
        teacher_weight,
        None,
        targets,
        2.0,
        4,
    )
    (ce + kd).backward()

    assert teacher_weight.device.type == "cpu"
    assert ce.device.type == "cuda" and kd.device.type == "cuda"
    assert student_weight.grad is not None


def test_compute_loss_applies_the_axolotl_input_fixups():
    """Replacing `compute_loss` outright must not drop the shared preprocessing."""
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(copy.deepcopy(student), logits_weight=0.5)
    trainer.args.include_tkps = True
    student.train()

    trainer.compute_loss(student, _batch())

    assert float(trainer.state.tokens["total"].item()) == 2 * 7


def test_compute_loss_does_not_mutate_the_callers_batch():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(copy.deepcopy(student), logits_weight=0.5)
    batch = _batch()
    keys = set(batch)

    trainer.compute_loss(student, batch)

    assert set(batch) == keys


# --------------------------------------------------------- anchored KD schedule


def test_distill_schedule_defaults_to_the_constant_blend():
    trainer = _bare_trainer()
    assert trainer.distill_schedule == "constant"
    assert trainer.anchor_start == 0.9
    assert trainer.distill_multiplier == 1.0


def test_anchored_schedule_starts_with_every_kd_term_off():
    trainer = _bare_trainer(schedule="anchored", anchor_start=0.8)
    assert trainer.distill_multiplier == 0.0
    assert trainer.anchor_start == 0.8


def test_loss_is_pure_ce_before_the_anchor():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(
        _tiny_llama(seed=1),
        schedule="anchored",
        logits_weight=0.5,
        hidden_weight=1.5,
        attn_relation_layer=-1,
    )
    batch = _batch()

    loss = trainer.compute_loss(student, batch)

    assert loss.item() == pytest.approx(_reference_ce(student, batch).item(), rel=1e-6)
    # no teacher forward means no relation hooks and no captures either
    assert not student.model.layers[-1].self_attn.q_proj._forward_hooks
    assert trainer._captured == {}


def test_the_pure_ce_step_keeps_the_trimmed_outputs_and_gradients():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(schedule="anchored")

    loss, outputs = trainer.compute_loss(student, _batch(), return_outputs=True)
    loss.backward()

    assert outputs.logits.shape[1] == 1
    assert any(param.grad is not None for param in student.parameters())


def test_the_teacher_is_not_loaded_before_the_anchor(monkeypatch):
    """The CE phase is the memory win: nothing may pull the teacher in early."""
    student = _tiny_llama(seed=0)
    calls: list[str] = []
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda name, **kwargs: calls.append(name) or _tiny_llama(seed=1),
    )
    trainer = _bare_trainer(teacher_model="dummy/teacher", schedule="anchored")

    for _ in range(3):
        trainer.compute_loss(student, _batch())

    assert calls == []
    assert trainer._teacher is None
    assert trainer._teacher_ready is False

    trainer.set_distill_multiplier(0.5)
    trainer.compute_loss(student, _batch())

    assert calls == ["dummy/teacher"]
    assert trainer._teacher is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_ce_phase_never_allocates_the_teacher(monkeypatch):
    student = _tiny_llama(seed=0).cuda()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda name, **kwargs: _tiny_llama(seed=1),
    )
    trainer = _bare_trainer(teacher_model="dummy/teacher", schedule="anchored")
    batch = {key: value.cuda() for key, value in _batch().items()}
    teacher_bytes = sum(
        param.numel() * param.element_size()
        for param in _tiny_llama(seed=1).parameters()
    )

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with torch.no_grad():
        trainer.compute_loss(student, batch)
        torch.cuda.synchronize()
        after_ce = torch.cuda.memory_allocated()
        trainer.set_distill_multiplier(1.0)
        trainer.compute_loss(student, batch)
        torch.cuda.synchronize()

    assert after_ce - before < teacher_bytes
    assert torch.cuda.memory_allocated() - after_ce >= teacher_bytes


def test_every_kd_term_follows_the_schedule_multiplier():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(
        _tiny_llama(seed=1),
        schedule="anchored",
        logits_weight=0.5,
        hidden_weight=1.5,
        attn_relation_layer=-1,
    )
    batch = _batch()

    with torch.no_grad():
        pure_ce = trainer.compute_loss(student, batch).item()
        trainer.set_distill_multiplier(1.0)
        blended = trainer.compute_loss(student, batch).item()
        trainer.set_distill_multiplier(0.5)
        mid_ramp = trainer.compute_loss(student, batch).item()

    assert blended != pytest.approx(pure_ce, rel=1e-3)
    # every term is linear in the multiplier, logits/hidden/relation alike
    assert mid_ramp == pytest.approx((pure_ce + blended) / 2, rel=1e-5)


def test_the_anchored_tail_matches_the_constant_blend_exactly():
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    batch = _batch()
    constant = _bare_trainer(
        copy.deepcopy(teacher), logits_weight=0.5, hidden_weight=1.5
    )
    anchored = _bare_trainer(
        copy.deepcopy(teacher),
        schedule="anchored",
        logits_weight=0.5,
        hidden_weight=1.5,
    )
    anchored.set_distill_multiplier(1.0)

    with torch.no_grad():
        assert (
            anchored.compute_loss(student, batch).item()
            == constant.compute_loss(student, batch).item()
        )


def test_constant_schedule_ignores_the_anchor_knobs():
    student = _tiny_llama(seed=0)
    teacher = _tiny_llama(seed=1)
    trainer = _bare_trainer(teacher, logits_weight=0.5, anchor_start=0.1)
    batch = _batch()

    loss = trainer.compute_loss(student, batch)

    reference = _reference_loss(student, copy.deepcopy(teacher), batch, 0.5, 2.0)
    assert loss.item() == pytest.approx(reference.item(), rel=1e-5, abs=1e-6)
    assert set(trainer._stored_metrics["train"]) == {"ternary/ce", "ternary/kd_logits"}


def test_anchored_metrics_carry_the_schedule_weight():
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(
        _tiny_llama(seed=1), schedule="anchored", logits_weight=0.5, hidden_weight=1.0
    )

    trainer.compute_loss(student, _batch())
    assert set(trainer._stored_metrics["train"]) == {
        "ternary/ce",
        "ternary/distill_weight",
    }

    trainer.set_distill_multiplier(0.25)
    trainer.compute_loss(student, _batch())
    assert set(trainer._stored_metrics["train"]) == {
        "ternary/ce",
        "ternary/kd_logits",
        "ternary/kd_hidden",
        "ternary/distill_weight",
    }
    assert trainer._stored_metrics["train"]["ternary/distill_weight"]["values"] == [
        0.0,
        0.25,
    ]


def test_train_defers_the_teacher_load_under_the_anchored_schedule(monkeypatch):
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(None, schedule="anchored")
    trainer.model = student
    loaded: list[str] = []
    monkeypatch.setattr(
        TernaryDistillTrainer,
        "_load_teacher",
        lambda self, s: loaded.append("loaded") or _tiny_llama(seed=1),
    )
    monkeypatch.setattr(transformers.Trainer, "train", lambda self, *args, **kwargs: 0)

    trainer.train()

    assert loaded == []
    assert trainer._teacher is None


def test_train_loads_the_teacher_eagerly_under_deepspeed(monkeypatch, caplog):
    """ZeRO-3 partitions a mid-run `from_pretrained`, so the deferral cannot apply."""
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(None, schedule="anchored")
    trainer.model = student
    trainer.is_deepspeed_enabled = True
    loaded: list[str] = []
    monkeypatch.setattr(
        TernaryDistillTrainer,
        "_load_teacher",
        lambda self, s: loaded.append("loaded") or _tiny_llama(seed=1),
    )
    monkeypatch.setattr(transformers.Trainer, "train", lambda self, *args, **kwargs: 0)

    with caplog.at_level("WARNING"):
        trainer.train()

    assert loaded == ["loaded"]
    assert any("DeepSpeed" in record.message for record in caplog.records)


def test_the_anchored_trainer_installs_the_schedule_callback(tmp_path):
    from axolotl.core.training_args import AxolotlTrainingArguments
    from axolotl.integrations.ternary.callbacks import DistillAnchorCallback

    def _trainer(**knobs):
        args = AxolotlTrainingArguments(
            output_dir=str(tmp_path), max_steps=1, report_to=[], save_strategy="no"
        )
        for key, value in knobs.items():
            setattr(args, f"ternary_distill_{key}", value)
        return TernaryDistillTrainer(
            model=_tiny_llama(seed=0), args=args, train_dataset=None
        )

    anchored = _trainer(schedule="anchored", anchor_start=0.75)
    callbacks = [
        callback
        for callback in anchored.callback_handler.callbacks
        if isinstance(callback, DistillAnchorCallback)
    ]
    assert len(callbacks) == 1
    assert callbacks[0].trainer is anchored
    assert callbacks[0].anchor_start == 0.75
    assert anchored.distill_multiplier == 0.0

    constant = _trainer()
    assert not any(
        isinstance(callback, DistillAnchorCallback)
        for callback in constant.callback_handler.callbacks
    )
    assert constant.distill_multiplier == 1.0


def test_the_schema_values_reach_the_trainer_knobs():
    from axolotl.integrations.ternary.args import TernaryDistillConfig

    config = TernaryDistillConfig(
        mode="inprocess", schedule="anchored", anchor_start=0.95
    )
    trainer = _bare_trainer(schedule=config.schedule, anchor_start=config.anchor_start)
    assert (trainer.distill_schedule, trainer.anchor_start) == ("anchored", 0.95)
    assert trainer.distill_multiplier == 0.0

    # the schema refuses to anchor a KD term that no distill mode ever computes
    with pytest.raises(pydantic.ValidationError, match="anchors a KD term"):
        TernaryDistillConfig(schedule="anchored")


def test_train_loads_the_teacher_before_the_loop_and_releases_hooks(monkeypatch):
    """Loading inside `training_step` puts the teacher inside ZeRO-3's init scope."""
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(None, attn_relation_layer=-1)
    trainer.model = student
    loaded: list[str] = []
    monkeypatch.setattr(
        TernaryDistillTrainer,
        "_load_teacher",
        lambda self, s: loaded.append("loaded") or _tiny_llama(seed=1),
    )
    monkeypatch.setattr(
        transformers.Trainer,
        "train",
        lambda self, *args, **kwargs: (
            self._register_relation_hooks(student, self._teacher),
            "done",
        )[1],
    )

    assert trainer.train() == "done"

    assert loaded == ["loaded"]
    assert trainer._hook_handles == []


# ------------------------------------------------------- hidden-loss variants


HIDDEN_LOSSES = ("cosine", "mse", "huber")


def _hidden(loss: str, student, teacher, mask=None, **knobs):
    trainer = _bare_trainer(hidden_weight=1.0, hidden_loss=loss, **knobs)
    return float(trainer.hidden_feature_kd(student, teacher, mask=mask))


def test_hidden_loss_defaults_to_cosine():
    trainer = _bare_trainer()

    assert trainer.hidden_loss == "cosine"
    assert trainer.hidden_huber_delta == 1.0


def test_the_default_dispatches_to_the_cosine_implementation():
    """The pre-existing behaviour has to stay byte-identical."""
    torch.manual_seed(0)
    student = torch.randn(2, 5, 16)
    teacher = torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    trainer = _bare_trainer(hidden_weight=1.0)

    assert torch.equal(
        trainer.hidden_feature_kd(student, teacher, mask=mask),
        trainer.hidden_cosine_kd(student, teacher, mask=mask),
    )


def test_only_the_raw_losses_see_a_pure_magnitude_change():
    """Same direction, three times the norm: cosine is blind to it by construction."""
    torch.manual_seed(0)
    student = torch.randn(1, 4, 8)
    teacher = student * 3.0

    assert _hidden("cosine", student, teacher) == pytest.approx(0.0, abs=1e-6)
    assert _hidden("mse", student, teacher) > 0.0
    assert _hidden("huber", student, teacher) > 0.0


def test_huber_bounds_an_outlier_channel_that_mse_lets_dominate():
    torch.manual_seed(0)
    student = torch.randn(1, 4, 32)
    teacher = student.clone()
    teacher[..., 0] += 50.0

    mse = _hidden("mse", student, teacher)
    huber = _hidden("huber", student, teacher)

    assert huber < mse
    # squared vs linear past the elbow: the gap grows with the outlier
    further = teacher.clone()
    further[..., 0] += 50.0
    assert (
        _hidden("mse", student, further) / mse
        > _hidden("huber", student, further) / huber
    )


def test_the_huber_elbow_moves_the_loss():
    torch.manual_seed(0)
    student = torch.randn(1, 4, 8)
    teacher = student + 4.0

    tight = _hidden("huber", student, teacher, hidden_huber_delta=0.5)
    loose = _hidden("huber", student, teacher, hidden_huber_delta=4.0)

    assert tight < loose


def test_huber_matches_mse_below_the_elbow():
    """Inside the quadratic region huber is `0.5·(s - t)²`."""
    student = torch.zeros(1, 1, 4)
    teacher = torch.full((1, 1, 4), 0.25)

    huber = _hidden("huber", student, teacher, hidden_huber_delta=1.0)
    mse = _hidden("mse", student, teacher)

    assert huber == pytest.approx(0.5 * mse)


@pytest.mark.parametrize("loss", HIDDEN_LOSSES)
def test_every_variant_is_zero_on_identical_streams(loss):
    torch.manual_seed(0)
    student = torch.randn(2, 3, 8)

    assert _hidden(loss, student, student.clone()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("loss", HIDDEN_LOSSES)
def test_masking_is_identical_across_variants(loss):
    """Which tokens count must not depend on which variant reduced the features."""
    torch.manual_seed(0)
    student = torch.randn(1, 4, 8)
    teacher = torch.randn(1, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0]])

    masked = _hidden(loss, student, teacher, mask=mask)
    kept_only = _hidden(loss, student[:, :2], teacher[:, :2])

    assert masked == pytest.approx(kept_only, rel=1e-6)


@pytest.mark.parametrize("loss", HIDDEN_LOSSES)
def test_every_variant_carries_a_gradient(loss):
    torch.manual_seed(0)
    student = torch.randn(1, 4, 8, requires_grad=True)
    teacher = torch.randn(1, 4, 8)
    trainer = _bare_trainer(hidden_weight=1.0, hidden_loss=loss)

    trainer.hidden_feature_kd(student, teacher).backward()

    assert student.grad is not None
    assert float(student.grad.abs().sum()) > 0.0


def test_an_unknown_variant_raises():
    trainer = _bare_trainer(hidden_weight=1.0, hidden_loss="cosine")
    trainer.hidden_loss = "kullback"

    with pytest.raises(ValueError, match="unknown ternary.distill.hidden_loss"):
        trainer.hidden_feature_kd(torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))


@pytest.mark.parametrize("loss", HIDDEN_LOSSES)
def test_the_variant_reaches_the_loss_through_compute_loss(loss):
    student = _tiny_llama(seed=0)
    trainer = _bare_trainer(
        _tiny_llama(seed=1), logits_weight=0.5, hidden_weight=1.0, hidden_loss=loss
    )

    loss_value = trainer.compute_loss(student, _batch())

    assert torch.isfinite(loss_value)
    assert trainer._stored_metrics["train"]["ternary/kd_hidden"]["values"]


def test_chunked_harmonizes_fp32_hidden_with_low_precision_heads():
    """accelerate returns fp32 hidden states while pure-bf16 models keep bf16 heads."""
    torch.manual_seed(3)
    tokens, hidden, vocab = 11, 16, 29
    student_hidden = torch.randn(tokens, hidden, requires_grad=True)
    teacher_hidden = torch.randn(tokens, hidden)
    head = nn.Linear(hidden, vocab, bias=False).to(torch.bfloat16)
    teacher_head = nn.Linear(hidden, vocab, bias=False).to(torch.bfloat16)
    targets = torch.randint(0, vocab, (tokens,))

    ce, kd = _chunked_ce_kd(
        student_hidden,
        teacher_hidden,
        head.weight,
        None,
        teacher_head.weight,
        None,
        targets,
        1.0,
        8,
    )
    assert torch.isfinite(ce) and torch.isfinite(kd)
    (ce + kd).backward()
    assert student_hidden.grad is not None
    assert head.weight.grad is not None
    assert torch.isfinite(student_hidden.grad).all()
