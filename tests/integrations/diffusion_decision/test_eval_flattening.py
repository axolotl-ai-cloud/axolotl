"""Token-budget (flattened) evaluation batches and final-pass-only evaluation reads."""

from __future__ import annotations

import random

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard

from axolotl.integrations.diffusion_decision._util import token_budget_eval_packing
from axolotl.integrations.diffusion_decision.datasets import DecisionDataset
from axolotl.integrations.diffusion_decision.loss import decision_example_from_canvas
from axolotl.integrations.diffusion_decision.trainer import (
    _TokenBudgetEvalBatchSampler,
    _uses_token_budget_batches,
)
from axolotl.integrations.diffusion_decision.training_collator import (
    DecisionTrainingCollator,
)

from tests.integrations.diffusion_decision.helpers import make_canvas
from tests.integrations.diffusion_decision.test_trainer import (
    _full_sequence_inputs,
    _loop_args,
    _LoopTrainer,
    _multistep,
    _SelectedTinyNativeModel,
    _spec,
    _TinyNativeModel,
)
from tests.native_source_fixtures import native_source_fixture_path

_SEQUENCE_LEN = 40
_EVAL_BATCH = 2


def _batches(sampler) -> list[list[int]]:
    return [list(batch) for batch in sampler]


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_token_budget_batches_keep_order_budget_and_rank_multiple(world_size):
    lengths = [random.Random(index).randint(1, 50) for index in range(97)]

    batches = _batches(_TokenBudgetEvalBatchSampler(lengths, 64, world_size))

    assert [index for batch in batches for index in batch] == list(range(97))
    assert all(sum(lengths[index] for index in batch) <= 64 for batch in batches)
    assert len(batches) % world_size == 0
    assert len(batches) < len(lengths)


def test_token_budget_batches_fill_greedily_without_splitting_rows():
    batches = _batches(_TokenBudgetEvalBatchSampler([30, 30, 5, 40, 60, 1], 64))

    assert batches == [[0, 1], [2, 3], [4, 5]]


def test_token_budget_batches_reject_rows_over_budget():
    with pytest.raises(ValueError, match="evaluation row 1 has 65 tokens"):
        _TokenBudgetEvalBatchSampler([3, 65], 64)


@pytest.mark.parametrize("world_size", [2, 4])
def test_accelerate_shards_token_budget_batches_without_repeating_rows(world_size):
    lengths = [random.Random(index).randint(1, 50) for index in range(53)]
    sampler = _TokenBudgetEvalBatchSampler(lengths, 64, world_size)

    shards = [
        _batches(
            BatchSamplerShard(
                sampler,
                num_processes=world_size,
                process_index=rank,
                even_batches=True,
            )
        )
        for rank in range(world_size)
    ]

    seen = [index for shard in shards for batch in shard for index in batch]
    assert sorted(seen) == list(range(53))
    assert len({len(shard) for shard in shards}) == 1


@pytest.mark.parametrize(
    ("sample_packing", "eval_sample_packing", "expected"),
    [
        (False, None, False),
        (False, False, False),
        (False, True, True),
        (True, True, False),
        (True, None, False),
    ],
)
def test_eval_sample_packing_alone_selects_token_budget_eval_batches(
    sample_packing, eval_sample_packing, expected
):
    cfg = {"sample_packing": sample_packing, "batch_flattening": True}
    if eval_sample_packing is not None:
        cfg["eval_sample_packing"] = eval_sample_packing

    assert token_budget_eval_packing(cfg) is expected


@pytest.mark.parametrize(
    ("sample_packing", "eval_sample_packing", "expected"),
    [
        (False, None, "sequential"),
        (False, False, "sequential"),
        (False, True, "token_budget"),
        (True, True, "core_multipack"),
    ],
)
def test_exactly_one_eval_packing_mechanism_applies(
    monkeypatch, sample_packing, eval_sample_packing, expected
):
    from torch.utils.data import SequentialSampler

    from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer

    from tests.integrations.diffusion_decision.test_trainer import _harness

    monkeypatch.setattr(
        AxolotlDiffusionTrainer,
        "_get_eval_sampler",
        lambda _self, dataset: "core_multipack",
    )
    dataset = DecisionDataset(_eval_rows(5, 128), {"per_batch_stratified": False})
    trainer = _harness()
    trainer.args.sample_packing = sample_packing
    trainer.args.eval_sample_packing = eval_sample_packing
    trainer.args.per_device_eval_batch_size = _EVAL_BATCH
    trainer.axolotl_cfg = {
        "sequence_len": _SEQUENCE_LEN,
        "eval_batch_size": _EVAL_BATCH,
        "sample_packing": sample_packing,
        "eval_sample_packing": eval_sample_packing,
        "diffusion_decision": {},
    }

    sampler = trainer._get_eval_sampler(dataset)

    observed = (
        "core_multipack"
        if sampler == "core_multipack"
        else "token_budget"
        if isinstance(sampler, _TokenBudgetEvalBatchSampler)
        else "sequential"
        if isinstance(sampler, SequentialSampler)
        else sampler
    )
    assert observed == expected


@pytest.mark.parametrize("selected", [False, True])
def test_evaluation_reads_only_the_final_pass_when_nothing_is_committed(selected):
    model_class = _SelectedTinyNativeModel if selected else _TinyNativeModel
    reference_model = _TinyNativeModel()
    model = model_class()
    model.load_state_dict(reference_model.state_dict())
    decision = {"labels": {"label_softmax": "both", "brier_weight": 0.1}}
    inputs = _full_sequence_inputs()

    with torch.no_grad():
        torch.manual_seed(3)
        reference = _multistep(1, k_max=1, decision=decision).compute_loss(
            reference_model, inputs
        )
        model.eval()
        torch.manual_seed(3)
        trainer = _multistep(2, k_max=2, decision=decision)
        loss = trainer.compute_loss(model, inputs)

    assert len(reference_model.forward_states) == 1
    assert len(model.forward_states) == 1
    torch.testing.assert_close(
        model.forward_states[0], reference_model.forward_states[0]
    )
    torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-6)
    eval_totals = trainer._decision_metric_totals("eval")["decision/all"]
    assert eval_totals["examples"] == 1
    if selected:
        assert model.selected_shapes == [torch.Size([1, 2, 11])]


def test_training_skips_unroll_passes_that_commit_nothing():
    model = _TinyNativeModel()

    _multistep(2, k_max=2).compute_loss(model, _full_sequence_inputs())

    assert model.forward_grad_enabled == [True]


def _eval_rows(count: int, vocab_size: int) -> list[dict[str, object]]:
    generator = random.Random(11)
    rows = []
    for index in range(count):
        prompt = tuple(
            generator.randrange(3, vocab_size - 30)
            for _ in range(generator.choice([2, 3, 9, 17, 31]))
        )
        canvas_ids = tuple(generator.randrange(3, vocab_size - 30) for _ in range(6))
        questions = 1 + index % 3
        positions = tuple(range(1, 1 + 2 * questions, 2))
        targets = tuple(
            {"kind": "hard", "gold_idx": (index + question) % 3}
            if (index + question) % 2
            else {"kind": "dist", "probs": (0.2, 0.5, 0.3)}
            for question in range(questions)
        )
        canvas = make_canvas(
            prompt,
            canvas_ids,
            positions,
            f"eval-{index}",
            allowed=(5, 6, 7),
            targets=targets,
        )
        rows.append(
            {
                "canvas": canvas,
                "length": len(prompt) + len(canvas_ids),
                "decision_example": decision_example_from_canvas(
                    canvas, source_weight=1.0 + (index % 4) / 4
                ),
                "source": ("alpha", "beta", "gamma")[index % 3],
            }
        )
    return rows


class _MultistepEvalTrainer(_LoopTrainer):
    """Real HF evaluation loop with K=2 reads sampled at two passes."""

    def _native_unroll_settings(self):
        return 2, False

    @staticmethod
    def _sample_native_unroll_steps(k_max, device):
        del k_max, device
        return 2


def _tiny_nemotron():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

    source = native_source_fixture_path("nemotron")
    if source is None:
        pytest.skip("native Nemotron source fixture unavailable")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(5)
    return resolve_nemotron_model_class(str(source))(config).float()


def _evaluate(tmp_path, model, rows, *, flatten: bool, selected: bool):
    spec = _spec()
    dataset = DecisionDataset(rows, {"per_batch_stratified": False})
    collator = DecisionTrainingCollator(spec)
    if not selected and getattr(model, "supports_selected_logits", False):
        model.supports_selected_logits = False
    trainer = _MultistepEvalTrainer(
        model=model,
        args=_loop_args(tmp_path, per_device_eval_batch_size=_EVAL_BATCH),
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=collator,
        eval_data_collator=collator,
        spec=spec,
    )
    trainer.axolotl_cfg = {
        "seed": 29,
        "sequence_len": _SEQUENCE_LEN,
        "eval_batch_size": _EVAL_BATCH,
        "batch_flattening": True,
        "sample_packing": False,
        "eval_sample_packing": flatten,
        "diffusion_decision": {
            "read_fraction": 1.0,
            "labels": {"label_softmax": "both", "brier_weight": 0.1},
        },
    }
    trainer.args.eval_sample_packing = flatten
    trainer.post_set_axolotl_cfg()
    dataloader = trainer.get_eval_dataloader()
    assert _uses_token_budget_batches(dataloader) is flatten
    budget = _SEQUENCE_LEN * _EVAL_BATCH
    batches = list(dataloader)
    assert all(batch["input_ids"].shape[0] == 1 for batch in batches)
    assert all(batch["input_ids"].shape[1] <= budget for batch in batches)
    if flatten:
        assert len(batches) < (len(rows) + _EVAL_BATCH - 1) // _EVAL_BATCH
    else:
        assert len(batches) == (len(rows) + _EVAL_BATCH - 1) // _EVAL_BATCH
    metrics = trainer.evaluate()
    return metrics, len(batches)


def _comparable(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if not any(
            marker in key
            for marker in ("runtime", "per_second", "epoch", "preparation_time")
        )
    }


@pytest.mark.parametrize("model_kind", ["bias", "nemotron"])
def test_flattened_selected_evaluation_metrics_match_fixed_full_logit_batches(
    tmp_path, model_kind
):
    def make_model():
        if model_kind == "nemotron":
            return _tiny_nemotron()
        model = _SelectedTinyNativeModel(vocab_size=128)
        model.slopes = False
        return model

    rows = _eval_rows(23, 128)
    results = {}
    for flatten in (False, True):
        for selected in (False, True):
            metrics, steps = _evaluate(
                tmp_path / f"{flatten}-{selected}",
                make_model(),
                rows,
                flatten=flatten,
                selected=selected,
            )
            results[(flatten, selected)] = _comparable(metrics)

    reference = results[(False, False)]
    assert reference["eval_decision/all/examples"] == len(rows)
    assert {"eval_decision/alpha/loss", "eval_decision/gamma/brier_loss"} <= set(
        reference
    )
    assert reference["eval_loss"] == pytest.approx(
        reference["eval_decision/all/loss"], rel=1e-6, abs=1e-6
    )
    for key, observed in results.items():
        assert observed.keys() == reference.keys(), key
        for name, value in reference.items():
            assert observed[name] == pytest.approx(value, rel=1e-5, abs=1e-5), (
                key,
                name,
            )


@pytest.mark.parametrize("batch_flattening", [False, True])
def test_eval_sample_packing_alone_caps_the_eval_collator_at_the_eval_budget(
    batch_flattening,
):
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision.training_collator import (
        decision_collator_for_config,
    )
    from axolotl.utils.dict import DictDefault

    _collator, kwargs = decision_collator_for_config(
        DictDefault(
            {
                "model_config_type": "nemotron_labs_diffusion",
                "tokenizer": SimpleNamespace(pad_token_id=0),
                "sample_packing": False,
                "eval_sample_packing": True,
                "batch_flattening": batch_flattening,
                "micro_batch_size": 1,
                "eval_batch_size": 8,
                "sequence_len": 128,
                "diffusion_lm": {"from_causal_lm": False},
            }
        ),
        is_eval=True,
    )
    assert kwargs["physical_payload_capacity"] == 1024
