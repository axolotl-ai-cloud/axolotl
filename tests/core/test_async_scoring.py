"""Tests for trainer-side scoring on async GRPO and async EBFT on top of it."""

import itertools
import multiprocessing as mp
import queue
import types
from collections import defaultdict

import numpy as np
import pytest
import torch
from async_test_utils import StubTransfer, StubWorker, reference_tree_attention
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    TrainerCallback,
)
from trl.experimental.async_grpo import async_grpo_trainer, packing
from trl.experimental.async_grpo.async_rollout_worker import RolloutGroup
from trl.trainer.utils import pad

from axolotl.core.trainers.ebft.async_rollout import (
    EBFTRolloutLoop,
    reference_ids,
)
from axolotl.core.trainers.ebft.async_trainer import (
    AxolotlAsyncEBFTConfig,
    AxolotlAsyncEBFTTrainer,
    EBFTScorer,
    feature_layer_indices,
)
from axolotl.core.trainers.ebft.trainer import EBFTMixin
from axolotl.core.trainers.grpo.async_progress import RolloutProgress, RolloutStop
from axolotl.core.trainers.grpo.async_rollout import SCORE_PAYLOAD
from axolotl.core.trainers.grpo.async_scoring import (
    GroupedQueue,
    ScoredRow,
    StepPlanner,
    TrainerScorer,
    chunk_inputs,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPTS = [
    [{"role": "user", "content": "What is two plus two?"}],
    [{"role": "user", "content": "Name a primary color."}],
]
COMPLETIONS = [
    ["The answer is four.", "Four", "It is four, of course."],
    ["Blue", "Red is a primary color.", "Yellow, like the sun."],
]
GROUND_TRUTH = ["Two plus two is four.", "Blue is a primary color."]
LAYER_FRACTIONS = [0.25, 0.5, 0.75]


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    return tokenizer


@pytest.fixture(autouse=True)
def _cpu_tree_attention(monkeypatch):
    from accelerate.state import PartialState

    PartialState()
    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    packing.register_tree_attention()


def tiny_model(tokenizer, attn_implementation="sdpa"):
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        attn_implementation=attn_implementation,
    ).eval()


def rollout_loop(tokenizer, rows, **kwargs):
    return EBFTRolloutLoop(
        model_name="m",
        dataset=Dataset.from_list(rows),
        reward_funcs=[],
        processing_class=tokenizer,
        eos_token_ids=[tokenizer.eos_token_id],
        rollout_buffer=queue.Queue(),
        model_version_value=mp.Value("i", 0),
        heartbeat_value=mp.Value("d", 0.0),
        failed_event=mp.Event(),
        exception_info_queue=mp.Queue(),
        metrics_queue=mp.Queue(),
        trainer_scored=True,
        **kwargs,
    )


def run_group(loop, group_id, row, sample, num_generations):
    """Generate and score one group in-process, with ``sample`` standing in for vLLM."""
    loop._complete = sample
    loop._group_rows[group_id] = row

    async def go():
        results = [
            await loop._generate_one(row["prompt"], {}, [], group_id)
            for _ in range(num_generations)
        ]
        group = RolloutGroup(
            prompts=[row["prompt"]] * num_generations,
            reward_kwargs={
                key: [value] * num_generations
                for key, value in row.items()
                if key != "prompt"
            },
            completions=[r[0] for r in results],
            completions_ids=[r[1] for r in results],
            completions_sequences=[r[2] for r in results],
            tool_call_counts=[0] * num_generations,
            tool_failure_counts=[0] * num_generations,
            model_version=0,
            group_id=group_id,
            env_rewards=[],
            rollout_rewards=[None] * num_generations,
        )
        return await loop._score_group(group)

    return loop._loop.run_until_complete(go())


def scripted_sample(tokenizer, replies):
    """A stand-in for vLLM that answers each prompt with the next of its replies."""
    queues = defaultdict(list)
    for prompt_ids, text in replies:
        queues[tuple(prompt_ids)].append(
            tokenizer(text, add_special_tokens=False)["input_ids"]
        )

    async def sample(prompt_ids, max_tokens):
        ids = queues[tuple(prompt_ids)].pop(0) + [tokenizer.eos_token_id]
        return ids, [-0.5] * len(ids)

    return sample


def score_with(scorer, samples):
    """Collate one row of ``samples`` and score its groups, as one rank would."""
    row = ScoredRow()
    uids = itertools.count()
    by_group = defaultdict(list)
    for sample in samples:
        payload = sample.metrics[SCORE_PAYLOAD]
        entry = {
            "input_ids": sample.input_ids,
            "completion_mask": sample.completion_mask,
            "group_id": sample.group_id,
            "score_payload": payload,
            "score_uid": next(uids),
        }
        row.add(entry)
        by_group[sample.group_id].append(entry)
    for group_id, entries in by_group.items():
        for extra in scorer.group_extras(entries):
            row.extras.append(
                {**extra, "scorer": 0, "group_id": group_id, "uid": next(uids)}
            )
    inputs = {
        key: pad(tensors, padding_value=padding)
        for key, (tensors, padding) in scorer.collate([row], 0).items()
    }
    features = {}
    for chunk in chunk_inputs(inputs):
        uid_list, chunk_features = scorer.features(None, chunk)
        features.update(zip(uid_list.tolist(), chunk_features, strict=True))
    rewards = []
    for group_id, entries in by_group.items():
        reference = [e["uid"] for e in row.extras if e["group_id"] == group_id]
        group_rewards, _ = scorer.group_rewards(
            torch.stack([features[e["score_uid"]] for e in entries]),
            torch.stack([features[uid] for uid in reference]),
        )
        rewards.append(group_rewards)
    return torch.cat(rewards)


def feature_matching_reward(tokenizer, model, embed_method, use_whitening, **kwargs):
    """``EBFTMixin._feature_matching_reward`` on a stand-in trainer."""
    args = types.SimpleNamespace(
        ebft_embed_method=embed_method,
        ebft_use_whitening=use_whitening,
        ebft_alignment_coef=1.0,
        ebft_diversity_coef=0.5,
        max_completion_length=32,
        temperature=1.0,
        generation_kwargs=None,
    )
    trainer = types.SimpleNamespace(
        accelerator=types.SimpleNamespace(device=torch.device("cpu")),
        args=args,
        num_generations=len(COMPLETIONS[0]),
        processing_class=tokenizer,
        _share_feature_weights=False,
        feature_network=model,
        feature_layer_indices=feature_layer_indices(model, LAYER_FRACTIONS),
        model=types.SimpleNamespace(training=True),
        _metrics={"train": defaultdict(list)},
        **kwargs,
    )
    trainer._sequential_rollout = types.MethodType(
        EBFTMixin._sequential_rollout, trainer
    )
    prompts = [p for p, cs in zip(PROMPTS, COMPLETIONS, strict=True) for _ in cs]
    completions = [c for cs in COMPLETIONS for c in cs]
    ground_truth = [
        g for g, cs in zip(GROUND_TRUTH, COMPLETIONS, strict=True) for _ in cs
    ]
    return torch.tensor(
        EBFTMixin._feature_matching_reward(
            trainer,
            prompts,
            completions,
            ground_truth=ground_truth,
            **kwargs.get("reward_kwargs", {}),
        )
    )


# A budget below a group's packed sequences scores them over several forwards.
@pytest.mark.parametrize("scoring_budget", [0, 48])
@pytest.mark.parametrize("use_whitening", [False, True])
@pytest.mark.parametrize(
    "embed_method", ["last_token", "mean_pooling", "completion_mean", "concat"]
)
def test_rewards_match_feature_matching_reward(
    tokenizer, embed_method, use_whitening, scoring_budget
):
    model = tiny_model(tokenizer)
    expected = feature_matching_reward(tokenizer, model, embed_method, use_whitening)

    tree_model = tiny_model(tokenizer, attn_implementation=packing.TREE_ATTENTION)
    tree_model.load_state_dict(model.state_dict())
    scorer = EBFTScorer(
        feature_layer_indices(model, LAYER_FRACTIONS),
        embed_method,
        use_whitening,
        alignment_coef=1.0,
        diversity_coef=0.5,
        feature_network=tree_model,
    )
    scorer.token_budget = scoring_budget
    samples = []
    for group_id, (prompt, completions, truth) in enumerate(
        zip(PROMPTS, COMPLETIONS, GROUND_TRUTH, strict=True)
    ):
        prompt_ids = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_dict=False
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
        for text in completions:
            # Identical inputs: sampled tokens that re-tokenize to themselves.
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            assert (
                prompt_ids + ids
                == tokenizer(prompt_text + text, add_special_tokens=False)["input_ids"]
            )
        loop = rollout_loop(tokenizer, [{"prompt": prompt, "ground_truth": truth}])
        samples += run_group(
            loop,
            group_id,
            {"prompt": prompt, "ground_truth": truth},
            scripted_sample(tokenizer, [(prompt_ids, text) for text in completions]),
            len(completions),
        )

    torch.testing.assert_close(
        score_with(scorer, samples),
        expected,
        atol=1e-3 if use_whitening else 1e-4,
        rtol=1e-3 if use_whitening else 1e-4,
    )


@pytest.mark.parametrize("scoring_budget", [0, 48])
def test_multi_turn_rewards_match_feature_matching_reward(tokenizer, scoring_budget):
    turns = [
        {"role": "user", "content": "And three plus three?"},
        {"role": "assistant", "content": "Six."},
    ]
    later = "Three plus three is six."
    model = tiny_model(tokenizer)

    class Client:
        def chat(self, messages, **kwargs):
            return {
                "completion_ids": [
                    tokenizer(later, add_special_tokens=False)["input_ids"]
                ]
            }

    turns_per_prompt = [turns] * len(PROMPTS)
    expected = feature_matching_reward(
        tokenizer,
        model,
        "mean_pooling",
        False,
        vllm_generation=types.SimpleNamespace(vllm_client=Client()),
        reward_kwargs={"remaining_turns": turns_per_prompt},
    )

    tree_model = tiny_model(tokenizer, attn_implementation=packing.TREE_ATTENTION)
    tree_model.load_state_dict(model.state_dict())
    scorer = EBFTScorer(
        feature_layer_indices(model, LAYER_FRACTIONS),
        "mean_pooling",
        False,
        alignment_coef=1.0,
        diversity_coef=0.5,
        feature_network=tree_model,
    )
    scorer.token_budget = scoring_budget
    samples = []
    for group_id, (prompt, completions, truth) in enumerate(
        zip(PROMPTS, COMPLETIONS, GROUND_TRUTH, strict=True)
    ):
        row = {"prompt": prompt, "ground_truth": truth, "remaining_turns": turns}
        prompt_ids = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_dict=False
        )
        replies = []
        for text in completions:
            replies.append((prompt_ids, text))
            conversation = [*prompt, {"role": "assistant", "content": text}, turns[0]]
            replies.append(
                (
                    tokenizer.apply_chat_template(
                        conversation, add_generation_prompt=True, return_dict=False
                    ),
                    later,
                )
            )
        loop = rollout_loop(tokenizer, [row])
        new = run_group(
            loop, group_id, row, scripted_sample(tokenizer, replies), len(completions)
        )
        # Only the first assistant turn is trained.
        for sample, text in zip(new, completions, strict=True):
            trained = sample.input_ids[sample.completion_mask.astype(bool)].tolist()
            assert tokenizer.decode(trained, skip_special_tokens=True) == text
        samples += new

    torch.testing.assert_close(
        score_with(scorer, samples), expected, atol=1e-4, rtol=1e-4
    )


def test_loops_that_never_parse_leave_the_tokenizer_alone():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    loop = rollout_loop(tokenizer, [{"prompt": PROMPTS[0], "ground_truth": "x"}])
    assert loop.tokenizer is tokenizer
    assert tokenizer.response_template is None
    assert "parse_response" not in tokenizer.__dict__


def test_reference_ids_match_the_old_tokenization(tokenizer):
    prompt, truth = PROMPTS[0], GROUND_TRUTH[0]
    ids, start = reference_ids(tokenizer, prompt, truth)
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    assert ids == tokenizer(prompt_text + truth, add_special_tokens=False)["input_ids"]
    assert start == len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])


def _pending(group_id, length, group_rows, offset=0):
    ids = np.arange(offset, offset + length, dtype=np.int64)
    return {
        "input_ids": ids,
        "completion_mask": np.r_[np.zeros(2), np.ones(length - 2)].astype(np.int64),
        "old_log_probs": np.zeros(length, dtype=np.float32),
        "advantage": float("nan"),
        "group_id": group_id,
        "metrics": {
            SCORE_PAYLOAD: {"group_rows": group_rows, "reward_names": [], "rewards": []}
        },
    }


class _ReferenceScorer(TrainerScorer):
    name = "ref"

    def group_extras(self, samples):
        return [{"input_ids": samples[0]["input_ids"][:2]}]


class TestGroupedQueue:
    def test_releases_interleaved_groups_whole(self):
        inner = queue.Queue()
        for i in range(3):
            for group_id in (0, 1):
                sample = types.SimpleNamespace(
                    group_id=group_id, metrics={SCORE_PAYLOAD: {"group_rows": 3}}, i=i
                )
                inner.put(sample)
        grouped = GroupedQueue(inner)
        order = [grouped.get(timeout=1) for _ in range(6)]
        assert [s.group_id for s in order] == [0, 0, 0, 1, 1, 1]
        assert [s.i for s in order] == [0, 1, 2, 0, 1, 2]
        with pytest.raises(queue.Empty):
            grouped.get(timeout=0.05)


class TestStepPlanner:
    @staticmethod
    def _plan(
        groups,
        steps,
        *,
        num_processes=1,
        token_budget=0,
        per_row=2,
        n=None,
        progress=None,
    ):
        metrics = defaultdict(list)
        planner = StepPlanner(
            iter([s for group in groups for s in group]),
            num_processes=num_processes,
            steps=steps,
            token_budget=token_budget,
            samples_per_row=per_row,
            scorers=[_ReferenceScorer()],
            metrics=metrics,
            progress=progress,
        )
        return list(itertools.islice(iter(planner), n)), metrics

    @staticmethod
    def _steps(micro_batches, steps):
        return [
            [row for mb in micro_batches[i : i + steps] for row in mb]
            for i in range(0, len(micro_batches), steps)
        ]

    def test_groups_stay_inside_one_step(self):
        # 19 trained tokens per group: one group per row, four per step. A step closes
        # when the next group does not fit, so two steps read nine groups.
        groups = [
            [_pending(g, 6 + (i % 3), 4, offset=100 * g) for i in range(4)]
            for g in range(9)
        ]
        micro_batches, _ = self._plan(
            groups, steps=2, num_processes=2, token_budget=24, n=4
        )
        seen = set()
        for step in self._steps(micro_batches, 2):
            assert all(row.samples for row in step)
            group_ids = [s["group_id"] for row in step for s in row.samples]
            for group_id in set(group_ids):
                assert group_ids.count(group_id) == 4
                assert group_id not in seen
                seen.add(group_id)
            for row in step:
                assert row.trained <= 24 and len(row.tokens) <= 24
                for extra in row.extras:
                    assert extra["group_id"] in row.groups
        assert seen == set(range(8))
        uids = [
            s["score_uid"] for mb in micro_batches for row in mb for s in row.samples
        ]
        assert len(uids) == len(set(uids))

    def test_count_mode_and_filler_rows(self):
        groups = [
            [_pending(g, 5, 3, offset=10 * g) for _ in range(3)] for g in range(4)
        ]
        micro_batches, metrics = self._plan(groups, steps=4, per_row=1, n=8)
        for step in self._steps(micro_batches, 4):
            assert all(len(row.samples) == 1 for row in step)
            real = [row.samples[0] for row in step if row.samples[0]["score_uid"] >= 0]
            assert len({s["group_id"] for s in real}) == 1 and len(real) == 3
            (filler,) = [
                row.samples[0] for row in step if row.samples[0]["score_uid"] < 0
            ]
            assert not filler["completion_mask"].any()
        assert sum(metrics["batch/filler_rows_total"]) == 2

    def test_drops_a_group_that_fits_no_step(self):
        groups = [
            [_pending(0, 30, 2), _pending(0, 30, 2, offset=50)],
            *[[_pending(g, 5, 2, offset=10 * g) for _ in range(2)] for g in (1, 2, 3)],
        ]
        progress = RolloutProgress(set(), epochs_done=lambda: False, stall_groups=4)
        micro_batches, metrics = self._plan(
            groups, steps=1, token_budget=16, n=1, progress=progress
        )
        assert sum(metrics["batch/dropped_oversize_group_total"]) == 1
        assert {
            s["group_id"] for mb in micro_batches for row in mb for s in row.samples
        } == {1, 2}
        assert progress.consumed == {0} and progress.dropped == {"oversize": 1}

    def test_stalls_when_no_group_fits_a_step(self):
        """Groups too long for any step are consumed, and a pass of them ends the
        stream instead of dropping rollouts forever."""
        groups = [
            [
                _pending(g, 30, 2, offset=100 * g),
                _pending(g, 30, 2, offset=100 * g + 50),
            ]
            for g in range(6)
        ]
        progress = RolloutProgress(set(), epochs_done=lambda: False, stall_groups=5)
        with pytest.raises(RolloutStop, match="5 did not fit token_budget"):
            self._plan(groups, steps=1, token_budget=16, progress=progress)
        assert progress.consumed == set(range(5))


def test_collate_splits_rows_into_equal_numbers_of_budgeted_chunks():
    scorer = EBFTScorer([0], "mean_pooling", False, 1.0, 0.5)
    scorer.token_budget = 20

    def scored(uid, length, offset):
        feature_ids = list(range(offset, offset + length))
        return {
            "score_uid": uid,
            "score_payload": {"feature_ids": feature_ids, "pool_start": 0},
        }

    long_row, short_row = ScoredRow(), ScoredRow()
    long_row.samples += [scored(0, 10, 0), scored(1, 10, 100)]
    long_row.extras.append(
        {"scorer": 0, "uid": 2, "input_ids": list(range(200, 230)), "pool_start": 0}
    )
    short_row.samples.append(scored(3, 5, 300))

    inputs = {
        key: pad(tensors, padding_value=padding)
        for key, (tensors, padding) in scorer.collate([long_row, short_row], 0).items()
    }
    chunks = chunk_inputs(inputs)

    def uids(row, chunk):
        return chunk["uids"][row, : int(chunk["sizes"][row, 3]) - 1].tolist()

    # Both rows run two forwards; the short row's second scores no sample.
    assert [[uids(row, chunk) for chunk in chunks] for row in (0, 1)] == [
        [[0, 1], [2]],
        [[3], [-1]],
    ]
    # The 30-token reference exceeds the budget alone; everything else fits it.
    assert [int(chunk["sizes"][0, 0]) for chunk in chunks] == [20, 30]
    assert [int(chunk["sizes"][1, 0]) for chunk in chunks] == [5, 1]


def _ebft_cpu_trainer(tokenizer, tmp_path, monkeypatch, passes=1, **overrides):
    from peft import LoraConfig, get_peft_model

    monkeypatch.setattr(
        async_grpo_trainer, "get_peak_flops_per_device", lambda *a, **k: None
    )
    model = get_peft_model(
        tiny_model(tokenizer).train(),
        LoraConfig(r=8, target_modules=["q_proj", "v_proj"]),
    )
    rows = [
        {"prompt": prompt, "ground_truth": truth}
        for prompt, truth in zip(PROMPTS * 4, GROUND_TRUTH * 4, strict=True)
    ]
    rng = np.random.default_rng(0)

    async def sample(prompt_ids, max_tokens):
        ids = [
            *rng.integers(100, 1000, size=rng.integers(3, 8)).tolist(),
            tokenizer.eos_token_id,
        ]
        return ids, [-1.0] * len(ids)

    loop = rollout_loop(tokenizer, rows, max_tokens=16, adaptive_max_tokens=False)
    samples = [
        s
        for group_id, row in enumerate(rows * passes)
        for s in run_group(loop, group_id, row, sample, 4)
    ]
    assert all(np.isnan(s.advantage) for s in samples)

    args = AxolotlAsyncEBFTConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        learning_rate=1e-3,
        bf16=False,
        use_cpu=True,
        report_to=[],
        save_strategy="no",
        logging_steps=1,
        gradient_checkpointing=False,
        ebft_feature_layers=LAYER_FRACTIONS,
        ebft_embed_method="mean_pooling",
        ebft_diversity_coef=0.5,
        **{"max_steps": 2, "token_budget": 0, **overrides},
    )
    trainer = AxolotlAsyncEBFTTrainer(
        model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        rollout_worker=StubWorker(samples),
        weight_transfer=StubTransfer(),
    )
    assert isinstance(trainer.rollout_queue._inner, GroupedQueue)
    return trainer


def _trainable(trainer):
    return {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }


@pytest.mark.parametrize("scoring_token_budget", [None, 16])
def test_trains_ebft_on_cpu(tokenizer, tmp_path, monkeypatch, scoring_token_budget):
    """Pending EBFT rollouts are planned, scored on the trainer and trained on, in
    one feature forward per row or, under a small scoring budget, several."""
    from axolotl.integrations.base import PluginManager
    from axolotl.utils.dict import DictDefault

    calls = []

    class Plugin:
        def on_rollouts_scored(self, cfg, trainer, *args):
            calls.append(args)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    plugin = Plugin()
    manager = PluginManager.get_instance()
    monkeypatch.setitem(manager.plugins, "recorder", plugin)
    monkeypatch.setattr(manager, "_cfg", DictDefault({"rl": "ebft"}))
    trainer = _ebft_cpu_trainer(
        tokenizer, tmp_path, monkeypatch, scoring_token_budget=scoring_token_budget
    )
    before = _trainable(trainer)
    trainer.train()

    trained = _trainable(trainer)
    assert all(not torch.equal(before[n], p) for n, p in trained.items())
    assert all(torch.isfinite(p).all() for p in trained.values())
    assert {0, 1, 2, 3} <= trainer._trained_groups
    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    assert losses and all(np.isfinite(losses))
    logs = {key for log in trainer.state.log_history for key in log}
    assert {"reward", "rewards/ebft", "ebft/alignment", "ebft/cfm_loss"} <= logs

    # Trace capture sees each step's two groups, with the rewards scored on the
    # trainer.
    assert len(calls) == 2
    for prompts, completions, rewards, advantages in calls:
        assert len(prompts) == len(completions) == len(advantages) == 8
        assert list(rewards) == ["ebft"] and len(rewards["ebft"]) == 8
        assert np.isfinite(advantages).all() and np.isfinite(rewards["ebft"]).all()


def test_groups_too_long_for_any_step_fail_the_run(tokenizer, tmp_path, monkeypatch):
    """No group fits token_budget: a pass over the prompts fails the run, naming the
    cause, where the step planner used to drop rollouts forever."""
    trainer = _ebft_cpu_trainer(tokenizer, tmp_path, monkeypatch, token_budget=8)

    def drained(stale_after_s):
        raise RuntimeError("waited for rollouts past the run's prompts")

    trainer.rollout_worker.check_health = drained
    with pytest.raises(RuntimeError, match=": 8 did not fit token_budget"):
        trainer.train()
    assert trainer.state.global_step == 0


def test_steps_with_no_trainable_samples_take_a_zero_loss(
    tokenizer, tmp_path, monkeypatch
):
    """Constant rewards zero every advantage, so scoring excludes every token."""
    group_rewards = EBFTScorer.group_rewards

    def constant(self, *args, **kwargs):
        rewards, metrics = group_rewards(self, *args, **kwargs)
        return torch.ones_like(rewards), metrics

    monkeypatch.setattr(EBFTScorer, "group_rewards", constant)
    trainer = _ebft_cpu_trainer(tokenizer, tmp_path, monkeypatch)
    before = _trainable(trainer)
    trainer.train()

    assert trainer.state.global_step == 2
    trained = _trainable(trainer)
    assert all(torch.equal(before[n], p) for n, p in trained.items())
    step_logs = [log for log in trainer.state.log_history if "loss" in log]
    assert [log["loss"] for log in step_logs] == [0.0, 0.0]
    assert all(log["sample/excluded_frac"] == 1.0 for log in step_logs)
    assert not any("ratio" in log or "kl" in log for log in step_logs)
    # The cold sync only: neither step changed the weights.
    assert trainer.weight_transfer.syncs == 1


def test_constant_rewards_fail_the_run_after_a_pass_over_the_prompts(
    tokenizer, tmp_path, monkeypatch
):
    """Scoring takes every sample out of the loss, so each step trains nothing. A
    pass over the 8 prompts in such steps fails the run, as dropped rollouts do,
    rather than stepping until max_steps."""
    group_rewards = EBFTScorer.group_rewards

    def constant(self, *args, **kwargs):
        rewards, metrics = group_rewards(self, *args, **kwargs)
        return torch.ones_like(rewards), metrics

    monkeypatch.setattr(EBFTScorer, "group_rewards", constant)
    trainer = _ebft_cpu_trainer(
        tokenizer, tmp_path, monkeypatch, passes=2, max_steps=50
    )

    def drained(stale_after_s):
        raise RuntimeError("waited for rollouts past the stub's")

    trainer.rollout_worker.check_health = drained
    with pytest.raises(RuntimeError, match=": 8 had a zero advantage"):
        trainer.train()
    # Steps 1-4 hold the 8 groups. Step 5 was planned before step 4 was scored, so
    # the stream stops while planning step 6.
    assert trainer.state.global_step == 5
    assert trainer.weight_transfer.syncs == 1


def test_a_step_that_trains_nothing_updates_neither_weights_nor_policy_version(
    tokenizer, tmp_path, monkeypatch
):
    """Step 1 trains. Step 2's groups get constant rewards, so scoring excludes every
    sample. Step 2 still counts (global step and learning-rate schedule advance),
    but Adam's momentum from step 1 must not move the weights, its state stays as
    step 1 left it, and with nothing to sync the policy version stays."""
    group_rewards = EBFTScorer.group_rewards
    calls = itertools.count()

    def constant_after_the_first_step(self, *args, **kwargs):
        rewards, metrics = group_rewards(self, *args, **kwargs)
        # A step holds two groups, scored one call each.
        if next(calls) >= 2:
            rewards = torch.ones_like(rewards)
        return rewards, metrics

    monkeypatch.setattr(EBFTScorer, "group_rewards", constant_after_the_first_step)
    trainer = _ebft_cpu_trainer(tokenizer, tmp_path, monkeypatch)
    after_first: dict[str, torch.Tensor] = {}

    class AfterFirstStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                after_first.update(_trainable(trainer))

    trainer.add_callback(AfterFirstStep())
    before = _trainable(trainer)
    trainer.train()

    assert trainer.state.global_step == 2
    # LoRA's B starts at zero, so only it moves in step 1, gaining momentum.
    assert any(not torch.equal(before[n], p) for n, p in after_first.items())
    assert all(torch.equal(after_first[n], p) for n, p in _trainable(trainer).items())
    optimizer_state = trainer.optimizer.optimizer.state
    assert optimizer_state and all(
        int(param_state["step"]) == 1 for param_state in optimizer_state.values()
    )
    assert trainer.lr_scheduler.last_epoch == 2
    # The cold sync and step 1's.
    assert trainer.weight_transfer.syncs == 2
    assert trainer.rollout_worker.versions == [1, 2]
    first, second = [log for log in trainer.state.log_history if "loss" in log]
    assert "batch/empty_steps_total" not in first and first["loss"] != 0.0
    assert second["batch/empty_steps_total"] == 1.0 and second["loss"] == 0.0
    assert second["perf/weight_sync_skipped_total"] == 1.0


def test_ebft_strategy_routes_async_prefetch_to_trl(tmp_path):
    from axolotl.core.trainers.ebft import EBFTStrategy
    from axolotl.core.trainers.ebft.trainer import AxolotlEBFTTrainer
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "rl": "ebft",
            "ebft": {"mode": "structured", "embed_method": "mean_pooling"},
            "trl": {
                "async_prefetch": True,
                "num_generations": 4,
                "scale_rewards": True,
                "generation_kwargs": {"stop_token_ids": [7]},
            },
            "vllm": None,
        }
    )
    assert EBFTStrategy.get_trainer_class(cfg) is AxolotlAsyncEBFTTrainer
    assert EBFTStrategy.get_training_args_class(cfg) is AxolotlAsyncEBFTConfig
    args = AxolotlAsyncEBFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        **EBFTStrategy.set_training_args_kwargs(cfg),
    )
    assert args.ebft_embed_method == "mean_pooling"
    assert args.scale_rewards == "group"
    assert args.generation_kwargs == {"stop_token_ids": [7]}

    cfg.trl.async_prefetch = False
    assert EBFTStrategy.get_trainer_class(cfg) is AxolotlEBFTTrainer
