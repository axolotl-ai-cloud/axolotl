"""Tests for async GRPO on TRL's AsyncGRPOTrainer and axolotl's extensions to it."""

import asyncio
import multiprocessing as mp
import queue
import time
import types
from collections import defaultdict

import numpy as np
import pytest
import torch
from async_test_utils import StubTransfer, StubWorker, reference_tree_attention
from trl.experimental.async_grpo.async_rollout_worker import (
    RolloutGroup,
    RolloutSample,
    TrainingSequence,
)

from axolotl.core.trainers.grpo import GRPOStrategy
from axolotl.core.trainers.grpo.async_rollout import (
    AxolotlRolloutLoop,
    RewardProcessPool,
    compute_group_advantages,
)
from axolotl.core.trainers.grpo.async_trainer import (
    AxolotlAsyncGRPOConfig,
    AxolotlAsyncGRPOTrainer,
    SampleFilterQueue,
    negative_advantage_veto_counts,
)
from axolotl.utils.dict import DictDefault

ONES = np.ones(1)


class TestComputeGroupAdvantages:
    def test_matches_trl_default(self):
        rewards = np.array([[0.0, 1.0, 1.0, 3.0]])
        weighted, advantages = compute_group_advantages(rewards, ONES)
        expected = (rewards[0] - rewards[0].mean()) / (rewards[0].std() + 1e-8)
        np.testing.assert_allclose(weighted, rewards[0])
        np.testing.assert_allclose(advantages, expected)

    def test_constant_group_is_exactly_zero(self):
        # 0.1 * 3 / 3 != 0.1 in floating point, so centering alone leaves ~1e-9.
        _, advantages = compute_group_advantages(np.full((1, 3), 0.1), ONES)
        assert (advantages == 0.0).all()

    def test_unscored_completions_keep_nan_reward_and_zero_advantage(self):
        rewards = np.array([[1.0, np.nan, 0.0], [np.nan, np.nan, 1.0]])
        weighted, advantages = compute_group_advantages(rewards, np.ones(2))
        assert np.isnan(weighted[1])
        assert advantages[1] == 0.0
        np.testing.assert_allclose(weighted[[0, 2]], [1.0, 1.0])
        assert (advantages[[0, 2]] == 0.0).all()

    def test_reward_weights(self):
        rewards = np.array([[1.0, 0.0], [0.0, 1.0]])
        weighted, advantages = compute_group_advantages(rewards, np.array([2.0, 0.5]))
        np.testing.assert_allclose(weighted, [2.0, 0.5])
        assert advantages[0] > 0 > advantages[1]

    def test_scale_none_only_centers(self):
        _, advantages = compute_group_advantages(
            np.array([[0.0, 4.0]]), ONES, scale_rewards="none"
        )
        np.testing.assert_allclose(advantages, [-2.0, 2.0])

    def test_normalize_then_sum_balances_reward_scales(self):
        # The second reward is 100x larger; GDPO normalizes each before summing.
        rewards = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 100.0, 100.0]])
        _, gdpo = compute_group_advantages(
            rewards, np.ones(2), aggregation="normalize_then_sum"
        )
        _, summed = compute_group_advantages(rewards, np.ones(2))
        np.testing.assert_allclose(gdpo, [0.0, -np.sqrt(2), 0.0, np.sqrt(2)], atol=1e-6)
        assert summed[0] < 0  # the large reward dominates the plain sum

    def test_rejects_batch_scaling(self):
        with pytest.raises(ValueError, match="scale_rewards"):
            compute_group_advantages(np.zeros((1, 2)), ONES, scale_rewards="batch")


def _sample(group_id, advantage, last_token=2):
    return types.SimpleNamespace(
        group_id=group_id,
        advantage=advantage,
        input_ids=np.array([5, 6, last_token]),
    )


class TestSampleFilterQueue:
    def _filter(self, samples, **kwargs):
        inner = queue.Queue()
        for sample in samples:
            inner.put(sample)
        consumed, metrics = set(), defaultdict(list)
        options = {"skip_zero_advantage": True, "stop_token_ids": None, **kwargs}
        return SampleFilterQueue(inner, consumed, metrics, **options), consumed, metrics

    def test_drops_zero_advantage_and_counts_group_consumed(self):
        filtered, consumed, metrics = self._filter(
            [_sample(0, 0.0), _sample(0, 0.0), _sample(1, 0.5)]
        )
        assert filtered.get(timeout=1).group_id == 1
        assert consumed == {0}
        assert metrics["sample/dropped_zero_adv_total"] == [1.0, 1.0]
        assert metrics["sample/dropped_frac"] == [(1.0, 1.0), (1.0, 1.0), (0.0, 1.0)]

    def test_keeps_zero_advantage_when_disabled(self):
        filtered, _, _ = self._filter([_sample(0, 0.0)], skip_zero_advantage=False)
        assert filtered.get(timeout=1).advantage == 0.0

    def test_drops_truncated(self):
        filtered, consumed, metrics = self._filter(
            [_sample(3, 1.0, last_token=9), _sample(4, 1.0, last_token=2)],
            stop_token_ids=frozenset({2}),
        )
        assert filtered.get(timeout=1).group_id == 4
        assert consumed == {3}
        assert metrics["sample/dropped_truncated_total"] == [1.0]

    def test_passes_samples_on_without_signal_once_the_epochs_are_consumed(self):
        inner = queue.Queue()
        for sample in [
            _sample(0, 0.0),
            _sample(1, 0.0),
            _sample(2, 1.0, last_token=9),
            _sample(3, 0.5),
        ]:
            inner.put(sample)
        consumed: set[int] = set()
        filtered = SampleFilterQueue(
            inner,
            consumed,
            defaultdict(list),
            skip_zero_advantage=True,
            stop_token_ids=frozenset({2}),
            epoch_done=lambda: len(consumed) >= 1,
        )
        # Group 0 completes the epochs; later drops would leave the last step unfilled.
        assert filtered.get(timeout=1).group_id == 1
        truncated = filtered.get(timeout=1)
        assert (truncated.group_id, truncated.advantage) == (2, 0.0)
        assert filtered.get(timeout=1).advantage == 0.5
        assert consumed == {0}

    def test_timeout_spans_dropped_samples(self):
        filtered, _, _ = self._filter([_sample(0, 0.0)])
        start = time.monotonic()
        with pytest.raises(queue.Empty):
            filtered.get(timeout=0.2)
        assert time.monotonic() - start < 1.0


def _loss_stub(loss_type="dapo", mode="token_truncate", cap=3.0, epsilon_high=0.2):
    return types.SimpleNamespace(
        epsilon_low=0.2,
        epsilon_high=epsilon_high,
        args=types.SimpleNamespace(
            loss_type=loss_type,
            vllm_importance_sampling_mode=mode,
            vllm_importance_sampling_cap=cap,
        ),
    )


class TestPolicyLoss:
    # Ratios 0.5, 1.0, 10.0 against the sampler.
    log_probs = torch.log(torch.tensor([0.5, 1.0, 10.0]))
    old = torch.zeros(3)

    def _loss(self, stub, advantages):
        log_probs = self.log_probs.clone().requires_grad_(True)
        loss, ratio, log_ratio = AxolotlAsyncGRPOTrainer._policy_loss(
            stub, log_probs, self.old, advantages
        )
        loss.sum().backward()
        assert not ratio.requires_grad and not log_ratio.requires_grad
        torch.testing.assert_close(ratio, torch.tensor([0.5, 1.0, 10.0]))
        return loss.detach(), log_probs.grad

    def test_dapo_is_trls_clipped_surrogate(self):
        ratio = torch.tensor([0.5, 1.0, 10.0])
        for advantages in (torch.ones(3), -torch.ones(3)):
            loss, grad = self._loss(_loss_stub(), advantages)
            expected = -torch.min(
                ratio * advantages, ratio.clamp(0.8, 1.2) * advantages
            )
            torch.testing.assert_close(loss, expected)
        # Clipped where the ratio left the trust region on the advantage's side: the
        # token whose probability rose 10x since sampling gets no gradient.
        _, grad = self._loss(_loss_stub(), torch.ones(3))
        torch.testing.assert_close(grad, torch.tensor([-0.5, -1.0, 0.0]))
        _, grad = self._loss(_loss_stub(), -torch.ones(3))
        torch.testing.assert_close(grad, torch.tensor([0.0, 1.0, 10.0]))

    def test_dapo_honors_epsilon_high(self):
        _, grad = self._loss(_loss_stub(epsilon_high=10.0), torch.ones(3))
        torch.testing.assert_close(grad, torch.tensor([-0.5, -1.0, -10.0]))

    def test_cispo_truncate_weights_by_capped_ratio(self):
        weights = torch.tensor([0.5, 1.0, 3.0])
        for advantages in (torch.ones(3), -torch.ones(3)):
            loss, grad = self._loss(_loss_stub("cispo"), advantages)
            torch.testing.assert_close(loss, -advantages * weights)
            torch.testing.assert_close(grad, -advantages * weights)

    def test_cispo_keeps_the_gradient_of_moved_tokens(self):
        _, grad = self._loss(_loss_stub("cispo", cap=20.0), torch.ones(3))
        torch.testing.assert_close(grad, torch.tensor([-0.5, -1.0, -10.0]))

    def test_cispo_mask_drops_tokens_beyond_cap(self):
        advantages = torch.tensor([1.0, -1.0, 1.0])
        loss, grad = self._loss(_loss_stub("cispo", mode="token_mask"), advantages)
        assert loss[2] == 0.0 and grad[2] == 0.0
        assert loss[1] == pytest.approx(1.0)
        assert grad[0] == pytest.approx(-0.5)


class TestNegativeAdvantageVetoCounts:
    def test_counts_negative_samples_below_each_threshold(self):
        # Per-sample min ratios: 1e-5, 1e-7 and 0.5 (negative), 1e-8 (positive).
        ratio = torch.tensor([1.0, 1e-5, 0.9, 1e-7, 0.5, 2.0, 1e-8])
        segment_id = torch.tensor([0, 0, 1, 1, 2, 2, 3])
        advantages = torch.tensor([-1.0, -1.0, -0.5, -0.5, -2.0, -2.0, 1.0])
        counts = negative_advantage_veto_counts(
            ratio, advantages, segment_id, (1e-4, 1e-6)
        )
        assert counts.tolist() == [3.0, 2.0, 1.0]

    def test_no_trained_tokens(self):
        empty = torch.zeros(0)
        counts = negative_advantage_veto_counts(
            empty, empty, torch.zeros(0, dtype=torch.long), (1e-4, 1e-6)
        )
        assert counts.tolist() == [0.0, 0.0, 0.0]


class TestConfig:
    def test_defaults_to_trls_objective(self, tmp_path):
        args = AxolotlAsyncGRPOConfig(output_dir=str(tmp_path), use_cpu=True)
        assert args.loss_type == "dapo"
        assert not hasattr(args, "vllm_importance_sampling_correction")

    def test_rejects_other_loss_types(self, tmp_path):
        with pytest.raises(ValueError, match="loss_type"):
            AxolotlAsyncGRPOConfig(
                output_dir=str(tmp_path), use_cpu=True, loss_type="grpo"
            )

    def test_rejects_cap_at_most_one(self, tmp_path):
        with pytest.raises(ValueError, match="must exceed 1"):
            AxolotlAsyncGRPOConfig(
                output_dir=str(tmp_path),
                use_cpu=True,
                vllm_importance_sampling_cap=1.0,
            )

    def test_rejects_sequence_modes(self, tmp_path):
        with pytest.raises(ValueError, match="vllm_importance_sampling_mode"):
            AxolotlAsyncGRPOConfig(
                output_dir=str(tmp_path),
                use_cpu=True,
                vllm_importance_sampling_mode="sequence_mask",
            )


class _Tokenizer:
    response_template = "unused"
    pad_token_id = 0


def _rollout_loop(reward_funcs, processing_class=None, **kwargs):
    from datasets import Dataset

    return AxolotlRolloutLoop(
        model_name="m",
        dataset=Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]]}),
        reward_funcs=reward_funcs,
        processing_class=processing_class or _Tokenizer(),
        eos_token_ids=[0],
        rollout_buffer=queue.Queue(),
        model_version_value=mp.Value("i", 0),
        heartbeat_value=mp.Value("d", 0.0),
        failed_event=mp.Event(),
        exception_info_queue=mp.Queue(),
        metrics_queue=mp.Queue(),
        **kwargs,
    )


def _group(group_id, contents):
    seq = TrainingSequence([1, 2, 3], [0, 1, 1], [0.0, -1.0, -1.0], "r")
    size = len(contents)
    return RolloutGroup(
        prompts=[[{"role": "user", "content": "q"}]] * size,
        reward_kwargs={"answer": ["x"] * size},
        completions=[[{"role": "assistant", "content": c}] for c in contents],
        completions_ids=[[2, 3]] * size,
        completions_sequences=[[seq]] * size,
        tool_call_counts=[0] * size,
        tool_failure_counts=[0] * size,
        model_version=0,
        group_id=group_id,
        env_rewards=[],
        rollout_rewards=[],
        queued_at=time.monotonic(),
    )


class TestRolloutLoop:
    @pytest.fixture(autouse=True)
    def _accelerate_state(self):
        from accelerate.state import PartialState

        PartialState()

    def test_score_group_applies_weights(self):
        from async_grpo_rewards import length_reward

        def constant_reward(completions, **kwargs):
            return [1.0] * len(completions)

        loop = _rollout_loop(
            [length_reward, constant_reward], reward_weights=[0.5, 2.0]
        )
        samples = loop._loop.run_until_complete(
            loop._score_group(_group(7, ["a", "abc"]))
        )
        assert [s.metrics["reward"] for s in samples] == [2.5, 3.5]
        assert samples[0].advantage < 0 < samples[1].advantage
        assert samples[0].metrics["rewards/length_reward"] == 1.0
        assert {s.group_id for s in samples} == {7}

    @pytest.fixture
    def chatml_tokenizer(self, download_smollm2_135m_instruct_model):
        from transformers import AutoTokenizer

        from axolotl.utils.chat_templates import get_chat_template

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        tokenizer.chat_template = get_chat_template("chatml")
        return tokenizer

    def test_chat_template_trl_cannot_parse_decodes_completions(self, chatml_tokenizer):
        from async_grpo_rewards import length_reward

        loop = _rollout_loop([length_reward], processing_class=chatml_tokenizer)
        prompt = [{"role": "user", "content": "What is 2+2?"}]
        answer = chatml_tokenizer(
            "The answer is 4.<|im_end|>", add_special_tokens=False
        )["input_ids"]

        async def generate_one_turn(prompt_ids):
            return answer, [-0.5] * len(answer)

        loop._generate_one_turn = generate_one_turn
        completion, completion_ids, sequences, *_ = loop._loop.run_until_complete(
            loop._generate_one(prompt, tool_dict={}, tools=[])
        )
        assert completion == [{"role": "assistant", "content": "The answer is 4."}]
        assert completion_ids == answer
        prompt_ids = chatml_tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_dict=False
        )
        assert sequences[0].input_ids == prompt_ids + answer

    def test_chat_template_trl_cannot_parse_still_fails_with_tools(
        self, chatml_tokenizer
    ):
        from async_grpo_rewards import length_reward, lookup_tool

        with pytest.raises(ValueError, match="Unrecognized chat template"):
            _rollout_loop(
                [length_reward], processing_class=chatml_tokenizer, tools=[lookup_tool]
            )

    def test_reward_weights_length_checked(self):
        from async_grpo_rewards import length_reward

        with pytest.raises(ValueError, match="reward_weights"):
            _rollout_loop([length_reward], reward_weights=[1.0, 2.0])

    @staticmethod
    def _score_groups(loop, num_groups):
        async def drive():
            async def produce():
                for group_id in range(num_groups):
                    await loop._groups_to_score.put(_group(group_id, ["a", "b"]))
                await loop._groups_to_score.put(None)

            await asyncio.gather(loop._score_loop(asyncio.Event()), produce())

        try:
            loop._loop.run_until_complete(drive())
        finally:
            loop._reward_pool.close()
        return [loop.rollout_buffer.get_nowait() for _ in range(2 * num_groups)]

    def test_reward_pool_runs_rewards_on_worker_main_threads(self):
        from async_grpo_rewards import alarm_reward, pid_reward

        loop = _rollout_loop([pid_reward, alarm_reward], reward_num_workers=2)
        samples = self._score_groups(loop, 3)
        assert sorted({s.group_id for s in samples}) == [0, 1, 2]
        assert mp.current_process().pid not in {
            s.metrics["rewards/pid_reward"] for s in samples
        }
        assert all(s.metrics["rewards/alarm_reward"] == 1.0 for s in samples)

    def test_score_loop_scores_groups_concurrently(self, tmp_path, monkeypatch):
        from async_grpo_rewards import overlap_reward

        monkeypatch.setenv("TRL_ASYNC_TEST_REWARD_RENDEZVOUS", str(tmp_path))
        loop = _rollout_loop([overlap_reward], reward_num_workers=2)
        samples = self._score_groups(loop, 4)
        assert sorted({s.group_id for s in samples}) == [0, 1, 2, 3]
        assert all(s.metrics["rewards/overlap_reward"] == 1.0 for s in samples)

    def test_reward_pool_surfaces_reward_errors(self):
        from async_grpo_rewards import failing_reward

        pool = RewardProcessPool([failing_reward], 1)
        try:
            with pytest.raises(RuntimeError, match="ZeroDivisionError"):
                asyncio.run(pool.call(0, {"completions": []}))
        finally:
            pool.close()


class TestGRPOStrategy:
    def _cfg(self, **trl):
        return DictDefault({"trl": DictDefault(trl), "vllm": None})

    def test_is_async(self):
        assert not GRPOStrategy.is_async(self._cfg())
        assert GRPOStrategy.is_async(self._cfg(async_prefetch=True))
        cfg = self._cfg(async_prefetch=True)
        cfg.nemo_gym_enabled = True
        assert GRPOStrategy.is_async(cfg)

    def test_plugins_add_reward_funcs(self, monkeypatch):
        from axolotl.integrations.base import BasePlugin, PluginManager

        def plugin_reward(completions, **kwargs):
            return [0.0] * len(completions)

        class RewardPlugin(BasePlugin):
            def get_reward_funcs(self, cfg):
                return [plugin_reward]

        manager = PluginManager.get_instance()
        monkeypatch.setitem(manager.plugins, "reward_plugin", RewardPlugin())
        cfg = self._cfg(reward_funcs=["async_grpo_rewards.length_reward"])
        (reward_funcs,) = GRPOStrategy.set_trainer_args(cfg)
        assert [f.__name__ for f in reward_funcs] == ["length_reward", "plugin_reward"]
        (reward_funcs,) = GRPOStrategy.set_trainer_args(self._cfg())
        assert reward_funcs == [plugin_reward]

    def test_trainer_and_args_classes(self):
        assert (
            GRPOStrategy.get_trainer_class(async_grpo=True) is AxolotlAsyncGRPOTrainer
        )
        assert (
            GRPOStrategy.get_training_args_class(async_grpo=True)
            is AxolotlAsyncGRPOConfig
        )

    def test_async_training_args_kwargs(self):
        cfg = self._cfg(
            async_prefetch=True,
            vllm_server_host="0.0.0.0",  # nosec B104
            vllm_server_port=8123,
            vllm_sync_interval=2,
            max_staleness=1,
            scale_rewards=False,
            reward_weights=[1.0, 0.5],
            num_generations=8,
            reward_num_workers=4,
        )
        kwargs = GRPOStrategy.set_async_training_args_kwargs(cfg)
        assert kwargs["vllm_server_base_url"] == "http://localhost:8123"
        assert kwargs["weight_sync_steps"] == 2
        assert kwargs["max_staleness"] == 1
        assert kwargs["scale_rewards"] == "none"
        assert kwargs["reward_weights"] == [1.0, 0.5]
        assert kwargs["reward_num_workers"] == 4
        assert "beta" not in kwargs
        assert "loss_type" not in kwargs

    def test_async_cispo_kwargs(self):
        cfg = self._cfg(
            async_prefetch=True,
            loss_type="cispo",
            vllm_importance_sampling_correction=True,
            vllm_importance_sampling_mode="token_mask",
            vllm_importance_sampling_cap=5.0,
        )
        kwargs = GRPOStrategy.set_async_training_args_kwargs(cfg)
        assert kwargs["loss_type"] == "cispo"
        assert kwargs["vllm_importance_sampling_mode"] == "token_mask"
        assert kwargs["vllm_importance_sampling_cap"] == 5.0
        assert "vllm_importance_sampling_correction" not in kwargs


class _TinyLM(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(32, 16, dtype=dtype)
        self.proj = torch.nn.Linear(16, 16, bias=True, dtype=dtype)
        self.head = torch.nn.Linear(16, 32, bias=False, dtype=dtype)


def _peft_model(dtype=torch.bfloat16, **lora_kwargs):
    from peft import LoraConfig, get_peft_model

    options = {"r": 4, "target_modules": ["embed", "proj"], **lora_kwargs}
    model = get_peft_model(_TinyLM(dtype), LoraConfig(**options))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name or "lora_embedding_B" in name:
                param.normal_(std=0.5)
    return model


def _stream(model):
    stub = types.SimpleNamespace(
        model=model,
        accelerator=types.SimpleNamespace(
            unwrap_model=lambda m: m, device=torch.device("cpu")
        ),
    )
    return dict(AxolotlAsyncGRPOTrainer._streaming_iter(stub))


def _manifest(model):
    from trl.experimental.async_grpo.async_grpo_trainer import _iter_vllm_named_params

    return {
        name: (tuple(param.shape), param.dtype)
        for name, param in _iter_vllm_named_params(model)
    }


class TestMergedWeightSync:
    """Merged sync streams W + delta without merging into the trained weights."""

    def test_streams_merged_weights_and_leaves_model_untouched(self):
        model = _peft_model()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        for _ in range(3):
            streamed = _stream(model)

        for name, param in model.named_parameters():
            assert torch.equal(param, before[name]), name
        assert {n: (tuple(t.shape), t.dtype) for n, t in streamed.items()} == (
            _manifest(model)
        )
        proj = model.base_model.model.proj
        delta = (
            proj.lora_B["default"].weight.float()
            @ proj.lora_A["default"].weight.float()
        ) * proj.scaling["default"]
        expected = (proj.base_layer.weight.float() + delta).to(torch.bfloat16)
        torch.testing.assert_close(streamed["proj.weight"], expected, rtol=0, atol=0)
        assert not torch.equal(streamed["proj.weight"], proj.base_layer.weight)
        assert torch.equal(streamed["proj.bias"], proj.base_layer.bias)
        assert torch.equal(streamed["head.weight"], model.base_model.model.head.weight)

    @pytest.mark.parametrize(
        "lora_kwargs",
        [
            {},
            {"lora_bias": True, "target_modules": ["proj"]},
            {"use_dora": True, "target_modules": ["proj"]},
        ],
        ids=["lora", "lora_bias", "dora"],
    )
    def test_matches_peft_merge(self, lora_kwargs):
        import copy

        model = _peft_model(dtype=torch.float32, **lora_kwargs)
        if lora_kwargs.get("lora_bias"):
            with torch.no_grad():
                model.base_model.model.proj.lora_B["default"].bias.normal_()
        reference = copy.deepcopy(model).merge_and_unload().state_dict()
        streamed = _stream(model)
        for name, tensor in streamed.items():
            torch.testing.assert_close(tensor, reference[name], msg=name)

    def test_fsdp_sharded_weights(self, tmp_path):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        expected = _stream(_peft_model())
        model = _peft_model()
        dist.init_process_group(
            "gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1
        )
        try:
            mesh = init_device_mesh("cpu", (1,))
            for module in list(model.modules()):
                for name, param in list(module.named_parameters(recurse=False)):
                    sharded = distribute_tensor(param.detach(), mesh, [Shard(0)])
                    setattr(
                        module,
                        name,
                        torch.nn.Parameter(sharded, requires_grad=param.requires_grad),
                    )
            streamed = _stream(model)
        finally:
            dist.destroy_process_group()
        assert streamed.keys() == expected.keys()
        for name, tensor in streamed.items():
            assert not isinstance(tensor, torch.distributed.tensor.DTensor)
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def _cpu_trainer(tmp_path, monkeypatch, old_log_probs, num_groups=16, **args_kwargs):
    """A SmolLM2 LoRA trainer on CPU over 16 prompts and ``num_groups`` groups of 4
    stub rollouts.

    ``old_log_probs(g)`` sets the sampled log-probabilities of a group's ``g``-th
    completion; group ids divisible by 4 carry no reward signal.
    """
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.experimental.async_grpo import async_grpo_trainer, packing

    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    monkeypatch.setattr(
        async_grpo_trainer, "get_peak_flops_per_device", lambda *a, **k: None
    )
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))

    prompt = [{"role": "user", "content": "What is 2+2?"}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )
    rng = np.random.default_rng(0)
    samples = []
    for group_id in range(num_groups):
        for g in range(4):
            completion = [*rng.integers(100, 1000, size=4 + g), tokenizer.eos_token_id]
            ids = np.asarray(prompt_ids + completion, dtype=np.int64)
            samples.append(
                RolloutSample(
                    prompt=prompt,
                    completion=[{"role": "assistant", "content": "x"}],
                    input_ids=ids,
                    completion_mask=np.asarray(
                        [0] * len(prompt_ids) + [1] * len(completion), dtype=np.int64
                    ),
                    old_log_probs=np.full(len(ids), old_log_probs(g), np.float32),
                    advantage=0.0 if group_id % 4 == 0 else float(g) - 1.5,
                    model_version=0,
                    group_id=group_id,
                    metrics={"reward": float(g), "reward_std": 1.0},
                )
            )

    args = AxolotlAsyncGRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        bf16=False,
        use_cpu=True,
        token_budget=0,
        report_to=[],
        save_strategy="no",
        gradient_checkpointing=False,
        **{"max_steps": 3, **args_kwargs},
    )
    transfer = StubTransfer()
    worker = StubWorker(samples)
    trainer = AxolotlAsyncGRPOTrainer(
        model,
        reward_funcs=[],
        args=args,
        train_dataset=Dataset.from_dict({"prompt": [prompt] * 16}),
        processing_class=tokenizer,
        rollout_worker=worker,
        weight_transfer=transfer,
    )
    return trainer, worker, transfer, samples


def test_filtering_every_sample_still_ends_the_epochs(
    download_smollm2_135m_instruct_model, tmp_path, monkeypatch
):
    """No group has a reward signal, yet consuming the prompts once ends the run.

    The stub worker holds a second pass over the prompts, as TRL's worker cycles them;
    waiting for more rollouts than that fails the test instead of hanging it.
    """
    trainer, worker, _, samples = _cpu_trainer(
        tmp_path,
        monkeypatch,
        old_log_probs=lambda g: 0.0,
        num_groups=32,
        max_steps=-1,
        num_train_epochs=1,
    )
    for sample in samples:
        sample.advantage = 0.0
    before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }

    def drained(stale_after_s):
        raise RuntimeError("waited for rollouts past the run's prompts")

    worker.check_health = drained
    trainer.train()

    assert trainer.state.global_step == 1
    assert set(range(16)) <= trainer._trained_groups
    assert all(
        torch.equal(before[n], p.detach())
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    )


def test_trains_axolotl_loaded_lora_model_on_cpu(
    download_smollm2_135m_instruct_model, tmp_path, monkeypatch
):
    """The axolotl-loaded PEFT model runs TRL's packing, loss and weight sync."""
    # The negative-advantage half was sampled far less likely than the policy now
    # finds it, so its ratio exceeds cispo's cap.
    trainer, worker, transfer, _ = _cpu_trainer(
        tmp_path,
        monkeypatch,
        old_log_probs=lambda g: -30.0 if g < 2 else 0.0,
        loss_type="cispo",
    )
    assert trainer.model.get_base_model().config._attn_implementation == "tree_fa3"
    assert isinstance(trainer.rollout_queue, SampleFilterQueue)

    before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    frozen_before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if not p.requires_grad
    }
    trainer.train()

    assert all(
        not torch.equal(before[n], p.detach())
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    )
    # Four merged syncs later, the frozen base is bit-for-bit what was loaded.
    assert all(
        torch.equal(frozen_before[n], p.detach())
        for n, p in trainer.model.named_parameters()
        if not p.requires_grad
    )
    # Cold sync plus one per optimizer step, of merged base weights only.
    assert transfer.syncs == 4 and worker.versions == [1, 2, 3, 4]
    assert not any("lora" in name for name in transfer.names)
    # Groups 0 and 4 have no reward signal: dropped, yet counted as consumed.
    assert set(range(8)) <= trainer._trained_groups
    logs = trainer.state.log_history
    assert sum(log.get("sample/dropped_zero_adv_total", 0) for log in logs) >= 8
    assert any(log.get("sampling/capped_token_frac", 0) > 0 for log in logs)
    # The positive-advantage half was sampled with probability 1, so its ratio is the
    # policy's own probability; the negative half was sampled far less likely.
    ratio_mins = [
        log["sampling/ratio_min"] for log in logs if "sampling/ratio_min" in log
    ]
    assert ratio_mins and max(ratio_mins) < 1.0
    vetoes = [
        log[key] for log in logs for key in log if key.startswith("sampling/nav_")
    ]
    assert len(vetoes) == 2 * len(ratio_mins) and all(v == 0.0 for v in vetoes)


def test_default_loss_is_trls(
    download_smollm2_135m_instruct_model, tmp_path, monkeypatch
):
    """With the default loss_type, compute_loss is TRL's AsyncGRPOTrainer's."""
    from trl.experimental.async_grpo.async_grpo_trainer import (
        AsyncGRPOTrainer,
        DataCollatorForRollout,
    )
    from trl.experimental.async_grpo.packing import TreeRow

    # Sampled log-probabilities around the policy's put ratios on both sides of the
    # trust region, for both advantage signs.
    trainer, _, _, samples = _cpu_trainer(
        tmp_path, monkeypatch, old_log_probs=lambda g: (-12.0, -8.0, -12.0, -8.0)[g]
    )
    assert trainer.args.loss_type == "dapo"
    row = TreeRow()
    for sample in samples[4:12]:
        row.add(
            {
                "input_ids": sample.input_ids,
                "completion_mask": sample.completion_mask,
                "old_log_probs": sample.old_log_probs,
                "advantage": sample.advantage,
                "group_id": sample.group_id,
                "metrics": sample.metrics,
            }
        )
    batch = DataCollatorForRollout(pad_token_id=trainer.processing_class.pad_token_id)(
        [[row]]
    )
    model = trainer.model
    model.train()
    trainer.current_gradient_accumulation_steps = 1
    params = [p for p in model.parameters() if p.requires_grad]
    results = []
    for compute_loss in (
        AxolotlAsyncGRPOTrainer.compute_loss,
        AsyncGRPOTrainer.compute_loss,
    ):
        model.zero_grad()
        trainer._metrics["train"].clear()
        loss = compute_loss(trainer, model, batch)
        loss.backward()
        metrics = {k: v[0] for k, v in trainer._metrics["train"].items()}
        results.append((loss.detach(), [p.grad.clone() for p in params], metrics))
    (loss, grads, metrics), (trl_loss, trl_grads, trl_metrics) = results
    torch.testing.assert_close(loss, trl_loss, rtol=0, atol=0)
    for grad, trl_grad in zip(grads, trl_grads, strict=True):
        torch.testing.assert_close(grad, trl_grad, rtol=0, atol=0)
    assert trl_metrics["clip_ratio/low_mean"] > 0
    assert trl_metrics["clip_ratio/high_mean"] > 0
    for key, value in trl_metrics.items():
        if key.startswith(("clip_ratio/", "ratio", "kl", "entropy")):
            assert metrics[key] == pytest.approx(value), key


class TestTreeFlashAttn:
    @pytest.fixture
    def tree_flash_attn(self, monkeypatch):
        import kernels

        from axolotl.core.trainers.grpo import async_trainer

        loaded = []
        monkeypatch.setattr(
            kernels, "get_kernel", lambda repo, **kw: loaded.append((repo, kw)) or repo
        )
        async_trainer._tree_flash_attn.cache_clear()
        yield async_trainer, loaded
        async_trainer._tree_flash_attn.cache_clear()

    def test_falls_back_to_flash_attn2_where_flash_attn3_has_no_build(
        self, tree_flash_attn, monkeypatch
    ):
        async_trainer, loaded = tree_flash_attn

        def no_build():
            raise RuntimeError(
                "CUDA capability 12.0 of the current device is not supported by "
                "the architectures of the build: 8.0, 9.0a"
            )

        monkeypatch.setattr(async_trainer, "_trl_tree_flash_attn", no_build)
        assert async_trainer._tree_flash_attn().kernel == (
            "kernels-community/flash-attn2"
        )
        assert loaded == [("kernels-community/flash-attn2", {"version": 3})]

    def test_keeps_flash_attn3_where_it_loads(self, tree_flash_attn, monkeypatch):
        async_trainer, loaded = tree_flash_attn
        monkeypatch.setattr(async_trainer, "_trl_tree_flash_attn", lambda: "fa3")
        assert async_trainer._tree_flash_attn() == "fa3"
        assert loaded == []

    def test_flash_attn2_gets_values_as_wide_as_queries(self):
        from axolotl.core.trainers.grpo.async_trainer import _WideValues

        seen = []

        def flash_attn_varlen_func(query, key, value, *args, **kwargs):
            seen.append(value.shape[-1])
            return value * 2, "lse"

        kernel = _WideValues(
            types.SimpleNamespace(flash_attn_varlen_func=flash_attn_varlen_func)
        )
        # MLA: 48-wide queries and keys, 32-wide values
        query, key, value = (
            torch.randn(5, 2, 48),
            torch.randn(5, 2, 48),
            torch.randn(5, 2, 32),
        )
        out, lse = kernel.flash_attn_varlen_func(query, key, value)
        assert seen == [48] and lse == "lse"
        torch.testing.assert_close(out, value * 2)


def test_flops_estimate_reads_unknown_moe_layouts_as_dense(monkeypatch):
    from axolotl.core.trainers.grpo import async_trainer

    def trl_estimate(config, seq_len):
        if config.num_experts_per_tok is not None:
            return config.decoder_sparse_step  # what TRL reads for MoE configs
        return config.hidden_size * seq_len

    monkeypatch.setattr(async_trainer, "_trl_flops_per_token", trl_estimate)
    # Nemotron-H: num_experts_per_tok without the Qwen-MoE fields
    hybrid = types.SimpleNamespace(num_experts_per_tok=2, hidden_size=8)
    assert async_trainer._flops_per_token(hybrid, 3) == 24
    moe = types.SimpleNamespace(num_experts_per_tok=2, decoder_sparse_step=1)
    assert async_trainer._flops_per_token(moe, 3) == 1


class TestTreeAttentionSwitch:
    def _model(self, modules, switch):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(modules)
                self.config = types.SimpleNamespace(_attn_implementation="sdpa")

            def set_attn_implementation(self, implementation):
                switch(self, implementation)

        return Model()

    def _attention(self):
        return type("MyAttention", (torch.nn.Module,), {})()

    def test_switches_attention_models(self):
        from axolotl.core.trainers.grpo.async_trainer import (
            TREE_ATTENTION,
            _set_tree_attention,
        )

        def switch(model, implementation):
            model.config._attn_implementation = implementation

        model = self._model([self._attention()], switch)
        _set_tree_attention(model)
        assert model.config._attn_implementation == TREE_ATTENTION

    def test_refuses_attention_that_keeps_another_implementation(self):
        from axolotl.core.trainers.grpo.async_trainer import _set_tree_attention

        model = self._model([self._attention()], lambda model, implementation: None)
        with pytest.raises(ValueError, match="kept attention implementation 'sdpa'"):
            _set_tree_attention(model)

    def test_models_without_attention_have_nothing_to_route(self):
        from axolotl.core.trainers.grpo.async_trainer import _set_tree_attention

        def unsupported(model, implementation):
            raise ValueError(
                "Mamba2ForCausalLM does not support an attention implementation"
            )

        _set_tree_attention(self._model([torch.nn.Linear(2, 2)], unsupported))


class TestAliasingVLLMClient:
    def test_sync_drains_and_moves_the_alias(self, monkeypatch):
        from trl.experimental.async_grpo import vllm_client

        from axolotl.core.trainers.grpo import async_trainer

        calls = []

        def post(url, json=None, params=None, timeout=None):
            calls.append((url.rsplit("/", 1)[-1], json or params))
            return types.SimpleNamespace(status_code=200, text="")

        monkeypatch.setattr(async_trainer.requests, "post", post)
        monkeypatch.setattr(vllm_client.requests, "post", post)
        client = async_trainer.AliasingVLLMClient("http://vllm", alias="policy")
        client.pause()
        client.load_lora_adapter("trl-policy-v2", "/adapters/v2")
        assert calls == [
            ("pause", {"mode": "wait"}),
            (
                "load_lora_adapter",
                {
                    "lora_name": "trl-policy-v2",
                    "lora_path": "/adapters/v2",
                    "load_inplace": False,
                },
            ),
            ("unload_lora_adapter", {"lora_name": "policy"}),
            (
                "load_lora_adapter",
                {
                    "lora_name": "policy",
                    "lora_path": "/adapters/v2",
                    "load_inplace": False,
                },
            ),
        ]


class _RolloutRewardLoop(AxolotlRolloutLoop):
    _provides_rollout_reward = True

    def __init__(self, *, agent_url, **kwargs):
        super().__init__(**kwargs)
        self.agent_url = agent_url


def test_custom_rollout_loop_and_alias_wiring(tmp_path, monkeypatch):
    """`rollout_loop` reaches TRL's rollout worker, `vllm_lora_alias` its client."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
    from trl.experimental.async_grpo import async_grpo_trainer, async_rollout_worker

    from axolotl.core.trainers.grpo.async_trainer import AliasingVLLMClient

    monkeypatch.setattr(async_rollout_worker, "is_vllm_available", lambda: True)
    monkeypatch.setattr(
        async_grpo_trainer, "get_peak_flops_per_device", lambda *a, **k: None
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    args = AxolotlAsyncGRPOConfig(
        output_dir=str(tmp_path),
        max_steps=1,
        use_cpu=True,
        bf16=False,
        report_to=[],
        rollout_loop=f"{__name__}._RolloutRewardLoop",
        rollout_loop_kwargs={"agent_url": "http://agent"},
        vllm_lora_alias="policy",
        reward_weights=[1.0],
    )
    trainer = AxolotlAsyncGRPOTrainer(
        model,
        reward_funcs=[],
        args=args,
        train_dataset=Dataset.from_dict(
            {"prompt": [[{"role": "user", "content": "q"}]] * 4}
        ),
        processing_class=tokenizer,
        weight_transfer=StubTransfer(),
    )
    worker = trainer.rollout_worker
    assert worker._loop_cls is _RolloutRewardLoop
    assert worker._loop_kwargs["agent_url"] == "http://agent"
    assert worker._loop_kwargs["reward_weights"] == [1.0]
    assert isinstance(trainer.vllm_client, AliasingVLLMClient)
    assert trainer.vllm_client.alias == "policy"
    # The patch is scoped to construction.
    assert async_grpo_trainer.VLLMClient is not AliasingVLLMClient

    args.rollout_loop = None
    with pytest.raises(ValueError, match="needs reward functions"):
        AxolotlAsyncGRPOTrainer(
            model,
            reward_funcs=[],
            args=args,
            train_dataset=Dataset.from_dict(
                {"prompt": [[{"role": "user", "content": "q"}]] * 4}
            ),
            processing_class=tokenizer,
            weight_transfer=StubTransfer(),
        )


def test_rollout_loop_must_be_an_axolotl_rollout_loop(tmp_path):
    with pytest.raises(ValueError, match="AxolotlRolloutLoop"):
        AxolotlAsyncGRPOConfig(
            output_dir=str(tmp_path), use_cpu=True, rollout_loop="queue.Queue"
        )


class _RecordingPlugin:
    """Stands in for a trace-capture plugin."""

    def __init__(self):
        self.calls = []

    def on_rollouts_scored(self, cfg, trainer, *args):
        self.calls.append(args)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _LiveWorker(StubWorker):
    """Generates without end like a rollout worker: each sample is tagged with the
    live policy version, and the first two after every sync are one version old."""

    def __init__(self, make_sample):
        super().__init__([])
        self._make_sample = make_sample
        self._stale_due = 0
        self._fresh = 0
        self.rollout_buffer = self

    def update_model_version(self, version):
        super().update_model_version(version)
        self._stale_due = 2

    def qsize(self):
        return 0

    def get(self, timeout=None):
        version = self.versions[-1]
        if self._stale_due:
            self._stale_due -= 1
            return self._make_sample(version - 1, index=-1)
        self._fresh += 1
        return self._make_sample(version, index=self._fresh - 1)


def test_staleness_zero_trains_on_fresh_samples_and_reports_rollouts(
    tmp_path, monkeypatch
):
    """`use_data_producer` maps onto max_staleness 0: samples of an older policy are
    dropped, training does not stall, and plugins see every step's rollouts."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
    from trl.experimental.async_grpo import async_grpo_trainer, packing

    from axolotl.integrations.base import PluginManager

    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    monkeypatch.setattr(
        async_grpo_trainer, "get_peak_flops_per_device", lambda *a, **k: None
    )
    plugin = _RecordingPlugin()
    manager = PluginManager.get_instance()
    monkeypatch.setitem(manager.plugins, "recorder", plugin)
    monkeypatch.setattr(manager, "_cfg", DictDefault({"rl": "grpo"}))

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )
    answer = tokenizer("Four", add_special_tokens=False)["input_ids"]

    completion = [*answer, tokenizer.eos_token_id]

    def make_sample(version, index):
        g = index % 4
        return RolloutSample(
            prompt=prompt,
            completion=[{"role": "assistant", "content": "Four"}],
            input_ids=np.asarray(prompt_ids + completion, dtype=np.int64),
            completion_mask=np.asarray(
                [0] * len(prompt_ids) + [1] * len(completion), dtype=np.int64
            ),
            old_log_probs=np.full(len(prompt_ids) + len(completion), -1.0),
            advantage=float(g) - 1.5,
            model_version=version,
            group_id=index // 4,
            metrics={"reward": float(g), "rewards/len": float(g)},
        )

    args = AxolotlAsyncGRPOConfig(
        output_dir=str(tmp_path),
        max_steps=3,
        max_staleness=0,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        bf16=False,
        use_cpu=True,
        token_budget=0,
        report_to=[],
        save_strategy="no",
        logging_steps=1,
        gradient_checkpointing=False,
    )
    worker = _LiveWorker(make_sample)
    trainer = AxolotlAsyncGRPOTrainer(
        model,
        reward_funcs=[],
        args=args,
        train_dataset=Dataset.from_dict({"prompt": [prompt] * 4}),
        processing_class=tokenizer,
        rollout_worker=worker,
        weight_transfer=StubTransfer(),
    )
    trainer.train()

    assert trainer.state.global_step == 3
    assert worker.versions == [1, 2, 3, 4]
    # TRL counts a group when it is collated; the dataloader fetches a micro-batch
    # ahead, so the next step's group is counted too.
    assert {0, 1, 2} <= trainer._trained_groups
    logs = trainer.state.log_history
    assert sum(log.get("sample/dropped_stale_total", 0) for log in logs) >= 4

    assert len(plugin.calls) == 3
    for prompts, completions, rewards, advantages in plugin.calls:
        assert prompts == [tokenizer.decode(prompt_ids, skip_special_tokens=True)] * 4
        assert completions == ["Four"] * 4
        assert rewards == {"len": [0.0, 1.0, 2.0, 3.0]}
        assert advantages == [-1.5, -0.5, 0.5, 1.5]
