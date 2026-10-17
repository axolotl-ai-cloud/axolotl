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
from trl.experimental.async_grpo.async_rollout_worker import (
    RolloutGroup,
    RolloutSample,
    TrainingSequence,
)

from axolotl.core.trainers.grpo import GRPOStrategy
from axolotl.core.trainers.grpo.trl_async_rollout import (
    AxolotlRolloutLoop,
    RewardProcessPool,
    compute_group_advantages,
)
from axolotl.core.trainers.grpo.trl_async_trainer import (
    AxolotlAsyncGRPOConfig,
    AxolotlAsyncGRPOTrainer,
    SampleFilterQueue,
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

    def test_timeout_spans_dropped_samples(self):
        filtered, _, _ = self._filter([_sample(0, 0.0)])
        start = time.monotonic()
        with pytest.raises(queue.Empty):
            filtered.get(timeout=0.2)
        assert time.monotonic() - start < 1.0


def _loss_stub(correction=True, mode="token_truncate", cap=3.0):
    return types.SimpleNamespace(
        epsilon_low=0.2,
        epsilon_high=0.2,
        args=types.SimpleNamespace(
            vllm_importance_sampling_correction=correction,
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
        loss, coef, _, capped = AxolotlAsyncGRPOTrainer._policy_loss(
            stub, log_probs, self.old, advantages
        )
        loss.sum().backward()
        return loss.detach(), log_probs.grad, coef.detach(), capped

    def test_without_correction_matches_ppo_clip(self):
        advantages = torch.tensor([-1.0, -1.0, -1.0])
        loss, _, coef, capped = self._loss(_loss_stub(correction=False), advantages)
        expected = -torch.min(coef * advantages, coef.clamp(0.8, 1.2) * advantages)
        torch.testing.assert_close(loss, expected)
        assert not capped.any()

    def test_truncate_caps_negative_advantage_ratio(self):
        advantages = torch.tensor([-1.0, -1.0, -1.0])
        loss, grad, _, capped = self._loss(_loss_stub(), advantages)
        assert capped.tolist() == [False, False, True]
        assert loss[2] == pytest.approx(3.0)
        assert grad[2] == 0.0
        _, uncapped_grad, _, _ = self._loss(_loss_stub(correction=False), advantages)
        assert uncapped_grad[2] == pytest.approx(10.0)

    def test_truncate_leaves_positive_advantages_to_ppo(self):
        advantages = torch.tensor([1.0, 1.0, 1.0])
        loss, _, _, _ = self._loss(_loss_stub(), advantages)
        reference, _, _, _ = self._loss(_loss_stub(correction=False), advantages)
        torch.testing.assert_close(loss, reference)

    def test_mask_drops_tokens_beyond_cap(self):
        advantages = torch.tensor([1.0, -1.0, 1.0])
        loss, grad, _, _ = self._loss(_loss_stub(mode="token_mask"), advantages)
        assert loss[2] == 0.0 and grad[2] == 0.0
        assert loss[1] == pytest.approx(1.0)


class TestConfig:
    def test_rejects_cap_inside_trust_region(self, tmp_path):
        with pytest.raises(ValueError, match="1 \\+ epsilon_high"):
            AxolotlAsyncGRPOConfig(
                output_dir=str(tmp_path), vllm_importance_sampling_cap=1.1
            )

    def test_rejects_sequence_modes(self, tmp_path):
        with pytest.raises(ValueError, match="vllm_importance_sampling_mode"):
            AxolotlAsyncGRPOConfig(
                output_dir=str(tmp_path), vllm_importance_sampling_mode="sequence_mask"
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
        from trl_async_grpo_rewards import length_reward

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
        from trl_async_grpo_rewards import length_reward

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
        from trl_async_grpo_rewards import length_reward, lookup_tool

        with pytest.raises(ValueError, match="Unrecognized chat template"):
            _rollout_loop(
                [length_reward], processing_class=chatml_tokenizer, tools=[lookup_tool]
            )

    def test_reward_weights_length_checked(self):
        from trl_async_grpo_rewards import length_reward

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
        from trl_async_grpo_rewards import alarm_reward, pid_reward

        loop = _rollout_loop([pid_reward, alarm_reward], reward_num_workers=2)
        samples = self._score_groups(loop, 3)
        assert sorted({s.group_id for s in samples}) == [0, 1, 2]
        assert mp.current_process().pid not in {
            s.metrics["rewards/pid_reward"] for s in samples
        }
        assert all(s.metrics["rewards/alarm_reward"] == 1.0 for s in samples)

    def test_score_loop_scores_groups_concurrently(self):
        from trl_async_grpo_rewards import start_time_reward

        loop = _rollout_loop([start_time_reward], reward_num_workers=2)
        samples = self._score_groups(loop, 4)
        starts = sorted({s.metrics["rewards/start_time_reward"] for s in samples})
        # One group at a time would start each 0.3 s after the previous one.
        assert len(starts) == 4
        assert min(b - a for a, b in zip(starts, starts[1:], strict=False)) < 0.15

    def test_reward_pool_surfaces_reward_errors(self):
        from trl_async_grpo_rewards import failing_reward

        pool = RewardProcessPool([failing_reward], 1)
        try:
            with pytest.raises(RuntimeError, match="ZeroDivisionError"):
                asyncio.run(pool.call(0, {"completions": []}))
        finally:
            pool.close()


class TestGRPOStrategy:
    def _cfg(self, **trl):
        return DictDefault({"trl": DictDefault(trl), "vllm": None})

    def test_async_backend(self):
        assert GRPOStrategy.get_async_backend(self._cfg()) is None
        assert GRPOStrategy.get_async_backend(self._cfg(async_prefetch=True)) == "trl"
        assert (
            GRPOStrategy.get_async_backend(self._cfg(use_data_producer=True))
            == "data_producer"
        )
        cfg = self._cfg(async_prefetch=True)
        cfg.nemo_gym_enabled = True
        assert GRPOStrategy.get_async_backend(cfg) == "trl"

    def test_plugins_add_reward_funcs(self, monkeypatch):
        from axolotl.integrations.base import BasePlugin, PluginManager

        def plugin_reward(completions, **kwargs):
            return [0.0] * len(completions)

        class RewardPlugin(BasePlugin):
            def get_reward_funcs(self, cfg):
                return [plugin_reward]

        manager = PluginManager.get_instance()
        monkeypatch.setitem(manager.plugins, "reward_plugin", RewardPlugin())
        cfg = self._cfg(reward_funcs=["trl_async_grpo_rewards.length_reward"])
        (reward_funcs,) = GRPOStrategy.set_trainer_args(cfg)
        assert [f.__name__ for f in reward_funcs] == ["length_reward", "plugin_reward"]
        (reward_funcs,) = GRPOStrategy.set_trainer_args(self._cfg())
        assert reward_funcs == [plugin_reward]

    def test_trainer_and_args_classes(self):
        from axolotl.core.trainers.grpo import (
            AxolotlDataProducerGRPOConfig,
            AxolotlDataProducerGRPOTrainer,
        )

        assert (
            GRPOStrategy.get_trainer_class(async_grpo=True) is AxolotlAsyncGRPOTrainer
        )
        assert (
            GRPOStrategy.get_trainer_class(data_producer=True)
            is AxolotlDataProducerGRPOTrainer
        )
        assert (
            GRPOStrategy.get_training_args_class(async_grpo=True)
            is AxolotlAsyncGRPOConfig
        )
        assert (
            GRPOStrategy.get_training_args_class(data_producer=True)
            is AxolotlDataProducerGRPOConfig
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


def reference_tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    tree_kv_index,
    tree_cu_seqlens_q,
    tree_cu_seqlens_k,
    tree_max_seqlen_q,
    tree_max_seqlen_k,
    scaling=None,
    **kwargs,
):
    """Pure-torch stand-in for TRL's FA3 tree attention."""
    q, k, v = (x.transpose(1, 2)[0] for x in (query, key, value))
    k, v = k.index_select(0, tree_kv_index), v.index_select(0, tree_kv_index)
    groups = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    for s in range(len(tree_cu_seqlens_q) - 1):
        qs, qe = int(tree_cu_seqlens_q[s]), int(tree_cu_seqlens_q[s + 1])
        ks, ke = int(tree_cu_seqlens_k[s]), int(tree_cu_seqlens_k[s + 1])
        lq, lk = qe - qs, ke - ks
        mask = torch.arange(lk)[None, :] <= (lk - lq + torch.arange(lq))[:, None]
        out[qs:qe] = torch.nn.functional.scaled_dot_product_attention(
            q[qs:qe].transpose(0, 1),
            k[ks:ke].repeat_interleave(groups, 1).transpose(0, 1),
            v[ks:ke].repeat_interleave(groups, 1).transpose(0, 1),
            attn_mask=mask,
            scale=scaling,
        ).transpose(0, 1)
    return out.unsqueeze(0), None


class _StubWorker:
    def __init__(self, samples):
        self.rollout_buffer = queue.Queue()
        self.metrics_queue = queue.Queue()
        self._samples = samples
        self.versions = []

    def start(self):
        for sample in self._samples:
            self.rollout_buffer.put(sample)

    def stop(self):
        pass

    def update_model_version(self, version):
        self.versions.append(version)

    def check_health(self, stale_after_s):
        pass


class _StubTransfer:
    syncs = 0

    def init_weight_transfer(self):
        pass

    def pause(self):
        pass

    def send_weights(self, iterator):
        self.names = [name for name, _ in iterator]
        self.syncs += 1

    def resume(self):
        pass

    def destroy(self):
        pass


def test_trains_axolotl_loaded_lora_model_on_cpu(
    download_smollm2_135m_instruct_model, tmp_path, monkeypatch
):
    """The axolotl-loaded PEFT model runs TRL's packing, loss and weight sync."""
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
    for group_id in range(16):
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
                    # The negative-advantage half was sampled far less likely than the
                    # policy now finds it, so its ratio exceeds the cap.
                    old_log_probs=np.full(
                        len(ids), -30.0 if g < 2 else 0.0, np.float32
                    ),
                    advantage=0.0 if group_id % 4 == 0 else float(g) - 1.5,
                    model_version=0,
                    group_id=group_id,
                    metrics={"reward": float(g), "reward_std": 1.0},
                )
            )

    args = AxolotlAsyncGRPOConfig(
        output_dir=str(tmp_path),
        max_steps=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        bf16=False,
        use_cpu=True,
        token_budget=0,
        report_to=[],
        save_strategy="no",
        gradient_checkpointing=False,
    )
    transfer = _StubTransfer()
    worker = _StubWorker(samples)
    trainer = AxolotlAsyncGRPOTrainer(
        model,
        reward_funcs=[],
        args=args,
        train_dataset=Dataset.from_dict({"prompt": [prompt] * 16}),
        processing_class=tokenizer,
        rollout_worker=worker,
        weight_transfer=transfer,
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


class TestAliasingVLLMClient:
    def test_sync_drains_and_moves_the_alias(self, monkeypatch):
        from trl.experimental.async_grpo import vllm_client

        from axolotl.core.trainers.grpo import trl_async_trainer

        calls = []

        def post(url, json=None, params=None, timeout=None):
            calls.append((url.rsplit("/", 1)[-1], json or params))
            return types.SimpleNamespace(status_code=200, text="")

        monkeypatch.setattr(trl_async_trainer.requests, "post", post)
        monkeypatch.setattr(vllm_client.requests, "post", post)
        client = trl_async_trainer.AliasingVLLMClient("http://vllm", alias="policy")
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

    from axolotl.core.trainers.grpo.trl_async_trainer import AliasingVLLMClient

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
        weight_transfer=_StubTransfer(),
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
            weight_transfer=_StubTransfer(),
        )


def test_rollout_loop_must_be_an_axolotl_rollout_loop(tmp_path):
    with pytest.raises(ValueError, match="AxolotlRolloutLoop"):
        AxolotlAsyncGRPOConfig(
            output_dir=str(tmp_path), use_cpu=True, rollout_loop="queue.Queue"
        )
