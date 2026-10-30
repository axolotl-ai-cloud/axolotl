"""Tests for NeMo Gym on async GRPO: the agent rollout loop, the /verify reward and
the plugin's trainer wiring.

Agents and resource servers are local aiohttp apps; nothing needs NeMo Gym, vLLM
or a GPU.
"""

import asyncio
import contextlib
import multiprocessing as mp
import queue
import time

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from datasets import Dataset

from axolotl.integrations.nemo_gym.rewards import NemoGymVerifyReward
from axolotl.integrations.nemo_gym.rollout import (
    NemoGymRolloutLoop,
    agent_request,
    messages_from_agent_output,
    turns_from_agent_output,
)
from axolotl.utils.dict import DictDefault

EOS = 0
PROMPT = [{"role": "user", "content": "What is the weather in SF?"}]


@contextlib.asynccontextmanager
async def serve(**routes):
    """Serve ``{path: handler}`` (``/`` spelled ``_``) on a free local port."""
    app = web.Application()
    for name, handler in routes.items():
        app.router.add_post("/" + name, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def agent_output():
    """Two model calls around a tool call, as NeMo Gym's /run returns them."""
    return [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
            "prompt_token_ids": [1, 2, 3],
            "generation_token_ids": [10, 11],
            "generation_log_probs": [-0.1, -0.2],
        },
        {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "It is sunny."}],
            "prompt_token_ids": [1, 2, 3, 10, 11, 20, 21],
            "generation_token_ids": [12, EOS],
            "generation_log_probs": [-0.3, -0.4],
        },
    ]


def row(agent=None):
    extra = {"responses_create_params": {"input": PROMPT, "max_output_tokens": None}}
    if agent:
        extra["agent_ref"] = {"name": agent}
    return {
        "prompt": PROMPT,
        "resources_server_ref": {"name": "weather"},
        "verify_extra": extra,
    }


class _Tokenizer:
    response_template = "unused"
    pad_token_id = EOS


def nemo_gym_loop(rows, **kwargs):
    kwargs.setdefault("agent_servers", {"unused": "http://127.0.0.1:1"})
    kwargs.setdefault("reward_funcs", [])
    return NemoGymRolloutLoop(
        model_name="m",
        dataset=Dataset.from_list(rows),
        processing_class=_Tokenizer(),
        eos_token_ids=[EOS],
        rollout_buffer=queue.Queue(),
        model_version_value=mp.Value("i", 0),
        heartbeat_value=mp.Value("d", 0.0),
        failed_event=mp.Event(),
        exception_info_queue=mp.Queue(),
        metrics_queue=mp.Queue(),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _accelerate_state():
    from accelerate.state import PartialState

    PartialState()


class TestAgentProtocol:
    def test_request_copies_the_row_and_sets_sampling(self):
        source = row("a")
        body = agent_request(source, max_tokens=64, temperature=0.7, top_p=0.9)
        params = body["responses_create_params"]
        assert params["max_output_tokens"] == 64
        assert (params["temperature"], params["top_p"]) == (0.7, 0.9)
        assert body["agent_ref"] == {"name": "a"}
        params["input"].append({"role": "user", "content": "more"})
        assert source["verify_extra"]["responses_create_params"]["input"] == PROMPT
        assert (
            source["verify_extra"]["responses_create_params"]["max_output_tokens"]
            is None
        )

    def test_request_falls_back_to_the_prompt(self):
        body = agent_request(
            {"prompt": PROMPT, "verify_extra": None},
            max_tokens=8,
            temperature=1.0,
            top_p=1.0,
        )
        assert body["responses_create_params"]["input"] == PROMPT

    def test_one_turn_per_model_call(self):
        turns = turns_from_agent_output(agent_output())
        assert [t.output_ids for t in turns] == [[10, 11], [12, EOS]]
        assert turns[1].prompt_ids == [1, 2, 3, 10, 11, 20, 21]
        assert turns[1].output_log_probs == [-0.3, -0.4]

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "message", "content": [{"type": "output_text", "text": "x"}]},
            {
                "type": "message",
                "prompt_token_ids": [1],
                "generation_token_ids": [2, 3],
                "generation_log_probs": [-0.1],
            },
        ],
    )
    def test_output_without_token_information_is_rejected(self, item):
        with pytest.raises(ValueError, match="return_token_id_information"):
            turns_from_agent_output([item])

    def test_empty_output_has_no_turns(self):
        assert turns_from_agent_output([]) == []

    def test_messages(self):
        assert messages_from_agent_output(agent_output()) == [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ]


class TestNemoGymRolloutLoop:
    def test_agent_reward_is_the_last_reward_column(self):
        def length_reward(completions, **kwargs):
            return [float(len(c[-1]["content"])) for c in completions]

        loop = nemo_gym_loop(
            [row()], reward_funcs=[length_reward], reward_weights=[0.0, 1.0]
        )
        assert loop.reward_func_names == ["length_reward", "nemo_gym"]
        with pytest.raises(ValueError, match="reward_weights"):
            nemo_gym_loop([row()], reward_funcs=[length_reward], reward_weights=[1.0])

    def test_rolls_out_through_agents(self):
        seen = {"a": [], "b": []}

        def agent(name, scale):
            async def run(request):
                seen[name].append((await request.json(), dict(request.cookies)))
                response = web.json_response(
                    {
                        "response": {"output": agent_output()},
                        "reward": scale * len(seen[name]),
                    }
                )
                response.set_cookie("session", name)
                return response

            return run

        loop = nemo_gym_loop(
            [row("a"), row("b")], num_generations=2, max_inflight_tasks=2
        )

        async def drive():
            async with (
                serve(run=agent("a", 1.0)) as url_a,
                serve(run=agent("b", 10.0)) as url_b,
            ):
                loop._agent_servers = {"a": url_a, "b": url_b}
                stop = asyncio.Event()
                task = asyncio.create_task(loop._run_loops(stop))
                deadline = time.monotonic() + 20
                while loop.rollout_buffer.qsize() < 4:
                    assert not task.done(), task.exception()
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.01)
                stop.set()
                await task

        loop._loop.run_until_complete(drive())
        samples = [loop.rollout_buffer.get_nowait() for _ in range(4)]
        by_group = {}
        for sample in samples:
            by_group.setdefault(sample.group_id, []).append(sample)

        # Group 0 went to agent a, group 1 to agent b.
        assert {s.metrics["rewards/nemo_gym"] for s in by_group[0]} <= {1.0, 2.0, 3.0}
        assert {s.metrics["rewards/nemo_gym"] for s in by_group[1]} >= {10.0}
        for group in by_group.values():
            assert sorted(s.advantage for s in group) == pytest.approx([-1.0, 1.0])

        sample = samples[0]
        np.testing.assert_array_equal(
            sample.input_ids, [1, 2, 3, 10, 11, 20, 21, 12, EOS]
        )
        np.testing.assert_array_equal(
            sample.completion_mask, [0, 0, 0, 1, 1, 0, 0, 1, 1]
        )
        np.testing.assert_allclose(
            sample.old_log_probs, [0, 0, 0, -0.1, -0.2, 0, 0, -0.3, -0.4]
        )
        assert sample.metrics["tools/call_frequency"] == 1.0
        assert sample.completion[-1] == {"role": "assistant", "content": "It is sunny."}

        bodies = [body for name in seen for body, _ in seen[name]]
        assert all(
            body["responses_create_params"]["max_output_tokens"] == loop.max_tokens
            for body in bodies
        )
        # Every rollout is its own agent session.
        assert not any(cookies for name in seen for _, cookies in seen[name])
        assert not loop._group_rows.keys() & {0, 1}

    def _generate_one(self, loop, handler):
        async def go():
            async with (
                serve(run=handler) as url,
                aiohttp.ClientSession() as session,
            ):
                loop._agent_servers = {"a": url}
                loop._agent_session = session
                loop._group_rows[0] = row("a")
                return await loop._generate_one(PROMPT, {}, [], group_id=0)

        return loop._loop.run_until_complete(go())

    def test_failed_run_is_unscorable(self):
        async def broken(request):
            return web.Response(status=500, text="agent crashed")

        loop = nemo_gym_loop([row("a")])
        assert self._generate_one(loop, broken) == loop._UNSCORABLE
        assert loop._counters["nemo_gym/run_failure_total"] == 1

    @pytest.mark.parametrize(
        "reply",
        [
            web.Response(text="{not json", content_type="application/json"),
            web.json_response(["not", "an", "object"]),
        ],
        ids=["malformed", "not_an_object"],
    )
    def test_unreadable_run_is_unscorable(self, reply):
        async def agent(request):
            return reply

        loop = nemo_gym_loop([row("a")])
        assert self._generate_one(loop, agent) == loop._UNSCORABLE
        assert loop._counters["nemo_gym/run_failure_total"] == 1

    def test_non_numeric_reward_is_unscored(self):
        async def agent(request):
            return web.json_response(
                {"response": {"output": agent_output()}, "reward": "n/a"}
            )

        loop = nemo_gym_loop([row("a")])
        assert self._generate_one(loop, agent)[-1] is None

    def test_missing_token_information_stops_the_worker(self):
        async def no_tokens(request):
            output = [
                {"type": "message", "content": [{"type": "output_text", "text": "x"}]}
            ]
            return web.json_response({"response": {"output": output}, "reward": 1.0})

        loop = nemo_gym_loop([row("a")])
        with pytest.raises(ValueError, match="return_token_id_information"):
            self._generate_one(loop, no_tokens)


class TestNemoGymVerifyReward:
    def test_verifies_a_group_concurrently(self):
        calls = []

        async def verify(request):
            body = await request.json()
            calls.append(body)
            await asyncio.sleep(0.2)
            text = body["response"]["output"][0]["content"][0]["text"]
            if text == "error":
                return web.Response(status=500)
            return web.json_response({"reward": float(len(text))})

        async def go():
            async with serve(verify=verify) as url:
                reward = NemoGymVerifyReward(
                    {"weather": f"{url}/verify"}, model_name="policy"
                )
                start = time.monotonic()
                scores = await reward(
                    [
                        [{"role": "assistant", "content": text}]
                        for text in ("ab", "abcd", "error", "x")
                    ],
                    prompts=[PROMPT] * 4,
                    resources_server_ref=[{"name": "weather"}] * 3
                    + [{"name": "unknown"}],
                    verify_extra=[{"task": "t", "skipped": None}] * 4,
                )
                return scores, time.monotonic() - start

        scores, elapsed = asyncio.run(go())
        assert scores == [2.0, 4.0, None, None]
        assert elapsed < 0.5
        assert calls[0]["task"] == "t" and "skipped" not in calls[0]
        assert calls[0]["response"]["model"] == "policy"
        assert calls[0]["responses_create_params"]["input"] == PROMPT

    def test_unreadable_replies_are_unscored(self):
        replies = {
            "malformed": lambda: web.Response(
                text="{not json", content_type="application/json"
            ),
            "list": lambda: web.json_response([1.0]),
            "text": lambda: web.json_response({"reward": "n/a"}),
        }

        async def verify(request):
            body = await request.json()
            return replies[body["response"]["output"][0]["content"][0]["text"]]()

        async def go():
            async with serve(verify=verify) as url:
                reward = NemoGymVerifyReward({"weather": f"{url}/verify"})
                return await reward(
                    [[{"role": "assistant", "content": text}] for text in replies],
                    prompts=[PROMPT] * len(replies),
                    resources_server_ref=[{"name": "weather"}] * len(replies),
                    verify_extra=[{}] * len(replies),
                )

        assert asyncio.run(go()) == [None, None, None]

    def test_is_a_picklable_async_reward(self):
        import pickle

        from trl.trainer.utils import get_callable_name, is_async_callable

        reward = pickle.loads(pickle.dumps(NemoGymVerifyReward({"s": "http://s"})))
        assert get_callable_name(reward) == "nemo_gym"
        assert is_async_callable(reward)
        assert reward.verify_endpoints == {"s": "http://s"}


def _cfg(**overrides):
    cfg = {
        "nemo_gym_enabled": True,
        "nemo_gym_multi_turn": False,
        "nemo_gym_model_name": None,
        "nemo_gym_verify_timeout": 45,
        "nemo_gym_run_timeout": 600,
        "base_model": "Qwen/Qwen3-0.6B",
        "adapter": "lora",
        "trl": DictDefault({"async_prefetch": True}),
        "vllm": None,
    }
    cfg.update(overrides)
    return DictDefault(cfg)


def _plugin(multi_turn=False):
    from axolotl.integrations.nemo_gym.plugin import NemoGymPlugin

    plugin = NemoGymPlugin()
    plugin._verify_endpoints = {"weather": "http://127.0.0.1:1/verify"}
    plugin._agent_servers = {"weather_agent": "http://127.0.0.1:2"}
    plugin._multi_turn = multi_turn
    return plugin


class TestPluginWiring:
    def test_single_turn_passes_the_verify_reward(self):
        (reward,) = _plugin().get_reward_funcs(_cfg())
        assert isinstance(reward, NemoGymVerifyReward)
        assert reward.verify_endpoints == {"weather": "http://127.0.0.1:1/verify"}
        assert reward.model_name == "Qwen/Qwen3-0.6B"
        assert reward.timeout == 45
        assert _plugin().get_reward_funcs(_cfg(nemo_gym_enabled=False)) is None

    def test_multi_turn_runs_the_agent_loop(self):
        from axolotl.core.trainers.async_grpo.config import AxolotlAsyncGRPOConfig

        plugin = _plugin(multi_turn=True)
        cfg = _cfg(nemo_gym_multi_turn=True, nemo_gym_model_name="policy")
        assert plugin.get_reward_funcs(cfg) is None
        args = plugin.get_training_args(cfg)
        assert args["rollout_loop"] == (
            "axolotl.integrations.nemo_gym.rollout.NemoGymRolloutLoop"
        )
        assert args["rollout_loop_kwargs"] == {
            "agent_servers": {"weather_agent": "http://127.0.0.1:2"},
            "run_timeout": 600.0,
        }
        assert args["vllm_lora_alias"] == "policy"
        AxolotlAsyncGRPOConfig(output_dir="/tmp/unused", use_cpu=True, **args)  # nosec B108

        full_finetune = plugin.get_training_args(
            _cfg(nemo_gym_multi_turn=True, adapter=None)
        )
        assert "vllm_lora_alias" not in full_finetune

    def test_sync_grpo_keeps_the_vllm_passthrough(self):
        cfg = _cfg(
            trl=DictDefault({"async_prefetch": False}),
            vllm=DictDefault({"max_model_len": 2048}),
        )
        args = _plugin().get_training_args(cfg)
        assert args == {"vllm_max_model_length": 2048}


class TestConfigValidation:
    @staticmethod
    def _validate(**data):
        from axolotl.integrations.nemo_gym.args import NemoGymArgs

        data = {
            "nemo_gym_enabled": True,
            "nemo_gym_auto_start": False,
            "nemo_gym_datasets": [{"path": "x.jsonl"}],
            **data,
        }
        NemoGymArgs.model_validate(data)
        return data

    def test_multi_turn_needs_async_grpo(self):
        with pytest.raises(ValueError, match="async_prefetch"):
            self._validate(nemo_gym_multi_turn=True, trl={"num_generations": 4})
        self._validate(nemo_gym_multi_turn=True, trl={"async_prefetch": True})
        self._validate(nemo_gym_multi_turn=True, trl={"use_data_producer": True})

    def test_superseded_reward_funcs_are_dropped(self):
        data = self._validate(
            trl={
                "reward_funcs": [
                    "axolotl.integrations.nemo_gym.rewards.reward_nemo_gym_verify",
                    "my.rewards.format_reward",
                    "axolotl.integrations.nemo_gym.rewards.reward_env",
                ]
            }
        )
        assert data["trl"]["reward_funcs"] == ["my.rewards.format_reward"]
