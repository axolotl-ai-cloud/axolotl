"""Async GRPO rollouts from a CoreWeave gateway, served by a local aiohttp app."""

import asyncio
import contextlib
import multiprocessing as mp
import queue
import time

import aiohttp
import pytest
from aiohttp import web
from datasets import Dataset

from axolotl.integrations.coreweave.async_rollouts import CoreWeaveRolloutLoop

PROMPT = [{"role": "user", "content": "What is 2+2?"}]


def length_reward(completions, **kwargs):
    return [float(len(completion[-1]["content"])) for completion in completions]


@contextlib.asynccontextmanager
async def gateway(handler):
    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from accelerate.state import PartialState

    PartialState()
    monkeypatch.setenv("CW_API_TOKEN", "secret")


@pytest.fixture
def tokenizer(download_smollm2_135m_instruct_model):
    from transformers import AutoTokenizer

    from axolotl.utils.chat_templates import get_chat_template

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    tokenizer.chat_template = get_chat_template("chatml")
    return tokenizer


def coreweave_loop(tokenizer, version=-1, **kwargs):
    kwargs.setdefault("gateway_url", "http://127.0.0.1:1")
    return CoreWeaveRolloutLoop(
        model="served-policy",
        model_name="hf/model",
        dataset=Dataset.from_dict({"prompt": [PROMPT]}),
        reward_funcs=[length_reward],
        processing_class=tokenizer,
        eos_token_ids=[tokenizer.eos_token_id],
        rollout_buffer=queue.Queue(),
        model_version_value=mp.Value("i", version),
        heartbeat_value=mp.Value("d", 0.0),
        failed_event=mp.Event(),
        exception_info_queue=mp.Queue(),
        metrics_queue=mp.Queue(),
        **kwargs,
    )


def test_samples_from_the_gateway_once_a_publish_is_served(tokenizer):
    requests = []
    answers = ["4<|im_end|>", "four<|im_end|>"]

    async def complete(request):
        requests.append((request.headers.get("Authorization"), await request.json()))
        ids = tokenizer(answers[len(requests) % 2], add_special_tokens=False)[
            "input_ids"
        ]
        return web.json_response(
            {
                "choices": [
                    {
                        "text": "",
                        "logprobs": {
                            "tokens": [f"token_id:{i}" for i in ids],
                            "token_logprobs": [-0.25] * len(ids),
                        },
                    }
                ]
            }
        )

    loop = coreweave_loop(tokenizer, num_generations=2, max_inflight_tasks=2)

    async def drive():
        async with gateway(complete) as url:
            loop.vllm_server_url = url
            stop = asyncio.Event()
            task = asyncio.create_task(loop._run_loops(stop))
            await asyncio.sleep(0.5)
            # Nothing is sampled before the first publish is served.
            assert not requests and not task.done()
            loop._model_version_value.value = 5
            deadline = time.monotonic() + 20
            while loop.rollout_buffer.qsize() < 2:
                assert not task.done(), task.exception()
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
            stop.set()
            await task

    loop._loop.run_until_complete(drive())
    samples = [loop.rollout_buffer.get_nowait() for _ in range(2)]
    assert {sample.model_version for sample in samples} == {5}
    assert sorted(s.completion[-1]["content"] for s in samples) == ["4", "four"]
    prompt_ids = tokenizer.apply_chat_template(
        PROMPT, add_generation_prompt=True, return_dict=False
    )
    for sample in samples:
        assert list(sample.input_ids[: len(prompt_ids)]) == prompt_ids
        completion = list(sample.old_log_probs[len(prompt_ids) :])
        assert completion == [-0.25] * (len(sample.input_ids) - len(prompt_ids))

    authorization, body = requests[0]
    assert authorization == "Bearer secret"
    assert body["model"] == "served-policy" and body["prompt"] == prompt_ids
    assert body["return_tokens_as_token_ids"] is True
    assert not body.keys() & {"return_token_ids", "top_k", "repetition_penalty"}


def test_zero_probability_top_logprobs_arrive_as_null(tokenizer):
    async def complete(request):
        return web.json_response(
            {
                "choices": [
                    {
                        "logprobs": {
                            "tokens": ["token_id:7"],
                            "token_logprobs": [0.0],
                            "top_logprobs": [{"token_id:7": 0.0, "token_id:2": None}],
                        }
                    }
                ]
            }
        )

    loop = coreweave_loop(tokenizer)

    async def drive():
        async with gateway(complete) as url, aiohttp.ClientSession() as session:
            loop.vllm_server_url, loop.session = url, session
            return await loop._post("/v1/completions", {}, timeout=10)

    (choice,) = loop._loop.run_until_complete(drive())["choices"]
    assert choice["token_ids"] == [7]
    assert choice["logprobs"]["top_logprobs"] == [
        {"token_id:7": 0.0, "token_id:2": float("-inf")}
    ]


def test_a_rejected_request_fails_without_retrying(tokenizer):
    calls = []

    async def reject(request):
        calls.append(1)
        return web.Response(status=422, text="prompt must be a string")

    loop = coreweave_loop(tokenizer, version=0)

    async def go():
        async with gateway(reject) as url:
            loop.vllm_server_url = url
            import aiohttp

            async with aiohttp.ClientSession() as session:
                loop.session = session
                return await loop._post("/v1/completions", {"prompt": [1]}, 5)

    with pytest.raises(RuntimeError, match="rejected /v1/completions \\(422\\)"):
        loop._loop.run_until_complete(go())
    assert calls == [1]


def test_top_p_sampling_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="top_p"):
        coreweave_loop(tokenizer, top_p=0.9)


def test_the_api_token_is_required(tokenizer, monkeypatch):
    monkeypatch.delenv("CW_API_TOKEN")
    with pytest.raises(ValueError, match="CW_API_TOKEN"):
        coreweave_loop(tokenizer)


def test_the_worker_does_not_need_vllm_installed(monkeypatch):
    import trl.experimental.async_grpo.async_rollout_worker as trl_worker

    from axolotl.core.trainers.async_grpo.rollout import AxolotlAsyncRolloutWorker

    monkeypatch.setattr(trl_worker, "is_vllm_available", lambda: False)
    worker = AxolotlAsyncRolloutWorker(loop_cls=CoreWeaveRolloutLoop, model_name="m")
    assert worker._loop_cls is CoreWeaveRolloutLoop
    assert not trl_worker.is_vllm_available()
    with pytest.raises(ImportError, match="vLLM is required"):
        AxolotlAsyncRolloutWorker(model_name="m")
