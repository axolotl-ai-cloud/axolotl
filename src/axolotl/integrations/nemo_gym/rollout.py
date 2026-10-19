# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
Multi-turn NeMo Gym rollouts for async GRPO.

A NeMo Gym agent server runs a whole rollout behind its ``/run`` endpoint: it
generates against our vLLM server through its model proxy, executes tool calls on
resource servers, verifies the outcome and returns the prompt and generated token
IDs and logprobs of every model call along with the reward. ``NemoGymRolloutLoop``
runs in async GRPO's rollout process and sends each rollout to an agent instead of
generating itself. It rebuilds training rows from the returned model calls with
TRL's multi-turn reconciler and scores the agent's reward as a reward column.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from typing import Any

import aiohttp
from trl.experimental.async_grpo.async_rollout_worker import (
    TurnRecord,
    _chain_to_sequences,
)

from axolotl.core.trainers.grpo.trl_async_rollout import AxolotlRolloutLoop
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _drop_none(value: Any) -> Any:
    # Arrow fills keys that other rows have with None; NeMo Gym reads a None
    # `max_output_tokens` as "no limit" rather than as unset.
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


def agent_request(
    row: dict, *, max_tokens: int, temperature: float, top_p: float
) -> dict:
    """The ``/run`` body for one rollout of a NeMo Gym dataset row."""
    body = _drop_none(copy.deepcopy(row.get("verify_extra") or {}))
    params = body.setdefault("responses_create_params", {})
    params.setdefault("input", copy.deepcopy(row["prompt"]))
    params.setdefault("max_output_tokens", max_tokens)
    params["temperature"] = temperature
    params["top_p"] = top_p
    return body


CAPTURE_REQUIRED = (
    "The NeMo Gym agent returned model output without prompt token IDs, generated "
    "token IDs and one logprob per generated token. Set "
    "`return_token_id_information: true` on its vllm_model server."
)


def turns_from_agent_output(output: list[dict]) -> list[TurnRecord]:
    """One turn per model call: NeMo Gym puts a call's token IDs on its last item."""
    turns = []
    for item in output:
        generated = item.get("generation_token_ids")
        if not generated:
            continue
        prompt_ids = item.get("prompt_token_ids")
        logprobs = item.get("generation_log_probs")
        if not prompt_ids or logprobs is None or len(logprobs) != len(generated):
            raise ValueError(CAPTURE_REQUIRED)
        turns.append(TurnRecord(list(prompt_ids), list(generated), list(logprobs)))
    if not turns and any(
        item.get("type") in ("message", "function_call") for item in output
    ):
        raise ValueError(CAPTURE_REQUIRED)
    return turns


def messages_from_agent_output(output: list[dict]) -> list[dict]:
    """The rollout as chat messages, for reward functions and completion logs."""
    messages: list[dict] = []
    for item in output:
        kind = item.get("type")
        if kind == "message":
            text = "".join(
                part.get("text") or ""
                for part in item.get("content") or []
                if isinstance(part, dict)
            )
            messages.append({"role": item.get("role") or "assistant", "content": text})
        elif kind == "function_call":
            call = {
                "id": item.get("call_id"),
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                },
            }
            last = messages[-1] if messages else None
            if last is not None and last["role"] == "assistant":
                last.setdefault("tool_calls", []).append(call)
            else:
                messages.append(
                    {"role": "assistant", "content": "", "tool_calls": [call]}
                )
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output"),
                }
            )
    return messages


class NemoGymRolloutLoop(AxolotlRolloutLoop):
    """Rolls out each prompt through a NeMo Gym agent's ``/run`` (see module doc).

    Args:
        agent_servers: Agent name to base URL. A row's ``agent_ref.name`` picks its
            agent; rows without one go to the first.
        run_timeout: Seconds a ``/run`` may go without sending data.
    """

    _provides_rollout_reward = True
    _rollout_reward_name = "nemo_gym"
    _parses_responses = False
    # Failed rollouts train nothing and stay out of the group baseline.
    _UNSCORABLE: tuple = ([], [], [], 0, 0, None)

    def __init__(
        self,
        *,
        agent_servers: dict[str, str],
        run_timeout: float = 300.0,
        **kwargs,
    ):
        if not agent_servers:
            raise ValueError("NemoGymRolloutLoop needs at least one agent server")
        super().__init__(**kwargs)
        self._agent_servers = dict(agent_servers)
        self._run_timeout = run_timeout
        self._agent_session: aiohttp.ClientSession | None = None

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        # Agents keep a session per cookie, which a shared jar would hand to every
        # rollout.
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.max_inflight_tasks),
            cookie_jar=aiohttp.DummyCookieJar(),
        ) as session:
            self._agent_session = session
            await super()._run_loops(stop_event)

    def _agent_url(self, body: dict) -> str:
        name = (body.get("agent_ref") or {}).get("name") or ""
        return self._agent_servers.get(name) or next(iter(self._agent_servers.values()))

    async def _run_agent(self, body: dict) -> dict | None:
        url = f"{self._agent_url(body)}/run"
        assert self._agent_session is not None
        try:
            async with self._agent_session.post(
                url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=self._run_timeout),
            ) as response:
                if response.status != 200:
                    LOG.warning(
                        f"NeMo Gym {url} returned {response.status}: "
                        f"{(await response.text())[:200]}"
                    )
                    return None
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            LOG.warning(f"NeMo Gym {url} failed: {type(error).__name__}: {error}")
            return None

    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        t_dispatch = time.monotonic()
        body = agent_request(
            self._group_rows[group_id],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        result = await self._run_agent(body)
        if result is None:
            self._counters["nemo_gym/run_failure_total"] += 1
            return self._UNSCORABLE
        output = (result.get("response") or {}).get("output") or []
        turns = turns_from_agent_output(output)
        if not turns:
            self._counters["nemo_gym/empty_rollout_total"] += 1
            return self._UNSCORABLE
        sequences, tally = _chain_to_sequences(
            turns, uuid.uuid4().hex, self._fork_threshold_tokens
        )
        completion_ids = [token for turn in turns for token in turn.output_ids]
        self._push_rollout_metrics(
            turns=len(turns),
            sequences=len(sequences),
            completion_ids=completion_ids,
            tally=tally,
            loop_exhausted=False,
            duration_s=time.monotonic() - t_dispatch,
        )
        tool_calls = sum(item.get("type") == "function_call" for item in output)
        reward = result.get("reward")
        return (
            messages_from_agent_output(output),
            completion_ids,
            sequences,
            tool_calls,
            0,
            None if reward is None else float(reward),
        )
