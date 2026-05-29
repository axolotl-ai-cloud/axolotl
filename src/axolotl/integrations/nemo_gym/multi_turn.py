# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
Multi-turn rollout function for NeMo Gym environments.

Delegates multi-turn orchestration to NeMo Gym's agent servers via the /run
endpoint. The agent handles generation (by calling our vLLM server), tool
execution, session management, and reward computation.

This follows the same pattern as TRL's reference implementation at
examples/scripts/nemo_gym/train_multi_environment.py.

Architecture:
  rollout_func(prompts, trainer)
    -> expand prompts by num_generations
    -> async POST /run to agent servers (one per sample)
    -> parse response: prompt_ids, completion_ids, logprobs, env_mask, reward
    -> return to TRL for GRPO training
"""

from __future__ import annotations

import asyncio
from typing import Any

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def create_nemo_gym_rollout_func(
    agent_servers: dict[str, str],
    dataset_lookup: dict[int, dict],
    request_timeout: float = 10800,
):
    """Create a TRL-compatible rollout_func that delegates to NeMo Gym agents.

    Args:
        agent_servers: Mapping of agent_name → agent URL (e.g., {"simple_agent": "http://host:port"}).
        dataset_lookup: Mapping of dataset index → full JSONL row dict.
        request_timeout: HTTP timeout for /run requests.

    Returns:
        A rollout_func with signature (prompts: list[str], trainer) -> dict.
    """

    def rollout_func(prompts: list[str], trainer) -> dict[str, Any]:
        is_training = trainer.model.training
        num_generations = (
            trainer.num_generations
            if is_training
            else getattr(trainer, "num_generations_eval", 1)
        )
        temperature = trainer.temperature
        top_p = getattr(trainer, "top_p", None) or 0.999
        max_completion_length = trainer.max_completion_length
        eos_token_id = trainer.processing_class.eos_token_id

        # Expand prompts: each prompt index repeated num_generations times
        expanded_items = []
        expanded_prompt_indices = []
        for prompt_str in prompts:
            # Prompts from TRL are chat-templated strings. Find the dataset item
            # by matching against dataset_lookup keys (raw user message text).
            full_item = None
            for key, val in dataset_lookup.items():
                if isinstance(key, str) and prompt_str == key:
                    full_item = val
                    break

            if full_item is None:
                full_item = {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": prompt_str}]
                    }
                }

            for _ in range(num_generations):
                # Preserve agent_ref for routing in _call_agents
                dispatched: dict = full_item.get("verify_extra", full_item)  # type: ignore[assignment]
                if isinstance(dispatched, dict) and "agent_ref" not in dispatched:
                    agent_ref = full_item.get("agent_ref")
                    if agent_ref:
                        dispatched = {**dispatched, "agent_ref": agent_ref}
                expanded_items.append(dispatched)
                expanded_prompt_indices.append(prompt_str)

        # Call NeMo Gym agents
        loop = asyncio.new_event_loop()
        try:
            responses = loop.run_until_complete(
                _call_agents(
                    dataset_items=expanded_items,
                    agent_servers=agent_servers,
                    timeout=request_timeout,
                    max_completion_length=max_completion_length,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        finally:
            loop.close()

        # Parse responses into rollout format
        all_prompt_ids = []
        all_completion_ids = []
        all_env_masks = []
        all_logprobs = []
        all_rewards = []
        all_num_turns = []

        for _i, response in enumerate(responses):
            result = _parse_agent_response(response, eos_token_id)
            all_prompt_ids.append(result["prompt_ids"])
            all_completion_ids.append(result["completion_ids"])
            all_env_masks.append(result["env_mask"])
            all_logprobs.append(result["logprobs"])
            all_rewards.append(result["reward"])
            all_num_turns.append(result["num_turns"])

        # TRL expects prompt_ids to be unique (one per original prompt, not per generation)
        unique_prompt_ids = all_prompt_ids[::num_generations]

        # Wrap logprobs for TRL: list[list[list[float]]]
        def _normalize(lp):
            while isinstance(lp, (list, tuple)) and len(lp) > 0:
                lp = lp[0]
            return float(lp) if lp is not None else 0.0

        wrapped_logprobs = [[[_normalize(lp)] for lp in seq] for seq in all_logprobs]

        return {
            "prompt_ids": unique_prompt_ids,
            "completion_ids": all_completion_ids,
            "env_mask": all_env_masks,
            "logprobs": wrapped_logprobs,
            "logprob_token_ids": None,  # nosec B105
            "env_reward": all_rewards,
            "num_turns": all_num_turns,
        }

    return rollout_func


async def _call_agents(
    dataset_items: list[dict],
    agent_servers: dict[str, str],
    timeout: float,
    max_completion_length: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.999,
) -> list[dict]:
    """Async batch POST to NeMo Gym agent /run endpoints."""
    import aiohttp

    results = []
    connector = aiohttp.TCPConnector(limit_per_host=64, limit=256)
    # Use sock_read for per-request timeout (not total session timeout).
    # This ensures a single stuck generation doesn't block all other requests.
    client_timeout = aiohttp.ClientTimeout(total=None, sock_read=timeout)

    async with aiohttp.ClientSession(
        connector=connector, timeout=client_timeout, cookie_jar=aiohttp.DummyCookieJar()
    ) as session:
        tasks = []
        for item in dataset_items:
            agent_ref = item.get("agent_ref", {})
            agent_name = agent_ref.get("name", "")
            agent_url = agent_servers.get(agent_name, "")

            if not agent_url:
                # Fallback: try first available agent
                if agent_servers:
                    agent_url = next(iter(agent_servers.values()))
                else:
                    results.append(
                        {
                            "response": {"output": []},
                            "reward": 0.0,
                            "error": "No agent server",
                        }
                    )
                    continue

            # Build request body
            request_body = dict(item)
            params = request_body.setdefault("responses_create_params", {})
            params.setdefault("max_output_tokens", max_completion_length)
            params["temperature"] = temperature
            params["top_p"] = top_p

            tasks.append(_post_run(session, agent_url, request_body))

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for resp in responses:
                if isinstance(resp, BaseException):
                    LOG.warning(f"Agent /run failed: {resp}")
                    results.append(
                        {"response": {"output": []}, "reward": 0.0, "error": str(resp)}
                    )
                else:
                    results.append(resp)

    return results


async def _post_run(session, agent_url: str, body: dict) -> dict:
    """POST to agent /run endpoint."""
    async with session.post(f"{agent_url}/run", json=body) as resp:
        if resp.status == 200:
            return await resp.json()
        text = await resp.text()
        return {
            "response": {"output": []},
            "reward": 0.0,
            "error": f"HTTP {resp.status}: {text[:200]}",
        }


def _parse_agent_response(response: dict, eos_token_id: int) -> dict:
    """Parse NeMo Gym agent /run response into rollout format.

    The agent returns:
      response.output[]: list of turns, each with prompt_token_ids,
        generation_token_ids, generation_log_probs
      reward: float
    """
    # Defaults for failed/empty responses
    defaults = {
        "prompt_ids": [eos_token_id],
        "completion_ids": [eos_token_id],
        "env_mask": [0],
        "logprobs": [0.0],
        "reward": 0.0,
        "num_turns": 0,
    }

    if not isinstance(response, dict) or "error" in response:
        return defaults

    output_items = response.get("response", {}).get("output", [])
    reward = float(response.get("reward", 0.0))

    if not output_items:
        defaults["reward"] = reward
        return defaults

    # Check at least one valid output
    has_valid = False
    for item in output_items:
        if item.get("type") == "function_call":
            has_valid = True
            break
        if item.get("type") == "message":
            for c in item.get("content", []):
                if (
                    isinstance(c, dict)
                    and c.get("type") == "output_text"
                    and c.get("text", "").strip()
                ):
                    has_valid = True
                    break
        if has_valid:
            break

    if not has_valid:
        defaults["reward"] = reward
        return defaults

    # Extract multi-turn token sequences
    first_prompt_ids = None
    completion_ids = []
    env_mask = []
    logprobs = []
    seen_token_ids = []
    num_turns = 0

    for item in output_items:
        prompt_token_ids = item.get("prompt_token_ids", [])
        generation_token_ids = item.get("generation_token_ids", [])
        generation_log_probs = item.get("generation_log_probs", [])

        if not generation_token_ids:
            continue

        num_turns += 1

        # First turn: capture prompt
        if first_prompt_ids is None:
            first_prompt_ids = list(prompt_token_ids)
            seen_token_ids = list(prompt_token_ids)
        else:
            # Subsequent turns: extract tool result tokens (between turns)
            if len(prompt_token_ids) > len(seen_token_ids):
                tool_result_tokens = prompt_token_ids[len(seen_token_ids) :]
                # Tool result tokens are NOT trained on (env_mask = 0)
                completion_ids.extend(tool_result_tokens)
                env_mask.extend([0] * len(tool_result_tokens))
                logprobs.extend([0.0] * len(tool_result_tokens))

        # Add generation tokens (trained on, env_mask = 1)
        completion_ids.extend(generation_token_ids)
        env_mask.extend([1] * len(generation_token_ids))

        # Pad logprobs if shorter than generation tokens
        gen_logprobs = list(generation_log_probs) if generation_log_probs else []
        if len(gen_logprobs) < len(generation_token_ids):
            gen_logprobs.extend([0.0] * (len(generation_token_ids) - len(gen_logprobs)))
        logprobs.extend(gen_logprobs[: len(generation_token_ids)])

        # Update seen tokens
        seen_token_ids = list(prompt_token_ids) + list(generation_token_ids)

    if first_prompt_ids is None:
        return defaults

    return {
        "prompt_ids": first_prompt_ids,
        "completion_ids": completion_ids,
        "env_mask": env_mask,
        "logprobs": logprobs,
        "reward": reward,
        "num_turns": num_turns,
    }
