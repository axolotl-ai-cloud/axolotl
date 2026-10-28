# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
NeMo Gym reward functions.

The plugin scores single-turn rollouts with :class:`NemoGymVerifyReward` and
multi-turn rollouts with the agent's own reward, so configs list neither.
``reward_nemo_gym_verify`` remains for using ``/verify`` without the plugin.
"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import requests

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def parse_reward(value: Any, source: str) -> float | None:
    """A NeMo Gym reward, or None (unscored) when it is missing or not a number."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        LOG.warning(f"{source} returned a non-numeric reward: {value!r}")
        return None


# ---------------------------------------------------------------------------
# Multi-turn passthrough reward
# ---------------------------------------------------------------------------


def reward_env(completions, prompts=None, **kwargs):
    """Deprecated passthrough of a precomputed ``env_reward`` column.

    Multi-turn rollouts now score the agent's ``/run`` reward themselves, and the
    plugin drops this function from ``trl.reward_funcs``.
    """
    env_rewards = kwargs.get("env_reward")
    if env_rewards is not None:
        if isinstance(env_rewards, (list, tuple)):
            return [float(r) for r in env_rewards]
        return [float(env_rewards) for _ in completions]
    return [0.0 for _ in completions]


# ---------------------------------------------------------------------------
# Single-turn /verify reward
# ---------------------------------------------------------------------------

# Module-level cache for discovered verify URLs
_verify_urls: dict[str, str] = {}
_verify_urls_lock = __import__("threading").Lock()


def _get_verify_urls(head_port: int = 11000) -> dict[str, str]:
    """Discover verify endpoints from the NeMo Gym head server.

    Results are cached so that the HTTP round-trip only happens once per
    process.  A lock guards against concurrent discovery from multiple
    threads (e.g. async_prefetch background thread + main training thread).
    """
    global _verify_urls
    if _verify_urls:
        return _verify_urls

    with _verify_urls_lock:
        # Double-check after acquiring lock
        if _verify_urls:
            return _verify_urls

        import yaml

        try:
            resp = requests.get(
                f"http://127.0.0.1:{head_port}/global_config_dict_yaml", timeout=5
            )
            config = yaml.safe_load(resp.text)
            if isinstance(config, str):
                config = yaml.safe_load(config)
            for _name, cfg in config.items():
                if not isinstance(cfg, dict):
                    continue
                for srv_name, srv_cfg in cfg.get("resources_servers", {}).items():
                    if (
                        isinstance(srv_cfg, dict)
                        and "host" in srv_cfg
                        and "port" in srv_cfg
                    ):
                        _verify_urls[srv_name] = (
                            f"http://{srv_cfg['host']}:{srv_cfg['port']}/verify"
                        )
        except Exception as exc:
            LOG.warning(f"Failed to discover NeMo Gym verify endpoints: {exc}")

    return _verify_urls


def verify_request(
    extra: dict | None, prompt: str, completion: str, model_name: str
) -> dict:
    """The ``/verify`` body scoring ``completion`` as the response to ``prompt``."""
    request = {k: v for k, v in (extra or {}).items() if v is not None}
    request["responses_create_params"] = {
        "input": [{"role": "user", "content": prompt}]
    }
    request["response"] = {
        "id": "resp",
        "created_at": 0,
        "model": model_name,
        "object": "response",
        "output": [
            {
                "id": "msg",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": completion, "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    return request


def _text(messages, index: int) -> str:
    if not messages:
        return ""
    return messages[index].get("content") or ""


def reward_nemo_gym_verify(completions, prompts=None, **kwargs):
    """Call NeMo Gym ``/verify`` endpoint for each completion (single-turn).

    Requires ``resources_server_ref`` and ``verify_extra`` kwargs, which the
    NeMo Gym dataset loader injects automatically. Discovers the endpoints from a
    head server on the default port; the plugin passes a configured
    :class:`NemoGymVerifyReward` instead.
    """
    verify_urls = _get_verify_urls()
    refs = kwargs.get("resources_server_ref", [])
    extras = kwargs.get("verify_extra", [])
    scores = []

    for i, completion in enumerate(completions):
        srv_name = (
            refs[i]["name"] if i < len(refs) and isinstance(refs[i], dict) else ""
        )
        url = verify_urls.get(srv_name, "")

        if not url:
            scores.append(0.0)
            continue

        req = verify_request(
            extras[i] if i < len(extras) else None,
            _text(prompts[i], 0) if prompts and i < len(prompts) else "",
            _text(completion, 0),
            "axolotl",
        )
        try:
            resp = requests.post(url, json=req, timeout=30)
            reward = resp.json().get("reward", 0.0) if resp.ok else 0.0
        except Exception as exc:
            LOG.warning(f"Verify request to {url} failed: {exc}")
            reward = 0.0

        scores.append(float(reward))

    return scores


class NemoGymVerifyReward:
    """Scores single-turn completions with the resource servers' ``/verify``.

    An instance pickles with its endpoint map, so async GRPO can score with it in
    its rollout process, and it verifies a group's completions concurrently. A
    completion it cannot get a verdict for scores ``None``, which keeps it out of
    its group's baseline.

    Args:
        verify_endpoints: Resource server name to ``/verify`` URL.
        model_name: Model name reported in the verified response.
        timeout: Seconds per ``/verify`` request.
    """

    def __init__(
        self,
        verify_endpoints: dict[str, str],
        model_name: str = "axolotl-model",
        timeout: float = 30.0,
    ):
        self.__name__ = "nemo_gym"
        self.verify_endpoints = dict(verify_endpoints)
        self.model_name = model_name
        self.timeout = timeout

    async def __call__(
        self, completions, prompts=None, **kwargs: Any
    ) -> list[float | None]:
        refs = kwargs.get("resources_server_ref") or []
        extras = kwargs.get("verify_extra") or []
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            cookie_jar=aiohttp.DummyCookieJar(),
        ) as session:
            return list(
                await asyncio.gather(
                    *[
                        self._verify(
                            session,
                            refs[i]["name"] if i < len(refs) and refs[i] else None,
                            verify_request(
                                extras[i] if i < len(extras) else None,
                                _text(prompts[i], 0) if prompts else "",
                                _text(completion, 0),
                                self.model_name,
                            ),
                        )
                        for i, completion in enumerate(completions)
                    ]
                )
            )

    async def _verify(
        self, session: aiohttp.ClientSession, server: str | None, request: dict
    ) -> float | None:
        url = self.verify_endpoints.get(server or "")
        if url is None:
            LOG.warning(f"No NeMo Gym verify endpoint for server {server!r}")
            return None
        try:
            async with session.post(url, json=request) as response:
                if response.status != 200:
                    LOG.warning(
                        f"Verify request to {url} returned {response.status}: "
                        f"{(await response.text())[:200]}"
                    )
                    return None
                body = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            LOG.warning(f"Verify request to {url} failed: {error}")
            return None
        if not isinstance(body, dict):
            LOG.warning(f"Verify request to {url} returned a {type(body).__name__}")
            return None
        return parse_reward(body.get("reward"), f"Verify request to {url}")
