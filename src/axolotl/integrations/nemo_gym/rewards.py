# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
NeMo Gym reward functions.

Provides ready-to-use reward functions for axolotl configs::

    trl:
      reward_funcs:
        # Multi-turn: passthrough reward from agent /run
        - axolotl.integrations.nemo_gym.rewards.reward_env
        # Single-turn: call /verify endpoints directly
        - axolotl.integrations.nemo_gym.rewards.reward_nemo_gym_verify
"""

from __future__ import annotations

from typing import Any

import numpy as np
import requests

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Multi-turn passthrough reward
# ---------------------------------------------------------------------------


def reward_env(completions, prompts=None, **kwargs):
    """Passthrough: extract pre-computed reward from NeMo Gym agent /run response.

    The ``NemoGymDataProducer`` injects ``env_reward`` into each sample's
    kwargs after the agent returns from ``/run``.  This function simply
    forwards that value so TRL can log it alongside other reward signals.

    Use this in your config when ``nemo_gym_multi_turn: true``::

        trl:
          reward_funcs:
            - axolotl.integrations.nemo_gym.rewards.reward_env
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


def reward_nemo_gym_verify(completions, prompts=None, **kwargs):
    """Call NeMo Gym ``/verify`` endpoint for each completion (single-turn).

    Requires ``resources_server_ref`` and ``verify_extra`` kwargs, which the
    NeMo Gym dataset loader injects automatically.

    Use this in your config when ``nemo_gym_multi_turn: false``::

        trl:
          reward_funcs:
            - axolotl.integrations.nemo_gym.rewards.reward_nemo_gym_verify
    """
    verify_urls = _get_verify_urls()
    refs = kwargs.get("resources_server_ref", [])
    extras = kwargs.get("verify_extra", [])
    scores = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if completion else ""
        prompt = prompts[i][0]["content"] if prompts and i < len(prompts) else ""
        srv_name = (
            refs[i]["name"] if i < len(refs) and isinstance(refs[i], dict) else ""
        )
        url = verify_urls.get(srv_name, "")

        if not url:
            scores.append(0.0)
            continue

        extra = extras[i] if i < len(extras) else {}
        req = {k: v for k, v in extra.items() if v is not None}
        req["responses_create_params"] = {
            "input": [{"role": "user", "content": prompt}]
        }
        req["response"] = {
            "id": "resp",
            "created_at": 0,
            "model": "axolotl",
            "object": "response",
            "output": [
                {
                    "id": "msg",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        try:
            resp = requests.post(url, json=req, timeout=30)
            reward = resp.json().get("reward", 0.0) if resp.ok else 0.0
        except Exception as exc:
            LOG.warning(f"Verify request to {url} failed: {exc}")
            reward = 0.0

        scores.append(float(reward))

    return scores


# ---------------------------------------------------------------------------
# Factory used internally by the plugin
# ---------------------------------------------------------------------------


def create_nemo_gym_reward_fn(
    global_config: dict,
    verify_endpoints: dict[str, str],
    model_name: str = "axolotl-model",
    verify_timeout: int = 30,
):
    """Create a reward function bound to specific verify endpoints.

    Used internally by ``NemoGymPlugin._wire_single_turn()`` to inject a
    reward function that already knows the endpoint map (no discovery needed).
    """

    def reward_fn(
        completions: list[list[dict[str, str]]],
        prompts: list[list[dict[str, str]]] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        resources_server_refs = kwargs.get("resources_server_ref", [])
        verify_extras = kwargs.get("verify_extra", [])

        scores = []
        for i, completion in enumerate(completions):
            completion_text = completion[0]["content"]
            task_prompt = prompts[i][0]["content"] if prompts else ""

            server_name = (
                resources_server_refs[i]["name"]
                if i < len(resources_server_refs)
                else None
            )

            if server_name is None or server_name not in verify_endpoints:
                LOG.warning(
                    f"No verify endpoint for server '{server_name}', returning 0 reward"
                )
                scores.append(0.0)
                continue

            verify_endpoint = verify_endpoints[server_name]

            verify_request = (
                {k: v for k, v in verify_extras[i].items() if v is not None}
                if i < len(verify_extras)
                else {}
            )
            verify_request["responses_create_params"] = {
                "input": [{"role": "user", "content": task_prompt}]
            }
            verify_request["response"] = {
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
                            {
                                "type": "output_text",
                                "text": completion_text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }

            try:
                resp = requests.post(
                    verify_endpoint, json=verify_request, timeout=verify_timeout
                )
                if resp.status_code == 200:
                    reward = resp.json().get("reward", 0.0)
                else:
                    LOG.warning(
                        f"Verify request returned status {resp.status_code}: {resp.text[:200]}"
                    )
                    reward = 0.0
            except requests.exceptions.RequestException as exc:
                LOG.warning(f"Verify request failed: {exc}")
                reward = 0.0

            scores.append(float(reward))

        return np.array(scores)

    return reward_fn
