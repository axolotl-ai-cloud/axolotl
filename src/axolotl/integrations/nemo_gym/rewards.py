# Copyright 2024 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
NeMo Gym reward functions.

Creates TRL-compatible reward functions that call NeMo Gym /verify endpoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import requests

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def create_nemo_gym_reward_fn(
    global_config: dict,
    verify_endpoints: dict[str, str],
    model_name: str = "axolotl-model",
    verify_timeout: int = 30,
):
    """Create a TRL-compatible reward function that calls NeMo Gym /verify endpoints.

    The reward function expects extra kwargs passed through from the dataset:
    - resources_server_ref: list[dict] with {"name": server_name}
    - verify_extra: list[dict] with the original JSONL data for verify requests

    Args:
        global_config: The NeMo Gym global config dict.
        verify_endpoints: Mapping of server_name -> verify endpoint URL.
        model_name: Model name to report in verify requests.
        verify_timeout: Timeout for each /verify request.

    Returns:
        A callable reward function with signature (completions, prompts=None, **kwargs).
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

            # Build the verify request from the original JSONL data
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
