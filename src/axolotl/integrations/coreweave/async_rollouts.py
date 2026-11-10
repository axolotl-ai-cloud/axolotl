"""Async GRPO rollouts from a CoreWeave Dedicated Inference deployment.

Runs in async GRPO's CUDA-free rollout process. Turns are sampled by TRL's own
``/v1/completions`` request, sent to the deployment's gateway with its bearer token.
The trainer publishes the merged weights itself (``checkpoint_publish``) and hot-loads
them at every weight sync; the rollout worker's version is the sync every replica
serves, set only once the hot-load converged, and no rollout starts before the first
one has.
"""

from __future__ import annotations

import asyncio
import os

import aiohttp

from axolotl.core.trainers.async_grpo.rollout import AxolotlRolloutLoop
from axolotl.utils.logging import get_logger

from .client import completion_token_ids

LOG = get_logger(__name__)

# Statuses a retry cannot fix; anything else (429, 5xx during a hot-load) is retried.
_REJECTED = frozenset(range(400, 500)) - {408, 429}
_VLLM_ONLY = frozenset({"return_token_ids", "top_k", "repetition_penalty", "min_p"})


class CoreWeaveRolloutLoop(AxolotlRolloutLoop):
    """Samples from a CoreWeave gateway (see module doc).

    Args:
        gateway_url: The deployment's OpenAI-compatible gateway.
        model: The model name the deployment serves.
        api_token_env: Environment variable holding the gateway's bearer token.
        ready_poll_s: How often to check whether the first publish is served.
    """

    _requires_vllm = False

    def __init__(
        self,
        *,
        gateway_url: str,
        model: str,
        api_token_env: str = "CW_API_TOKEN",  # nosec B107
        ready_poll_s: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.sampling_replay:
            raise ValueError(
                "CoreWeave rollouts sample through /v1/completions, which does not "
                "return the candidate set top_p/top_k sampled each token from; use "
                "top_p: 1.0 and top_k: 0."
            )
        token = os.environ.get(api_token_env)
        if not token:
            raise ValueError(f"Set {api_token_env} to the CoreWeave API token")
        self.vllm_server_url = gateway_url.rstrip("/")
        self._served_model = model
        self._token = token
        self._ready_poll_s = ready_poll_s

    @property
    def _request_model(self) -> str:
        return self._served_model

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        # The gateway has none of vLLM's /metrics or /server_info, so TRL's server
        # probes are skipped and the in-flight cap stays at max_inflight_tasks.
        connector = aiohttp.TCPConnector(limit=self.max_inflight_tasks + 1)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"Authorization": f"Bearer {self._token}"},
        ) as session:
            self.session = session
            LOG.info(
                f"CoreWeave rollouts from {self.vllm_server_url} ({self._served_model}): "
                f"num_generations={self.num_generations}, "
                f"max_inflight_tasks={self.max_inflight_tasks}, "
                f"temperature={self.temperature}"
            )
            await asyncio.gather(
                asyncio.create_task(self._generate_once_served(stop_event)),
                asyncio.create_task(self._score_loop(stop_event)),
            )

    async def _generate_once_served(self, stop_event: asyncio.Event) -> None:
        # The score loop keeps the heartbeat while the first publish uploads.
        while self.model_version < 0 and not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._ready_poll_s)
            except TimeoutError:
                pass
        if stop_event.is_set():
            return
        LOG.info(f"CoreWeave serves version {self.model_version}; starting rollouts")
        await self._generate_loop(stop_event)

    async def _post(
        self, path: str, payload: dict, timeout: float, max_retries: int = 3
    ) -> dict:
        # The sync rollout path's request: ids come back spelled `token_id:<id>`, and
        # vLLM-only sampling extras (neutral, see the plugin's checks) are left out.
        payload = {k: v for k, v in payload.items() if k not in _VLLM_ONLY}
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        async def post():
            assert self.session is not None
            async with self.session.post(
                f"{self.vllm_server_url}{path}", json=payload, timeout=client_timeout
            ) as response:
                if response.status in _REJECTED:
                    raise RuntimeError(
                        f"CoreWeave gateway rejected {path} ({response.status}): "
                        f"{(await response.text())[:500]}. Rollouts need token-ID "
                        "prompts and return_tokens_as_token_ids."
                    )
                response.raise_for_status()
                return await response.json() or {}

        content = await self._retry(
            post, label=f"POST {path}", max_attempts=max_retries
        )
        if path == "/v1/completions":
            for choice in content.get("choices", []):
                choice["token_ids"] = completion_token_ids(choice)
                # JSON has no -inf: the gateway sends a zero-probability candidate's
                # log-prob as null.
                for top in choice["logprobs"].get("top_logprobs") or []:
                    for token, logprob in top.items():
                        if logprob is None:
                            top[token] = float("-inf")
        return content
