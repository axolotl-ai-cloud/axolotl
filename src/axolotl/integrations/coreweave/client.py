"""CoreWeave management and OpenAI-compatible rollout API client."""

import os
import re
import time
from urllib.parse import quote

import requests

from .args import CoreWeaveConfig


class CoreWeaveClient:
    """Gate rollout generation on confirmed replica convergence."""

    def __init__(self, config: CoreWeaveConfig, session=None):
        self.config = config
        self.session = session or requests.Session()
        self.token = os.environ[config.api_token_env]
        self.ready_identity = None

    def _request(self, method, path, *, gateway=False, **kwargs):
        url = self.config.gateway_url if gateway else self.config.api_url
        response = self.session.request(
            method,
            url.rstrip("/") + path,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=min(self.config.timeout, 60),
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def hot_load(self, identity, *, base_identity=None):
        from axolotl.utils.delta_checkpoint import validate_identity

        validate_identity(identity)
        self.ready_identity = None
        body = {
            "deploymentId": self.config.deployment_id,
            "identity": identity,
            "type": "SNAPSHOT_TYPE_FULL",
            "promptCachePolicy": "PROMPT_CACHE_POLICY_RESET_ALL",
        }
        if base_identity is not None:
            validate_identity(base_identity)
            body["type"] = "SNAPSHOT_TYPE_INCREMENTAL"
            body["incrementalSnapshotMetadata"] = {
                "previousSnapshotIdentity": base_identity,
                "compressionFormat": "COMPRESSION_FORMAT_ZSTD",
                "checksumFormat": "CHECKSUM_FORMAT_ADLER32",
            }
        result = self._request("POST", "/v1alpha1/inference/hotLoads", json=body)
        operation = result["hotLoad"]["spec"]["id"]
        self.wait_for_hot_load(operation, identity)
        return operation

    def wait_for_hot_load(self, operation, identity):
        self.ready_identity = None
        path = f"/v1alpha1/inference/hotLoads/{quote(operation, safe='')}"
        deadline = time.monotonic() + self.config.timeout
        while time.monotonic() < deadline:
            result = self._request("GET", path)
            spec = result["hotLoad"]["spec"]
            if (
                spec["identity"] != identity
                or spec["deploymentId"] != self.config.deployment_id
            ):
                raise RuntimeError(
                    "Hot-load operation does not match the requested checkpoint"
                )
            state = result["hotLoad"]["status"]["state"]
            if state in {"HOT_LOAD_STATE_FAILED", "HOT_LOAD_STATE_CANCELED"}:
                raise RuntimeError(f"CoreWeave hot-load {operation}: {state}")
            if state == "HOT_LOAD_STATE_COMPLETED" and self._replicas_ready(
                path, identity, deadline
            ):
                self.ready_identity = identity
                return
            time.sleep(
                min(self.config.poll_interval, max(0, deadline - time.monotonic()))
            )
        raise TimeoutError(
            f"CoreWeave hot-load {operation} did not converge to {identity}"
        )

    def _replicas_ready(self, path, identity, deadline):
        token, seen, count, ready = "", set(), 0, True
        while time.monotonic() < deadline:
            page = self._request(
                "GET",
                path + "/replicas",
                params={"maxPageSize": 1000, "pageToken": token},
            )
            for replica in page.get("replicas", []):
                count += 1
                if replica["state"] == "ERROR":
                    raise RuntimeError(
                        f"CoreWeave replica failed: {replica.get('errorReason', 'unknown')}"
                    )
                ready &= (
                    replica["state"] == "READY"
                    and replica.get("currentSnapshotIdentity") == identity
                )
            token = page.get("nextPageToken", "")
            if not token:
                return count > 0 and ready
            if token in seen:
                raise RuntimeError("CoreWeave returned a repeated replica page token")
            seen.add(token)
        return False

    def sample(self, prompt_ids, *, n, max_tokens, temperature, top_p=1.0):
        if self.ready_identity is None:
            raise RuntimeError(
                "Publish weights and wait for replica convergence before sampling"
            )
        if temperature <= 0:
            raise ValueError("RL sampling requires a positive temperature")
        if not prompt_ids or any(
            type(token) is not int or token < 0 for token in prompt_ids
        ):
            raise ValueError("Rollout prompts must be nonempty lists of token IDs")
        try:
            result = self._request(
                "POST",
                "/v1/completions",
                gateway=True,
                json={
                    "model": self.config.model,
                    "prompt": prompt_ids,
                    "n": n,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": 1,
                    "return_tokens_as_token_ids": True,  # nosec B105
                    "echo": False,
                },
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in {
                400,
                404,
                405,
                422,
            }:
                raise RuntimeError(
                    "Gateway rejected token-based /v1/completions; use a deployment that supports "
                    "integer prompts and return_tokens_as_token_ids. No chat fallback was attempted."
                ) from exc
            raise
        choices = result["choices"]
        if len(choices) != n:
            raise ValueError("CoreWeave returned an unexpected number of completions")
        samples = []
        for choice in choices:
            logprobs = choice["logprobs"]["token_logprobs"]
            tokens = choice.get("token_ids")
            if tokens is None:
                encoded = choice["logprobs"].get("tokens", [])
                if not encoded or any(
                    not isinstance(token, str)
                    or not re.fullmatch(r"token_id:[0-9]+", token)
                    for token in encoded
                ):
                    raise ValueError(
                        "Gateway must support exact completion token IDs; text retokenization is unsafe"
                    )
                tokens = [int(token.split(":", 1)[1]) for token in encoded]
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("Gateway returned invalid completion token IDs")
            if len(tokens) != len(logprobs):
                raise ValueError(
                    "CoreWeave token IDs and log probabilities are not aligned"
                )
            samples.append({"tokens": tokens, "logprobs": logprobs})
        return samples
