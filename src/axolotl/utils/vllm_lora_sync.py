"""Publish LoRA adapters to a vLLM server through its native adapter routes."""

import os
import shutil

import requests

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _load_adapter(base_url: str, name: str, path: str, timeout: float) -> None:
    response = requests.post(
        f"{base_url}/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": path, "load_inplace": False},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"LoRA adapter sync failed: {response.status_code} {response.text}. "
            "The vLLM server must be started with `axolotl vllm-serve` "
            "(trl.vllm_lora_sync: true) or `vllm serve --enable-lora` with "
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True."
        )


def publish_lora_adapter(
    vllm_client,
    sync_dir: str,
    version: int,
    timeout: float,
    alias: str | None = None,
):
    """Register ``<sync_dir>/v<version>`` under a fresh versioned adapter name.

    Each version gets its own name because vLLM re-reads an adapter reloaded in
    place from disk on every request and keys its prefix cache on the name alone.
    In-flight requests are drained first so at most the previous version can
    still be referenced once generation resumes.

    ``alias`` is a fixed name that is re-pointed at every version, for clients
    that cannot follow the versioned name. It should be the served base model
    name: requests arriving between the unload and the reload then fall back to
    the base model instead of failing.
    """
    base_url = vllm_client.base_url
    base_model = getattr(vllm_client, "_lora_sync_base_model", None)
    if base_model is None:
        base_model = vllm_client.model
        vllm_client._lora_sync_base_model = base_model

    paused = (
        requests.post(
            f"{base_url}/pause", params={"mode": "wait"}, timeout=timeout
        ).status_code
        == 200
    )
    try:
        lora_name = f"{base_model}-v{version}"
        lora_path = os.path.join(sync_dir, f"v{version}")
        _load_adapter(base_url, lora_name, lora_path, timeout)
        if alias:
            requests.post(
                f"{base_url}/v1/unload_lora_adapter",
                json={"lora_name": alias},
                timeout=timeout,
            )
            _load_adapter(base_url, alias, lora_path, timeout)
        vllm_client.model = lora_name
    finally:
        if paused:
            requests.post(f"{base_url}/resume", timeout=timeout)

    stale = version - 2
    if stale > 0:
        requests.post(
            f"{base_url}/v1/unload_lora_adapter",
            json={"lora_name": f"{base_model}-v{stale}"},
            timeout=timeout,
        )
        shutil.rmtree(os.path.join(sync_dir, f"v{stale}"), ignore_errors=True)
    return lora_name
