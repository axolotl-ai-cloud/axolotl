# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
NeMo Gym server lifecycle management.

Handles cloning the NeMo Gym repo, starting resource servers,
waiting for readiness, and cleanup on exit.
"""

from __future__ import annotations

import atexit
import os
import subprocess  # nosec B404
import time

import requests
import yaml

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_ng_process = None
_ng_log_file = None


def ensure_gym_repo(gym_dir: str, auto_clone: bool = True) -> str:
    """Clone the NeMo Gym repo if it doesn't exist.

    Args:
        gym_dir: Path to the NeMo Gym directory.
        auto_clone: Whether to auto-clone if missing.

    Returns:
        Resolved path to the NeMo Gym directory.
    """
    gym_dir = os.path.expanduser(gym_dir)
    if os.path.exists(gym_dir):
        LOG.info(f"NeMo Gym directory exists at {gym_dir}")
        return gym_dir

    if not auto_clone:
        raise FileNotFoundError(
            f"NeMo Gym directory not found at {gym_dir} and auto_clone is disabled."
        )

    LOG.info(f"Cloning NeMo Gym to {gym_dir}...")
    subprocess.run(  # nosec
        ["git", "clone", "https://github.com/NVIDIA-NeMo/Gym.git", gym_dir],
        check=True,
    )
    return gym_dir


def ensure_gym_venv(gym_dir: str):
    """Set up the NeMo Gym Python venv if not present."""
    venv_python = os.path.join(gym_dir, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return

    LOG.info("Setting up NeMo Gym venv...")
    subprocess.run(["uv", "venv", "--python", "3.12"], cwd=gym_dir, check=True)  # nosec
    subprocess.run(  # nosec
        ["bash", "-c", "source .venv/bin/activate && uv sync"],
        cwd=gym_dir,
        check=True,
    )


def start_servers(
    gym_dir: str,
    config_paths: list[str],
    head_port: int = 11000,
    timeout: int = 360,
):
    """Start NeMo Gym resource servers via ng_run.

    Args:
        gym_dir: Path to the NeMo Gym directory.
        config_paths: List of config YAML paths relative to gym_dir.
        head_port: Port for the head server.
        timeout: Max seconds to wait for servers.
    """
    global _ng_process, _ng_log_file

    head_url = f"http://127.0.0.1:{head_port}/global_config_dict_yaml"

    # Check if already running
    try:
        requests.get(head_url, timeout=2)
        LOG.info(f"NeMo Gym servers already running on port {head_port}.")
        return
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass

    ng_run_bin = os.path.join(gym_dir, ".venv", "bin", "ng_run")
    config_arg = f"+config_paths=[{','.join(config_paths)}]"
    _ng_log_file = open(os.path.join(gym_dir, "ng_run.log"), "w")  # noqa: SIM115
    _ng_process = subprocess.Popen(  # nosec B603
        [ng_run_bin, config_arg, "+skip_venv_if_present=true"],
        cwd=gym_dir,
        stdout=_ng_log_file,
        stderr=subprocess.STDOUT,
    )

    atexit.register(_cleanup_servers)

    LOG.info("Waiting for NeMo Gym head server...")
    for _ in range(timeout // 3):
        try:
            requests.get(head_url, timeout=2)
            LOG.info("NeMo Gym head server is ready.")
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if _ng_process.poll() is not None:
                raise RuntimeError(
                    "NeMo Gym server process exited unexpectedly. "
                    f"Check {gym_dir}/ng_run.log for details."
                ) from None
            time.sleep(3)

    raise RuntimeError(
        f"NeMo Gym servers did not start within {timeout}s. "
        f"Check {gym_dir}/ng_run.log for details."
    )


def get_server_configs(head_port: int = 11000) -> dict:
    """Fetch the global config from the NeMo Gym head server.

    Returns:
        Dict mapping server_name -> server config.
    """
    response = requests.get(
        f"http://127.0.0.1:{head_port}/global_config_dict_yaml", timeout=5
    )
    response.raise_for_status()
    result = yaml.safe_load(response.text)
    # NeMo Gym head server double-encodes: YAML string inside a YAML string
    if isinstance(result, str):
        result = yaml.safe_load(result)
    return result


def get_agent_servers(
    global_config: dict, head_host: str = "127.0.0.1"
) -> dict[str, str]:
    """Discover NeMo Gym agent servers from the global config.

    Agent servers handle multi-turn orchestration via /run endpoint.
    Returns mapping of agent_name → URL (e.g., {"simple_agent": "http://host:port"}).
    """
    agents = {}
    for top_name, top_cfg in global_config.items():
        if not isinstance(top_cfg, dict):
            continue
        agent_dict = top_cfg.get("responses_api_agents", {})
        if not agent_dict:
            continue
        for _agent_name, agent_cfg in agent_dict.items():
            if not isinstance(agent_cfg, dict):
                continue
            host = agent_cfg.get("host", "127.0.0.1")
            port = agent_cfg.get("port")
            if not port:
                continue
            # Replace loopback with head_host for remote access
            host = _normalize_host(host, fallback=head_host)
            # Use the top-level config name (not the inner agent name)
            # because dataset agent_ref.name references the top-level name
            agents[top_name] = f"http://{host}:{port}"
    if agents:
        LOG.info(f"Discovered NeMo Gym agent servers: {agents}")
    return agents


def _normalize_host(host: str, fallback: str = "127.0.0.1") -> str:
    """Normalize bind-all and loopback addresses for reachability."""
    if host in ("0.0.0.0", "localhost"):  # nosec B104
        return fallback
    return host


def get_server_base_url(global_config: dict, server_name: str) -> str:
    """Get the base URL for a given resource server."""
    try:
        srv_cfg = global_config[server_name]["resources_servers"][server_name]
        host = _normalize_host(srv_cfg["host"])
        return f"http://{host}:{srv_cfg['port']}"
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Could not find resource server config for '{server_name}' in NeMo Gym. "
            f"Available servers: {list(global_config.keys())}"
        ) from exc


def get_verify_endpoint(global_config: dict, server_name: str) -> str:
    """Get the /verify endpoint URL for a given resource server."""
    return f"{get_server_base_url(global_config, server_name)}/verify"


def wait_for_resource_servers(global_config: dict, timeout: int = 180):
    """Wait for all resource servers in the config to become reachable."""
    for srv_name in global_config:
        try:
            srv_cfg = global_config[srv_name]["resources_servers"][srv_name]
        except (KeyError, TypeError):
            continue  # Skip non-server config entries silently

        host, port = _normalize_host(srv_cfg["host"]), srv_cfg["port"]
        LOG.info(f"Waiting for resource server '{srv_name}' at {host}:{port}...")
        for _ in range(timeout // 2):
            try:
                requests.get(f"http://{host}:{port}/", timeout=2)
                LOG.info(f"Resource server '{srv_name}' is ready.")
                break
            except requests.exceptions.ConnectionError:
                time.sleep(2)
        else:
            raise RuntimeError(
                f"Resource server '{srv_name}' at {host}:{port} "
                f"did not start within {timeout}s."
            )


def _cleanup_servers():
    """Terminate NeMo Gym server process on exit."""
    global _ng_process, _ng_log_file
    if _ng_process is not None and _ng_process.poll() is None:
        LOG.info("Terminating NeMo Gym servers...")
        _ng_process.terminate()
        try:
            _ng_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _ng_process.kill()
    if _ng_log_file is not None:
        _ng_log_file.close()
