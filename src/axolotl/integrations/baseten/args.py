"""Baseten image and registry configuration."""

from typing import Any

from axolotl.utils.schemas.cloud import CloudImageConfig


class BasetenImageConfig(CloudImageConfig):
    """Image source and native Truss DockerAuth configuration for Baseten."""

    docker_auth: dict[str, Any] | None = None
