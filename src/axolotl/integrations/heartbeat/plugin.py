"""Plugin for adding a heartbeat monitoring system to Axolotl training."""

import logging
from typing import Any, List

import pydantic
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin
from axolotl.utils.callbacks.heartbeat import heartbeat_callback_factory
from axolotl.utils.dict import DictDefault

logger = logging.getLogger(__name__)


class HeartbeatPluginConfig(pydantic.BaseModel):
    """Configuration for the HeartbeatPlugin."""

    enabled: bool = True
    port: int = 224209
    update_frequency: int = 10  # seconds

    class Config:
        """Pydantic config class."""

        extra = "forbid"


class HeartbeatPlugin(BasePlugin):
    """
    Plugin that adds a heartbeat monitoring system to Axolotl training.

    This plugin creates an HTTP endpoint at http://localhost:PORT/heartbeat that
    reports the training status.
    """

    def __init__(self):
        """Initialize the plugin."""
        super().__init__()
        self.config = None

    def register(self, cfg: DictDefault) -> None:
        """Register the plugin with Axolotl.

        Args:
            cfg: Axolotl configuration
        """
        logger.info("Registering HeartbeatPlugin")

        plugin_config = cfg.get("heartbeat", {})
        self.config = HeartbeatPluginConfig(**plugin_config)

        if not self.config.enabled:
            logger.info("HeartbeatPlugin is disabled")
            return

        logger.info(
            "Heartbeat monitoring will be available at "
            f"http://localhost:{self.config.port}/heartbeat"
        )

    def get_input_args(self) -> type:
        """Return the input arguments schema for this plugin."""
        return HeartbeatPluginConfig

    def add_callbacks_pre_trainer(self, cfg: DictDefault, model: Any) -> List[TrainerCallback]:
        """
        Add heartbeat callback before creating the trainer.

        Args:
            cfg: Axolotl configuration
            model: The model being trained

        Returns:
            List of callbacks to add to the trainer
        """
        if not getattr(self, "config", None) or not self.config.enabled:
            return []

        logger.info("Adding heartbeat callback to trainer")
        return [
            heartbeat_callback_factory(
                port=self.config.port,
                update_frequency=self.config.update_frequency,
            )
        ]
