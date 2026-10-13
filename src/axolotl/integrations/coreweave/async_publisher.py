"""CoreWeave model-file selection and publication for the shared background queue."""

import json
from pathlib import Path

from axolotl.core.checkpoint_publisher import (
    BackgroundCheckpointPublisher,
    retain_checkpoint,
)


def retain_model_checkpoint(source, root):
    """Retain only the HF weights and config needed by CoreWeave's publisher."""
    source = Path(source)
    index = source / "model.safetensors.index.json"
    if (source / "adapter_config.json").is_file():
        names = {"adapter_config.json", "adapter_model.safetensors"}
    elif index.is_file():
        manifest = json.loads(index.read_text(encoding="utf-8"))
        names = set(manifest["weight_map"].values()) | {index.name, "config.json"}
    else:
        names = {"model.safetensors", "config.json"}
    for name in names:
        if Path(name).name != name:
            raise ValueError(f"Invalid checkpoint file: {name}")
    return retain_checkpoint(source, root, files=names)


class CoreWeaveBackgroundPublisher(BackgroundCheckpointPublisher):
    """Own a lazy CoreWeave publisher and load snapshots only on the worker."""

    def __init__(self, factory, loader, directory):
        self.factory, self.loader = factory, loader
        self.publisher = None
        super().__init__(
            self.publish_checkpoint,
            directory,
            retain=retain_model_checkpoint,
            thread_name_prefix="coreweave-publish",
        )

    def publish_checkpoint(self, directory, step):
        if self.publisher is None:
            self.publisher = self.factory()
        weights, config = self.loader(directory)
        return self.publisher.publish(weights, config, step)
