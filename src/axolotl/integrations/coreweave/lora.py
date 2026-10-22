"""Merged LoRA publication against an exact, committed disk baseline."""

from pathlib import Path

from axolotl.core.publishing.exporters import MergedExporter
from axolotl.core.publishing.pinned_base import PinnedBase, pin_base

from .checkpoints import CheckpointPublisher


def adapter_checkpoint(directory):
    """Keep adapter tensors on disk until the ordered publication worker runs."""
    return Path(directory), None


class LoRACheckpointPublisher(CheckpointPublisher):
    """Merge every adapter against the pinned original base, never prior merges."""

    def __init__(
        self,
        *args,
        base_model,
        revision=None,
        merged_format="preserve",
        device="cpu",
        merge_workers=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.revision = revision
        self.merged_format, self.device = merged_format, device
        self.merge_workers = merge_workers
        self.pinned: PinnedBase | None = None
        self.exporter: MergedExporter | None = None

    def _pin_base(self) -> MergedExporter:
        if self.exporter is None:
            source = Path(self.base_model)
            if not source.is_dir():
                from huggingface_hub import snapshot_download

                source = Path(
                    snapshot_download(
                        self.base_model,
                        revision=self.revision,
                        allow_patterns=["*.safetensors", "*.json"],
                    )
                )
            self.pinned = pin_base(source, self.directory, revision=self.revision)
            self.exporter = MergedExporter(
                self.pinned,
                self.merged_format,
                device=self.device,
                workdir=self.directory,
                workers=self.merge_workers,
            )
        return self.exporter

    def fingerprint(self, adapter):
        return self._pin_base().fingerprint(Path(adapter))

    def publish(self, weights, model_config, step, *, fingerprint=None):
        self.check_ready()
        return self.publish_export(
            self._pin_base(),
            Path(weights),
            self.new_identity(step),
            fingerprint=fingerprint,
        )
