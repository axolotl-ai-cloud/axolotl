"""Turn a saved LoRA adapter into the tensors a chain publishes, in serving layout."""

import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open

from .adapters import (
    ADAPTER_WEIGHTS,
    AdapterTargets,
    load_adapter_config,
    validate_adapter,
    validate_adapter_config,
)
from .fingerprint import merge_fingerprint
from .pinned_base import FLOAT_DTYPES, PinnedBase, safetensors_layout

Sink = Callable[[str, torch.Tensor], None]
MERGE_PATH = "merge_lora_sharded_efficient"
# a change to any of these alters what the served adapter means, not its bytes
LORA_CONFIG_KEYS = (
    "r",
    "lora_alpha",
    "use_rslora",
    "rank_pattern",
    "alpha_pattern",
    "target_modules",
    "target_parameters",
    "fan_in_fan_out",
)


def _scale_mode(adapter_config: dict) -> str:
    from axolotl.cli.utils.lora_merge import _resolve_nvfp4_scale_mode

    return _resolve_nvfp4_scale_mode(adapter_config)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


class Exporter:
    """Produce ``(name, tensor)`` pairs for one adapter snapshot."""

    kind: str

    def validate(self, adapter: Path) -> None:
        raise NotImplementedError

    def fingerprint(self, adapter: Path) -> dict:
        raise NotImplementedError

    def layout(self, adapter: Path) -> dict[str, tuple[str, list[int]]] | None:
        """The layout a publish will have, or ``None`` if only known after export."""
        return None

    def export(self, adapter: Path, sink: Sink) -> dict:
        """Stream every tensor to ``sink`` and return the checkpoint's config."""
        raise NotImplementedError


class LoRAExporter(Exporter):
    """Publish the adapter tensors as they were saved."""

    kind = "lora"

    def __init__(self, base: PinnedBase | None = None, base_identity: str = ""):
        self.base = base
        self.base_identity = base.identity if base is not None else base_identity

    def validate(self, adapter):
        if self.base is not None:
            validate_adapter(
                adapter, self.base.layout, num_experts=self.base.num_experts
            )
        else:
            validate_adapter_config(load_adapter_config(Path(adapter)))

    def fingerprint(self, adapter):
        config = load_adapter_config(Path(adapter))
        return merge_fingerprint(
            export="lora",
            base_identity=self.base_identity,
            extra={"adapter": {key: config.get(key) for key in LORA_CONFIG_KEYS}},
        )

    def layout(self, adapter):
        return safetensors_layout(Path(adapter) / ADAPTER_WEIGHTS)

    def export(self, adapter, sink):
        adapter = Path(adapter)
        with safe_open(adapter / ADAPTER_WEIGHTS, framework="pt") as weights:
            for key in sorted(weights.keys()):
                sink(key, weights.get_tensor(key))
        return load_adapter_config(adapter)


class MergedExporter(Exporter):
    """Merge the adapter into the pinned base with the sharded merge.

    ``preserve`` writes the base's own layout (NVFP4 experts re-quantized,
    BF16 elsewhere) and checks every merged tensor against the pinned layout.
    ``bf16`` dequantizes every quantized weight.
    """

    kind = "merged"

    def __init__(
        self,
        base: PinnedBase,
        merged_format: str = "preserve",
        *,
        device: str = "cpu",
        workdir: Path | str | None = None,
        workers: int | None = None,
    ):
        if merged_format not in ("preserve", "bf16"):
            raise ValueError(f"Unknown merged format {merged_format!r}")
        self.base, self.merged_format, self.device = base, merged_format, device
        self.workdir = Path(workdir) if workdir is not None else None
        self.workers = workers

    @property
    def devices(self) -> list[str]:
        """One merge worker per entry; ``device: cuda`` uses every visible GPU."""
        from .parallel_merge import visible_devices

        if self.workers is None and torch.device(self.device).type != "cuda":
            return [self.device]
        return visible_devices(self.device, self.workers)

    @property
    def dequant(self) -> bool:
        return self.merged_format == "bf16" and self.base.format != "unquantized"

    def validate(self, adapter):
        adapter = Path(adapter)
        self.targets(adapter)
        scale_mode = _scale_mode(load_adapter_config(adapter))
        if self.dequant and scale_mode == "fresh":
            raise ValueError(
                "A merge-aware NVFP4 adapter must be published with merged_format: preserve"
            )

    def fingerprint(self, adapter):
        config = load_adapter_config(Path(adapter))
        return merge_fingerprint(
            export="merged",
            base_identity=self.base.identity,
            merge_path=MERGE_PATH + ("-dequant" if self.dequant else ""),
            merge_compile=None,
            scale_mode=_scale_mode(config),
            device=self.device,
            extra={"merged_format": self.merged_format},
        )

    def layout(self, adapter):
        return None if self.dequant else self.base.layout

    def _check(self, name: str, tensor: torch.Tensor):
        entry = (_dtype_name(tensor.dtype), list(tensor.shape))
        if not self.dequant:
            expected = self.base.layout.get(name)
            if entry != expected:
                raise ValueError(
                    f"Merged tensor {name} does not match the pinned base layout: "
                    f"{entry} vs {expected}"
                )
        elif entry[0] not in FLOAT_DTYPES:
            raise ValueError(f"bf16 merge left {name} quantized as {entry[0]}")

    def targets(self, adapter: Path) -> AdapterTargets:
        return validate_adapter(
            adapter, self.base.layout, num_experts=self.base.num_experts
        )

    def _check_report(self, adapter: Path, report: dict):
        targets = self.targets(adapter)
        if report.get("expert_lora_dropped"):
            raise RuntimeError(
                "The merge dropped the expert LoRA (torchao missing or experts not "
                "packed NVFP4); refusing to publish an incomplete merge"
            )
        if targets.expert_groups and (
            report.get("expert_groups") != targets.expert_groups
        ):
            raise RuntimeError(
                f"The merge applied {report.get('expert_groups')} of "
                f"{targets.expert_groups} expert LoRA groups"
            )
        if report.get("merged", 0) < targets.pairs:
            raise RuntimeError(
                f"The merge applied {report.get('merged', 0)} of {targets.pairs} "
                "LoRA targets"
            )

    def export(self, adapter, sink):
        adapter = Path(adapter)
        self.base.verify()
        seen: set[str] = set()

        def consume(tensors, destination, metadata):
            for name, tensor in tensors.items():
                if name in seen:
                    raise ValueError(f"Merge emitted {name} twice")
                self._check(name, tensor)
                seen.add(name)
                sink(name, tensor)

        workdir = self.workdir or self.base.path.parent
        workdir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".merge-", dir=workdir))
        report: dict = {}
        try:
            candidate = staging / "merged"
            devices = self.devices
            if len(devices) > 1:
                from .parallel_merge import merge_sharded_parallel

                merge_sharded_parallel(
                    self.base.path,
                    adapter,
                    candidate,
                    devices=devices,
                    dequant=self.dequant,
                    shard_sink=consume,
                    merge_report=report,
                )
            else:
                from axolotl.cli.utils import lora_merge

                lora_merge.merge_lora_sharded_efficient(
                    self.base.path,
                    adapter,
                    candidate,
                    device=devices[0],
                    dequant=self.dequant,
                    shard_sink=consume,
                    merge_report=report,
                )
            self._check_report(adapter, report)
            if not self.dequant and (missing := self.base.layout.keys() - seen):
                raise ValueError(
                    f"Merged checkpoint is missing {len(missing)} base tensors, "
                    f"e.g. {sorted(missing)[0]}"
                )
            config = json.loads((candidate / "config.json").read_text())
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        if self.dequant:
            config.pop("quantization_config", None)
        return config
