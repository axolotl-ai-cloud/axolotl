"""Export pipeline: baked bf16 master → packed deployment formats.

`master_bf16` is produced during training by `TernaryLinear._post_training`; the
packers here re-derive codes from that master (exact by construction) and each
export is gated by `parity.run_parity_gate`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from ..args import EmbeddingDtype, ExportFormat, resolve_ternary_config
from ..swap import SwapManifest
from . import bake, parity

if TYPE_CHECKING:
    from .gguf_tq import GGUFTernaryType

LOG = get_logger(__name__)

GGUF_QUANT_TYPES: dict[str, GGUFTernaryType] = {
    "gguf_tq2_0": "tq2_0",
    "gguf_tq1_0": "tq1_0",
}


def run_export(
    cfg: DictDefault,
    formats: Sequence[ExportFormat] | None = None,
    master_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    manifest: SwapManifest | None = None,
) -> dict[str, Path]:
    """Write every requested export format from the baked master.

    Args:
        cfg: The axolotl config; `ternary.export` supplies the defaults.
        formats: Formats to write; defaults to `ternary.export.formats`.
        master_dir: Where the baked master lives; defaults to `cfg.output_dir`.
        output_dir: Where exports are written; defaults to `master_dir`.
        manifest: The live manifest from the run that produced the master (it also
            carries the λ the bake happened at); read from `master_dir` when `None`.

    Returns:
        Mapping of format name to the artifact path (a file or a directory).

    Raises:
        FileNotFoundError: If the master or its swap manifest is missing.
        ImportError: If a requested GGUF format's optional `gguf` package is absent.
        ValueError: If a format is unknown or the manifest cannot represent it.
        RuntimeError: If a parity gate fails and `ternary.export.run_parity_gate`
            is enabled.
    """
    ternary_cfg = resolve_ternary_config(cfg)
    requested = list(formats if formats is not None else ternary_cfg.export.formats)
    if not requested:
        return {}

    source = Path(
        master_dir if master_dir is not None else cfg.get("output_dir") or "."
    )
    output = Path(output_dir) if output_dir is not None else source
    if manifest is None:
        manifest = SwapManifest.load(source)
    gate = ternary_cfg.export.run_parity_gate
    if gate and manifest.final_lambda is not None and manifest.final_lambda < 1.0:
        LOG.warning(
            f"ternary: the master was baked at lambda={manifest.final_lambda:g}; the "
            "exports below are fully ternary models that were never trained or "
            "evaluated in that state"
        )

    # every packer re-derives codes from the master, so it has to be baked first;
    # bake_directory is idempotent, leaving a master baked at save time byte-identical
    master = bake.bake_directory(source, manifest=manifest)

    artifacts: dict[str, Path] = {}
    for fmt in requested:
        artifacts[fmt] = _write_artifact(
            fmt, master, output, manifest, gate, ternary_cfg.export.embedding_dtype
        )
        if gate:
            _gate_artifact(fmt, master, artifacts[fmt], manifest)
    return artifacts


def _write_artifact(
    fmt: ExportFormat,
    master: Path,
    output: Path,
    manifest: SwapManifest,
    gate: bool,
    embedding_dtype: EmbeddingDtype = "bf16",
) -> Path:
    from ..subln import assert_subln_exportable

    assert_subln_exportable(manifest, fmt)
    if fmt == "master_bf16":
        if output.resolve() == master.resolve():
            return master
        return bake.bake_directory(master, output / fmt, manifest)
    if fmt == "hf_bitnet":
        from . import hf_bitnet

        return hf_bitnet.export_hf_bitnet(master, output / fmt, manifest)
    if fmt in GGUF_QUANT_TYPES:
        from . import gguf_tq

        return gguf_tq.export_gguf_tq(
            master,
            output / fmt,
            manifest,
            GGUF_QUANT_TYPES[fmt],
            gate=gate,
            embedding_dtype=embedding_dtype,
        )
    if fmt == "mask_sign":
        from . import mask_sign

        return mask_sign.export_mask_sign(master, output / fmt, manifest)
    if fmt == "i2_s":
        from . import i2s

        return i2s.export_i2s(
            master, output / fmt, manifest, gate=gate, embedding_dtype=embedding_dtype
        )
    raise ValueError(f"unknown ternary export format {fmt!r}")


def _gate_artifact(
    fmt: ExportFormat, master: Path, artifact: Path, manifest: SwapManifest
) -> None:
    if fmt in parity.BLOCK_GATED_FORMATS:
        # gated byte-for-byte while it was packed: the container holds exactly the
        # blocks that round-tripped back to the master's codes
        return
    if fmt not in parity.EXTRACTORS:
        raise RuntimeError(
            f"ternary: {fmt} has no parity unpacker and no block gate, so it cannot be "
            "verified against the master; set ternary.export.run_parity_gate: false to "
            "write it unverified"
        )
    # the master is the gate's own reference, so a smoke eval would compare a model
    # against itself; unpacking it still proves it is baked
    report = parity.run_parity_gate(
        master, artifact, fmt, manifest, run_smoke=fmt != "master_bf16"
    )
    if not report.passed:
        raise RuntimeError(
            f"ternary: the {fmt} export failed its parity gate: "
            + "; ".join(report.failures[:5])
        )
    LOG.info(
        f"ternary: {fmt} parity gate passed ({report.tensors_checked} tensors, "
        f"max dequant error {report.max_dequant_error:.3e})"
    )
    smoke = report.smoke
    if smoke is not None and smoke.cross_quantizer_logit_delta is not None:
        # reported, never gated: a deep swapped model amplifies the ulp-level
        # disagreement between two equivalent quantizers to O(1) logits
        LOG.info(
            f"ternary: {fmt} deployment activation quantizer differs from the "
            f"training one on {smoke.runtime_code_mismatch:.4%} of codes "
            f"({smoke.probes_compared} probes), shifting fixed-prompt logits by up "
            f"to {smoke.cross_quantizer_logit_delta:.3e}"
        )


__all__ = ["run_export"]
