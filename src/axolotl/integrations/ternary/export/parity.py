"""Parity gates: every export must reproduce the master's codes exactly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from axolotl.utils.logging import get_logger

from .. import quant
from ..args import ExportFormat
from ..swap import SwapManifest
from . import bake, hf_bitnet

LOG = get_logger(__name__)

# half an f16 mantissa ulp: the most a per-tensor scale can drift when an export
# stores it as f16, which bounds the whole dequantization error since |code| <= 1
F16_HALF_ULP: float = 2.0**-11

SMOKE_TOKENS: int = 16
SMOKE_LOGIT_TOL: float = 1e-2

# format -> (artifact path, manifest) -> {module name: (int8 codes, fp32 scale)}
CodeExtractor = Callable[
    [Path, SwapManifest], dict[str, tuple[torch.Tensor, torch.Tensor]]
]
EXTRACTORS: dict[str, CodeExtractor] = {}

# formats whose blocks are gated as they are packed: the container is written by an
# external library, so the artifact cannot be re-read without a GGUF parser, but the
# bytes handed to it are round-tripped through our own unpacker first
BLOCK_GATED_FORMATS: frozenset[str] = frozenset({"gguf_tq2_0", "gguf_tq1_0", "i2_s"})

# format -> packed uint8 bytes, (rows, cols) -> (int8 codes, fp32 block scales)
BlockDecoder = Callable[
    [torch.Tensor, "tuple[int, int]"], "tuple[torch.Tensor, torch.Tensor]"
]
BLOCK_DECODERS: dict[str, BlockDecoder] = {}


def register_block_decoder(fmt: str, decoder: BlockDecoder) -> None:
    """Register the byte-level unpacker a block-gated format is verified with."""
    BLOCK_DECODERS[fmt] = decoder


def gate_block_bytes(
    fmt: str, packed: torch.Tensor, master_weight: torch.Tensor
) -> list[str]:
    """Round-trip one tensor's packed bytes back to codes and gate them on the master.

    The container these bytes go into is written by an external library, so this is
    where a hand-rolled block layout is verified: exactly the byte string handed to
    the writer is decoded again and compared to the codes and `s16` the master
    yields.

    Args:
        fmt: Export format name, as registered by `register_block_decoder`.
        packed: The uint8 byte string handed to the container writer.
        master_weight: The baked master tensor the bytes were packed from.

    Returns:
        A list of failure descriptions; empty when the bytes round-trip exactly.
    """
    decoder = BLOCK_DECODERS.get(fmt)
    if decoder is None:
        return [f"no block unpacker is registered for {fmt}"]
    codes, scale = bake.derive_codes_and_scale(master_weight)
    shape = (int(master_weight.shape[0]), int(master_weight.shape[1]))
    try:
        unpacked, block_scales = decoder(packed, shape)
    except ValueError as exc:
        return [f"could not unpack the {fmt} blocks: {exc}"]

    failures: list[str] = []
    if unpacked.shape != codes.shape:
        return [
            f"unpacked shape {tuple(unpacked.shape)} does not match the master's "
            f"{tuple(codes.shape)}"
        ]
    mismatches = int((codes != unpacked.to(codes.device)).sum())
    if mismatches:
        failures.append(
            f"{mismatches} of {codes.numel()} codes differ from the master after "
            f"a {fmt} block round-trip"
        )
    reference = float(scale.reshape(()))
    drift = float((block_scales.to(torch.float32) - reference).abs().amax())
    if drift > reference * F16_HALF_ULP:
        failures.append(
            f"block scale drifted {drift:.3e} from the master's s16 {reference:.3e}, "
            f"beyond the f16 rounding bound {reference * F16_HALF_ULP:.3e}"
        )
    return failures


@dataclass
class ParityReport:
    """Outcome of gating one exported artifact against the baked master."""

    format: str
    tensors_checked: int = 0
    code_mismatches: int = 0
    max_dequant_error: float = 0.0
    dequant_error_bound: float = 0.0
    smoke_max_logit_delta: float | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the artifact cleared every gate."""
        return (
            not self.failures
            and self.code_mismatches == 0
            and self.max_dequant_error <= self.dequant_error_bound
        )


def register_extractor(fmt: str, extractor: CodeExtractor) -> None:
    """Register the unpacker a format's parity gate reads its codes with."""
    EXTRACTORS[fmt] = extractor


def check_code_parity(master_weight: torch.Tensor, unpacked_codes: torch.Tensor) -> int:
    """Return how many codes differ between the master-derived codes and `unpacked_codes`."""
    return _code_mismatches(master_weight, unpacked_codes, group_size=None)


def check_dequant_error(
    master_weight: torch.Tensor, dequantized: torch.Tensor
) -> tuple[float, float]:
    """Return `(max_abs_error, bound)` for a round-tripped tensor.

    The bound is the f16-rounding error of the scale: an export that stores the same
    codes and an f16 scale cannot drift further than that.
    """
    master = master_weight.detach().to(torch.float32)
    if master.shape != dequantized.shape:
        raise ValueError(
            f"shape mismatch: master {tuple(master.shape)} vs round-trip "
            f"{tuple(dequantized.shape)}"
        )
    error = (master - dequantized.to(torch.float32)).abs().amax()
    return float(error), float(master.abs().amax()) * F16_HALF_ULP


def bitlinear_act_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token activation quantization as `BitNetForCausalLM`'s BitLinear runs it.

    Deliberately not `quant.act_quant`: the runtime scales by `127 / amax` (flooring
    `amax`, not the scale) and clamps to `[-128, 127]`, where training divides by
    `amax / 127` (flooring the scale) and clamps symmetrically. Reproducing it is the
    whole point of the smoke eval — it is the one gate that sees the deployment
    numerics rather than the packing.
    """
    scale = quant.ACT_QMAX / x.float().abs().amax(dim=-1, keepdim=True).clamp_min(
        quant.SCALE_EPS
    )
    codes = (x.float() * scale).round().clamp_(-quant.ACT_QMAX - 1, quant.ACT_QMAX)
    return (codes / scale).to(x.dtype)


# format -> the activation quantizer its runtime applies before each ternary linear
RUNTIME_ACT_QUANT: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "master_bf16": lambda x: quant.act_quant(x, 1.0),
    "hf_bitnet": bitlinear_act_quant,
}


def _install_act_quant(
    model: torch.nn.Module, names: list[str], fn: Callable[[torch.Tensor], torch.Tensor]
) -> list[torch.utils.hooks.RemovableHandle]:
    def pre_hook(_module, args):
        return (fn(args[0]),) + tuple(args[1:])

    return [
        model.get_submodule(name).register_forward_pre_hook(pre_hook) for name in names
    ]


def run_smoke_eval(
    master_dir: str | Path, artifact: str | Path, fmt: ExportFormat
) -> float | None:
    """Return the max logit delta between the master and the artifact on fixed prompts.

    The reference is the baked master run the way training ran it (λ=1, per-token
    activation quantization); the comparison side substitutes the weights decoded
    from the artifact *and* the activation quantization the artifact's own runtime
    performs. The delta is therefore the deployment mismatch, not a re-run of the
    code-parity check.

    Returns `None` when the format has no in-process runtime available.
    """
    extractor = EXTRACTORS.get(fmt)
    runtime_act_quant = RUNTIME_ACT_QUANT.get(fmt)
    if extractor is None or runtime_act_quant is None:
        return None
    try:
        from transformers import AutoModelForCausalLM

        manifest = SwapManifest.load(master_dir)
        model = AutoModelForCausalLM.from_pretrained(master_dir, dtype=torch.float32)
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        LOG.warning(f"ternary: skipping the {fmt} smoke eval ({exc})")
        return None

    codes = extractor(Path(artifact), manifest)
    names = [entry.name for entry in manifest.entries if entry.name in codes]
    model.eval()
    input_ids = (torch.arange(SMOKE_TOKENS) % model.config.vocab_size).unsqueeze(0)
    quantize_acts = manifest.activation_bits == 8
    with torch.no_grad():
        handles = (
            _install_act_quant(model, names, RUNTIME_ACT_QUANT["master_bf16"])
            if quantize_acts
            else []
        )
        try:
            reference = model(input_ids=input_ids).logits.float()
        finally:
            for handle in handles:
                handle.remove()

        for name, (module_codes, scale) in codes.items():
            weight = model.get_parameter(f"{name}.weight")
            weight.data = quant.dequantize_codes(module_codes, scale, weight.dtype)
        handles = (
            _install_act_quant(model, names, runtime_act_quant) if quantize_acts else []
        )
        try:
            roundtrip = model(input_ids=input_ids).logits.float()
        finally:
            for handle in handles:
                handle.remove()
    return float((reference - roundtrip).abs().amax())


def run_parity_gate(
    master_dir: str | Path,
    artifact: str | Path,
    fmt: ExportFormat,
    manifest: SwapManifest,
    run_smoke: bool = True,
) -> ParityReport:
    """Unpack an exported artifact and gate it against the baked master.

    Raises:
        ValueError: If `fmt` has no registered unpacker.
    """
    extractor = EXTRACTORS.get(fmt)
    if extractor is None:
        raise ValueError(
            f"no parity unpacker registered for format {fmt!r} "
            f"(registered: {sorted(EXTRACTORS)})"
        )
    report = ParityReport(format=fmt)
    master = bake.load_tensors(master_dir)
    try:
        extracted = extractor(Path(artifact), manifest)
    except ValueError as exc:
        report.failures.append(f"could not unpack the {fmt} artifact: {exc}")
        return report

    for entry in manifest.entries:
        key = f"{entry.name}.weight"
        if key not in master:
            report.failures.append(f"{key} is missing from the master")
            continue
        if entry.name not in extracted:
            report.failures.append(f"{entry.name} is missing from the {fmt} artifact")
            continue
        codes, scale = extracted[entry.name]
        try:
            mismatches = _code_mismatches(master[key], codes, entry.group_size)
            dequantized = quant.dequantize_codes(codes, scale, torch.float32)
            error, bound = check_dequant_error(master[key], dequantized)
        except ValueError as exc:
            report.failures.append(f"{entry.name}: {exc}")
            continue
        report.tensors_checked += 1
        report.code_mismatches += mismatches
        report.max_dequant_error = max(report.max_dequant_error, error)
        report.dequant_error_bound = max(report.dequant_error_bound, bound)
        if mismatches:
            report.failures.append(
                f"{entry.name}: {mismatches} of {master[key].numel()} codes differ "
                "from the master"
            )
        if error > bound:
            report.failures.append(
                f"{entry.name}: dequant error {error:.3e} exceeds the f16 bound {bound:.3e}"
            )

    if run_smoke and not report.failures:
        delta = run_smoke_eval(master_dir, artifact, fmt)
        report.smoke_max_logit_delta = delta
        if delta is not None and delta > SMOKE_LOGIT_TOL:
            report.failures.append(
                f"smoke eval: max logit delta {delta:.3e} exceeds {SMOKE_LOGIT_TOL:.3e}"
            )
    return report


def _code_mismatches(
    master_weight: torch.Tensor, unpacked_codes: torch.Tensor, group_size: int | None
) -> int:
    codes, _ = bake.derive_codes_and_scale(master_weight, group_size)
    if codes.shape != unpacked_codes.shape:
        raise ValueError(
            f"shape mismatch: master codes {tuple(codes.shape)} vs unpacked "
            f"{tuple(unpacked_codes.shape)}"
        )
    return int((codes != unpacked_codes.to(codes.device)).sum())


def _extract_master_bf16(
    artifact: Path, manifest: SwapManifest
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    tensors = bake.load_tensors(artifact)
    extracted: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for entry in manifest.entries:
        weight = tensors.get(f"{entry.name}.weight")
        if weight is not None:
            extracted[entry.name] = bake.derive_codes_and_scale(
                weight, entry.group_size
            )
    return extracted


def _extract_hf_bitnet(
    artifact: Path, manifest: SwapManifest
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    tensors = bake.load_tensors(artifact)
    extracted: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for entry in manifest.entries:
        packed = tensors.get(f"{entry.name}.weight")
        scale = tensors.get(f"{entry.name}.{hf_bitnet.WEIGHT_SCALE_SUFFIX}")
        if packed is None or scale is None:
            continue
        codes = hf_bitnet.unpack_hf_bitnet(packed, entry.out_features)
        extracted[entry.name] = (
            codes,
            scale.to(torch.float32).reshape(()).reciprocal(),
        )
    return extracted


def _decode_block(fmt: str):
    def decode(packed: torch.Tensor, shape: tuple[int, int]):
        if fmt == "i2_s":
            from . import i2s

            return i2s.decode_i2s(packed, shape)
        from . import gguf_tq

        decoder = gguf_tq.decode_tq2_0 if fmt == "gguf_tq2_0" else gguf_tq.decode_tq1_0
        return decoder(packed, shape)

    return decode


register_extractor("master_bf16", _extract_master_bf16)
register_extractor("hf_bitnet", _extract_hf_bitnet)
for _fmt in BLOCK_GATED_FORMATS:
    register_block_decoder(_fmt, _decode_block(_fmt))


__all__ = [
    "BLOCK_DECODERS",
    "BLOCK_GATED_FORMATS",
    "EXTRACTORS",
    "F16_HALF_ULP",
    "SMOKE_LOGIT_TOL",
    "ParityReport",
    "check_code_parity",
    "bitlinear_act_quant",
    "check_dequant_error",
    "gate_block_bytes",
    "register_block_decoder",
    "register_extractor",
    "run_parity_gate",
    "run_smoke_eval",
]
