"""Parity gates: every export must reproduce the master's codes exactly."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
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

# `onebitllms_bf16` stores a latent rather than a grid, so its consumer re-derives the
# scale as the mean of bf16-rounded inflated values. Each value carries up to half a
# bf16 mantissa ulp (2^-9 relative), and the mean of same-signed roundings does not
# cancel, so the re-derived scale can sit that far from the master's. Measured at
# ~1e-3 relative on real tensors; the bound is one binade of headroom over that.
LATENT_SCALE_RTOL: float = 1e-2

# formats whose consumer re-derives the scale from a stored latent
LATENT_SCALE_FORMATS: frozenset[str] = frozenset({"onebitllms_bf16"})

SMOKE_TOKENS: int = 16

# The weight path is a deterministic fp32 graph fed bit-identical weights, so its
# logit delta is 0 in practice; the bound only absorbs kernel-selection noise.
WEIGHT_PATH_LOGIT_TOL: float = 1e-5

# Runtime-quantizer equivalence, checked on the activations themselves rather than
# through the network: a swapped ternary net is chaotic, so an ulp-level per-layer
# difference between two numerically-equivalent quantizers amplifies to O(1) logits
# after ~30 layers. End-to-end logit deltas cannot separate that from a real bug.
#
# - codes may differ only where `x / s` lands on a rounding tie, which is O(ulp) of
#   the elements in practice; a systematic fault (inverted scale, wrong qmax) flips
#   most of the tensor, so 0.5% sits orders of magnitude above the noise and far
#   below any real fault,
# - a tie moves one code, hence one scale step; 1.5 steps leaves headroom for the
#   scale's own rounding without admitting a two-step drift,
# - the reachable code range must match exactly: a clamp-range fault ([-127, 126])
#   moves too few elements to register as a fraction, but it always truncates the
#   extreme code that per-token absmax scaling is guaranteed to produce.
RUNTIME_CODE_MISMATCH_TOL: float = 5e-3
RUNTIME_DEQUANT_STEP_TOL: float = 1.5
RUNTIME_PROBE_TENSORS: int = 128

# a Q8_0 code is `round(w / d)`, so dequantization lands within half a block scale
Q8_0_STEP_TOL: float = 0.5

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


def gate_q8_0_bytes(packed: torch.Tensor, master_weight: torch.Tensor) -> list[str]:
    """Round-trip a Q8_0 embedding tensor's packed bytes and gate them on the master.

    Same contract as `gate_block_bytes`, with the bound a per-block int8 quantizer
    admits instead of a ternary one: exact code equality, block scales within f16
    rounding, and a dequantization error of at most half a quantization step.

    Args:
        packed: The uint8 byte string handed to the container writer.
        master_weight: The embedding or head weight it was packed from.

    Returns:
        A list of failure descriptions; empty when the bytes round-trip exactly.
    """
    from .gguf_tq import decode_q8_0, dequantize_q8_0, quantize_q8_0

    shape = (int(master_weight.shape[0]), int(master_weight.shape[1]))
    try:
        codes, scales = quantize_q8_0(master_weight)
        unpacked, block_scales = decode_q8_0(packed, shape)
    except ValueError as exc:
        return [f"could not unpack the q8_0 blocks: {exc}"]

    failures: list[str] = []
    mismatches = int((codes != unpacked).sum())
    if mismatches:
        failures.append(
            f"{mismatches} of {codes.numel()} codes differ from the master after a "
            "q8_0 block round-trip"
        )
    drift = (block_scales - scales).abs()
    bound = scales.abs() * F16_HALF_ULP
    if bool((drift > bound).any()):
        failures.append(
            f"q8_0 block scale drifted {float(drift.amax()):.3e} from the master's, "
            f"beyond the f16 rounding bound {float(bound.amax()):.3e}"
        )
    if not failures:
        error = (
            master_weight.to(torch.float32) - dequantize_q8_0(unpacked, block_scales)
        ).abs()
        step = block_scales.repeat_interleave(
            master_weight.shape[-1] // block_scales.shape[-1], dim=-1
        )
        if bool((error > step * Q8_0_STEP_TOL).any()):
            failures.append(
                f"q8_0 dequantization is off by up to "
                f"{float((error / step.clamp_min(torch.finfo(torch.float32).tiny)).amax()):.3f} "
                f"block scales, over the {Q8_0_STEP_TOL} half-step bound"
            )
    return failures


@dataclass
class SmokeReport:
    """Outcome of the forward-path gates; see `run_smoke_eval`.

    Attributes:
        weight_path_logit_delta: Max logit delta with the *same* activation quantizer
            on both sides — gated, and 0 for a correct packer.
        runtime_code_mismatch: Largest fraction of int8 activation codes on which the
            deployment and training quantizers disagree — gated.
        runtime_dequant_steps: Largest dequantized disagreement between them, in
            units of the training per-token scale — gated.
        runtime_code_range_matches: Whether both quantizers reach the same extreme
            codes — gated.
        cross_quantizer_logit_delta: Max logit delta between the two quantizers
            end-to-end. **Reported only**: a deep swapped model amplifies their
            ulp-level disagreement to O(1), so it bounds nothing.
        probes_compared: How many captured activation tensors the runtime gate saw.
    """

    format: str
    weight_path_logit_delta: float | None = None
    runtime_code_mismatch: float | None = None
    runtime_dequant_steps: float | None = None
    runtime_code_range_matches: bool | None = None
    cross_quantizer_logit_delta: float | None = None
    probes_compared: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether both forward-path gates held."""
        return not self.failures

    def gate_weight_path(self) -> None:
        """Fail when substituting the artifact's weights moved the logits at all."""
        delta = self.weight_path_logit_delta
        if delta is not None and delta > WEIGHT_PATH_LOGIT_TOL:
            self.failures.append(
                f"weight path: max logit delta {delta:.3e} exceeds "
                f"{WEIGHT_PATH_LOGIT_TOL:.3e} with the training activation quantizer "
                "on both sides — the artifact does not decode to the master's weights"
            )

    def gate_runtime_quantizer(self, activations) -> None:
        """Fail when the deployment activation quantizer is not the training one."""
        mismatch = steps = 0.0
        same_range = True
        for tensor in activations:
            tensor_mismatch, tensor_steps, tensor_range = compare_act_quantizers(tensor)
            mismatch = max(mismatch, tensor_mismatch)
            steps = max(steps, tensor_steps)
            same_range = same_range and tensor_range
            self.probes_compared += 1
        if not self.probes_compared:
            return

        self.runtime_code_mismatch = mismatch
        self.runtime_dequant_steps = steps
        self.runtime_code_range_matches = same_range
        if mismatch > RUNTIME_CODE_MISMATCH_TOL:
            self.failures.append(
                f"runtime quantizer: {mismatch:.3%} of activation codes differ from "
                f"the training quantizer, over the {RUNTIME_CODE_MISMATCH_TOL:.3%} "
                "rounding-tie allowance"
            )
        if steps > RUNTIME_DEQUANT_STEP_TOL:
            self.failures.append(
                f"runtime quantizer: activations drift {steps:.2f} scale steps from "
                f"the training quantizer, over the {RUNTIME_DEQUANT_STEP_TOL} allowed"
            )
        if not same_range:
            self.failures.append(
                "runtime quantizer: it does not reach the same extreme activation "
                "codes as the training quantizer (a clamp-range mismatch)"
            )


@dataclass
class ParityReport:
    """Outcome of gating one exported artifact against the baked master."""

    format: str
    tensors_checked: int = 0
    code_mismatches: int = 0
    max_dequant_error: float = 0.0
    dequant_error_bound: float = 0.0
    smoke: SmokeReport | None = None
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
    master_weight: torch.Tensor,
    dequantized: torch.Tensor,
    rtol: float = F16_HALF_ULP,
) -> tuple[float, float]:
    """Return `(max_abs_error, bound)` for a round-tripped tensor.

    The default bound is the f16-rounding error of the scale: an export that stores
    the same codes and an f16 scale cannot drift further than that. Formats that
    store a *latent* and have their consumer re-derive the scale pass their own
    `rtol` — see `LATENT_SCALE_RTOL`.
    """
    master = master_weight.detach().to(torch.float32)
    if master.shape != dequantized.shape:
        raise ValueError(
            f"shape mismatch: master {tuple(master.shape)} vs round-trip "
            f"{tuple(dequantized.shape)}"
        )
    error = (master - dequantized.to(torch.float32)).abs().amax()
    return float(error), float(master.abs().amax()) * rtol


def bitlinear_act_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token activation quantization as `BitNetForCausalLM`'s BitLinear runs it.

    Deliberately not `quant.act_quant_int8`: the runtime multiplies by `127 / amax`
    (flooring `amax`, not the scale) and clamps to `[-128, 127]`, where training
    divides by `amax / 127` (flooring the scale) and clamps symmetrically.

    Returns:
        `(codes, scale)` with the runtime's own orientation — `scale` is the
        *multiplier*, so dequantization divides by it.
    """
    scale = quant.ACT_QMAX / x.float().abs().amax(dim=-1, keepdim=True).clamp_min(
        quant.SCALE_EPS
    )
    codes = (x.float() * scale).round().clamp_(-quant.ACT_QMAX - 1, quant.ACT_QMAX)
    return codes.to(torch.int8), scale


def bitlinear_act_quant(x: torch.Tensor) -> torch.Tensor:
    """Dequantized BitLinear activations, in `x`'s dtype."""
    codes, scale = bitlinear_act_quant_int8(x)
    return (codes.float() / scale).to(x.dtype)


def training_act_quant(x: torch.Tensor) -> torch.Tensor:
    """The activation quantizer the swapped model trained and evaluated with."""
    return quant.act_quant(x, 1.0)


# format -> the activation quantizer its runtime applies before each ternary linear
RUNTIME_ACT_QUANT: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "master_bf16": training_act_quant,
    "hf_bitnet": bitlinear_act_quant,
}


def compare_act_quantizers(x: torch.Tensor) -> tuple[float, float, bool]:
    """Compare the deployment activation quantizer against the training one on `x`.

    Args:
        x: A real pre-linear activation tensor, `(..., in_features)`.

    Returns:
        `(code_mismatch_fraction, max_dequant_steps, code_range_matches)` — the
        fraction of int8 codes that differ, the largest dequantized deviation in
        units of the training per-token scale, and whether both quantizers reach the
        same extreme codes.
    """
    train_codes, train_scale = quant.act_quant_int8(x)
    runtime_codes, runtime_scale = bitlinear_act_quant_int8(x)
    if not train_codes.numel():
        return 0.0, 0.0, True

    mismatch = float((train_codes != runtime_codes).to(torch.float32).mean())
    deviation = (
        train_codes.to(torch.float32) * train_scale
        - runtime_codes.to(torch.float32) / runtime_scale
    ).abs()
    steps = float((deviation / train_scale).amax())
    same_range = int(train_codes.amin()) == int(runtime_codes.amin()) and int(
        train_codes.amax()
    ) == int(runtime_codes.amax())
    return mismatch, steps, same_range


def _install_act_quant(
    model: torch.nn.Module,
    names: list[str],
    fn: Callable[[torch.Tensor], torch.Tensor] | None,
    captured: dict[str, torch.Tensor] | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Quantize (and optionally record) the input of every named linear."""

    def make_hook(name: str):
        def pre_hook(_module, args):
            if captured is not None and name not in captured:
                captured[name] = args[0].detach().to(torch.float32).cpu()
            if fn is None:
                return None
            return (fn(args[0]),) + tuple(args[1:])

        return pre_hook

    return [
        model.get_submodule(name).register_forward_pre_hook(make_hook(name))
        for name in names
    ]


@quant.pinned_fp32_precision()
def run_smoke_eval(
    master_dir: str | Path, artifact: str | Path, fmt: ExportFormat
) -> "SmokeReport | None":
    """Run the artifact's forward-path gates against the baked master.

    Two orthogonal checks, because one end-to-end number cannot serve both:

    1. **Weight path** — both sides run the *same* (training) activation quantizer,
       so the only variable is the weights decoded from the artifact. Those are
       bit-identical to the master's when the packer is correct, making this a
       deterministic fp32 graph fed identical inputs: the logit delta is 0, and a
       layout transpose or an inverted scale blows it up by orders of magnitude.
    2. **Runtime quantizer** — the deployment activation quantizer is compared to the
       training one *directly*, on the real pre-linear activations captured from the
       reference forward. Running that comparison through the network instead would
       measure chaos amplification, not equivalence.

    The end-to-end delta between the two quantizers is still measured and reported,
    but it is not a bound: on a deep swapped model it is O(1) even when the two
    quantizers agree to the last representable code.

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
    storage = bake.tensor_dtypes(master_dir)
    model.eval()
    input_ids = (torch.arange(SMOKE_TOKENS) % model.config.vocab_size).unsqueeze(0)
    act_quant = training_act_quant if manifest.activation_bits == 8 else None
    stride = max(1, -(-len(names) // RUNTIME_PROBE_TENSORS))
    probes = set(names[::stride][:RUNTIME_PROBE_TENSORS])
    captured: dict[str, torch.Tensor] = {}

    report = SmokeReport(format=fmt)
    with torch.no_grad():
        with _hooks(
            [
                *_install_act_quant(model, sorted(probes), act_quant, captured),
                *_install_act_quant(
                    model, [n for n in names if n not in probes], act_quant
                ),
            ]
        ):
            reference = model(input_ids=input_ids).logits.float()

        for name, (module_codes, scale) in codes.items():
            weight = model.get_parameter(f"{name}.weight")
            key = f"{name}.weight"
            # through the checkpoint's own storage dtype: reconstructing a bf16
            # master in fp32 lands an ulp off (the export stores a reciprocal
            # scale), and a deep swapped net amplifies that to O(1) logits
            weight.data = quant.dequantize_codes(
                module_codes, scale, storage.get(key, weight.dtype)
            ).to(weight.dtype)

        with _hooks(_install_act_quant(model, names, act_quant)):
            same_quantizer = model(input_ids=input_ids).logits.float()
        report.weight_path_logit_delta = float(
            (reference - same_quantizer).abs().amax()
        )

        if act_quant is not None and runtime_act_quant is not training_act_quant:
            with _hooks(_install_act_quant(model, names, runtime_act_quant)):
                cross = model(input_ids=input_ids).logits.float()
            report.cross_quantizer_logit_delta = float((reference - cross).abs().amax())

    report.gate_weight_path()
    if act_quant is not None and runtime_act_quant is not training_act_quant:
        report.gate_runtime_quantizer(captured.values())
    return report


@contextmanager
def _hooks(handles: list[torch.utils.hooks.RemovableHandle]):
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@quant.pinned_fp32_precision()
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
            mismatches = _code_mismatches(
                master[key],
                codes,
                entry.group_size,
                entry.weight_scale,
                manifest.scales_for(entry.name),
            )
            # in the master's own dtype: a free-sum value is a *sum* of two scales,
            # so an fp32 reconstruction of a bf16 master lands a storage ulp away
            dequantized = bake.dequantize_derived(
                codes, scale, entry.weight_scale, master[key].dtype
            )
            error, bound = check_dequant_error(
                master[key], dequantized, _dequant_rtol(fmt)
            )
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
        report.smoke = run_smoke_eval(master_dir, artifact, fmt)
        if report.smoke is not None:
            report.failures.extend(report.smoke.failures)
    return report


def _dequant_rtol(fmt: str) -> float:
    """Relative bound on a format's dequantized weights against the master."""
    return LATENT_SCALE_RTOL if fmt in LATENT_SCALE_FORMATS else F16_HALF_ULP


def _code_mismatches(
    master_weight: torch.Tensor,
    unpacked_codes: torch.Tensor,
    group_size: int | None,
    weight_scale: str = "absmean",
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> int:
    codes, _ = bake.derive_codes_and_scale(
        master_weight, group_size, weight_scale, scales
    )
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
                weight,
                entry.group_size,
                entry.weight_scale,
                manifest.scales_for(entry.name),
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


def _extract_mask_sign(
    artifact: Path, manifest: SwapManifest
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    from . import mask_sign

    tensors = bake.load_tensors(artifact)
    extracted: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for entry in manifest.entries:
        packed = tensors.get(f"{entry.name}.{mask_sign.PACKED_SUFFIX}")
        if packed is None:
            continue
        extracted[entry.name] = mask_sign.decode_mask_sign(
            packed, (entry.out_features, entry.in_features)
        )
    return extracted


def gate_bitplane_bytes(
    fmt: str,
    packed: torch.Tensor,
    scales: torch.Tensor,
    master_weight: torch.Tensor,
    weight_scale: str,
) -> list[str]:
    """Round-trip an `fv5`/`tp9` tensor's bytes and gate them on the master.

    Same contract as `gate_block_bytes`, over a two-plane grid: the codes must come
    back exactly, and the values they reconstruct must equal the master's *in the
    master's own storage dtype* — a free sum of two scales rounds differently in fp32,
    which is not a packing fault.

    Args:
        fmt: `fv5` or `tp9`.
        packed: The byte string being written.
        scales: The `(rows, 2)` scale tensor written beside it.
        master_weight: The baked master tensor it was packed from.
        weight_scale: The master's scale mode, `dual` or `trit_planes`.

    Returns:
        A list of failure descriptions; empty when the bytes round-trip exactly.
    """
    from . import bitplanes

    shape = (int(master_weight.shape[0]), int(master_weight.shape[1]))
    decode = (
        bitplanes.decode_fv5 if fmt == bitplanes.FV5_FORMAT else bitplanes.decode_tp9
    )
    try:
        codes, got_scales = decode(packed, scales, shape)
    except ValueError as exc:
        return [f"could not unpack the {fmt} planes: {exc}"]

    failures: list[str] = []
    try:
        expected, expected_scales = bake.derive_codes_and_scale(
            master_weight, None, weight_scale, _scale_pair(scales)
        )
    except ValueError as exc:
        # the stored scales do not describe a grid the master sits on, which is what
        # a corrupted scale vector looks like from here
        return [f"the {fmt} scale vectors do not reproduce the master: {exc}"]
    if codes.shape != expected.shape:
        return [
            f"unpacked shape {tuple(codes.shape)} does not match the master's "
            f"{tuple(expected.shape)}"
        ]
    mismatches = int((codes != expected.to(codes.device)).sum())
    if mismatches:
        failures.append(
            f"{mismatches} of {expected.numel()} codes differ from the master after "
            f"a {fmt} round-trip"
        )
    if not torch.equal(got_scales, expected_scales.to(torch.float32)):
        failures.append(
            f"the {fmt} scale vectors differ from the grid the master was baked on"
        )
    if not failures:
        rebuilt = bake.dequantize_derived(
            codes, got_scales, weight_scale, master_weight.dtype
        )
        if not bool(torch.equal(rebuilt, master_weight)):
            drift = float(
                (rebuilt.to(torch.float32) - master_weight.to(torch.float32))
                .abs()
                .amax()
            )
            failures.append(
                f"{fmt} reconstruction differs from the master by up to {drift:.3e}"
            )
    return failures


def _scale_pair(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a stored `(rows, 2)` scale tensor into the pair the deriver wants."""
    dense = scales.to(torch.float32)
    return dense[:, :1], dense[:, 1:]


def _extract_bitplanes(fmt: str):
    """Build the artifact extractor for a bit-plane format."""

    def extract(
        artifact: Path, manifest: SwapManifest
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        from . import bitplanes

        suffix = (
            bitplanes.FV5_SUFFIX
            if fmt == bitplanes.FV5_FORMAT
            else bitplanes.TP9_SUFFIX
        )
        decode = (
            bitplanes.decode_fv5
            if fmt == bitplanes.FV5_FORMAT
            else bitplanes.decode_tp9
        )
        tensors = bake.load_tensors(artifact)
        extracted: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for entry in manifest.entries:
            packed = tensors.get(f"{entry.name}.{suffix}")
            scales = tensors.get(f"{entry.name}.{bitplanes.SCALES_SUFFIX}")
            if packed is None or scales is None:
                continue
            extracted[entry.name] = decode(
                packed, scales, (entry.out_features, entry.in_features)
            )
        return extracted

    return extract


def _extract_onebitllms(
    artifact: Path, manifest: SwapManifest
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Read the artifact the way `BitNetLinear` will: run their quantizer on it.

    The stored tensors are latents, not a grid, so the gate has to re-derive the grid
    exactly as the consumer does. That is what makes it catch the trap this format
    exists for: handing the *raw* master to `use_onebitllms` keeps the codes but
    shrinks every scale by the density, and running their quantizer here surfaces that
    as a dequant-scale mismatch rather than a silent 0.67x on every linear.
    """
    from . import onebitllms

    tensors = bake.load_tensors(artifact)
    extracted: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for entry in manifest.entries:
        weight = tensors.get(f"{entry.name}.weight")
        if weight is not None:
            extracted[entry.name] = onebitllms.onebitllms_quantize(weight)
    return extracted


def _decode_block(fmt: str):
    def decode(packed: torch.Tensor, shape: tuple[int, int]):
        if fmt == "i2_s":
            from . import i2s

            return i2s.decode_i2s(packed, shape)
        if fmt == "mask_sign":
            from . import mask_sign

            return mask_sign.decode_mask_sign(packed, shape)
        from . import gguf_tq

        decoder = gguf_tq.decode_tq2_0 if fmt == "gguf_tq2_0" else gguf_tq.decode_tq1_0
        return decoder(packed, shape)

    return decode


register_extractor("master_bf16", _extract_master_bf16)
register_extractor("hf_bitnet", _extract_hf_bitnet)
register_extractor("mask_sign", _extract_mask_sign)
register_extractor("onebitllms_bf16", _extract_onebitllms)
register_extractor("fv5", _extract_bitplanes("fv5"))
register_extractor("tp9", _extract_bitplanes("tp9"))
# mask_sign is re-read from disk like the other extractor formats, but its bytes are
# also gated as they are packed, so a corrupt write never reaches the container
for _fmt in (*BLOCK_GATED_FORMATS, "mask_sign"):
    register_block_decoder(_fmt, _decode_block(_fmt))


__all__ = [
    "BLOCK_DECODERS",
    "BLOCK_GATED_FORMATS",
    "EXTRACTORS",
    "F16_HALF_ULP",
    "LATENT_SCALE_FORMATS",
    "LATENT_SCALE_RTOL",
    "Q8_0_STEP_TOL",
    "RUNTIME_CODE_MISMATCH_TOL",
    "RUNTIME_DEQUANT_STEP_TOL",
    "WEIGHT_PATH_LOGIT_TOL",
    "ParityReport",
    "SmokeReport",
    "bitlinear_act_quant",
    "bitlinear_act_quant_int8",
    "check_code_parity",
    "check_dequant_error",
    "compare_act_quantizers",
    "gate_bitplane_bytes",
    "gate_block_bytes",
    "gate_q8_0_bytes",
    "register_block_decoder",
    "register_extractor",
    "run_parity_gate",
    "run_smoke_eval",
    "training_act_quant",
]
