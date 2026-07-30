"""Sequential per-block local distillation — healing a model too large to hold.

End-to-end KD needs the whole student resident plus optimizer state, which a 128B
ternary master does not fit into any single box. Blockwise healing trades that for
time: one teacher block and one student block on the GPU at a time, with the
activations that connect them living on disk.

The design decision that matters most is which activations the student trains *on*.
The teacher's block-`k` inputs are the obvious choice and the wrong one: they are the
activations of a model that has no quantization error, so every block would be healed
against inputs it will never see at inference. Instead the student is fed its **own**
propagated activations — the outputs of the blocks already healed — while the *targets*
stay the teacher's block-`k` outputs. Block `k` therefore learns to map the drifted
input it will actually receive back onto the output the teacher would have produced,
which is what lets accumulated quantization error be compensated downstream rather than
compounding.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .. import swap
from ..args import resolve_ternary_config
from ..modules import SCALE_ATTRS, iter_quantized_modules
from .calibrate import decoder_layers
from .stream import PARTIAL_SUFFIX

LOG = get_logger(__name__)

RECORD_FILENAME: str = "ternary_heal_records.json"

# activation chunks are written one captured batch at a time, so a resume never has to
# trust a partially written buffer
_CHUNK_TEMPLATE: str = "{index:05d}.safetensors"

_TEACHER_TAG: str = "teacher"
_STUDENT_TAG: str = "student"

_HASH_CHUNK: int = 1 << 20


@dataclass
class BlockRecord:
    """One healed block, and the digests proving its outputs are the healed ones."""

    block: int
    steps: int
    tokens: int
    initial_loss: float
    final_loss: float
    seconds: float
    weights_sha256: str
    student_sha256: str


@dataclass
class BlockwiseHealReport:
    """What a heal pass did, block by block."""

    master_dir: str
    output_dir: str
    blocks: int
    codebook: str = "ternary"
    blocks_healed: int = 0
    blocks_skipped: int = 0
    seconds: float = 0.0
    records: dict[str, BlockRecord] = field(default_factory=dict)

    @property
    def mean_final_loss(self) -> float:
        """Mean per-block final MSE, the number to watch across a campaign."""
        if not self.records:
            return 0.0
        return sum(r.final_loss for r in self.records.values()) / len(self.records)

    @property
    def mean_improvement(self) -> float:
        """Mean fraction of the initial block MSE the heal removed."""
        ratios = [
            1.0 - r.final_loss / r.initial_loss
            for r in self.records.values()
            if r.initial_loss > 0
        ]
        return sum(ratios) / len(ratios) if ratios else 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serializable view."""
        data = asdict(self)
        data["mean_final_loss"] = self.mean_final_loss
        data["mean_improvement"] = self.mean_improvement
        return data


class ActivationCache:
    """Two rolling on-disk buffers of block activations, keyed by block index.

    Chunks are written to a temporary name and renamed, so a chunk that exists is a
    chunk that finished — the same discipline the shard writer uses.
    """

    def __init__(self, root: str | Path) -> None:
        """Open (creating if needed) a cache rooted at `root`."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, tag: str, block: int) -> Path:
        """Directory holding one buffer's chunks."""
        return self.root / f"{tag}-{block:04d}"

    def exists(self, tag: str, block: int) -> bool:
        """Whether a complete buffer is on disk for `(tag, block)`."""
        return (self.path(tag, block) / "count.json").is_file()

    def write(self, tag: str, block: int, chunks: Iterator[torch.Tensor]) -> int:
        """Write a buffer, one chunk at a time, and return how many landed."""
        directory = self.path(tag, block)
        directory.mkdir(parents=True, exist_ok=True)
        count = 0
        for index, chunk in enumerate(chunks):
            destination = directory / _CHUNK_TEMPLATE.format(index=index)
            partial = destination.with_suffix(destination.suffix + PARTIAL_SUFFIX)
            save_file(
                {"activations": chunk.to(torch.bfloat16).contiguous().cpu()}, partial
            )
            os.replace(partial, destination)
            count = index + 1
        (directory / "count.json").write_text(json.dumps({"chunks": count}))
        return count

    def read(self, tag: str, block: int) -> Iterator[torch.Tensor]:
        """Yield a buffer's chunks in order, one resident at a time."""
        directory = self.path(tag, block)
        count = json.loads((directory / "count.json").read_text())["chunks"]
        for index in range(count):
            yield load_file(directory / _CHUNK_TEMPLATE.format(index=index))[
                "activations"
            ]

    def digest(self, tag: str, block: int) -> str:
        """Return a content digest over a buffer's chunks, for resume equivalence."""
        directory = self.path(tag, block)
        count = json.loads((directory / "count.json").read_text())["chunks"]
        running = hashlib.sha256()
        for index in range(count):
            path = directory / _CHUNK_TEMPLATE.format(index=index)
            with open(path, "rb") as handle:
                for piece in iter(lambda: handle.read(_HASH_CHUNK), b""):
                    running.update(piece)
        return running.hexdigest()

    def drop(self, tag: str, block: int) -> None:
        """Delete a buffer that no later block needs."""
        directory = self.path(tag, block)
        if not directory.is_dir():
            return
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()

    def prune(self, keep: set[int]) -> None:
        """Drop every buffer whose block index is not in `keep`.

        Kills and resumes strand buffers outside the rolling window (each one is
        `num_sequences x sequence_len x hidden` on disk — tens of GB at 1.7B and
        worse with width), and no later block will ever read them.
        """
        for directory in self.root.iterdir():
            name = directory.name
            if not directory.is_dir() or "-" not in name:
                continue
            tag, _, suffix = name.rpartition("-")
            if tag in (_TEACHER_TAG, _STUDENT_TAG) and suffix.isdigit():
                if int(suffix) not in keep:
                    self.drop(tag, int(suffix))


def heal_blocks(
    master_dir: str | Path,
    output_dir: str | Path,
    cfg: DictDefault,
    source_dir: str | Path | None = None,
    device: str | torch.device = "cuda",
    resume: bool = True,
) -> BlockwiseHealReport:
    """Heal a ternary master block by block against its full-precision teacher.

    Args:
        master_dir: The fit-stream master, whose ternary latents are healed.
        output_dir: Where the healed master is written.
        cfg: The axolotl config carrying `ternary.blockwise`.
        source_dir: The full-precision (or fp8) teacher. Defaults to the config's
            `base_model`.
        device: Device one teacher block and one student block are resident on.
        resume: Skip blocks a previous pass already healed.

    Returns:
        The report, one record per healed block.

    Raises:
        ValueError: If `ternary.blockwise` is not configured.
    """
    ternary_cfg = resolve_ternary_config(cfg)
    plan = ternary_cfg.blockwise
    if plan is None or not plan.enabled:
        raise ValueError(
            "blockwise healing needs ternary.blockwise.enabled: true and a corpus"
        )

    master = Path(master_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    teacher_dir = Path(source_dir) if source_dir else Path(cfg.base_model)
    cache = ActivationCache(plan.cache_dir or (output / "activation-cache"))

    student_model = _meta_model(master, cfg, swapped=True)
    blocks = decoder_layers(student_model)
    prefixes = {
        index: _module_prefix(student_model, block)
        for index, block in enumerate(blocks)
    }
    report = BlockwiseHealReport(
        master_dir=str(master),
        output_dir=str(output),
        blocks=len(blocks),
        codebook=ternary_cfg.codebook,
    )
    records = _load_records(output) if resume else {}

    started = time.monotonic()
    _release(student_model)

    # Resume position comes from the weights digests alone: the rolling window
    # drops old activation buffers by design, so demanding a cache per healed
    # block silently re-heals everything past the window (and re-seeding then
    # re-materializes the whole teacher chain — hundreds of GB of scratch).
    # Only the boundary needs its buffer, verified against the previous
    # block's recorded student digest.
    resume_at = 0
    for index in range(len(blocks)):
        record = records.get(str(index))
        if record is None or not _weights_valid(record, output, index):
            break
        resume_at = index + 1
    if resume_at:
        previous = records[str(resume_at - 1)]
        if (
            not cache.exists(_STUDENT_TAG, resume_at)
            or cache.digest(_STUDENT_TAG, resume_at) != previous.student_sha256
        ):
            raise ValueError(
                f"resume needs the student buffer for block {resume_at} "
                f"(digest {previous.student_sha256[:12]}…) in {cache.root}; it is "
                "missing or stale. Re-run with --no-resume to heal from scratch."
            )
    cache.prune(keep={resume_at - 1, resume_at, resume_at + 1})
    for index in range(resume_at):
        report.blocks_skipped += 1
        report.records[str(index)] = records[str(index)]

    forward_kwargs = _seed_block_zero(
        master,
        cfg,
        plan,
        cache,
        tuple(prefix.rstrip(".") for prefix in prefixes.values()),
        device,
        write_buffers=resume_at == 0,
    )

    for index in range(resume_at, len(blocks)):
        record = _heal_one_block(
            index=index,
            master=master,
            teacher_dir=teacher_dir,
            output=output,
            cfg=cfg,
            plan=plan,
            cache=cache,
            forward_kwargs=forward_kwargs,
            prefix=prefixes[index],
            device=device,
        )
        records[str(index)] = record
        _write_records(output, records)
        report.blocks_healed += 1
        report.records[str(index)] = record
        # the buffers feeding this block are dead once it has produced its own
        if index:
            cache.drop(_TEACHER_TAG, index - 1)
            cache.drop(_STUDENT_TAG, index - 1)

    report.seconds = time.monotonic() - started
    _finish_master(master, output, prefixes)
    LOG.info(
        f"ternary: blockwise heal wrote {output} over {report.blocks_healed}/"
        f"{report.blocks} blocks ({report.blocks_skipped} skipped), mean block MSE "
        f"{report.mean_final_loss:.6f}, mean improvement {report.mean_improvement:.1%}"
    )
    return report


def _heal_one_block(
    *,
    index: int,
    master: Path,
    teacher_dir: Path,
    output: Path,
    cfg: DictDefault,
    plan,
    cache: ActivationCache,
    forward_kwargs: dict[str, Any],
    prefix: str,
    device: str | torch.device,
) -> BlockRecord:
    """Heal block `index`, then propagate the student's own outputs to `index + 1`."""
    started = time.monotonic()

    # 1. teacher targets: one teacher block resident, streamed over its cached inputs
    if not cache.exists(_TEACHER_TAG, index + 1):
        teacher = _load_block(teacher_dir, cfg, prefix, device)
        cache.write(
            _TEACHER_TAG,
            index + 1,
            _apply_block(teacher, cache.read(_TEACHER_TAG, index), forward_kwargs),
        )
        _release(teacher)

    # 2. the student block, on its own drifted inputs, against those targets
    student = _load_block(master, cfg, prefix, device, student=True)
    initial, final, steps, tokens = _train_block(
        student, cache, index, forward_kwargs, plan, device
    )

    # 3. what the *student* hands the next block — not what the teacher would have
    cache.write(
        _STUDENT_TAG,
        index + 1,
        _apply_block(student, cache.read(_STUDENT_TAG, index), forward_kwargs),
    )
    weights = _save_block(student, output, index)
    _release(student)

    return BlockRecord(
        block=index,
        steps=steps,
        tokens=tokens,
        initial_loss=initial,
        final_loss=final,
        seconds=time.monotonic() - started,
        weights_sha256=_file_sha256(weights),
        student_sha256=cache.digest(_STUDENT_TAG, index + 1),
    )


def _build_optimizer(parameters, plan) -> torch.optim.Optimizer:
    if plan.optimizer == "adamw_torch_8bit":
        try:
            from torchao.optim import AdamW8bit
        except ImportError as err:
            raise ImportError(
                "ternary.blockwise.optimizer: adamw_torch_8bit needs torchao "
                "(pip install torchao)"
            ) from err
        return AdamW8bit(parameters, lr=plan.learning_rate)
    return torch.optim.AdamW(parameters, lr=plan.learning_rate)


def _train_block(
    student: nn.Module,
    cache: ActivationCache,
    index: int,
    forward_kwargs: dict[str, Any],
    plan,
    device: str | torch.device,
) -> tuple[float, float, int, int]:
    """Fit one student block onto the teacher's outputs with MSE, and report the arc."""
    parameters = [p for p in student.parameters() if p.requires_grad]
    optimizer = _build_optimizer(parameters, plan)
    dtype = parameters[0].dtype if parameters else torch.bfloat16
    # like-for-like improvement metric: a fixed slice measured before and after,
    # instead of first-batch-loss vs last-batch-loss over different data
    initial = _slice_loss(student, cache, index, forward_kwargs, device)
    student.train()

    final = 0.0
    steps = 0
    tokens = 0
    for _epoch in range(max(1, plan.epochs)):
        pairs = zip(
            cache.read(_STUDENT_TAG, index),
            cache.read(_TEACHER_TAG, index + 1),
            strict=False,
        )
        for inputs, targets in pairs:
            if plan.max_steps and steps >= plan.max_steps:
                break
            hidden = inputs.to(device=device, dtype=dtype)
            wanted = targets.to(device)
            output = _block_forward(student, hidden, forward_kwargs)
            loss = torch.nn.functional.mse_loss(output.float(), wanted.float())
            loss.backward()
            if plan.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(parameters, plan.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            tokens += int(hidden.shape[0] * hidden.shape[1])
        if plan.max_steps and steps >= plan.max_steps:
            break
    student.eval()
    final = _slice_loss(student, cache, index, forward_kwargs, device)
    return initial, final, steps, tokens


_EVAL_SLICE_CHUNKS = 4


@torch.no_grad()
def _slice_loss(
    student: nn.Module,
    cache: "ActivationCache",
    index: int,
    forward_kwargs: dict[str, Any],
    device: str | torch.device,
) -> float:
    """Mean block MSE over the first few cached chunks — a fixed yardstick."""
    was_training = student.training
    student.eval()
    dtype = next(student.parameters()).dtype
    total = 0.0
    seen = 0
    pairs = zip(
        cache.read(_STUDENT_TAG, index),
        cache.read(_TEACHER_TAG, index + 1),
        strict=False,
    )
    for inputs, targets in pairs:
        if seen >= _EVAL_SLICE_CHUNKS:
            break
        hidden = inputs.to(device=device, dtype=dtype)
        wanted = targets.to(device)
        output = _block_forward(student, hidden, forward_kwargs)
        total += float(torch.nn.functional.mse_loss(output.float(), wanted.float()))
        seen += 1
    if was_training:
        student.train()
    return total / max(seen, 1)


@torch.no_grad()
def _apply_block(
    block: nn.Module, chunks: Iterator[torch.Tensor], forward_kwargs: dict[str, Any]
) -> Iterator[torch.Tensor]:
    """Yield `block(chunk)` for each chunk, never holding two outputs at once."""
    reference = next(block.parameters())
    for chunk in chunks:
        staged = chunk.to(device=reference.device, dtype=reference.dtype)
        yield _block_forward(block, staged, forward_kwargs).cpu()


def _block_forward(
    block: nn.Module, hidden: torch.Tensor, forward_kwargs: dict[str, Any]
) -> torch.Tensor:
    """Run one decoder block, unwrapping the tuple older signatures return."""
    output = block(hidden, **forward_kwargs)
    return output[0] if isinstance(output, tuple) else output


def _seed_block_zero(
    master: Path,
    cfg: DictDefault,
    plan,
    cache: ActivationCache,
    block_prefixes: tuple[str, ...],
    device: str | torch.device,
    write_buffers: bool = True,
) -> dict[str, Any]:
    """Capture the corpus's block-0 inputs, which both models share at the start.

    Only the pre-block modules are materialized — the embedding and whatever rotary
    state the architecture keeps beside it — so seeding a 128B costs an embedding, not
    a model. Those modules are kept full precision, so the master carries the teacher's
    own copies and both models' block-0 inputs are the same tensor; the student buffer
    starts as a copy of the teacher's and only diverges once block 0 has trained.
    """
    from .calibrate import collect_layer_inputs

    # swapped, because the block-input catcher walks the ternary block list — and the
    # master's pre-block modules *are* the teacher's, since they are kept full precision
    model = _meta_model(master, cfg, swapped=True)
    _materialize_pre_block(model, master, block_prefixes, device)

    inputs, forward_kwargs = collect_layer_inputs(
        model, cfg, plan.num_sequences, plan.sequence_len, plan.dataset
    )
    if write_buffers:
        if not cache.exists(_TEACHER_TAG, 0):
            cache.write(_TEACHER_TAG, 0, iter(inputs))
        if not cache.exists(_STUDENT_TAG, 0):
            cache.write(_STUDENT_TAG, 0, iter(inputs))
    _release(model)
    return forward_kwargs


def _materialize_pre_block(
    model: nn.Module,
    directory: Path,
    block_prefixes: tuple[str, ...],
    device: str | torch.device,
) -> None:
    """Give real storage to everything the forward touches before block 0."""
    model.to_empty(device=device)
    state = {
        key: value
        for key, value in _read_checkpoint(directory).items()
        if not any(key.startswith(f"{prefix}.") for prefix in block_prefixes)
    }
    model.load_state_dict(state, strict=False, assign=False)
    _rebuild_nonpersistent_buffers(model, device)
    for param in model.parameters():
        if param.is_floating_point() and param.dtype is not torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)


def _rebuild_nonpersistent_buffers(
    model: nn.Module, device: str | torch.device
) -> None:
    """Recompute buffers `to_empty` wiped that no checkpoint can restore.

    Non-persistent buffers (rotary `inv_freq` above all) are excluded from the
    state dict, so after `to_empty` they hold uninitialized memory — and rotary
    cos/sin computed from garbage made every heal a different artifact. Rebuild
    the ones with a known recipe; refuse the ones without, loudly.
    """
    for name, module in model.named_modules():
        stale = set(module._non_persistent_buffers_set)
        if not stale:
            continue
        init_fn = getattr(module, "rope_init_fn", None)
        if init_fn is None and hasattr(module, "compute_default_rope_parameters"):
            init_fn = module.compute_default_rope_parameters
            if getattr(module, "rope_type", "default") != "default":
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                init_fn = ROPE_INIT_FUNCTIONS[module.rope_type]
        if init_fn is not None:
            inv_freq, attention_scaling = init_fn(module.config, torch.device(device))
            module.inv_freq = inv_freq
            module.attention_scaling = attention_scaling
            if hasattr(module, "original_inv_freq"):
                module.original_inv_freq = inv_freq
            stale -= {"inv_freq", "original_inv_freq"}
        if stale:
            raise ValueError(
                f"{name}: non-persistent buffers {sorted(stale)} were wiped by "
                "to_empty and blockwise healing has no recipe to rebuild them"
            )


def _load_block(
    directory: Path,
    cfg: DictDefault,
    prefix: str,
    device: str | torch.device,
    student: bool = False,
) -> nn.Module:
    """Materialize one decoder block's weights from a checkpoint, and nothing else.

    The rest of the model stays on `meta`, so a 128B teacher costs one block of VRAM
    rather than 256GB of it. The block is addressed by its state-dict prefix, taken
    from the student: the teacher has the same architecture but is not ternary-swapped,
    and the swapped block list is the only one `decoder_layers` recognizes.
    """
    model = _meta_model(directory, cfg, swapped=student)
    block = model.get_submodule(prefix.rstrip("."))
    _materialize(block, prefix, directory, device, skip_scales=student)
    if student:
        _restore_scales(block, prefix, directory)
    return block


def _restore_scales(block: nn.Module, prefix: str, master: Path) -> None:
    """Give a swapped block the learnable scales its master was fitted with.

    fit-stream keeps them in the manifest's sidecar rather than the shards, because a
    free-sum grid cannot be recovered from the values alone. Re-seeding from the latent
    would silently substitute a different quantizer for the two-plane modes.
    """
    from ..swap import SwapManifest

    try:
        manifest = SwapManifest.load(master)
    except FileNotFoundError:
        manifest = None
    for name, module in block.named_modules():
        if not hasattr(module, "refresh_scale_from_weight"):
            continue
        # `to_empty` left every scale holding uninitialized memory; each one the
        # module actually has must be written here or the heal is nondeterministic
        present = [a for a in SCALE_ATTRS if getattr(module, a, None) is not None]
        scales = manifest.scales_for(f"{prefix}{name}") if manifest else None
        if scales and len(scales) == len(present):
            with torch.no_grad():
                for attr, value in zip(present, scales, strict=True):
                    param = getattr(module, attr)
                    param.copy_(value.reshape(param.shape).log().to(param.device))
        elif len(present) > 1:
            raise ValueError(
                f"{prefix}{name}: master sidecar has {len(scales) if scales else 0} "
                f"scale(s) for a {len(present)}-scale module; a free-sum grid cannot "
                "be re-derived from the latent, refusing to substitute a different "
                "quantizer"
            )
        else:
            module.refresh_scale_from_weight()


def _meta_model(directory: Path, cfg: DictDefault, swapped: bool) -> nn.Module:
    """Instantiate the architecture on `meta`, optionally ternary-swapped."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    if swapped:
        swap.convert_model(model, cfg)
    return model


def _materialize(
    module: nn.Module,
    prefix: str,
    directory: Path,
    device: str | torch.device,
    skip_scales: bool = False,
) -> nn.Module:
    """Give one submodule real storage and load just its tensors off disk."""
    module.to_empty(device=device)
    state = _read_block_state(directory, prefix)
    missing, _ = module.load_state_dict(state, strict=False, assign=False)
    _rebuild_nonpersistent_buffers(module, device)
    real = [
        key
        for key in missing
        if _is_real_parameter(module, key)
        and not (skip_scales and key.rsplit(".", 1)[-1] in SCALE_ATTRS)
    ]
    if real:
        raise ValueError(
            f"{prefix or '<root>'} is missing {sorted(real)[:5]} in {directory}; the "
            "checkpoint and the config disagree about its parameters"
        )
    # trainable (student) blocks hold fp32 latents: healing updates at sane
    # learning rates sit below bf16's ULP at typical weight scales, and bf16
    # training degrades near-optimal blocks by rounding random-walk — the same
    # fp32-latent requirement the end-to-end trainer gets from its upcast
    train_dtype = torch.float32 if skip_scales else torch.bfloat16
    for param in module.parameters():
        if param.is_floating_point() and param.dtype is not train_dtype:
            param.data = param.data.to(train_dtype)
    return module


def _is_real_parameter(module: nn.Module, key: str) -> bool:
    """Whether a missing key names a stored parameter rather than a derived buffer."""
    names = {name for name, _ in module.named_parameters()}
    return key in names


def _read_checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    """Read every tensor of a checkpoint, fp8 payloads dequantized on the way."""
    return _read_block_state(directory, "")


def _read_block_state(directory: Path, prefix: str) -> dict[str, torch.Tensor]:
    """Read exactly one block's tensors out of a sharded checkpoint.

    fp8 payloads are dequantized on the way through, so a quantized teacher reaches
    the healer as float weights like any other.
    """
    from . import fp8

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(
            {weight_map[key] for key in weight_map if key.startswith(prefix)}
        )
    else:
        shards = sorted(path.name for path in directory.glob("*.safetensors"))

    collected: dict[str, torch.Tensor] = {}
    for name in shards:
        tensors = load_file(directory / name)
        wanted = {
            key: value for key, value in tensors.items() if key.startswith(prefix)
        }
        for key, value in fp8.dequantized_items(wanted):
            collected[key[len(prefix) :]] = value
    return collected


def _module_prefix(model: nn.Module, target: nn.Module) -> str:
    """Return the state-dict prefix of `target` inside `model`, e.g. `model.layers.3.`."""
    for name, module in model.named_modules():
        if module is target and name:
            return f"{name}."
    raise ValueError("the module is not reachable by name from the model")


def _save_block(block: nn.Module, output: Path, index: int) -> Path:
    """Write one healed block's tensors, temp-then-rename."""
    directory = output / "blocks"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"block-{index:04d}.safetensors"
    partial = destination.with_suffix(destination.suffix + PARTIAL_SUFFIX)
    state = {
        key: value.detach().to(torch.bfloat16).contiguous().cpu()
        for key, value in block.state_dict().items()
    }
    save_file(state, partial)
    os.replace(partial, destination)
    return destination


def _finish_master(master: Path, output: Path, prefixes: dict[int, str]) -> None:
    """Fold the healed blocks back over a copy of the master, shard by shard."""
    from ..export import bake

    bake.copy_aux_files(master, output, skip={RECORD_FILENAME})
    replacements: dict[str, torch.Tensor] = {}
    for path in sorted((output / "blocks").glob("block-*.safetensors")):
        index = int(path.stem.rsplit("-", 1)[1])
        prefix = prefixes.get(index)
        if prefix is None:
            raise ValueError(f"no state-dict prefix known for healed block {index}")
        for key, value in load_file(path).items():
            replacements[f"{prefix}{key}"] = value
    if not replacements:
        return

    # learnable scales live in the manifest's sidecar, not the shards — the same split
    # fit-stream writes, because a free-sum grid is not recoverable from the values
    scales: dict[str, list[torch.Tensor]] = {}
    for key in [k for k in replacements if k.rsplit(".", 1)[-1] in SCALE_ATTRS]:
        module, _, attr = key.rpartition(".")
        scales.setdefault(module, [None, None])[SCALE_ATTRS.index(attr)] = (
            replacements.pop(key).float().exp().reshape(-1)
        )
    if scales:
        _persist_scales(master, output, scales)

    landed: set[str] = set()
    for path in sorted(master.glob("*.safetensors")):
        tensors = load_file(path)
        for key in list(tensors):
            if key in replacements:
                tensors[key] = replacements[key]
                landed.add(key)
        destination = output / path.name
        partial = destination.with_suffix(destination.suffix + PARTIAL_SUFFIX)
        save_file(tensors, partial)
        os.replace(partial, destination)

    stranded = sorted(set(replacements) - landed)
    if stranded:
        raise ValueError(
            f"{len(stranded)} healed tensors match no key in the master "
            f"({stranded[:5]}); the healed blocks would be silently dropped"
        )


def _persist_scales(master: Path, output: Path, scales: dict[str, list]) -> None:
    """Write the healed learnable scales into the output manifest's sidecar."""
    from ..swap import SwapManifest

    try:
        manifest = SwapManifest.load(master)
    except FileNotFoundError:
        return
    for name, values in scales.items():
        present = [value for value in values if value is not None]
        if present:
            manifest.record_scales(name, *present)
    manifest.save(output)


def _release(module: nn.Module | None) -> None:
    """Drop a block's VRAM before the next one is materialized."""
    if module is None:
        return
    for param in module.parameters():
        # same dtype and device: `set_data` refuses a meta tensor over a real one
        param.grad = None
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()


def _weights_valid(record: BlockRecord, output: Path, index: int) -> bool:
    """Whether a recorded block's healed weights are still the recorded ones.

    Deliberately weights-only: the rolling window drops old activation buffers,
    so healed blocks far behind the frontier cannot and need not prove their
    outputs — only the resume boundary's buffer is checked, by the caller,
    against the previous record's student digest.
    """
    weights = output / "blocks" / f"block-{index:04d}.safetensors"
    return weights.is_file() and _file_sha256(weights) == record.weights_sha256


def _load_records(output: Path) -> dict[str, BlockRecord]:
    """Read the per-block records a previous pass wrote."""
    path = output / RECORD_FILENAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {key: BlockRecord(**value) for key, value in data.items()}


def _write_records(output: Path, records: dict[str, BlockRecord]) -> Path:
    """Persist the per-block records, temp-then-rename."""
    path = output / RECORD_FILENAME
    partial = path.with_suffix(path.suffix + PARTIAL_SUFFIX)
    partial.write_text(
        json.dumps({key: asdict(value) for key, value in records.items()}, indent=2)
    )
    os.replace(partial, path)
    return path


def _file_sha256(path: Path) -> str:
    """Return the sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trainable_block_parameters(block: nn.Module) -> list[torch.Tensor]:
    """Return the block parameters a heal trains: ternary latents, scales and norms."""
    return [param for param in block.parameters() if param.requires_grad]


def block_lambda(block: nn.Module, value: float) -> None:
    """Set λ on every quantized module inside one block."""
    for _, module in iter_quantized_modules(block):
        module.set_lambda(value)


__all__ = [
    "ActivationCache",
    "BlockRecord",
    "BlockwiseHealReport",
    "block_lambda",
    "heal_blocks",
    "trainable_block_parameters",
]
