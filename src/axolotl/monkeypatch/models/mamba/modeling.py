"""Sample-packing patches for the pure-SSM transformers models: Mamba, Mamba2, Falcon-Mamba.

None of these forwards accept ``position_ids``, so the ForCausalLM wrapper turns
them into ``seq_idx`` and stashes the boundaries on every block, which passes them
to its mixer as a kwarg. The mixers forward kwargs into their kernels, so Mamba2
needs nothing more than a live kernel that takes ``seq_idx``.

Mamba1's selective scan has no such argument, so a packed row is scattered into
right-padded per-document batches for the scan alone (the conv resets through
``seq_idx``) and gathered back: exact for real tokens, padding costs only the
scan. Its fused training kernel bakes in the whole row, so it is disabled for
packed batches and the forward continues on the unfused branch.
"""

import contextlib
import functools
import importlib
from dataclasses import dataclass, field

import torch

from axolotl.monkeypatch.models.mamba_utils import (
    get_seq_idx,
    kernel_accepts,
    require_seq_idx_kernels,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# a document group may pad to at most this multiple of its real tokens
PAD_FACTOR = 2.0

_FAMILIES = {
    "mamba": {
        "prefix": "Mamba",
        "seq_idx_kernels": ("causal_conv1d_fn",),
        "fused": "mamba_inner_fn",
        "scan": "mamba_selective_scan",
    },
    "falcon_mamba": {
        "prefix": "FalconMamba",
        "seq_idx_kernels": ("causal_conv1d_fn",),
        "fused": "mamba_inner_fn",
        "scan": "mamba_selective_scan",
    },
    "mamba2": {
        "prefix": "Mamba2",
        "seq_idx_kernels": (
            "causal_conv1d_fn",
            "mamba2_split_conv1d_scan_combined",
            "mamba2_chunk_scan",
        ),
        "fused": None,
        "scan": None,
    },
}


@dataclass
class PackedSegments:
    """Document boundaries of a packed batch, with a lazily built scatter plan."""

    seq_idx: torch.Tensor  # [B, T] int32
    _plan: list[tuple[torch.Tensor, torch.Tensor]] | None = field(
        default=None, repr=False
    )

    @property
    def plan(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``[(index, mask)]`` groups, each ``[docs, max_len]`` into the flat ``B*T`` axis."""
        if self._plan is None:
            self._plan = build_segment_plan(self.seq_idx)
        return self._plan


def build_segment_plan(
    seq_idx: torch.Tensor, pad_factor: float = PAD_FACTOR
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Group documents by length so no group pads beyond ``pad_factor`` of its tokens."""
    batch_size, seq_len = seq_idx.shape
    change = torch.ones_like(seq_idx, dtype=torch.bool)
    change[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
    starts = change.reshape(-1).nonzero().squeeze(-1)
    ends = torch.cat([starts[1:], starts.new_tensor([batch_size * seq_len])])
    lengths = ends - starts

    order = torch.argsort(lengths, descending=True)
    sorted_lengths = lengths[order].tolist()

    groups: list[list[int]] = []
    group_tokens = 0
    for position, length in enumerate(sorted_lengths):
        if groups:
            longest = sorted_lengths[groups[-1][0]]
            if longest * (len(groups[-1]) + 1) <= pad_factor * (group_tokens + length):
                groups[-1].append(position)
                group_tokens += length
                continue
        groups.append([position])
        group_tokens = length

    plan = []
    for positions in groups:
        docs = order[positions]
        max_len = sorted_lengths[positions[0]]
        offsets = torch.arange(max_len, device=seq_idx.device)
        mask = offsets[None, :] < lengths[docs][:, None]
        index = (starts[docs][:, None] + offsets[None, :]) * mask
        plan.append((index, mask))
    return plan


def packed_selective_scan(
    scan_fn, segments: PackedSegments, u, delta, A, B, C, D, z, delta_bias, **kwargs
):
    """Run ``scan_fn`` on each packed document separately; returns ``[B, D, T]``."""
    batch_size, _, seq_len = u.shape
    flat_out = None

    def to_docs(x, index, mask):  # [B, C, T] -> [docs, C, max_len], zero padded
        flat = x.transpose(1, 2).reshape(batch_size * seq_len, -1)
        return (
            (flat[index] * mask[..., None].to(flat.dtype)).transpose(1, 2).contiguous()
        )

    for index, mask in segments.plan:
        out = scan_fn(
            to_docs(u, index, mask),
            to_docs(delta, index, mask),
            A,
            to_docs(B, index, mask),
            to_docs(C, index, mask),
            D,
            to_docs(z, index, mask) if z is not None else None,
            delta_bias,
            **kwargs,
        )
        if isinstance(out, tuple):
            out = out[0]
        if flat_out is None:
            flat_out = out.new_zeros(batch_size * seq_len, out.shape[1])
        flat_out[index[mask]] = out.transpose(1, 2)[mask]

    return flat_out.view(batch_size, seq_len, -1).transpose(1, 2)


@contextlib.contextmanager
def _unfused_packed_scan(mod, segments: PackedSegments, fused: str, scan: str):
    """Disable the fused row kernel and split the selective scan per document.

    The mixer looks these up as module globals at call time, so swapping the
    attributes for the duration of the call is enough, and a gradient
    checkpointing recompute re-enters through the same patched mixer forward.
    """
    original_fused = getattr(mod, fused)
    original_scan = getattr(mod, scan)

    @functools.wraps(original_scan)
    def packed_shim(u, delta, A, B, C, D=None, z=None, delta_bias=None, **kwargs):
        kwargs.pop("return_last_state", None)
        return packed_selective_scan(
            original_scan, segments, u, delta, A, B, C, D, z, delta_bias, **kwargs
        )

    # returning None is the stock "no fused kernel" signal; the forward then
    # continues on the unfused branch
    setattr(mod, fused, lambda *args, **kwargs: None)
    setattr(mod, scan, packed_shim)
    try:
        yield
    finally:
        setattr(mod, fused, original_fused)
        setattr(mod, scan, original_scan)


def _binarize(attention_mask):
    # multipack encodes one id per document; the mixers only need 0/1 for padding
    if attention_mask is None:
        return None
    return (attention_mask != 0).to(attention_mask.dtype)


def _patch_causal_lm(causal_lm_cls) -> None:
    """Turn ``position_ids`` into ``PackedSegments`` stashed on every block.

    The blocks read the stash at call time, so a gradient-checkpointing recompute
    sees the same boundaries as the original forward.
    """
    if getattr(causal_lm_cls.forward, "_axolotl_seq_idx_patch", False):
        return
    original_forward = causal_lm_cls.forward

    @functools.wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        segments = None
        if position_ids is not None and kwargs.get("cache_params") is None:
            segments = PackedSegments(get_seq_idx(position_ids))
            # a packed eval batch would otherwise get a fresh cache and the stock path
            kwargs["use_cache"] = False
            kwargs["attention_mask"] = _binarize(kwargs.get("attention_mask"))
        for block in self.backbone.layers:
            block._axolotl_segments = segments
        return original_forward(self, *args, **kwargs)

    patched_forward._axolotl_seq_idx_patch = True
    causal_lm_cls.forward = patched_forward


def _patch_block(block_cls) -> None:
    """Hand the stashed segments to the mixer (Mamba1's block forwards no kwargs)."""
    if getattr(block_cls.forward, "_axolotl_seq_idx_patch", False):
        return
    original_forward = block_cls.forward

    @functools.wraps(original_forward)
    def patched_forward(
        self, hidden_states, cache_params=None, attention_mask=None, **kwargs
    ):
        segments = getattr(self, "_axolotl_segments", None)
        if segments is None or cache_params is not None:
            return original_forward(
                self,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            segments=segments,
            **kwargs,
        )
        return residual + hidden_states

    patched_forward._axolotl_seq_idx_patch = True
    block_cls.forward = patched_forward


def _assert_packed_ready(mixer, mod, family: dict, model_type: str) -> None:
    """Fail closed before a packed batch reaches a kernel that would drop seq_idx."""
    if "cuda" not in mixer.in_proj.weight.device.type:
        raise RuntimeError(
            f"{model_type} sample packing needs the CUDA kernels; the torch fallbacks "
            "have no document boundaries."
        )
    if getattr(mod, "_axolotl_seq_idx_verified", False):
        return
    for name in family["seq_idx_kernels"]:
        if kernel_accepts(getattr(mod, name), "seq_idx") is False:
            raise RuntimeError(
                f"{model_type} sample packing: `{name}` is the transformers torch "
                "fallback, which drops seq_idx and mixes state across packed samples. "
                "Install mamba-ssm and causal-conv1d or set `use_kernels: true`."
            )
    mod._axolotl_seq_idx_verified = True


def _patch_mixer(mod, mixer_cls, family: dict, model_type: str) -> None:
    if getattr(mixer_cls.forward, "_axolotl_seq_idx_patch", False):
        return
    original_forward = mixer_cls.forward

    @functools.wraps(original_forward)
    def patched_forward(
        self,
        hidden_states,
        cache_params=None,
        attention_mask=None,
        segments=None,
        **kwargs,
    ):
        if segments is None or cache_params is not None:
            return original_forward(
                self,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        _assert_packed_ready(self, mod, family, model_type)
        kwargs["seq_idx"] = segments.seq_idx
        if family["scan"] is None:
            return original_forward(
                self,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        with _unfused_packed_scan(mod, segments, family["fused"], family["scan"]):
            return original_forward(
                self,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )

    patched_forward._axolotl_seq_idx_patch = True
    mixer_cls.forward = patched_forward


def _import(model_type):
    try:
        return importlib.import_module(
            f"transformers.models.{model_type}.modeling_{model_type}"
        )
    except ImportError:
        LOG.warning(f"{model_type} not found in transformers, skipping packing patches")
        return None


def _apply(model_type: str, kernels_enabled: bool) -> None:
    family = _FAMILIES[model_type]
    mod = _import(model_type)
    if mod is None:
        return
    require_seq_idx_kernels(mod, family["seq_idx_kernels"], model_type, kernels_enabled)

    prefix = family["prefix"]
    _patch_causal_lm(getattr(mod, f"{prefix}ForCausalLM"))
    _patch_block(getattr(mod, f"{prefix}Block"))
    _patch_mixer(mod, getattr(mod, f"{prefix}Mixer"), family, model_type)

    LOG.info(f"Applied {prefix} sample packing patch (seq_idx threading into the SSM)")


def patch_mamba_modeling_packing(kernels_enabled: bool = False) -> None:
    _apply("mamba", kernels_enabled)


def patch_falcon_mamba_modeling_packing(kernels_enabled: bool = False) -> None:
    _apply("falcon_mamba", kernels_enabled)


def patch_mamba2_modeling_packing(kernels_enabled: bool = False) -> None:
    _apply("mamba2", kernels_enabled)
