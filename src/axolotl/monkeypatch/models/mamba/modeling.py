"""Sample-packing patches for the pure-SSM transformers models: Mamba, Mamba2, Falcon-Mamba.

None of these forwards accept ``position_ids``, so the ForCausalLM wrapper turns
them into ``seq_idx`` and hands it to every block, which passes it to its mixer.

Mamba2's kernels take ``seq_idx`` directly. Mamba1's selective scan has no such
argument, so a packed row is scattered into a right-padded per-document batch
for the scan alone (the conv resets through ``seq_idx``); outputs are gathered
back, so real tokens are exact and padding costs only the scan.
"""

import contextlib
import functools
import importlib
from dataclasses import dataclass

import torch

from axolotl.monkeypatch.models.mamba_utils import (
    get_seq_idx,
    mamba2_seq_idx_kernels_available,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class PackedSegments:
    """Document boundaries of a packed batch, with a lazily built scatter plan."""

    seq_idx: torch.Tensor  # [B, T] int32
    _plan: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def plan(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(index, mask)`` of shape ``[num_docs, max_len]`` into the flattened ``B*T`` axis."""
        if self._plan is None:
            self._plan = build_segment_plan(self.seq_idx)
        return self._plan


def build_segment_plan(seq_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = seq_idx.shape
    change = torch.ones_like(seq_idx, dtype=torch.bool)
    change[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
    starts = change.reshape(-1).nonzero().squeeze(-1)
    ends = torch.cat([starts[1:], starts.new_tensor([batch_size * seq_len])])
    lengths = ends - starts
    max_len = int(lengths.max())
    offsets = torch.arange(max_len, device=seq_idx.device)
    mask = offsets[None, :] < lengths[:, None]
    index = (starts[:, None] + offsets[None, :]) * mask
    return index, mask


def packed_selective_scan(
    scan_fn, segments: PackedSegments, u, delta, A, B, C, D, z, delta_bias, **kwargs
):
    """Run ``scan_fn`` on each packed document separately; returns ``[B, D, T]``."""
    index, mask = segments.plan
    batch_size, _, seq_len = u.shape

    def to_docs(x):  # [B, C, T] -> [num_docs, C, max_len], zero padded
        flat = x.transpose(1, 2).reshape(batch_size * seq_len, -1)
        return (
            (flat[index] * mask[..., None].to(flat.dtype)).transpose(1, 2).contiguous()
        )

    out = scan_fn(
        to_docs(u),
        to_docs(delta),
        A,
        to_docs(B),
        to_docs(C),
        D,
        to_docs(z) if z is not None else None,
        delta_bias,
        **kwargs,
    )
    if isinstance(out, tuple):
        out = out[0]

    flat = out.new_zeros(batch_size * seq_len, out.shape[1])
    flat[index[mask]] = out.transpose(1, 2)[mask]
    return flat.view(batch_size, seq_len, -1).transpose(1, 2)


@contextlib.contextmanager
def _kernels_with_seq_idx(mod, segments: PackedSegments, packed_scan_attr: str | None):
    """Swap the module-level kernels for ones that carry the document boundaries.

    The stock forwards pass ``seq_idx=None`` explicitly, so the shims override the
    kwarg instead of merely defaulting it.
    """
    seq_idx = segments.seq_idx
    saved = {}

    def inject(name):
        original = getattr(mod, name, None)
        if original is None:
            return

        @functools.wraps(original)
        def shim(*args, **kwargs):
            kwargs["seq_idx"] = seq_idx
            return original(*args, **kwargs)

        saved[name] = original
        setattr(mod, name, shim)

    inject("causal_conv1d_fn")
    inject("mamba_chunk_scan_combined")
    inject("mamba_split_conv1d_scan_combined")

    if packed_scan_attr is not None:
        original_scan = getattr(mod, packed_scan_attr)

        @functools.wraps(original_scan)
        def packed_shim(u, delta, A, B, C, D=None, z=None, delta_bias=None, **kwargs):
            kwargs.pop("return_last_state", None)
            return packed_selective_scan(
                original_scan, segments, u, delta, A, B, C, D, z, delta_bias, **kwargs
            ), None

        saved[packed_scan_attr] = original_scan
        setattr(mod, packed_scan_attr, packed_shim)

    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(mod, name, original)


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
        cache_params = kwargs.get("cache_params")
        segments = None
        if position_ids is not None and cache_params is None:
            segments = PackedSegments(get_seq_idx(position_ids))
            # a packed eval batch would otherwise get a fresh cache and the stock path
            kwargs["use_cache"] = False
        for block in self.backbone.layers:
            block._axolotl_segments = segments
        return original_forward(self, *args, **kwargs)

    patched_forward._axolotl_seq_idx_patch = True
    causal_lm_cls.forward = patched_forward


def _patch_block(block_cls) -> None:
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
            attention_mask=_binarize(attention_mask),
            segments=segments,
        )
        return residual + hidden_states

    patched_forward._axolotl_seq_idx_patch = True
    block_cls.forward = patched_forward


def _patch_mixer(
    mod, mixer_cls, packed_scan_attr: str | None, force_unfused: bool
) -> None:
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
        if not (
            _fast_path_available(mod) and "cuda" in self.in_proj.weight.device.type
        ):
            raise RuntimeError(
                f"{mixer_cls.__name__} sample packing needs the CUDA fast path: the torch "
                "fallback has no document boundaries. Install mamba-ssm and "
                "causal-conv1d (or set `use_kernels: true`) and train on CUDA."
            )
        # Mamba1's fused training kernel bakes in the whole row; the unfused
        # branch is the one whose conv and scan can be told about boundaries.
        was_training = self.training
        if force_unfused:
            self.training = False
        try:
            with _kernels_with_seq_idx(mod, segments, packed_scan_attr):
                return self.cuda_kernels_forward(
                    hidden_states, cache_params, attention_mask
                )
        finally:
            if force_unfused:
                self.training = was_training

    patched_forward._axolotl_seq_idx_patch = True
    mixer_cls.forward = patched_forward


def _fast_path_available(mod) -> bool:
    # Mamba2 keeps this as a module global; Mamba1 recomputes it from the kernels
    available = getattr(mod, "is_fast_path_available", None)
    if available is not None:
        return bool(available)
    return all(
        getattr(mod, name, None) is not None
        for name in ("selective_scan_fn", "causal_conv1d_fn")
    )


def _import(model_type):
    try:
        return importlib.import_module(
            f"transformers.models.{model_type}.modeling_{model_type}"
        )
    except ImportError:
        LOG.warning(f"{model_type} not found in transformers, skipping packing patches")
        return None


def mamba1_packing_kernels_available() -> bool:
    try:
        importlib.import_module("causal_conv1d")
        importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return True


def _require_kernels(model_type, kernels_enabled, available):
    if not (kernels_enabled or available):
        raise RuntimeError(
            f"{model_type} sample packing requires the Mamba kernels: the transformers "
            "torch fallbacks have no document boundaries, which silently mixes SSM "
            "state across packed samples. Either install them (`pip install "
            "mamba-ssm causal-conv1d`) or set `use_kernels: true`."
        )


def patch_mamba_modeling_packing(kernels_enabled: bool = False) -> None:
    _apply(
        "mamba",
        "Mamba",
        kernels_enabled,
        mamba1_packing_kernels_available(),
        packed_scan_attr="selective_scan_fn",
        force_unfused=True,
    )


def patch_falcon_mamba_modeling_packing(kernels_enabled: bool = False) -> None:
    _apply(
        "falcon_mamba",
        "FalconMamba",
        kernels_enabled,
        mamba1_packing_kernels_available(),
        packed_scan_attr="selective_scan_fn",
        force_unfused=True,
    )


def patch_mamba2_modeling_packing(kernels_enabled: bool = False) -> None:
    _apply(
        "mamba2",
        "Mamba2",
        kernels_enabled,
        mamba2_seq_idx_kernels_available(),
        packed_scan_attr=None,
        force_unfused=False,
    )


def _apply(
    model_type, cls_prefix, kernels_enabled, available, packed_scan_attr, force_unfused
):
    mod = _import(model_type)
    if mod is None:
        return
    _require_kernels(model_type, kernels_enabled, available)

    _patch_causal_lm(getattr(mod, f"{cls_prefix}ForCausalLM"))
    _patch_block(getattr(mod, f"{cls_prefix}Block"))
    _patch_mixer(
        mod, getattr(mod, f"{cls_prefix}Mixer"), packed_scan_attr, force_unfused
    )

    LOG.info(
        f"Applied {cls_prefix} sample packing patch (seq_idx threading into the SSM)"
    )
