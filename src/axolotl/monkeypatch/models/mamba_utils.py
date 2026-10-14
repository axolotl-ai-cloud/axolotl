"""Shared utilities for Mamba2 SSM sample-packing and context-parallelism patches.

Used by: nemotron_h, falcon_h1, granite_moe_hybrid
"""

import functools

import torch
import torch.distributed as dist

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def get_seq_idx(position_ids: torch.Tensor) -> torch.Tensor:
    """Convert position_ids [B, T] → seq_idx [B, T] int32 for mamba-ssm kernels.

    Example: position_ids [[0,1,2,3,0,1,2]] → seq_idx [[0,0,0,0,1,1,1]]

    Under context parallelism a rank may receive a chunk that begins mid-sample
    (position_ids[0] != 0), so the raw cumsum starts at 0 and subtracting 1
    would yield -1 — an invalid value for the Mamba kernels.  Subtracting the
    first element of the cumsum instead normalises every chunk to start at 0
    while still correctly incrementing at every intra-chunk sample boundary.

    Example (CP rank 1, chunk starts mid-sample):
        position_ids [[3,4,5,0,1,2]] → seq_idx [[0,0,0,1,1,1]]
    """
    cumsum = torch.cumsum((position_ids == 0).int(), dim=-1)
    return (cumsum - cumsum[..., :1]).to(torch.int32)


def is_cp_active() -> bool:
    """Return True if context parallelism (ring attention) is active on this rank.

    Zero-cost when CP is not configured: the import guard ensures we only touch
    the distributed group if ring_flash_attn is installed.
    """
    try:
        from axolotl.monkeypatch.ring_attn import get_ring_attn_group

        group = get_ring_attn_group()
        return group is not None and dist.get_world_size(group) > 1
    except (ImportError, RuntimeError):
        return False


def mamba2_seq_idx_kernels_available() -> bool:
    """True when the installed Mamba2 kernels honour ``seq_idx``.

    transformers resolves ``mamba-ssm`` / ``causal-conv1d`` ahead of its own
    torch fallbacks, and those fallbacks drop unknown kwargs — so without the
    real kernels ``seq_idx`` is silently ignored and SSM state leaks across
    packed samples.
    """
    import importlib
    import inspect

    try:
        conv_fn = importlib.import_module("causal_conv1d").causal_conv1d_fn
        scan_fn = importlib.import_module(
            "mamba_ssm.ops.triton.ssd_combined"
        ).mamba_chunk_scan_combined
    except Exception:  # pylint: disable=broad-exception-caught
        return False

    return all(
        "seq_idx" in inspect.signature(fn).parameters for fn in (conv_fn, scan_fn)
    )


def assert_mamba2_scan_honours_seq_idx(target_module) -> None:
    """Probe the resolved chunk-scan once, and raise if it drops ``seq_idx``.

    Which implementation ends up behind ``mamba2_chunk_scan`` is only settled
    once the model is built (pip package, Hub kernel, or the torch fallback),
    and the fallback drops unknown kwargs silently — training would then mix
    SSM state across packed samples. The probe is skipped when it cannot run
    (no CUDA device, or a kernel that rejects the probe shapes).
    """
    if getattr(target_module, "_axolotl_seq_idx_probed", False):
        return
    target_module._axolotl_seq_idx_probed = True

    scan = getattr(target_module, "mamba2_chunk_scan", None) or getattr(
        target_module, "mamba_chunk_scan_combined", None
    )
    if scan is None or not torch.cuda.is_available():
        return

    num_heads, head_dim, state_size, chunk_size = 2, 64, 16, 64
    seq_len = 2 * chunk_size
    kwargs = {"dtype": torch.bfloat16, "device": "cuda"}
    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(1, seq_len, num_heads, head_dim, generator=generator, **kwargs)
    dt = torch.rand(1, seq_len, num_heads, generator=generator, **kwargs)
    A = -torch.rand(num_heads, generator=generator, device="cuda")
    B = torch.randn(1, seq_len, 1, state_size, generator=generator, **kwargs)
    C = torch.randn(1, seq_len, 1, state_size, generator=generator, **kwargs)
    seq_idx = torch.zeros(1, seq_len, dtype=torch.int32, device="cuda")
    seq_idx[:, chunk_size:] = 1

    try:
        with torch.no_grad():
            without = scan(hidden, dt, A, B, C, chunk_size=chunk_size)
            with_seq_idx = scan(
                hidden, dt, A, B, C, chunk_size=chunk_size, seq_idx=seq_idx
            )
    except Exception as exception:  # pylint: disable=broad-exception-caught
        LOG.warning(
            "Could not probe the Mamba2 scan for seq_idx support: %s", exception
        )
        return

    if torch.equal(without, with_seq_idx):
        raise RuntimeError(
            "The Mamba2 chunk scan in use ignores seq_idx, so sample packing "
            "would mix SSM state across packed samples. Install the kernels "
            "(`pip install mamba-ssm causal-conv1d`) or disable sample_packing."
        )


def ensure_mamba_kernels_loaded(target_module):
    """Eagerly resolve mamba-ssm and causal-conv1d globals on *target_module*.

    Transformers >= 5.5 lazily loads these inside ``Mixer.__init__`` via
    ``lazy_load_kernel``.  Our monkeypatches run *before* model instantiation,
    so the module globals are still ``None``.  This helper triggers the kernel
    resolution early so the patched ``cuda_kernels_forward`` can reference them.
    """
    if getattr(target_module, "mamba_chunk_scan_combined", None) is not None:
        return

    try:
        from transformers.integrations.hub_kernels import lazy_load_kernel
        from transformers.utils.import_utils import resolve_internal_import
    except ImportError:
        return

    causal_conv1d = lazy_load_kernel("causal-conv1d")
    if causal_conv1d is not None:
        target_module.causal_conv1d_update = getattr(
            causal_conv1d, "causal_conv1d_update", None
        )
        target_module.causal_conv1d_fn = getattr(
            causal_conv1d, "causal_conv1d_fn", None
        )

    mamba_ssm = lazy_load_kernel("mamba-ssm")
    if mamba_ssm is not None:
        target_module.selective_state_update = resolve_internal_import(
            mamba_ssm,
            chained_path="ops.triton.selective_state_update.selective_state_update",
        )
        target_module.mamba_chunk_scan_combined = resolve_internal_import(
            mamba_ssm,
            chained_path="ops.triton.ssd_combined.mamba_chunk_scan_combined",
        )
        target_module.mamba_split_conv1d_scan_combined = resolve_internal_import(
            mamba_ssm,
            chained_path="ops.triton.ssd_combined.mamba_split_conv1d_scan_combined",
        )

    target_module.is_fast_path_available = all(
        (
            getattr(target_module, "selective_state_update", None),
            getattr(target_module, "mamba_chunk_scan_combined", None),
            getattr(target_module, "mamba_split_conv1d_scan_combined", None),
            getattr(target_module, "causal_conv1d_fn", None),
            getattr(target_module, "causal_conv1d_update", None),
        )
    )


def patch_model_forward_seq_idx(model_cls) -> None:
    """Inject ``seq_idx`` (from ``position_ids``) into ``model_cls.forward`` kwargs.

    For architectures whose mixers already accept ``seq_idx`` through ``**kwargs``
    but whose model forward never derives it, this is the only piece missing for
    sample packing.
    """
    if getattr(model_cls.forward, "_axolotl_seq_idx_patch", False):
        return

    original_forward = model_cls.forward

    @functools.wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is None and len(args) > 2:
            position_ids = args[2]

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None and len(args) > 3:
            past_key_values = args[3]

        is_decoding = (
            past_key_values is not None
            and hasattr(past_key_values, "has_previous_state")
            and past_key_values.has_previous_state
        )

        if position_ids is not None and not is_decoding and "seq_idx" not in kwargs:
            kwargs["seq_idx"] = get_seq_idx(position_ids)

        return original_forward(self, *args, **kwargs)

    patched_forward._axolotl_seq_idx_patch = True
    model_cls.forward = patched_forward


def kernel_accepts(fn, name: str) -> bool | None:
    """Whether a hub-wrapped kernel's live implementation takes ``name``.

    transformers resolves the pip kernel at import time and filters kwargs to
    its signature, so a torch fallback (which takes ``**kwargs``) drops
    ``seq_idx`` without a word. Once kernelize has swapped in a Hub kernel, its
    own signature is checked instead; the Hub layers declare ``**kwargs`` and pop
    the name inside, so their source is searched. Returns None when nothing can
    be told.
    """
    import inspect

    try:
        # kernels >= 0.16 turns a hub-decorated function into an nn.Module whose
        # forward closes over the transformers wrapper
        if not inspect.isroutine(fn) and hasattr(fn, "forward"):
            forward = fn.forward
            try:
                fn = inspect.getclosurevars(forward).nonlocals["func"]
            except (TypeError, KeyError):
                params = inspect.signature(forward).parameters
                if name in params:
                    return True
                if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
                    return False
                try:
                    return True if name in inspect.getsource(forward) else None
                except (OSError, TypeError):
                    return None
        implementation = inspect.getclosurevars(fn).nonlocals["implementation"]
        return name in inspect.signature(implementation).parameters
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def require_seq_idx_kernels(mod, names, model_type: str, kernels_enabled: bool) -> None:
    """Raise unless every kernel in ``names`` honours ``seq_idx`` (or Hub kernels will)."""
    if kernels_enabled:
        return
    dropped = [
        name
        for name in names
        if getattr(mod, name, None) is not None
        and kernel_accepts(getattr(mod, name), "seq_idx") is False
    ]
    if dropped:
        raise RuntimeError(
            f"{model_type} sample packing requires kernels that take seq_idx, but "
            f"{', '.join(dropped)} resolved to the transformers torch fallback, which "
            "silently mixes state across packed samples. Install them (`pip install "
            "mamba-ssm causal-conv1d`) or set `use_kernels: true`."
        )
