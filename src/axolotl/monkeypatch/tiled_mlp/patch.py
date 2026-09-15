"""Monkeypatch for Tiled MLP implementation"""

import math
import os

import torch
import torch.distributed as dist

from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Suffixes used to discover MoE block classes inside
# ``transformers.models.{model_type}.modeling_{model_type}``.
# Order matters — preferred names come first.
_MOE_BLOCK_SUFFIXES = ("SparseMoeBlock", "MoeMLP", "MoE")


def _resolve_moe_block_cls(module, model_cls_prefix):
    """Return the MoE block class for the model module, or ``None`` if dense."""
    for suffix in _MOE_BLOCK_SUFFIXES:
        cls = getattr(module, f"{model_cls_prefix}{suffix}", None)
        if cls is not None:
            return cls
    return None


def _build_tiled_forward(
    inner_forward,
    model_type,
    cfg_num_shards,
    is_moe_block,
):
    """Construct a ``tiled_mlp_forward`` closure.

    The returned forward shards inputs along the sequence dim and dispatches
    to the correct :class:`torch.autograd.Function` implementation based on
    the parallel-training backend in use.

    ``inner_forward`` is the un-tiled forward (either the dense MLP forward
    or the MoE block's routing+expert forward — possibly a kernels-substituted
    forward in the scattermoe-lora case).
    """
    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        TiledMLP as DeepSpeedTiledMLP,
    )

    from axolotl.monkeypatch.tiled_mlp.base import DeepSpeedTiledMLPMoE, TiledMLP

    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    def tiled_mlp_forward(self, x):
        input_shape = x.shape
        seqlen = input_shape[-2]
        if cfg_num_shards is None:
            # Target ~32K tokens per shard. The previous `ceil(seq / hidden)`
            # heuristic produced only ~2K tokens/shard at long context, well
            # below the MoE kernel's BLOCK_M sweet spot. An empirical sweep at
            # seq ∈ {64K, 128K, 256K, 512K} showed 3.2× speed-up at 64–256K
            # and 2.1× at 512K from raising per-shard tokens to ~32K, with
            # only a modest peak-mem cost (~5–10 GiB extra at seq=256K)
            # because the routed intermediate buffer dominates and scales
            # linearly with per-shard tokens. Operators can override via
            # cfg_num_shards for niche cases (smaller intermediate, larger
            # top_k) where the default is wrong.
            target_tokens_per_shard = 32768
            num_shards = max(1, math.ceil(seqlen / target_tokens_per_shard))
            if is_distributed:
                num_shards_tensor = torch.tensor(num_shards, device=x.device)
                dist.all_reduce(num_shards_tensor, op=dist.ReduceOp.MAX)
                num_shards = num_shards_tensor.item()
        else:
            num_shards = cfg_num_shards

        if not self._compute_params:
            self._compute_params = [p for p in self.parameters() if p.requires_grad]

        compute_params = self._compute_params
        if not self._tiled_mlp_dist_impl:
            uses_deepspeed = (
                self._compute_params
                and any(
                    hasattr(p, "ds_id") or hasattr(p, "param_idx_in_group")
                    for p in self._compute_params
                )
            ) or os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"

            if uses_deepspeed:
                # gpt_oss already used the MoE variant before this refactor;
                # extend the same treatment to every MoE block, since they
                # tend to return tuple outputs (hidden_states, router_logits)
                # the way gpt_oss does.
                if model_type == "gpt_oss" or is_moe_block:
                    self._tiled_mlp_dist_impl = DeepSpeedTiledMLPMoE
                else:
                    self._tiled_mlp_dist_impl = DeepSpeedTiledMLP
            else:
                self._tiled_mlp_dist_impl = TiledMLP

        return self._tiled_mlp_dist_impl.apply(
            inner_forward,
            self,
            x,
            num_shards,
            compute_params,
        )

    return tiled_mlp_forward


def _prepare_target_class(target_cls):
    """Initialize the bookkeeping attrs the tiled forward expects."""
    target_cls._compute_params = []
    target_cls._tiled_mlp_dist_impl = None


def patch_tiled_mlp(
    model_type,
    use_original_mlp=True,
    cfg_num_shards=None,
    use_scattermoe=False,
):
    """Install the class-level tiled MLP patch.

    For dense models this patches ``{prefix}MLP`` (falling back to
    ``{prefix}TextMLP`` for multimodal wrappers).

    For MoE models with scattermoe-lora active, the MoE block class
    (``{prefix}SparseMoeBlock`` / ``{prefix}MoeMLP`` / ``{prefix}MoE``) is the
    one whose forward does routing + expert invocation, so we patch that.
    Note that the ``kernels`` library installs scattermoe-lora's forward at
    the *instance* level during ``model.kernelize()``, so the class-level
    patch is shadowed at runtime. :func:`patch_tiled_mlp_moe_instances` is
    the companion post-model-load step that re-wraps each MoE block instance
    so the tiled forward runs on top of the kernels-installed forward.
    """
    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
    try:
        module = __import__(module_path, fromlist=[f"{model_cls_prefix}MLP"])
    except ImportError as e:
        raise RuntimeError(
            f"Could not import MLP class for model_type: {model_type}. Error: {str(e)}"
        ) from e

    # MoE block patch path: only walk into this branch when scattermoe-lora
    # is active. For non-scattermoe MoE models the dense MLP fallback applies
    # — we do not auto-enable MoE-block tiling because each model family's
    # block forward has different output-tuple semantics.
    moe_block_cls = (
        _resolve_moe_block_cls(module, model_cls_prefix) if use_scattermoe else None
    )

    if moe_block_cls is not None:
        original_forward = moe_block_cls.forward
        tiled_forward = _build_tiled_forward(
            inner_forward=original_forward,
            model_type=model_type,
            cfg_num_shards=cfg_num_shards,
            is_moe_block=True,
        )
        moe_block_cls.forward = tiled_forward
        _prepare_target_class(moe_block_cls)
        LOG.info(
            "Successfully monkey-patched TiledMLP for model_type: "
            f"{model_type} (MoE block: {moe_block_cls.__name__})"
        )
        return

    # Dense MLP path (existing behavior).
    try:
        mlp_cls = getattr(
            module,
            f"{model_cls_prefix}MLP",
            None,
        ) or getattr(module, f"{model_cls_prefix}TextMLP")
    except AttributeError as e:
        raise RuntimeError(
            f"Could not import MLP class for model_type: {model_type}. Error: {str(e)}"
        ) from e

    if use_original_mlp:
        mlp_forward = mlp_cls.forward
    else:

        def generic_mlp_forward(self_, hs):
            return self_.down_proj(
                self_.act_fn(self_.gate_proj(hs)) * self_.up_proj(hs)
            )

        mlp_forward = torch.compile(generic_mlp_forward)

    tiled_forward = _build_tiled_forward(
        inner_forward=mlp_forward,
        model_type=model_type,
        cfg_num_shards=cfg_num_shards,
        is_moe_block=False,
    )
    mlp_cls.forward = tiled_forward
    _prepare_target_class(mlp_cls)
    LOG.info(f"Successfully monkey-patched TiledMLP for model_type: {model_type}")


def patch_tiled_mlp_moe_instances(
    model,
    model_type,
    cfg_num_shards=None,
):
    """Re-wrap each MoE block instance's ``forward`` after model load.

    The ``kernels`` library installs scattermoe-lora's forward on each MoE
    block *instance* during ``model.kernelize()`` (called inside
    ``from_pretrained``). That instance-level binding shadows the class-level
    patch :func:`patch_tiled_mlp` installs, so without this step tiling is
    silently bypassed on every block. We capture each instance's current
    forward (the kernels-installed one) and rebind the instance to a tiled
    forward that delegates to it.

    Does nothing if no MoE block class exists for ``model_type`` or if
    ``model`` contains no instances of it.
    """
    from types import MethodType

    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
    try:
        module = __import__(module_path, fromlist=[model_cls_prefix])
    except ImportError:
        return 0

    moe_block_cls = _resolve_moe_block_cls(module, model_cls_prefix)
    if moe_block_cls is None:
        return 0

    wrapped = 0
    for sub in model.modules():
        if not isinstance(sub, moe_block_cls):
            continue
        # If there is no per-instance ``forward`` binding, the class-level
        # tiled patch from ``patch_tiled_mlp`` is still active; nothing to do.
        # Kernels (when scattermoe-lora kernelizes the model) installs a
        # bound method on the instance, which shows up in ``__dict__``.
        if "forward" not in sub.__dict__:
            continue
        # Snapshot the instance-level forward installed by kernels.
        bound_forward = sub.__dict__["forward"]
        # Convert bound method back to a plain function that takes (self, x)
        # so the tiled wrapper can pass `self` through to it.
        if hasattr(bound_forward, "__func__"):
            inner_fn = bound_forward.__func__
        else:
            # Instance-bound closure: wrap it so the (self, x) signature lines up.
            def _adapt(orig):
                def _call(self_, x):  # noqa: ARG001
                    return orig(x)

                return _call

            inner_fn = _adapt(bound_forward)

        tiled_forward = _build_tiled_forward(
            inner_forward=inner_fn,
            model_type=model_type,
            cfg_num_shards=cfg_num_shards,
            is_moe_block=True,
        )
        # Each instance needs its own bookkeeping (compute_params,
        # dist_impl) so concurrent forwards across blocks don't stomp.
        sub._compute_params = []
        sub._tiled_mlp_dist_impl = None
        sub.forward = MethodType(tiled_forward, sub)
        wrapped += 1

    if wrapped:
        LOG.info(
            f"Successfully wrapped TiledMLP around {wrapped} {moe_block_cls.__name__} "
            f"instance(s) for model_type: {model_type}"
        )
    return wrapped
