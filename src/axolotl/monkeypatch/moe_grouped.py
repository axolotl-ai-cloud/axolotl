import logging
from functools import wraps

import torch

from axolotl.common.architectures import MOE_ARCH_BLOCK
from axolotl.kernels.moe.backends import MOEBackend, get_moe_backend_name

_LOG = logging.getLogger("axolotl.moe.patch")


def _patch_block_forward(block_cls, grouped_fn):
    """Replace block_cls.forward with grouped_fn preserving signature."""
    block_cls.forward = grouped_fn


def apply_grouped_to_moe_blocks(cfg=None) -> None:
    """
    Attempt to patch all known MoE block classes to use the torch_grouped backend
    when cfg.moe_backend resolves to 'torch_grouped' and the op is available.
    Falls back to original forwards otherwise.
    """
    preferred = getattr(cfg, "moe_backend", None) if cfg is not None else None
    backend = get_moe_backend_name(preferred)
    if backend != MOEBackend.TORCH_GROUPED:
        _LOG.info(
            f"moe_backend is '{backend}', not 'torch_grouped'; skipping grouped patches"
        )
        return
    try:
        from axolotl.kernels.moe import torch_grouped as _tg
    except Exception:
        _LOG.warning("torch_grouped backend import failed; skipping grouped patches")
        return
    if not _tg.available():
        _LOG.warning(
            "torch_grouped requested but unavailable (op smoke test failed); skipping grouped patches"
        )
        return

    # Map of architecture key to (modeling module path, class name or list of class names)
    model_mods = {
        "mixtral": (
            "transformers.models.mixtral.modeling_mixtral",
            MOE_ARCH_BLOCK.get("mixtral"),
        ),
        "qwen2_moe": (
            "transformers.models.qwen2_moe.modeling_qwen2_moe",
            MOE_ARCH_BLOCK.get("qwen2_moe"),
        ),
        "qwen3_moe": (
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            MOE_ARCH_BLOCK.get("qwen3_moe"),
        ),
        "jamba": (
            "transformers.models.jamba.modeling_jamba",
            MOE_ARCH_BLOCK.get("jamba"),
        ),
        "deepseek_v2": (
            "transformers.models.deepseek_v2.modeling_deepseek_v2",
            MOE_ARCH_BLOCK.get("deepseek_v2"),
        ),
        # Others may not follow standard paths; best-effort import
        "dbrx": ("transformers.models.dbrx.modeling_dbrx", MOE_ARCH_BLOCK.get("dbrx")),
        "jetmoe": (
            "transformers.models.jetmoe.modeling_jetmoe",
            MOE_ARCH_BLOCK.get("jetmoe"),
        ),
        "gpt_oss": (
            "transformers.models.gpt_oss.modeling_gpt_oss",
            MOE_ARCH_BLOCK.get("gpt_oss"),
        ),
    }

    def make_grouped_forward(orig_forward):
        @wraps(orig_forward)
        def _grouped_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
            bsz, seqlen, hdim = hidden_states.shape
            y, router_logits = _tg.moe_ffn_forward_grouped(
                hidden_states, self.gate, self.experts, self.top_k
            )
            # One-time log per block instance indicating whether grouped engaged or fallback occurred
            if not getattr(self, "_ax_grouped_wrapper_logged", False):
                if y is None:
                    _LOG.warning(
                        f"Grouped wrapper active but fell back to naive for {self.__class__.__name__}"
                    )
                else:
                    _LOG.info(
                        f"Grouped wrapper engaged for {self.__class__.__name__} (top_k={self.top_k})"
                    )
                self._ax_grouped_wrapper_logged = True
            if y is None:
                return orig_forward(self, hidden_states, *args, **kwargs)
            return y, router_logits

        return _grouped_forward

    patched = 0
    for key, (mod_path, cls_names) in model_mods.items():
        if not cls_names:
            continue
        try:
            import importlib

            modeling = importlib.import_module(mod_path)
            names = cls_names if isinstance(cls_names, list) else [cls_names]
            for name in names:
                if not hasattr(modeling, name):
                    continue
                block_cls = getattr(modeling, name)
                orig_forward = getattr(block_cls, "forward", None)
                if orig_forward is None:
                    continue
                _patch_block_forward(block_cls, make_grouped_forward(orig_forward))
                patched += 1
                _LOG.info(f"Patched MoE block for grouped GEMM: {mod_path}.{name}")
        except Exception as e:
            # Best effort; log and skip this entry
            _LOG.warning(f"Skipping MoE patch for arch '{key}' ({mod_path}): {e}")
    if patched == 0:
        _LOG.warning(
            "No MoE blocks patched for grouped GEMM; model may not use known MoE classes"
        )
    else:
        _LOG.info(f"Grouped GEMM patches applied to {patched} MoE block class(es)")
