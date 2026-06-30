"""Post-load NVFP4 module swap — runs in post_model_load (post weights/merge/PEFT) so modules are in final position; FFT swaps raw nn.Linear, adapter modes swap the frozen lora base_layer."""

import os
import re

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def resolve_ce_mode(nvfp4) -> str:
    """Effective lm_head CE kernel."""
    return getattr(nvfp4, "lm_head_cross_entropy", None) or "off"


def fp4_ce_active(nvfp4) -> bool:
    """True when the resolved CE kernel reads the NVFP4-packed lm_head."""
    mode = resolve_ce_mode(nvfp4)
    if mode == "fp4":
        return True
    if mode == "auto":
        return bool(getattr(nvfp4, "quantize_lm_head", False))
    return False


def _as_args(nvfp4):
    """Return an NVFP4Args for the swap to read; rebuilds it after validate_config dict-ifies cfg, which would otherwise drop the @property bridge and silently disable the swap."""
    from .args import NVFP4Args

    return nvfp4 if isinstance(nvfp4, NVFP4Args) else NVFP4Args(**nvfp4)


def _module_name(model: PreTrainedModel, target) -> str | None:
    if target is None:
        return None
    return next((n for n, m in model.named_modules() if m is target), None)


def _model_ties_embeddings(model: PreTrainedModel) -> bool:
    # Weight identity is authoritative; config.tie_word_embeddings can be stale.
    try:
        out_emb = model.get_output_embeddings()
        in_emb = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        return bool(
            getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        )
    out_w = getattr(out_emb, "weight", None)
    in_w = getattr(in_emb, "weight", None)
    return out_w is not None and in_w is not None and out_w is in_w


def _tied_weight_trainable(model: PreTrainedModel) -> bool:
    try:
        in_w = getattr(model.get_input_embeddings(), "weight", None)
    except (AttributeError, NotImplementedError):
        return False
    return bool(getattr(in_w, "requires_grad", False))


def _block_exclusions(
    model: PreTrainedModel, skip_first: int, skip_last: int
) -> tuple[str, ...]:
    """Translate skip_first/last block counts into ``layers.<i>.`` name fragments."""
    if skip_first <= 0 and skip_last <= 0:
        return ()
    block_re = re.compile(r"(.*\blayers)\.(\d+)\.")
    prefixes: dict[str, set[int]] = {}
    for name, _ in model.named_modules():
        m = block_re.match(name)
        if m:
            prefixes.setdefault(m.group(1), set()).add(int(m.group(2)))
    fragments: list[str] = []
    for prefix, indices in prefixes.items():
        ordered = sorted(indices)
        skip = set(ordered[:skip_first])
        if skip_last > 0:
            skip |= set(ordered[len(ordered) - skip_last :])
        fragments.extend(f"{prefix}.{i}." for i in sorted(skip))
    return tuple(fragments)


def _resolve_keep_hp_counts(nvfp4, model: PreTrainedModel) -> tuple[int, int]:
    """(skip_first, skip_last). The 'paper' preset's tail (round(0.13*L)) needs the post-build block count L."""
    if getattr(nvfp4, "keep_hp_paper_preset", False):
        block_re = re.compile(r".*\blayers\.(\d+)\.")
        indices = {
            int(m.group(1))
            for name, _ in model.named_modules()
            if (m := block_re.match(name))
        }
        n_blocks = (max(indices) + 1) if indices else 0
        return 1, round(0.13 * n_blocks)
    return int(nvfp4.skip_first_n_blocks or 0), int(nvfp4.skip_last_n_blocks or 0)


def _check_lm_head_quantizable(cfg, nvfp4, model) -> None:
    """Reject unsafe `quantize: [lm_head]` combos: trainable tied weight (FP4-storing a shared trainable weight corrupts training) and cut_cross_entropy (reads the head weight directly)."""
    from .nvfp4_training import _is_swappable

    tied = _model_ties_embeddings(model)
    if tied:
        if _tied_weight_trainable(model):
            raise RuntimeError(
                "nvfp4_training quantize: [lm_head] with tied embeddings requires a "
                "FROZEN shared weight: FP4-storing a TRAINABLE shared weight would "
                "corrupt training. Freeze the embedding (e.g. use LoRA), or drop "
                "lm_head from `quantize`."
            )

    if cfg.cut_cross_entropy and not fp4_ce_active(nvfp4):
        raise RuntimeError(
            "nvfp4_training quantize: [lm_head] is incompatible with cut_cross_entropy "
            "(the fused linear CE reads the lm_head weight directly, bypassing the "
            "NVFP4 head). Disable one, or set nvfp4_training.cross_entropy.mode: fp4 "
            "(or auto) to use the FP4-aware fused cross-entropy."
        )

    if tied:
        return

    out_emb = model.get_output_embeddings()
    if not isinstance(out_emb, torch.nn.Linear) or not _is_swappable(out_emb):
        LOG.warning(
            "nvfp4_training quantize: [lm_head]: lm_head is not NVFP4-swappable "
            "(in=%s out=%s, both must be %%32); it will stay high precision.",
            getattr(out_emb, "in_features", "?"),
            getattr(out_emb, "out_features", "?"),
        )


def _swap_frozen_lm_head(model, recipe, base_mode: str) -> None:
    from .nvfp4_training import swap_frozen_linear_to_nvfp4

    out_emb = model.get_output_embeddings()
    if not isinstance(out_emb, nn.Linear):
        return
    name = _module_name(model, out_emb)
    if name is None:
        LOG.warning(
            "nvfp4_training: could not locate the lm_head module; leaving it HP."
        )
        return
    swap_frozen_linear_to_nvfp4(model, name, recipe, base_mode=base_mode)


def _apply_tied_or_lm_head(nvfp4, model, recipe, base_mode: str) -> None:
    """Route the tied / lm_head / embedding NVFP4 swaps post linear-conversion."""
    from .nvfp4_training import (
        swap_frozen_embedding_to_nvfp4,
        swap_frozen_lm_head_tileable,
        swap_tied_embedding_and_lm_head_to_nvfp4,
    )

    want_lm_head = bool(getattr(nvfp4, "quantize_lm_head", False))
    want_embed = bool(getattr(nvfp4, "quantize_embeddings", False))
    if not (want_lm_head or want_embed):
        return

    tied = _model_ties_embeddings(model)

    if tied and want_lm_head:
        in_name = _module_name(model, model.get_input_embeddings())
        out_name = _module_name(model, model.get_output_embeddings())
        if in_name and out_name:
            swap_tied_embedding_and_lm_head_to_nvfp4(model, in_name, out_name, recipe)
        return

    if want_lm_head:
        if fp4_ce_active(nvfp4):
            out_emb = model.get_output_embeddings()
            if isinstance(out_emb, nn.Linear):
                name = _module_name(model, out_emb)
                if name:
                    swap_frozen_lm_head_tileable(model, name, recipe)
        else:
            _swap_frozen_lm_head(model, recipe, base_mode)

    if want_embed:
        in_emb = model.get_input_embeddings()
        if isinstance(in_emb, nn.Embedding):
            in_name = _module_name(model, in_emb)
            if in_name:
                swap_frozen_embedding_to_nvfp4(model, in_name)


def _load_packed_sidecar(cfg, model: PreTrainedModel) -> None:
    """Restore FP4-packed weights from a save_packed sidecar, if one exists."""
    from .nvfp4_training import NVFP4_PACKED_SIDECAR, load_nvfp4_packed

    for cand in (cfg.resume_from_checkpoint, cfg.base_model):
        if not cand or not isinstance(cand, str):
            continue
        if os.path.isfile(os.path.join(cand, NVFP4_PACKED_SIDECAR)):
            load_nvfp4_packed(model, cand)
            return


def mark_ddp_ignore(cfg, model: PreTrainedModel) -> None:
    """Exclude NVFP4 frozen-base buffers from DDP sync (NCCL can't broadcast the packed dtypes; they're frozen and per-rank identical anyway)."""
    nvfp4 = cfg.nvfp4_training
    if not nvfp4:
        return
    nvfp4 = _as_args(nvfp4)
    if not nvfp4.enabled:
        return
    exotic = {torch.float8_e4m3fn, torch.float8_e5m2}
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is not None:
        exotic.add(fp4)
    ignore = [
        name
        for name, buf in model.named_buffers()
        if buf is not None
        and (type(buf).__name__ == "NVFP4Tensor" or buf.dtype in exotic)
    ]
    if not ignore:
        return
    existing = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
    model._ddp_params_and_buffers_to_ignore = list(dict.fromkeys(existing + ignore))
    LOG.info("NVFP4: excluded %d FP4 base buffers from DDP sync", len(ignore))


def apply_nvfp4_training(cfg: DictDefault, model: PreTrainedModel) -> None:
    """Swap eligible linears for NVFP4-GEMM linears (Blackwell FP4 compute)."""
    nvfp4 = getattr(cfg, "nvfp4_training", None)
    if not nvfp4:
        return
    nvfp4 = _as_args(nvfp4)
    if not nvfp4.enabled:
        return

    # The FP4 base exposes weight read-only, so an in-process merge would no-op; keep bf16 for merge.
    if cfg.merge_lora:
        return

    from .nvfp4_training import (
        NVFP4Recipe,
        convert_lora_base_to_nvfp4,
        convert_to_nvfp4_training,
    )

    recipe = NVFP4Recipe(
        stochastic_rounding=nvfp4.stochastic_rounding,
        hadamard=nvfp4.hadamard,
    )
    skip_first, skip_last = _resolve_keep_hp_counts(nvfp4, model)
    # lm_head/embeddings are handled by the keyword path, not as body linears.
    body_exclude = ("lm_head", "embed_tokens") + _block_exclusions(
        model, skip_first, skip_last
    )
    # [all] -> include=None (all eligible); fragment list -> only those; keyword-only -> no body swap.
    include = None if nvfp4.quantize_all_body else nvfp4.quantize_body_fragments
    want_body = nvfp4.quantize_all_body or bool(nvfp4.quantize_body_fragments)

    if getattr(nvfp4, "quantize_lm_head", False):
        _check_lm_head_quantizable(cfg, nvfp4, model)

    adapter = cfg.adapter
    if adapter in ("lora", "qlora"):
        base_mode = getattr(nvfp4, "base_mode", None)
        if base_mode is None:
            base_mode = (
                "storage"
                if (bool(nvfp4.quantize_base) or adapter == "qlora")
                else "compute"
            )
        if want_body:
            compute_base = base_mode == "compute"
            quantized_storage = base_mode == "storage"
            if compute_base and cfg.torch_compile:
                from .nvfp4_training import _mslk_available

                if not _mslk_available():
                    LOG.info(
                        "nvfp4_training compute-base under torch_compile is using the "
                        "torchao fallback (MSLK not installed); compile-safe but slower."
                    )
            use_fsdp = (quantized_storage or compute_base) and bool(cfg.fsdp_config)
            if (
                convert_lora_base_to_nvfp4(
                    model,
                    recipe,
                    quantized_storage=quantized_storage,
                    compute_base=compute_base,
                    fsdp=use_fsdp,
                    include=include,
                    exclude=body_exclude,
                )
                == 0
            ):
                LOG.warning(
                    "nvfp4_training: no eligible LoRA base layers matched the selected "
                    "`quantize` (is the model PEFT-wrapped?)"
                )
    else:
        base_mode = getattr(nvfp4, "base_mode", None) or "compute"
        if want_body and (
            convert_to_nvfp4_training(
                model, recipe, include=include, exclude=body_exclude
            )
            == 0
        ):
            LOG.warning(
                "nvfp4_training: no eligible nn.Linear layers matched the selected "
                "`quantize`"
            )

    _apply_tied_or_lm_head(nvfp4, model, recipe, base_mode)

    if getattr(nvfp4, "quantize_vision_tower", False):
        from .nvfp4_training import convert_vision_tower_to_nvfp4

        convert_vision_tower_to_nvfp4(model, recipe, base_mode=base_mode)

    _load_packed_sidecar(cfg, model)

    ce_mode = resolve_ce_mode(nvfp4)
    if ce_mode != "off":
        from .kernels.lm_head_ce import patch_lm_head_cross_entropy

        patch_lm_head_cross_entropy(
            model,
            mode=ce_mode,
            vocab_block=getattr(nvfp4, "fused_ce_vocab_block", None),
        )
