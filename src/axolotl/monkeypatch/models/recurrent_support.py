"""Which recurrent / linear-attention architectures can train on packed sequences.

A packed row concatenates documents, and attention isolates them through the
mask or ``cu_seqlens``. Recurrent mixers (Mamba, GatedDeltaNet, KDA, short
convolutions) carry state along the row instead, so each one needs a boundary
signal (``seq_idx`` or ``cu_seqlens``) threaded into its kernel, and each rank
under context parallelism needs the previous rank's final state. Neither
failure raises: the loss just quietly trains on cross-document state.
"""

# model types with a sample-packing boundary patch (or native handling)
PACKING_PATCHED = frozenset(
    {
        "bailing_hybrid",
        "bamba",
        "falcon_h1",
        "granitemoehybrid",
        "kimi_linear",
        "lfm2",
        "lfm2_moe",
        "nemotron_h",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_5_text",
        "qwen3_next",
        "qwen4_exp",
        "qwen4_exp_text",
    }
)

# recurrent model types whose mixers never see a document boundary
PACKING_UNSUPPORTED = frozenset(
    {
        "falcon_mamba",
        "inkling",
        "jamba",
        "mamba",
        "mamba2",
        "minimax",
        "olmo_hybrid",
        "zamba",
        "zamba2",
    }
)

# recurrent model types whose SSM state is exchanged across context-parallel ranks
CONTEXT_PARALLEL_SUPPORTED = frozenset({"falcon_h1", "granitemoehybrid", "nemotron_h"})

RECURRENT_MODEL_TYPES = PACKING_PATCHED | PACKING_UNSUPPORTED


def validate_recurrent_model_config(cfg) -> None:
    """Raise when a recurrent architecture would silently train on leaked state."""
    model_type = cfg.model_config_type or ""
    if model_type not in RECURRENT_MODEL_TYPES:
        return

    if (cfg.sample_packing or cfg.batch_flattening) and (
        model_type in PACKING_UNSUPPORTED
    ):
        raise ValueError(
            f"sample_packing / batch_flattening is not supported for model_type="
            f"{model_type}: its recurrent layers have no packed-sequence boundary "
            "handling, so state would leak across packed samples. Set "
            "`sample_packing: false` and `batch_flattening: false`."
        )

    if (cfg.context_parallel_size or 1) > 1 and (
        model_type not in CONTEXT_PARALLEL_SUPPORTED
    ):
        raise ValueError(
            f"context_parallel_size > 1 is not supported for model_type={model_type}: "
            "its recurrent layers carry state that is not exchanged across "
            "context-parallel ranks, so every rank would restart from zero. "
            "Set `context_parallel_size: 1`."
        )
