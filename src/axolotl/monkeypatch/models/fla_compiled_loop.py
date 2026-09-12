"""Shared helpers for the compile-friendly FLA GatedDeltaNet decoder loop (Qwen3.5 / Qwen3-Next)."""


def init_fla_compiled_ops(enabled: bool = True) -> bool:
    from axolotl.monkeypatch.models import gated_delta_net_ops as fla_ops

    return fla_ops.fla_ops_available() if enabled else False


def _call_self_attn(attn_module, **kwargs):
    return attn_module(**kwargs)


try:
    import torch._dynamo as _dynamo

    call_self_attn_disabled = _dynamo.disable(_call_self_attn)
except Exception:  # pragma: no cover
    call_self_attn_disabled = _call_self_attn
