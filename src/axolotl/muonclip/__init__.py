"""Helpers for MuonClip integration."""

from .attention import (
    auto_register_llama_attention,
    register_attention_module,
    record_attention_logits,
)
from .controller import MuonClipController, MuonClipContext
from .hooks import (
    ensure_llama_attention_instrumentation,
    ensure_qwen_attention_instrumentation,
)
from .math import muon_orthogonal_update, newton_schulz_orthogonalize
from .parameters import (
    MuonParameterInfo,
    ParameterTagSummary,
    tag_parameters_for_muon,
)
from .state import MuonStateStore

__all__ = [
    "tag_parameters_for_muon",
    "MuonParameterInfo",
    "ParameterTagSummary",
    "MuonStateStore",
    "MuonClipController",
    "MuonClipContext",
    "register_attention_module",
    "record_attention_logits",
    "auto_register_llama_attention",
    "ensure_llama_attention_instrumentation",
    "ensure_qwen_attention_instrumentation",
    "muon_orthogonal_update",
    "newton_schulz_orthogonalize",
]
