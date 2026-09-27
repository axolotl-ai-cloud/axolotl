"""Config-selected mixer replacements retaining Transformers parameter ownership."""

import os

from transformers.integrations.accelerate import force_accelerate_hooks
from transformers.models.mamba.modeling_mamba import MambaMixer as NativeMambaMixer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer as NativeMamba2Mixer


class _MambaBackend:
    def __init__(self, config, layer_idx, initialize_mixer_weights=True):
        backend = getattr(config, "mamba_backend", "transformers")
        if backend not in ("transformers", "fla"):
            raise ValueError("model_config.mamba_backend must be transformers or fla")
        super().__init__(config, layer_idx, initialize_mixer_weights)
        if backend == "fla":
            if os.environ.get("FLA_CONV_BACKEND", "cuda") != "cuda":
                raise ValueError("FLA Mamba integration requires FLA_CONV_BACKEND=cuda")
            from .adapters import enable_lora_projections
            from .fla_mixer import FlaMamba2MixerView, FlaMambaMixerView

            view = (
                FlaMambaMixerView
                if config.model_type == "mamba"
                else FlaMamba2MixerView
            )
            self.fla_mixer = view(self, config)
            enable_lora_projections()

    def forward(
        self,
        hidden_states,
        cache_params=None,
        attention_mask=None,
        segments=None,
        **kwargs,
    ):
        if hasattr(self, "fla_mixer"):
            return self._fla_forward(
                hidden_states,
                cache_params=cache_params,
                use_cache=cache_params is not None,
                attention_mask=attention_mask,
                _axolotl_segments=segments,
                **kwargs,
            )
        if segments is not None:
            if getattr(super().forward, "_axolotl_seq_idx_patch", False):
                kwargs["segments"] = segments
            elif (segments.seq_idx[:, 1:] != segments.seq_idx[:, :-1]).any():
                raise ValueError(
                    "Native Mamba packed inputs require the packing patches; "
                    "enable sample_packing or batch_flattening."
                )
        return super().forward(
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            **kwargs,
        )

    def _fla_forward(self, *args, **kwargs):
        return self.fla_mixer(*args, **kwargs)[0]


class MambaMixer(_MambaBackend, NativeMambaMixer):
    _fla_forward = force_accelerate_hooks(["conv1d", "x_proj", "dt_proj", "out_proj"])(
        _MambaBackend._fla_forward
    )


class Mamba2Mixer(_MambaBackend, NativeMamba2Mixer):
    _fla_forward = force_accelerate_hooks(["conv1d", "norm", "out_proj"])(
        _MambaBackend._fla_forward
    )
