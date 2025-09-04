# pylint: skip-file
import os
from collections import namedtuple
from functools import partial
from typing import Optional, Union

import torch
from mamba_ssm.models.mixer_seq_simple import MixerModel, _init_weights
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from torch import nn
from torch.nn import CrossEntropyLoss

from axolotl.models.mamba.configuration_mamba import MambaConfig


class MambaLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        initializer_cfg=None,
        pad_vocab_size_multiple: int = 1,
        device=None,
        dtype=None,
        **backbone_kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.config = MambaConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            pad_vocab_size_multiple=pad_vocab_size_multiple,
        )
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            initializer_cfg=initializer_cfg,
            **backbone_kwargs,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        labels=None,
        **kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

        loss = None
        if labels is not None:
            logits = lm_logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "loss"])
            print(loss)
            return CausalLMOutput(logits=lm_logits, loss=loss)

        else:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=lm_logits)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        state_dict: Optional[dict] = None,
        safe_serialization: Optional[bool] = None,  # pylint: disable=unused-argument
    ):
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config = load_config_hf(pretrained_model_name)
        model = cls(**config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(
            load_state_dict_hf(pretrained_model_name, device={"": device}, dtype=dtype)
        )
        return model
