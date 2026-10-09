"""Tie LoRA adapters on tied output embeddings to the input-embedding adapter."""

import torch.nn.functional as F
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class TiedTransposedLinear(nn.Module):
    """Linear whose weight is the live transpose of a parameter owned elsewhere."""

    def __init__(self, source: nn.ParameterDict, adapter_name: str):
        super().__init__()
        # plain attribute: registering the source here would hand the optimizer a second copy
        object.__setattr__(self, "_source", source)
        self.adapter_name = adapter_name
        self.bias = None
        self.register_state_dict_post_hook(_emit_tied_weight)
        self.register_load_state_dict_pre_hook(_drop_tied_weight)

    @property
    def weight(self):
        return self._source[self.adapter_name].t()

    @property
    def in_features(self) -> int:
        return self._source[self.adapter_name].shape[0]

    @property
    def out_features(self) -> int:
        return self._source[self.adapter_name].shape[1]

    def forward(self, x):
        return F.linear(x, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, tied=True"


def _emit_tied_weight(module, state_dict, prefix, local_metadata):
    # saved adapters keep PEFT's layout so loaders without this tie still get the head weights
    state_dict[prefix + "weight"] = module.weight.detach()


def _drop_tied_weight(module, state_dict, prefix, *args):
    state_dict.pop(prefix + "weight", None)


def tie_lora_output_embeddings(model) -> list[str]:
    """Point LoRA adapters on tied output embeddings at the input-embedding adapter.

    PEFT's ``ensure_weight_tying`` builds the head adapter as new ``Parameter``s over
    views of the embedding adapter. Adapter autocast on a half-precision model replaces
    each one's storage separately, and even when storage survives the optimizer keeps two
    states for one buffer. Replacing the head's ``lora_A``/``lora_B`` with parameterless
    views gives one set of weights whose gradient sums both uses.
    """
    peft_configs = getattr(model, "peft_config", None) or {}
    tied: list[str] = []
    for adapter_name, peft_config in peft_configs.items():
        targets = getattr(peft_config, "target_modules_to_tie", None) or []
        if not getattr(peft_config, "ensure_weight_tying", False) or not targets:
            continue
        embeddings = model.get_input_embeddings()
        emb_A = getattr(embeddings, "lora_embedding_A", {})
        emb_B = getattr(embeddings, "lora_embedding_B", {})
        if adapter_name not in emb_A or adapter_name not in emb_B:
            LOG.warning(
                "ensure_weight_tying requested for %s but the input embeddings carry no "
                "LoRA adapter %r; leaving the output adapter untied",
                targets,
                adapter_name,
            )
            continue
        for name, module in model.named_modules():
            if not any(name == t or name.endswith(f".{t}") for t in targets):
                continue
            lora_A = getattr(module, "lora_A", None)
            if lora_A is None or adapter_name not in lora_A:
                continue
            module.lora_A[adapter_name] = TiedTransposedLinear(emb_B, adapter_name)
            module.lora_B[adapter_name] = TiedTransposedLinear(emb_A, adapter_name)
            tied.append(name)
    if tied:
        LOG.info("Tied LoRA adapters on %s to the input-embedding adapter", tied)
    return tied
