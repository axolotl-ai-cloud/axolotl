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
    from torch.distributed.tensor import DTensor

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
            weights = (
                emb_A[adapter_name],
                emb_B[adapter_name],
                module.lora_A[adapter_name].weight,
                module.lora_B[adapter_name].weight,
            )
            if any(isinstance(weight, DTensor) for weight in weights) or any(
                getattr(layer.get_base_layer(), "_hf_tp_plan", None)
                for layer in (embeddings, module)
            ):
                raise ValueError(
                    "Tied embedding LoRA requires embed_tokens and lm_head outside "
                    "the tensor-parallel plan. TP on other layers is supported; "
                    "TP-sharded embedding/head adapters cannot be replaced safely."
                )
            module.lora_A[adapter_name] = TiedTransposedLinear(emb_B, adapter_name)
            module.lora_B[adapter_name] = TiedTransposedLinear(emb_A, adapter_name)
            tied.append(name)
    if tied:
        LOG.info("Tied LoRA adapters on %s to the input-embedding adapter", tied)
    return tied


def tied_lora_no_wrap_modules(model) -> set[nn.Module]:
    """Keep tied adapter owners and consumers in the root FSDP group."""
    modules = list(model.modules())
    tied = [module for module in modules if isinstance(module, TiedTransposedLinear)]
    if not tied:
        return set()
    sources = {module._source for module in tied}
    protected = set(tied)
    for module in modules:
        if any(child in sources for child in module.children()):
            protected.update(module.modules())
    # Base embedding weights may also be shared with the head.
    owners: dict[int, list[nn.Module]] = {}
    for module in modules:
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), []).append(module)
    for shared in owners.values():
        if len(shared) > 1:
            protected.update(shared)
    for module in reversed(modules):
        if any(child in protected for child in module.children()):
            protected.add(module)
    return protected


def add_tied_lora_state_dict_weights(
    model, state_dict, name_transform=lambda name: name
):
    """Rebuild head aliases from gathered embedding tensors without extra collectives."""
    sources = {id(module): name for name, module in model.named_modules()}
    for name, module in model.named_modules():
        if not isinstance(module, TiedTransposedLinear):
            continue
        source_name = sources[id(module._source)]
        source_key = name_transform(f"{source_name}.{module.adapter_name}")
        state_dict[name_transform(f"{name}.weight")] = state_dict[source_key].t()
