"""Make PEFT's LoRA merge -> unmerge restore floating-point base weights exactly.

PEFT merges in place with ``W += delta`` and unmerges with ``W -= delta``. In BF16/FP16
that does not round-trip: ``bf16(bf16(W + D) - bf16(D)) != W`` for some entries,
mostly where ``|W|`` is small next to ``|D|`` or ``W`` sits on a power of two. TRL's
vLLM weight sync merges and unmerges the adapter at every sync, so the frozen base
weights the trainer uses drift further with each sync. Each merge records the entries
the matching unmerge will get wrong, and the unmerge writes their original values back.
"""

from functools import wraps

import torch


def _exact_unmerge_supported(module, adapters) -> bool:
    weight = module.get_base_layer().weight
    return (
        type(weight) in (torch.Tensor, torch.nn.Parameter)
        and weight.is_floating_point()
        and weight.is_contiguous()
        and all(a not in module.lora_variant for a in adapters)
        and not any(module.lora_bias[a] for a in adapters)
    )


def _merge_groups(module) -> list:
    groups = module.__dict__.get("_axolotl_merge_groups")
    if groups is None:
        groups = module._axolotl_merge_groups = []
    return groups


def patch_peft_lora_exact_unmerge():
    from peft.tuners.lora.layer import Linear

    if getattr(Linear.unmerge, "_axolotl_exact_unmerge", False):
        return
    original_merge = Linear.merge
    original_unmerge = Linear.unmerge

    @wraps(original_merge)
    def merge(self, safe_merge=False, adapter_names=None):
        names = self.active_adapters if adapter_names is None else adapter_names
        if isinstance(names, str):
            return original_merge(self, safe_merge, adapter_names)
        pending = [
            a for a in names if a in self.lora_A and a not in self.merged_adapters
        ]
        before = None
        if pending and _exact_unmerge_supported(self, pending):
            before = self.get_base_layer().weight.detach().clone()
        merged_count = len(self.merged_adapters)
        original_merge(self, safe_merge, adapter_names)
        group = self.merged_adapters[merged_count:]
        if not group:
            return
        fix = None
        if before is not None:
            # Replays PEFT's unmerge arithmetic on a copy to find what it would not restore.
            restored = self.get_base_layer().weight.detach().clone()
            for adapter in reversed(group):
                restored -= self.get_delta_weight(adapter).to(restored.dtype)
            wrong = torch.nonzero((restored != before).view(-1)).squeeze(1)
            fix = (wrong, before.view(-1)[wrong])
        _merge_groups(self).append((len(group), fix))

    @wraps(original_unmerge)
    def unmerge(self):
        groups = _merge_groups(self)
        if not self.merged or sum(count for count, _ in groups) != len(
            self.merged_adapters
        ):
            # nothing merged (PEFT warns), or merged_adapters changed outside merge()
            # so the recorded fixes no longer line up
            groups.clear()
            return original_unmerge(self)
        while groups:
            count, fix = groups.pop()
            remaining = self.merged_adapters[:-count]
            self.merged_adapters = self.merged_adapters[-count:]
            original_unmerge(self)
            self.merged_adapters = remaining
            if fix is not None:
                wrong, values = fix
                self.get_base_layer().weight.data.view(-1)[wrong] = values

    merge._axolotl_exact_unmerge = True
    unmerge._axolotl_exact_unmerge = True
    Linear.merge = merge
    Linear.unmerge = unmerge
