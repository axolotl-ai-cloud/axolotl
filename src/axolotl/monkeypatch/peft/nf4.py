"""Materialize parametrized NF4 bases before PEFT mutates weights during merge."""

from functools import wraps

from torch import nn
from torch.nn.utils import parametrize

from axolotl.utils.nf4 import BnbNF4Parametrization, TorchaoNF4Parametrization


def _materialize(module, name):
    if not parametrize.is_parametrized(module, name):
        return
    chain = module.parametrizations[name]
    if len(chain) == 1 and isinstance(
        chain[0], (BnbNF4Parametrization, TorchaoNF4Parametrization)
    ):
        weight = getattr(module, name).detach()
        parametrize.remove_parametrizations(module, name, leave_parametrized=False)
        setattr(module, name, nn.Parameter(weight, requires_grad=False))
        if name == "weight" and hasattr(module, "_nf4_original_forward"):
            module.forward = module._nf4_original_forward
            del module._nf4_original_forward


def patch_nf4_merge():
    import peft.utils.save_and_load as save_and_load
    from peft.tuners.lora.layer import Linear, ParamWrapper
    from peft.tuners.lora.model import LoraModel

    original_tp = save_and_load._maybe_shard_state_dict_for_tp
    if not getattr(original_tp, "_axolotl_nf4", False):

        @wraps(original_tp)
        def shard_for_tp(model, *args, **kwargs):
            if getattr(model, "_axolotl_staged_nf4", False):
                return
            return original_tp(model, *args, **kwargs)

        shard_for_tp._axolotl_nf4 = True
        save_and_load._maybe_shard_state_dict_for_tp = shard_for_tp

    def wrap(original):
        @wraps(original)
        def merge(self, *args, **kwargs):
            base = self.get_base_layer()
            name = self.parameter_name if isinstance(self, ParamWrapper) else "weight"
            _materialize(base, name)
            return original(self, *args, **kwargs)

        merge._axolotl_nf4 = True
        return merge

    for cls in (Linear, ParamWrapper):
        if not getattr(cls.merge, "_axolotl_nf4", False):
            cls.merge = wrap(cls.merge)

    if not getattr(LoraModel.merge_and_unload, "_axolotl_nf4", False):
        original_unload = LoraModel.merge_and_unload

        @wraps(original_unload)
        def merge_and_unload(self, *args, **kwargs):
            model = original_unload(self, *args, **kwargs)
            for module in list(model.modules()):
                for name in list(getattr(module, "parametrizations", {})):
                    _materialize(module, name)
            return model

        merge_and_unload._axolotl_nf4 = True
        LoraModel.merge_and_unload = merge_and_unload
