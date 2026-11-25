"""Keep PEFT's adapter state-dict selection aligned with activation-checkpoint wrappers."""

from functools import wraps
from types import SimpleNamespace

_WRAPPER = "_checkpoint_wrapped_module"


def _strip_wrapper(name: str) -> str:
    return name.replace(f".{_WRAPPER}", "").replace(f"{_WRAPPER}.", "")


def patch_peft_checkpoint_wrapper_prefixes() -> None:
    """Strip the checkpoint-wrapper segment where PEFT matches module names to state-dict keys.

    Torch's checkpoint wrapper removes ``_checkpoint_wrapped_module.`` from state-dict
    keys but not from module names. Unpatched, PEFT's key prefixes select an empty
    adapter, and PEFT 0.22's DTensor reshard-on-load misses every key. Both prefix forms
    are kept because ``_mark_only_adapters_as_trainable`` filters ``named_parameters``.
    """
    from peft.tuners import tuners_utils
    from peft.utils import save_and_load

    original = getattr(tuners_utils, "_get_tuner_state_dict_key_prefixes", None)
    if original is not None and not getattr(original, "_axolotl_patched", False):

        @wraps(original)
        def prefixes(model, adapter_name=None):
            found = original(model, adapter_name)
            return found | {_strip_wrapper(prefix) for prefix in found}

        prefixes._axolotl_patched = True
        tuners_utils._get_tuner_state_dict_key_prefixes = prefixes

    reshard = getattr(save_and_load, "_reshard_dtensor_values_for_load", None)
    if reshard is not None and not getattr(reshard, "_axolotl_patched", False):

        @wraps(reshard)
        def reshard_unwrapped(model, state_dict):
            named = [(_strip_wrapper(n), p) for n, p in model.named_parameters()]
            # upstream only calls model.named_parameters()
            return reshard(SimpleNamespace(named_parameters=lambda: named), state_dict)

        reshard_unwrapped._axolotl_patched = True
        save_and_load._reshard_dtensor_values_for_load = reshard_unwrapped
