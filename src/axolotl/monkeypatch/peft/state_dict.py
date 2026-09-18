"""Keep PEFT's adapter state-dict selection aligned with activation-checkpoint wrappers."""

from functools import wraps

_WRAPPER = "_checkpoint_wrapped_module"


def patch_peft_checkpoint_wrapper_prefixes() -> None:
    """Strip the checkpoint-wrapper segment from PEFT 0.21's structural key prefixes.

    Torch's checkpoint wrapper removes ``_checkpoint_wrapped_module.`` from state-dict
    keys but not from module names, and PEFT 0.21 derives adapter key prefixes from
    module names, so an activation-checkpointed FSDP model saves and loads an empty
    adapter. Both forms are kept: the state-dict filters need the stripped prefix and
    ``_mark_only_adapters_as_trainable`` filters ``named_parameters``, which keeps
    the segment. Releases without the structural prefix builder are unaffected.
    """
    from peft.tuners import tuners_utils

    original = getattr(tuners_utils, "_get_tuner_state_dict_key_prefixes", None)
    if original is None or getattr(original, "_axolotl_patched", False):
        return

    @wraps(original)
    def prefixes(model, adapter_name=None):
        found = original(model, adapter_name)
        return found | {
            prefix.replace(f".{_WRAPPER}", "").replace(f"{_WRAPPER}.", "")
            for prefix in found
        }

    prefixes._axolotl_patched = True
    tuners_utils._get_tuner_state_dict_key_prefixes = prefixes
