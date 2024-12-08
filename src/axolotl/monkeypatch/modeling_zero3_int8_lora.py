"""
fix for zero3 8-bit lora
see https://github.com/huggingface/transformers/pull/32943/files
"""
import inspect
import logging

from transformers import modeling_utils

LOG = logging.getLogger("axolotl.monkeypatch.modeling_zero3_int8_lora")

ORIGINAL_LOAD_CODE = """
            if is_fsdp_enabled() or is_deepspeed_zero3_enabled():
                module, tensor_name = get_module_from_name(model, param_name)
                value = getattr(module, tensor_name)
                param_to = "cpu"
                if is_fsdp_enabled() and not is_local_dist_rank_0():
                    param_to = "meta"
                value = type(value)(value.data.to(param_to), **value.__dict__)
                setattr(module, tensor_name, value)
"""

PATCHED_LOAD_CODE = """
            if is_fsdp_enabled() or is_deepspeed_zero3_enabled():
                module, tensor_name = get_module_from_name(model, param_name)
                value = getattr(module, tensor_name)
                param_to = "cpu"
                if is_fsdp_enabled() and not is_local_dist_rank_0():
                    param_to = "meta"
                val_kwargs = {}
                if hasattr(module, "weight") and module.weight.__class__.__name__ == "Int8Params":
                    val_kwargs["requires_grad"] = False
                value = type(value)(value.data.to(param_to), **val_kwargs, **value.__dict__)
                setattr(module, tensor_name, value)
"""


def get_modeling_state_dict_code() -> str:
    load_code = inspect.getsource(
        modeling_utils._load_state_dict_into_meta_model  # pylint: disable=protected-access
    )
    return load_code


def check_modeling_state_dict_code_is_patchable() -> bool:
    load_code = get_modeling_state_dict_code()
    return ORIGINAL_LOAD_CODE in load_code


def patch_modeling_state_dict_code():
    """
    monkeypatch for fixing the meta model loader for zero3 8-bit lora
    """

    load_code = get_modeling_state_dict_code()
    modeling_utils._original_load_state_dict_into_meta_model = (  # pylint: disable=protected-access
        load_code
    )
    assert (
        ORIGINAL_LOAD_CODE in load_code
    ), "Original _load_state_dict_into_meta_model code not found"

    load_code = load_code.replace(ORIGINAL_LOAD_CODE, PATCHED_LOAD_CODE)
    load_code = load_code.replace(
        "def _load_state_dict_into_meta_model(",
        "def _fixed_load_state_dict_into_meta_model(",
        1,
    )

    items_to_import = []
    for item in dir(modeling_utils):
        if item in load_code:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.modeling_utils import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(load_code, globals())  # pylint: disable=exec-used  # nosec B102
    LOG.info("patching _load_state_dict_into_meta_model")
    modeling_utils._load_state_dict_into_meta_model = _fixed_load_state_dict_into_meta_model  # pylint: disable=protected-access,undefined-variable  # noqa: F821
