"""
Monkeypatch to add Params4bit and Int8Params support to FSDP2. This enables QLoRA + FSDP2
and 8-bit LoRA + FSDP2, as well as our LoRA / QLoRA Triton kernels to work with FSDP2.

This patch modifies the _init_sharded_param and init_unsharded_param methods in FSDPParam
to handle bitsandbytes Params4bit and Int8Params parameters, preserving their quantization
metadata through the FSDP2 shard/unshard cycle.
"""

import importlib
import inspect

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def apply_init_sharded_param_patch():
    """Apply patch to FSDPParam._init_sharded_param to support Params4bit."""
    if getattr(apply_init_sharded_param_patch, "_axolotl_patched", False):
        return
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

    # Get original source
    original_source = inspect.getsource(FSDPParam._init_sharded_param)
    original_source, _ = detab_code(original_source)

    # Define the replacement
    original_param_creation = """    self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
    self.sharded_param.requires_grad_(param.requires_grad)"""

    patched_param_creation = """    import bitsandbytes as bnb
    if isinstance(param, bnb.nn.modules.Params4bit):
        self.sharded_param = bnb.nn.modules.Params4bit(
            data=sharded_param,
            requires_grad=param.requires_grad,
            quant_state=param.quant_state,
            blocksize=param.blocksize,
            compress_statistics=param.compress_statistics,
            quant_type=param.quant_type,
            quant_storage=param.quant_storage,
            module=param.module,
            bnb_quantized=param.bnb_quantized,
        )
        self.sharded_param = self.to_sharded_dtensor(self.sharded_param)
    elif isinstance(param, bnb.nn.modules.Int8Params):
        self.sharded_param = bnb.nn.modules.Int8Params(
            data=sharded_param,
            requires_grad=param.requires_grad,
            has_fp16_weights=param.has_fp16_weights,
            CB=None,
            SCB=param.SCB,
        )
        self.sharded_param = self.to_sharded_dtensor(self.sharded_param)
    else:
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(sharded_param),
            requires_grad=param.requires_grad,
        )"""

    # Apply the replacement
    if original_param_creation in original_source:
        patched_source = original_source.replace(
            original_param_creation, patched_param_creation
        )
        patched_source = patched_source.replace(
            "def _init_sharded_param(",
            "def patched_init_sharded_param(",
            1,
        )

        # Load necessary imports
        module_name = FSDPParam.__module__
        module = importlib.import_module(module_name)

        items_to_import = []
        for item in dir(module):
            if item in patched_source:
                items_to_import.append(item)

        exec(  # nosec B102
            f"from {module_name} import ({', '.join(items_to_import)})",
            globals(),
        )
        exec(patched_source, globals())  # nosec B102

        # Replace the method
        FSDPParam._init_sharded_param = patched_init_sharded_param
        apply_init_sharded_param_patch._axolotl_patched = True
        LOG.info("Successfully applied FSDP _init_sharded_param patch")
    else:
        LOG.warning("Could not find target code for _init_sharded_param patching")


def apply_init_unsharded_param_patch():
    """Apply patch to FSDPParam.init_unsharded_param to support Params4bit."""
    if getattr(apply_init_unsharded_param_patch, "_axolotl_patched", False):
        return
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

    # Get original source
    original_source = inspect.getsource(FSDPParam.init_unsharded_param)
    original_source, _ = detab_code(original_source)

    # Define the replacement
    original_param_creation = """        self._unsharded_param = nn.Parameter(
            unsharded_param, requires_grad=self.sharded_param.requires_grad
        )"""

    patched_param_creation = """        import bitsandbytes as bnb
        local_tensor = self.sharded_param._local_tensor
        if isinstance(local_tensor, bnb.nn.modules.Params4bit):
            self._unsharded_param = bnb.nn.modules.Params4bit(
                data=unsharded_param,
                requires_grad=self.sharded_param.requires_grad,
                quant_state=local_tensor.quant_state,
                blocksize=local_tensor.blocksize,
                compress_statistics=local_tensor.compress_statistics,
                quant_type=local_tensor.quant_type,
                quant_storage=local_tensor.quant_storage,
                module=local_tensor.module,
                bnb_quantized=local_tensor.bnb_quantized,
            )
        elif isinstance(local_tensor, bnb.nn.modules.Int8Params):
            self._unsharded_param = bnb.nn.modules.Int8Params(
                data=unsharded_param,
                requires_grad=self.sharded_param.requires_grad,
                has_fp16_weights=local_tensor.has_fp16_weights,
                CB=unsharded_param,
                SCB=local_tensor.SCB,
            )
        else:
            self._unsharded_param = nn.Parameter(
                unsharded_param, requires_grad=self.sharded_param.requires_grad
            )"""

    # Apply the replacement
    if original_param_creation in original_source:
        patched_source = original_source.replace(
            original_param_creation, patched_param_creation
        )
        patched_source = patched_source.replace(
            "def init_unsharded_param(",
            "def patched_init_unsharded_param(",
            1,
        )

        # Load necessary imports
        module_name = FSDPParam.__module__
        module = importlib.import_module(module_name)

        items_to_import = []
        for item in dir(module):
            if item in patched_source:
                items_to_import.append(item)

        exec(  # nosec B102
            f"from {module_name} import ({', '.join(items_to_import)})",
            globals(),
        )
        exec(patched_source, globals())  # nosec B102

        # Replace the method
        FSDPParam.init_unsharded_param = patched_init_unsharded_param
        apply_init_unsharded_param_patch._axolotl_patched = True
        LOG.info("Successfully applied FSDP init_unsharded_param patch")
    else:
        LOG.warning("Could not find target code for patching")


def apply_linear8bitlt_save_patch():
    """Patch Linear8bitLt._save_to_state_dict to handle DTensor-wrapped Int8Params.

    After FSDP2 sharding, Linear8bitLt.weight is a DTensor wrapping Int8Params.
    BnB's _save_to_state_dict accesses self.weight.SCB directly, but DTensor
    doesn't proxy custom attribute access to its _local_tensor. This patch
    temporarily unwraps the DTensor during saving so BnB can find the SCB attribute.
    """
    if getattr(apply_linear8bitlt_save_patch, "_axolotl_patched", False):
        return
    import bitsandbytes as bnb
    from torch.distributed.tensor import DTensor

    original_save = bnb.nn.Linear8bitLt._save_to_state_dict

    def _patched_save_to_state_dict(self, destination, prefix, keep_vars):
        # Use _parameters dict directly to bypass nn.Module.__setattr__ type check.
        weight = self._parameters["weight"]
        unwrapped = False
        if isinstance(weight, DTensor) and hasattr(weight, "_local_tensor"):
            self._parameters["weight"] = weight._local_tensor
            unwrapped = True
        try:
            original_save(self, destination, prefix, keep_vars)
        finally:
            if unwrapped:
                self._parameters["weight"] = weight

    bnb.nn.Linear8bitLt._save_to_state_dict = _patched_save_to_state_dict
    apply_linear8bitlt_save_patch._axolotl_patched = True
    LOG.info("Patched Linear8bitLt._save_to_state_dict for DTensor compatibility")


def apply_init_dtype_attrs_patch():
    """Prevent FSDP2 mixed precision from casting non-float quantized params.

    When mixed precision is enabled (e.g., bf16), FSDP2's init_dtype_attrs sets
    param_dtype=bf16 for ALL params. During all-gather, _to_dtype_if_needed casts
    the sharded param to param_dtype. For non-float params (uint8 packed 4-bit,
    int8 quantized) without FSDP2 extensions, this destroys the quantized data.

    Params4bit handles this via fsdp_pre/post_all_gather extensions, but our
    parametrize-based expert quantization uses plain nn.Parameter(uint8/int8)
    without extensions.
    """
    if getattr(apply_init_dtype_attrs_patch, "_axolotl_patched", False):
        return
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

    original_init_dtype_attrs = FSDPParam.init_dtype_attrs

    def patched_init_dtype_attrs(self, mp_policy):
        original_init_dtype_attrs(self, mp_policy)
        # Skip casting non-float quantized params (uint8/int8) without FSDP2
        # extensions — the parametrization chain handles dequantization.
        if self.param_dtype is not None and not self.sharded_param.is_floating_point():
            local = self.sharded_param
            if hasattr(local, "_local_tensor"):
                local = local._local_tensor
            if not hasattr(local, "fsdp_pre_all_gather"):
                self.param_dtype = None

    FSDPParam.init_dtype_attrs = patched_init_dtype_attrs
    apply_init_dtype_attrs_patch._axolotl_patched = True
    LOG.info("Patched FSDPParam.init_dtype_attrs for non-float quantized params")
