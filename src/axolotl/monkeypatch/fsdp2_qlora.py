"""
Monkeypatch to add Params4bit support to FSDP2. This enables QLoRA + FSDP2, as well as
our LoRA / QLoRA Triton kernels to work with FSDP2.

This patch modifies the _init_sharded_param method in FSDPParam to handle bitsandbytes
Params4bit parameters.
"""

import importlib
import inspect

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def apply_init_sharded_param_patch():
    """Apply patch to FSDPParam._init_sharded_param to support Params4bit."""
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
    else:
        self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
        self.sharded_param.requires_grad_(param.requires_grad)"""

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
        LOG.info("Successfully applied FSDP _init_sharded_param patch")
    else:
        LOG.warning("Could not find target code for _init_sharded_param patching")


def apply_init_unsharded_param_patch():
    """Apply patch to FSDPParam.init_unsharded_param to support Params4bit."""
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
        LOG.info("Successfully applied FSDP init_unsharded_param patch")
    else:
        LOG.warning("Could not find target code for patching")
