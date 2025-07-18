"""Monkey patch for Accelerate to add support for ND parallelism."""

import inspect

import accelerate

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ORIGINAL_PREPARE_DATALOADER_CODE = """
            submesh_fsdp_size = 1
            submesh_dp_size = 1
            submesh_tp_size = 1
            if "tp" in torch_device_mesh.mesh_dim_names:
                submesh_tp_size = torch_device_mesh["tp"].size()
            if "dp" in torch_device_mesh.mesh_dim_names:
                submesh_dp_size = torch_device_mesh["dp"].size()
            if "fsdp" in torch_device_mesh.mesh_dim_names:
                submesh_fsdp_size = torch_device_mesh["fsdp"].size()
            process_index = process_index // submesh_tp_size""".lstrip(
    "\n"
)

NEW_PREPARE_DATALOADER_CODE = """
        submesh_dp_fsdp_size = 1
        submesh_fsdp_size = 1
        submesh_dp_size = 1
        submesh_tp_size = 1
        submesh_cp_size = 1
        if "tp" in torch_device_mesh.mesh_dim_names:
            submesh_tp_size = torch_device_mesh["tp"].size()
        if "cp" in torch_device_mesh.mesh_dim_names:
            submesh_cp_size = torch_device_mesh["cp"].size()
        if "dp" in torch_device_mesh.mesh_dim_names:
            submesh_dp_size = torch_device_mesh["dp"].size()
        if "fsdp" in torch_device_mesh.mesh_dim_names:
            submesh_fsdp_size = torch_device_mesh["fsdp"].size()
        try:
            submesh_dp_fsdp_size = torch_device_mesh["dp_fsdp"].size()
        except KeyError:
            pass
        process_index = process_index // (submesh_tp_size * submesh_cp_size)
        num_processes = submesh_dp_fsdp_size if submesh_dp_fsdp_size > 1 else submesh_fsdp_size * submesh_dp_size
""".strip(
    "\n"
)


def patch_prepare_data_loader():
    """Patch `accelerate.data_loader.prepare_data_loader` to respect the SP degree.

    Raises:
        RuntimeError: If source code to patch does not exist.
    """
    original_fn = accelerate.data_loader.prepare_data_loader
    original_source = inspect.getsource(original_fn)

    if ORIGINAL_PREPARE_DATALOADER_CODE not in original_source:
        raise RuntimeError(
            "SP patch failed - target snippet not found. "
            "Check accelerate's version or update the patch."
        )

    patched_source = original_source.replace(
        ORIGINAL_PREPARE_DATALOADER_CODE, NEW_PREPARE_DATALOADER_CODE
    )

    items_to_import = []
    for item in dir(accelerate.data_loader):
        if item in patched_source:
            items_to_import.append(item)

    # Create a new function from the patched source
    namespace = {}
    exec(  # pylint: disable=exec-used  # nosec B102
        f"from accelerate.data_loader import ({', '.join(items_to_import)})",
        globals(),
    )
    exec(  # pylint: disable=exec-used  # nosec B102
        patched_source, globals(), namespace
    )

    patched_function = namespace["prepare_data_loader"]
    original_fn.__code__ = patched_function.__code__

    LOG.info("Patched accelerate.data_loader.prepare_data_loader for SP support")
