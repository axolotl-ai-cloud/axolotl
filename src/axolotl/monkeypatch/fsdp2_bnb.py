"""
Monkeypatch to add Params4bit support to FSDP2. This enables QLoRA + FSDP2, as well
as our LoRA / QLoRA Triton kernels to work with FSDP2.

This patch modifies the _init_sharded_param method in FSDPParam to properly handle
bitsandbytes Params4bit parameters.
"""

from typing import Callable, cast

import torch
from torch import nn


# pylint: disable=protected-access
def patched_init_sharded_param(
    self,
    param: nn.Parameter,
    device: torch.device,
    shard_placement_fn: Callable | None,
):
    """
    Patched version of FSDPParam._init_sharded_param that supports Params4bit.
    """
    if param.device != device and param.device.type != "meta":
        raise AssertionError(
            f"Expects the parameter to already be moved to device {device} but got {param.device}"
        )
    if not param.is_contiguous():
        raise NotImplementedError(
            f"FSDP does not support non-contiguous parameters yet: {param.shape=} {param.stride()=}"
        )

    import bitsandbytes as bnb
    from torch.distributed.fsdp._fully_shard._fsdp_param import (
        HSDPMeshInfo,
        ShardedState,
        _chunk_with_empty,
        _get_dim_chunked_size,
        make_contiguous_strides_for,
    )
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
    from torch.distributed.tensor.device_mesh import _mesh_resources
    from torch.distributed.tensor.placement_types import _StridedShard

    fsdp_placement = shard_placement_fn(param) if shard_placement_fn else None
    if fsdp_placement is None:
        fsdp_placement = Shard(0)
    elif fsdp_placement.dim < 0:
        fsdp_placement = Shard(fsdp_placement.dim + param.ndim)
    assert isinstance(fsdp_placement, Shard), f"{fsdp_placement}"
    self.fsdp_placement = fsdp_placement
    shard_dim = fsdp_placement.dim

    self.is_dtensor = isinstance(param, DTensor)
    if self.is_dtensor:
        self._tp_spec = cast(DTensor, param)._spec
        dp_mesh, tp_mesh = (self.mesh_info.mesh, self._tp_spec.mesh)
        dp_global_mesh = _mesh_resources.get_root_mesh(dp_mesh)
        tp_global_mesh = _mesh_resources.get_root_mesh(tp_mesh)
        if dp_global_mesh != tp_global_mesh or (
            dp_global_mesh is None or tp_global_mesh is None
        ):
            raise AssertionError(
                "FSDP requires the DP and TP mesh to have the same parent mesh but got: \n"
                f"DP's global mesh: {dp_global_mesh}\nTP's global mesh: {tp_global_mesh}"
            )
        name_dims_error = "FSDP requires named DeviceMesh dims for ND parallelism"
        assert dp_mesh.mesh_dim_names is not None, name_dims_error
        assert tp_mesh.mesh_dim_names is not None, name_dims_error
        submesh_names = dp_mesh.mesh_dim_names + tp_mesh.mesh_dim_names
        self._spmd_mesh = dp_global_mesh[submesh_names]
        if len(self._tp_spec.placements) != 1:
            raise NotImplementedError(
                f"FSDP only supports 1D TP, not {self._tp_spec.placements}"
            )
        split_factor = self._tp_spec.num_shards_map[shard_dim]
        assert (
            2 <= self._spmd_mesh.ndim <= 3
        ), f"_spmd_mesh.ndim can only be 2 or 3 but got {self._spmd_mesh.ndim}."
        self._spmd_placements: tuple
        dp_shard_tp_placement = (
            (
                _StridedShard(shard_dim, split_factor=split_factor)
                if split_factor > 1
                else fsdp_placement
            ),
            self._tp_spec.placements[0],
        )
        if self._spmd_mesh.ndim == 2:
            self._spmd_placements = dp_shard_tp_placement
        else:
            assert self.mesh_info.replicate_mesh_dim == 0
            self._spmd_placements = (Replicate(),) + dp_shard_tp_placement
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=self._tp_spec.tensor_meta,
        )
        if split_factor > 1:
            num_shards = self._sharding_spec.num_shards_map[0]
            tensor_size_dim_0 = self._sharding_spec.shape[0]
            if tensor_size_dim_0 % num_shards != 0:
                raise NotImplementedError(
                    "FSDP+TP sharding does not support uneven sharding for now: "
                    f"tensor dim 0 has size {tensor_size_dim_0} which cannot be "
                    f"evenly sharded into {num_shards} shards."
                )
        param_data = cast(DTensor, param)._local_tensor
    else:
        self._spmd_mesh = self.mesh_info.mesh
        if isinstance(self.mesh_info, HSDPMeshInfo):
            self._spmd_placements = (Replicate(), fsdp_placement)
        else:
            self._spmd_placements = (fsdp_placement,)
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
        )
        param_data = param

    assert param_data.is_contiguous(), f"{param_data.shape=} {param_data.stride()=}"
    shard_dim = fsdp_placement.dim
    if shard_dim >= param_data.ndim:
        raise AssertionError(
            f"Shard dim {shard_dim} is invalid for {param_data.ndim}D tensor: {param.shape}"
        )

    self._orig_size = param_data.size()
    self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)
    shard_rank = self.mesh_info.shard_mesh_rank
    shard_world_size = self.mesh_info.shard_mesh_size
    if shard_dim > 0 and param_data.size(shard_dim) % shard_world_size != 0:
        raise NotImplementedError(
            f"FSDP does not support uneven sharding on dim {shard_dim}: "
            f"{param_data.size()} (world size: {shard_world_size})"
        )

    chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
    sharded_param = chunks[shard_rank]
    self.sharded_size = _get_dim_chunked_size(
        sharded_param, param_data.size(), dim=shard_dim
    )
    self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
    padded_sharded_size = chunks[0].size()
    self.padded_sharded_param_size = padded_sharded_size

    padded_sharded_param = param_data.new_zeros(padded_sharded_size)
    if sharded_param.numel() > 0:
        padded_sharded_param.narrow(
            dim=shard_dim, start=0, length=sharded_param.size(shard_dim)
        ).copy_(sharded_param)
    if self.offload_to_cpu and not padded_sharded_param.is_meta:
        padded_sharded_param = padded_sharded_param.cpu()
        if self.pin_memory:
            padded_sharded_param = padded_sharded_param.pin_memory(device=self.device)

    self._sharded_param_data = padded_sharded_param.view(-1)
    length = sharded_param.size(shard_dim) if sharded_param.numel() > 0 else 0
    sharded_param = padded_sharded_param.narrow(dim=shard_dim, start=0, length=length)

    assert sharded_param.is_contiguous(), f"{self.fsdp_placement=}"

    # PATCHED SECTION: Handle both regular nn.Parameter and bitsandbytes Params4bit
    if isinstance(param, bnb.nn.modules.Params4bit):
        # Create a new Params4bit with the sharded data, preserving quantization attributes
        # Note: Pass the raw tensor data, not the DTensor wrapper
        self.sharded_param = bnb.nn.modules.Params4bit(
            data=sharded_param,  # Use raw tensor, not DTensor
            requires_grad=param.requires_grad,
            quant_state=param.quant_state,
            blocksize=param.blocksize,
            compress_statistics=param.compress_statistics,
            quant_type=param.quant_type,
            quant_storage=param.quant_storage,
            module=param.module,
            bnb_quantized=param.bnb_quantized,
        )
        # Convert to DTensor after creating the Params4bit
        self.sharded_param = self.to_sharded_dtensor(self.sharded_param)
    else:
        # Regular nn.Parameter case
        self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
        self.sharded_param.requires_grad_(param.requires_grad)

    self._setattr_on_modules(self.sharded_param)
    self.sharded_state = ShardedState.SHARDED


# pylint: disable=protected-access
def apply_fsdp2_params4bit_patch():
    """Apply the monkeypatch to enable Params4bit support in FSDP."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

        # Store original method for potential restoration
        if not hasattr(FSDPParam, "_original_init_sharded_param"):
            FSDPParam._original_init_sharded_param = FSDPParam._init_sharded_param

        # Apply the patch
        FSDPParam._init_sharded_param = patched_init_sharded_param
        print("Successfully applied FSDP2 Params4bit patch")
    except ImportError as e:
        print(f"Failed to apply FSDP patch: {e}")


def patched_torch_function(cls, func, types, args=(), kwargs=None):
    """
    Patched version of Params4bit.__torch_function__ for preserving Params4bit
    class identity and attributes.
    """
    if kwargs is None:
        kwargs = {}

    if func in [torch.chunk, torch.split]:
        tensor = args[0]

        # Call the original tensor operation
        result = torch.Tensor.__torch_function__(func, types, args, kwargs)

        if isinstance(result, tuple):
            # Return tuple of new Params4bit instances
            return tuple(
                cls(
                    data=chunk,
                    requires_grad=tensor.requires_grad,
                    quant_state=tensor.quant_state,
                    blocksize=tensor.blocksize,
                    compress_statistics=tensor.compress_statistics,
                    quant_type=tensor.quant_type,
                    quant_storage=tensor.quant_storage,
                    module=tensor.module,
                    bnb_quantized=tensor.bnb_quantized,
                )
                for chunk in result
            )

        return cls(
            data=result,
            requires_grad=tensor.requires_grad,
            quant_state=tensor.quant_state,
            blocksize=tensor.blocksize,
            compress_statistics=tensor.compress_statistics,
            quant_type=tensor.quant_type,
            quant_storage=tensor.quant_storage,
            module=tensor.module,
            bnb_quantized=tensor.bnb_quantized,
        )

    return torch.Tensor.__torch_function__(func, types, args, kwargs)


# pylint: disable=protected-access
def apply_bnb_torch_function_patch():
    """Apply monkeypatch to Params4bit.__torch_function__."""
    try:
        from bitsandbytes.nn.modules import Params4bit

        # Store original method for potential restoration
        if not hasattr(Params4bit, "_original_torch_function"):
            Params4bit._original_torch_function = Params4bit.__torch_function__

        # Apply the patch
        Params4bit.__torch_function__ = classmethod(patched_torch_function)
        print("Successfully applied Params4bit __torch_function__ patch")
    except ImportError as e:
        print(f"Failed to apply Params4bit patch: {e}")


# pylint: disable=protected-access
def patched_init_unsharded_param(self):
    """
    Patched version of FSDPParam.init_unsharded_param that supports Params4bit.
    """
    # Import bitsandbytes conditionally
    try:
        import bitsandbytes as bnb

        has_bnb = True
    except ImportError:
        has_bnb = False
        bnb = None

    # Import required FSDP internals
    from torch.distributed._composable.fsdp._fsdp_common import (
        compiled_autograd_enabled,
    )
    from torch.distributed._tensor.api import _from_local_no_grad
    from torch.distributed.utils import alloc_storage

    if not compiled_autograd_enabled() and hasattr(
        self, "_unsharded_param"
    ):  # after the 1st all-gather
        inner_tensor = self._sharded_local_tensor
        if not hasattr(inner_tensor, "fsdp_post_all_gather"):
            return  # already initialized
        for tensor in self._unsharded_inner_tensors:
            alloc_storage(tensor)
        all_gather_outputs = self._unflatten_all_gather_outputs()
        inner_tensor.fsdp_post_all_gather(
            all_gather_outputs,
            self._extensions_data.all_gather_metadata,
            self.param_dtype or self.orig_dtype,
            out=self._unsharded_param,
        )
        self._extensions_data.clear()
        return

    inner_tensor = self._sharded_local_tensor
    if not compiled_autograd_enabled() and hasattr(
        inner_tensor, "fsdp_post_all_gather"
    ):
        all_gather_outputs = self._unflatten_all_gather_outputs()
        (
            unsharded_tensor,
            self._unsharded_inner_tensors,
        ) = inner_tensor.fsdp_post_all_gather(
            all_gather_outputs,
            self._extensions_data.all_gather_metadata,
            self.param_dtype or self.orig_dtype,
        )
        self._extensions_data.clear()
    else:
        # For the default path (no post-all-gather), the all-gather output
        # gives the unsharded parameter data directly
        assert len(self.all_gather_outputs) == 1, f"{len(self.all_gather_outputs)}"
        unsharded_tensor = self.all_gather_outputs[0]

    unsharded_param = torch.as_strided(
        unsharded_tensor,
        self._orig_size,
        self._contiguous_orig_stride,
        storage_offset=0,
    )

    if self.is_dtensor:
        unsharded_param = _from_local_no_grad(unsharded_param, self._tp_spec)

    if hasattr(self, "_unsharded_param"):
        assert compiled_autograd_enabled()
        with (
            torch.no_grad(),
            torch.autograd._unsafe_preserve_version_counter(self._unsharded_param),
        ):
            # NOTE: Under compile, if an unsharded param goes through
            # resize_(full) -> copy_ -> resize_(0) pattern, we will remove those
            # resize_ and copy_ ops in a compiler graph pass
            # `remove_fsdp2_unsharded_param_graph_input_usage` to recover performance.
            self._unsharded_param.untyped_storage().resize_(
                self._unsharded_param.numel() * self._unsharded_param.itemsize
            )
            torch.ops.fsdp.copy_(self._unsharded_param, unsharded_param)
    else:
        # PATCHED SECTION: Handle both regular nn.Parameter and bitsandbytes Params4bit
        if has_bnb and isinstance(self.sharded_param, bnb.nn.modules.Params4bit):
            # Create a new Params4bit with the unsharded data, preserving quantization attributes
            self._unsharded_param = bnb.nn.modules.Params4bit(
                data=unsharded_param,
                requires_grad=self.sharded_param.requires_grad,
                quant_state=self.sharded_param.quant_state,
                blocksize=self.sharded_param.blocksize,
                compress_statistics=self.sharded_param.compress_statistics,
                quant_type=self.sharded_param.quant_type,
                quant_storage=self.sharded_param.quant_storage,
                module=self.sharded_param.module,
                bnb_quantized=self.sharded_param.bnb_quantized,
            )
        else:
            # Regular nn.Parameter case
            self._unsharded_param = nn.Parameter(
                unsharded_param, requires_grad=self.sharded_param.requires_grad
            )


def apply_init_unsharded_param_patch():
    """Apply the monkeypatch to enable Params4bit support in FSDP init_unsharded_param."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

        # Store original method for potential restoration
        if not hasattr(FSDPParam, "_original_init_unsharded_param"):
            FSDPParam._original_init_unsharded_param = FSDPParam.init_unsharded_param

        # Apply the patch
        FSDPParam.init_unsharded_param = patched_init_unsharded_param
        print("Successfully applied FSDP init_unsharded_param Params4bit patch")

    except ImportError as e:
        print(f"Failed to apply FSDP init_unsharded_param patch: {e}")
        print("Make sure PyTorch with FSDP support is installed")
