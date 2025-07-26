"""
handle importing ParallelismConfig from accelerate with fallback
"""

# pylint: disable=protected-access,consider-iterating-dictionary,ungrouped-imports,unused-import,inconsistent-return-statements
try:
    from accelerate.utils.dataclasses import ParallelismConfig
except ImportError:
    from dataclasses import dataclass
    from typing import Union

    import torch

    @dataclass
    class TorchTensorParallelConfig:
        """
        Use this object in your [`Accelerator`] to customize your torch tensor parallelism.
        """

        enable_async_tp: bool = False

    @dataclass
    class ParallelismConfig:
        """
        A dataclass to configure parallelisms applied to the model. Inspired by torchtitan's `ParallelDims`
        https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py

        Args:
            dp_replicate_size (`int`, defaults to `1`):
                The size of the data parallel group. If `dp_replicate_size` is set to 1, the data parallel replication
                group will not be used.
            dp_shard_size (`int`, defaults to `1`):
                The size of the model shard group. If `dp_replicate_size > 1` and `tp_size > 1`, `dp_shard_size` must also
                be greater than 1, as composing DDP + TP is currently not supported.
            tp_size (`int`, defaults to `1`):
                The size of the tensor parallel group. If `tp_size` is set to `1`, the tensor parallel group will not be
                used.
            cp_size (`int`, defaults to `1`):
                The size of the context parallel group. Currently not supported, but reserved for future use and enabled
                for downstream libraries.
            tp_handler (`~utils.TorchTensorParallelConfig`, defaults to `None`):
                The handler for the tensor parallel group.

        You may obtain different distributed data parallel paradigms by configuring `dp_replicate_size` and `dp_shard_size`
        together:
            - `dp_replicate_size == 1` and `dp_shard_size > 1`, we obtain Fully Sharded Data Parallel (FSDP).
            - `dp_replicate_size > 1` and `dp_shard_size > 1`, we obtain Hybrid Sharded Data Parallel (HSDP).
            - `dp_replicate_size > 1` and `dp_shard_size == 1` is an invalid configuration, to use pure DP, use
              `DistributedDataParallelKwargs` instead.

        """

        dp_replicate_size: int = 1
        dp_shard_size: int = 1
        tp_size: int = 1
        cp_size: int = 1

        # we use Union because we might support other x parallel plugins (i.e. deepspeed, etc)
        tp_handler: Union[None, TorchTensorParallelConfig] = None

        def __repr__(self):
            return (
                "ParallelismConfig(\n "
                f"\tdp_replicate_size={self.dp_replicate_size},\n"
                f"\tdp_shard_size={self.dp_shard_size},\n"
                f"\ttp_size={self.tp_size},\n"
                f"\tcp_size={self.cp_size},\n"
                f"\ttotal_size={self.total_size}\n)"
            )

        @property
        def dp_dim_names(self):
            dims = []
            if self.dp_enabled:
                dims += ["dp_replicate"]
            if self.fsdp_enabled:
                dims += ["dp_shard"]
            return dims

        @property
        def non_dp_dim_names(self):
            dims = []
            if self.tp_enabled:
                dims += ["tp"]
            if self.cp_enabled:
                dims += ["cp"]
            return dims

        @property
        def dp_shard_cp_dim_names(self):
            dims = []
            if self.fsdp_enabled:
                dims += ["dp_shard"]
            if self.cp_enabled:
                dims += ["cp"]
            return dims

        @property
        def dp_cp_dim_names(self):
            dims = []
            if self.dp_enabled:
                dims += ["dp_replicate"]
            if self.fsdp_enabled:
                dims += ["dp_shard"]
            if self.cp_enabled:
                dims += ["cp"]
            return dims

        @property
        def model_shard_dim_names(self):
            dims = []
            if self.dp_enabled:
                dims += ["dp_replicate"]
            dims += ["dp_shard_cp"]
            return dims

        @property
        def total_size(self):
            return (
                self.dp_replicate_size
                * self.dp_shard_size
                * self.tp_size
                * self.cp_size
            )

        @property
        def dp_enabled(self):
            return self.dp_replicate_size > 1

        @property
        def fsdp_enabled(self):
            return self.dp_shard_size > 1

        @property
        def tp_enabled(self):
            return self.tp_size > 1

        @property
        def cp_enabled(self):
            return self.cp_size > 1

        @property
        def active_mesh_dims(self):
            return self.dp_dim_names + self.non_dp_dim_names

        def build_device_mesh(self, device_type: str):
            mesh = self.get_mesh()
            if not mesh:
                return
            mesh_dim_names, mesh_shape = mesh
            device_mesh = torch.distributed.init_device_mesh(
                device_type,
                mesh_shape,
                mesh_dim_names=mesh_dim_names,
            )
            if self.dp_dim_names:
                device_mesh[self.dp_dim_names]._flatten("dp")
            if self.dp_shard_cp_dim_names:
                device_mesh[self.dp_shard_cp_dim_names]._flatten("dp_shard_cp")
            if self.dp_cp_dim_names:
                device_mesh[self.dp_cp_dim_names]._flatten("dp_cp")

            return device_mesh

        def get_mesh(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
            """Generate mesh shape and dimension names for torch.distributed.init_device_mesh()."""

            # Build mesh dimensions dictionary
            mesh_dims = {
                parallelism: self._sizes[parallelism]
                for parallelism in self.active_mesh_dims
            }

            # Apply canonical ordering
            mesh_order = ["dp_replicate", "dp_shard", "cp", "tp"]
            sorted_items = sorted(
                mesh_dims.items(),
                key=lambda x: (mesh_order.index(x[0])),
            )
            return tuple(zip(*sorted_items))

        def __post_init__(self):
            # Basic size validation
            if self.dp_replicate_size < 1:
                raise ValueError(
                    f"dp_replicate_size must be at least 1, but got {self.dp_replicate_size}"
                )
            if self.dp_shard_size < 1:
                raise ValueError(
                    f"dp_shard_size must be at least 1, but got {self.dp_shard_size}"
                )
            if self.tp_size < 1:
                raise ValueError(f"tp_size must be at least 1, but got {self.tp_size}")
            if self.cp_size < 1:
                raise ValueError(f"cp_size must be at least 1, but got {self.cp_size}")

            if (
                (self.tp_size > 1 or self.cp_size > 1)
                and self.dp_replicate_size > 1
                and self.dp_shard_size == 1
            ):
                raise ValueError(
                    "Tensor/Context parallelism (tp/cp_size > 1) cannot be used with pure data parallelism (dp_replicate_size > 1 and dp_shard_size == 1). "
                    "Please set dp_shard_size > 1 and dp_replicate_size == 1 to compose FSDP + TP/CP for 2D parallel, "
                    "or set dp_replicate_size == 1 and dp_shard_size > 1 to compose HSDP + TP/CP for 3D parallel."
                )
            self._sizes = {
                "dp_replicate": self.dp_replicate_size,
                "dp_shard": self.dp_shard_size,
                "tp": self.tp_size,
                "cp": self.cp_size,
            }

        def _set_size(self, parallelism: str, size: int):
            assert (
                parallelism in self._sizes.keys()
            ), f"Parallelism must be one of {self._sizes.keys()}"
            self._sizes[parallelism] = size
            setattr(self, f"{parallelism}_size", size)
