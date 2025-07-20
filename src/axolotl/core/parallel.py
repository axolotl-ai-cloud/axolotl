import os
from dataclasses import dataclass, field

from test_dict import DictDefaultTest
from torch.distributed import init_device_mesh

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class DistParallel:
    """
    Class to manage distributed parallelism for training
    """

    dp_replicate_size: int | None = field(default=1)
    dp_shard_size: int | None = field(default=1)
    tp_size: int | None = field(default=1)
    cp_size: int | None = field(default=1)

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __repr__(self):
        return f"{self.__class__.__name__}(dp_replicate_size={self.dp_replicate_size}, dp_shard_size={self.dp_shard_size}, tp_size={self.tp_size}, cp_size={self.cp_size})"

    @classmethod
    def build(
        cls,
        dp_replicate_size: int | None = None,
        dp_shard_size: int | None = None,
        tp_size: int | None = None,
        cp_size: int | None = None,
        fsdp: bool = False,
        world_size: int | None = None,
    ):
        if not world_size:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

        dp_total_size = world_size
        if tp_size and tp_size > 1:
            dp_total_size = dp_total_size // tp_size
        if cp_size and cp_size > 1:
            dp_total_size = dp_total_size // cp_size

        if dp_shard_size and dp_shard_size > 1:
            dp_replicate_size = dp_total_size // dp_shard_size
        elif dp_replicate_size and dp_replicate_size > 1:
            dp_shard_size = dp_total_size // dp_replicate_size
        elif dp_shard_size is None and fsdp:
            # assume FSDP across all remaining dims
            dp_shard_size = dp_total_size
        elif (tp_size and tp_size > 1) or (cp_size and cp_size > 1):
            dp_replicate_size = dp_total_size
        else:
            raise ValueError("Unhandled distributed parallelism configuration")

        res = cls(
            dp_replicate_size=dp_replicate_size or 1,
            dp_shard_size=dp_shard_size or 1,
            tp_size=tp_size or 1,
            cp_size=cp_size or 1,
        )
        LOG.debug(res, main_process_only=True)
        return res

    def get_mesh(self):
        mesh_dims = ()
        mesh_dim_names = ()

        if self.dp_replicate_size > 1:
            mesh_dims += (self.dp_replicate_size,)
            mesh_dim_names += ("dp_replicate",)
        if self.dp_shard_size > 1:
            mesh_dims += (self.dp_shard_size,)
            mesh_dim_names += ("dp_shard",)
        if self.tp_size > 1:
            mesh_dims += (self.tp_size,)
            mesh_dim_names += ("tp",)
        if self.cp_size > 1:
            mesh_dims += (self.cp_size,)
            mesh_dim_names += ("cp",)

        return mesh_dims, mesh_dim_names

    def get_device_mesh(self):
        mesh_dims, mesh_dim_names = self.get_mesh()
        mesh = init_device_mesh("cuda", mesh_dims, mesh_dim_names=mesh_dim_names)

        dp_mesh_dim_names = []
        dp_shard_cp_mesh_dim_names = []
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_size > 1:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_size > 1:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_size > 1:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        if self.dp_shard_size > 1:
            # legacy support for fsdp
            mesh[("dp_shard",)]._flatten(mesh_dim_name="fsdp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )

        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        return mesh
