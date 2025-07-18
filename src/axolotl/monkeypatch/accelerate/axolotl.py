import accelerate
from accelerate import Accelerator


class AxolotlAccelerator(Accelerator):
    _world_device_mesh = None

    @property
    def world_device_mesh(self):
        return self._world_device_mesh

    @world_device_mesh.setter
    def world_device_mesh(self, value):
        self._world_device_mesh = value

    def _prepare_device_mesh(self):
        if (
            not (
                self.distributed_type
                == accelerate.accelerator.DistributedType.DEEPSPEED
                and hasattr(self.state, "ds_device_mesh")
            )
            and self.world_device_mesh is not None
        ):
            return self.world_device_mesh
        return super()._prepare_device_mesh()
