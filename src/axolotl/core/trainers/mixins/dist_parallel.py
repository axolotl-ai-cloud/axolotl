"""Axolotl Trainer mixin to patch Accelerator for distributed parallel training"""

import os

import transformers.trainer
from accelerate import PartialState
from accelerate.utils import TorchTensorParallelPlugin
from torch.distributed import DeviceMesh


class DistParallelMixin(transformers.trainer.Trainer):
    """
    Trainer mixin to patch Accelerator for distributed parallel training
    """

    def create_accelerator_and_postprocess(self):
        res = super().create_accelerator_and_postprocess()

        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            device_mesh: DeviceMesh = self.accelerator.state.device_mesh
            mesh_dim_names: tuple[str, ...] | None = device_mesh.mesh_dim_names
            if "tp" in mesh_dim_names and device_mesh["tp"].size() > 1:
                self.accelerator.state.distributed_type = "TP"
                PartialState().distributed_type = "TP"
                tp_plugin = TorchTensorParallelPlugin(
                    tp_size=device_mesh["tp"].size(), torch_device_mesh=device_mesh
                )
                self.accelerator.state.torch_tp_plugin = tp_plugin

        return res
