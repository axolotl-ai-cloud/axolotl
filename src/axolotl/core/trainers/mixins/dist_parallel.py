import transformers.trainer

from axolotl.core.parallel import DistParallel
from axolotl.monkeypatch.accelerate.axolotl import AxolotlAccelerator


class DistParallelMixin(transformers.trainer.Trainer):
    def create_accelerator_and_postprocess(self):
        transformers.trainer.Accelerator = AxolotlAccelerator
        res = super().create_accelerator_and_postprocess()

        if self.args.world_size > 1:
            dist_parallel_kwargs = {}
            if self.args.dist_parallel_dim_names and self.args.dist_parallel_dims:
                for name, dim in zip(
                    self.args.dist_parallel_dim_names, self.args.dist_parallel_dims
                ):
                    if dim > 1:
                        dist_parallel_kwargs[f"{name}_size"] = dim

            dist_parallel = DistParallel.build(
                fsdp=bool(self.args.fsdp_config),
                world_size=self.args.world_size,
                **dist_parallel_kwargs,
            )
            self.accelerator.world_device_mesh = dist_parallel.get_device_mesh()

        return res
