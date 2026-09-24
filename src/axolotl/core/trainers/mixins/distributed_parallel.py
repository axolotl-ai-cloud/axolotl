"""
Mixin for correctly saving fsdp
"""

from accelerate import PartialState
from transformers import Trainer


class DistributedParallelMixin(Trainer):
    """
    Mixin for correctly saving fsdp
    """

    def _wrap_model(self, model, *args, **kwargs):
        cfg = getattr(self, "axolotl_cfg", None)
        parallel = self.accelerator.parallelism_config
        distributed_type = self.accelerator.distributed_type
        pure_ddp = (
            getattr(distributed_type, "name", distributed_type) == "MULTI_GPU"
            and not any(
                (getattr(cfg, key, 1) or 1) > 1
                for key in (
                    "tensor_parallel_size",
                    "context_parallel_size",
                    "expert_parallel_size",
                )
            )
            and not any(
                getattr(parallel, key, False)
                for key in ("tp_enabled", "cp_enabled", "dp_shard_enabled")
            )
        )
        if pure_ddp and not getattr(model, "_axolotl_native_nvfp4_ddp_prepared", False):
            from torch.nn.parallel import DistributedDataParallel

            if not isinstance(model, DistributedDataParallel):
                from axolotl.monkeypatch.torchao_ddp import prepare_native_nvfp4_ddp

                if prepare_native_nvfp4_ddp(model, self.accelerator.device):
                    model._axolotl_native_nvfp4_ddp_prepared = True
        deepspeed = getattr(distributed_type, "name", distributed_type) == "DEEPSPEED"
        if deepspeed and not getattr(
            model, "_axolotl_native_nvfp4_deepspeed_prepared", False
        ):
            plugin = getattr(
                getattr(self.accelerator, "state", None), "deepspeed_plugin", None
            )
            config = getattr(plugin, "deepspeed_config", {}) or {}
            zero_stage = config.get("zero_optimization", {}).get("stage", 0)
            if zero_stage in (1, 2, 3):
                from axolotl.monkeypatch.torchao_deepspeed import (
                    prepare_native_nvfp4_deepspeed,
                )

                if prepare_native_nvfp4_deepspeed(
                    model, self.accelerator.device, zero_stage
                ):
                    model._axolotl_native_nvfp4_deepspeed_prepared = True
        return super()._wrap_model(model, *args, **kwargs)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        from axolotl.monkeypatch.torchao_deepspeed import (
            native_nvfp4_zero3_peft_state_dict,
        )

        if not getattr(self.model, "_axolotl_native_nvfp4_zero3_components", ()):
            return super().save_model(output_dir, _internal_call)
        state_dict = native_nvfp4_zero3_peft_state_dict(
            self.model, collect_on_this_rank=self.args.should_save
        )
        if state_dict is None:
            return super().save_model(output_dir, _internal_call)
        output_dir = output_dir or self.args.output_dir
        error = None
        if self.args.should_save:
            try:
                self._save(output_dir, state_dict=state_dict)
            except Exception as exc:  # pylint: disable=broad-except
                error = f"{type(exc).__name__}: {exc}"
        if self.accelerator.num_processes > 1:
            errors = [None] * self.accelerator.num_processes
            import torch.distributed as dist

            dist.all_gather_object(errors, error)
            error = next((item for item in errors if item is not None), None)
        if error is not None:
            raise RuntimeError(f"Native NVFP4 ZeRO-3 adapter export failed: {error}")
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(
                commit_message="Model save", revision=self.args.hub_revision
            )

    def _save(self, output_dir: str | None = None, state_dict=None):
        if (
            state_dict is None
            and self.accelerator.parallelism_config
            and self.accelerator.parallelism_config.dp_shard_enabled
        ):
            state_dict = self.accelerator.get_state_dict(self.model)
        super()._save(output_dir, state_dict=state_dict)

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        if (
            self.accelerator.distributed_type == "FSDP"
            and self.accelerator.state.fsdp_plugin is None
        ):
            # handle Context Parallelism without FSDP
            self.accelerator.state.distributed_type = "MULTI_GPU"
            self.accelerator.state._shared_state["distributed_type"] = "MULTI_GPU"
            PartialState().distributed_type = "MULTI_GPU"
