"""
Mixin for correctly saving fsdp
"""

import contextlib
import os

from accelerate import PartialState
from transformers import Trainer


def tp_save_joins_all_ranks(accelerator, is_fsdp_enabled: bool) -> bool:
    """Whether every TP rank must enter ``save_pretrained`` for the gather + barrier."""
    if not is_fsdp_enabled:
        return True
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    # under a sharded state dict rank 0 never calls _save, so the other ranks must not barrier
    return "FULL_STATE_DICT" in str(getattr(plugin, "state_dict_type", ""))


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
        if (
            deepspeed
            and getattr(
                model, "_axolotl_native_nvfp4_deepspeed_merge_aware_requested", False
            )
            and not getattr(model, "_axolotl_native_nvfp4_metadata_prepared", False)
        ):
            from axolotl.monkeypatch.torchao_nvfp4_merge_persistence import (
                prepare_sharded_native_metadata,
            )

            prepare_sharded_native_metadata(model)
            model._axolotl_native_nvfp4_metadata_prepared = True
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

    def _expert_parallel_enabled(self) -> bool:
        parallelism = getattr(self.accelerator, "parallelism_config", None)
        if parallelism is not None:
            return bool(getattr(parallelism, "ep_enabled", False))
        # pure EP builds no ParallelismConfig, only the env var
        return int(os.environ.get("PARALLELISM_CONFIG_EP_SIZE", "1") or 1) > 1

    def _global_mesh(self):
        return getattr(
            self.accelerator,
            "torch_device_mesh",
            getattr(getattr(self.accelerator, "state", None), "device_mesh", None),
        )

    def _cpu_offloaded(self) -> bool:
        from torch.distributed.fsdp import CPUOffloadPolicy

        plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
        return isinstance(getattr(plugin, "cpu_offload", None), CPUOffloadPolicy)

    def _clip_grad_norm(self, model):
        from axolotl.utils.gradient_clipping import (
            clip_grad_norm_ep_local_shards_,
            clip_grad_norm_local_shards_,
            ep_local_parameter_ids,
            has_cpu_offloaded_dtensor_gradients,
        )

        parameters = list(model.parameters())
        if self._expert_parallel_enabled():
            # experts live on their own mesh: one ownership-filtered global norm
            self.accelerator.unscale_gradients()
            return clip_grad_norm_ep_local_shards_(
                parameters,
                self.args.max_grad_norm,
                ep_local_parameters=ep_local_parameter_ids(model),
                global_mesh=self._global_mesh(),
            )
        if self._cpu_offloaded() and has_cpu_offloaded_dtensor_gradients(parameters):
            self.accelerator.unscale_gradients()
            return clip_grad_norm_local_shards_(parameters, self.args.max_grad_norm)
        return super()._clip_grad_norm(model)

    def _get_grad_norm(self, model, grad_norm=None):
        if grad_norm is not None:
            return super()._get_grad_norm(model, grad_norm)
        from axolotl.utils.gradient_clipping import (
            ep_local_parameter_ids,
            get_grad_norm_ep_local_shards_,
            get_grad_norm_local_shards_,
            has_cpu_offloaded_dtensor_parameters,
        )

        parameters = list(model.parameters())
        if self._expert_parallel_enabled():
            self.accelerator.unscale_gradients()
            return get_grad_norm_ep_local_shards_(
                parameters,
                ep_local_parameters=ep_local_parameter_ids(model),
                global_mesh=self._global_mesh(),
            )
        if self._cpu_offloaded() and has_cpu_offloaded_dtensor_parameters(parameters):
            self.accelerator.unscale_gradients()
            return get_grad_norm_local_shards_(parameters)
        return super()._get_grad_norm(model, grad_norm)

    def _save_model_native(
        self, output_dir: str | None = None, _internal_call: bool = False
    ):
        from axolotl.monkeypatch.torchao_deepspeed import (
            native_nvfp4_zero3_peft_state_dict,
        )
        from axolotl.monkeypatch.torchao_tp_lora import (
            native_nvfp4_tp_peft_state_dict,
        )

        state_dict = None
        plugin = getattr(
            getattr(getattr(self, "accelerator", None), "state", None),
            "fsdp_plugin",
            None,
        )
        if (
            getattr(self, "is_fsdp_enabled", False)
            and getattr(plugin, "fsdp_version", None) == 2
            and "FULL_STATE_DICT" in str(getattr(plugin, "state_dict_type", ""))
        ):
            from accelerate.utils.modeling import is_peft_model

            from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
                _model_needs_ownership,
                full_model_state,
            )

            if is_peft_model(self.model) and _model_needs_ownership(self.model):
                state_dict = full_model_state(self.model, adapter_only=True)
        if state_dict is None:
            state_dict = native_nvfp4_zero3_peft_state_dict(
                self.model, collect_on_this_rank=self.args.should_save
            )
        if state_dict is None:
            state_dict = native_nvfp4_tp_peft_state_dict(
                self.model, collect_on_this_rank=self.args.should_save
            )
        if state_dict is None:
            metadata = getattr(self.model, "_axolotl_native_nvfp4_metadata", None)
            result = super().save_model(output_dir, _internal_call)
            if metadata and self.args.should_save:
                from axolotl.monkeypatch.torchao_nvfp4_merge_persistence import (
                    clear_native_metadata,
                    native_metadata_valid_for_save,
                    write_native_metadata,
                )

                target = output_dir or self.args.output_dir
                if native_metadata_valid_for_save(self.model):
                    write_native_metadata(target, metadata)
                else:
                    clear_native_metadata(target)
            return result
        output_dir = output_dir or self.args.output_dir
        error = None
        if self.args.should_save:
            try:
                self._save(output_dir, state_dict=state_dict)
                from axolotl.monkeypatch.torchao_nvfp4_merge_persistence import (
                    persist_native_metadata_after_save,
                )

                persist_native_metadata_after_save(self.model, output_dir)
            except Exception as exc:  # pylint: disable=broad-except
                error = f"{type(exc).__name__}: {exc}"
        if self.accelerator.num_processes > 1:
            errors = [None] * self.accelerator.num_processes
            import torch.distributed as dist

            dist.all_gather_object(errors, error)
            error = next((item for item in errors if item is not None), None)
        if error is not None:
            raise RuntimeError(f"Distributed adapter export failed: {error}")
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(
                commit_message="Model save", revision=self.args.hub_revision
            )

    def _ep_full_param_experts(self) -> bool:
        cfg = getattr(self, "axolotl_cfg", None)
        if not cfg or (getattr(cfg, "expert_parallel_size", 1) or 1) <= 1:
            return False
        if getattr(cfg, "adapter", None) or not self.is_fsdp_enabled:
            return False
        from axolotl.integrations.expert_parallel.shard import _detect_experts_modules

        return any(
            getattr(m, "num_experts_global", m.num_experts) > m.num_experts
            for _n, m in _detect_experts_modules(self.model)
        )

    def _axolotl_tp_size(self) -> int:
        cfg = getattr(self, "axolotl_cfg", None)
        return int(getattr(cfg, "tensor_parallel_size", 1) or 1)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        previous = getattr(self, "_axolotl_saving_checkpoint", False)
        self._axolotl_saving_checkpoint = _internal_call
        try:
            if (
                self._axolotl_tp_size() > 1
                and not self._ep_full_param_experts()
                and tp_save_joins_all_ranks(self.accelerator, self.is_fsdp_enabled)
            ):
                result = self._save_model_native(output_dir, _internal_call)
                if not self.args.should_save:
                    # transformers gathers TP DTensors and barriers inside save_pretrained, so the
                    # non-writing ranks must call it too; under FSDP the state dict was already
                    # gathered by every rank, so they only need to join the barrier
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir or self.args.output_dir,
                        state_dict={} if self.is_fsdp_enabled else None,
                        is_main_process=False,
                    )
                return result
            return self._save_model_native(output_dir, _internal_call)
        finally:
            self._axolotl_saving_checkpoint = previous

    def _save_fsdp2_model_only_checkpoint(self, output_dir, state_dict):
        plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
        if (
            state_dict is not None
            and self.args.should_save
            and getattr(self.args, "save_only_model", False)
            and getattr(self, "_axolotl_saving_checkpoint", False)
            and getattr(plugin, "fsdp_version", None) == 2
            and "FULL_STATE_DICT" in str(getattr(plugin, "state_dict_type", ""))
        ):
            from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
                _model_needs_ownership,
            )

            if _model_needs_ownership(self.model):
                import torch

                target = output_dir or self.args.output_dir
                os.makedirs(target, exist_ok=True)
                torch.save(state_dict, os.path.join(target, "pytorch_model_fsdp.bin"))

    def _ep_sharded_checkpoint(self) -> bool:
        """FSDP with EP-sharded experts or expert LoRA (full-parameter or adapter runs)."""
        cfg = getattr(self, "axolotl_cfg", None)
        if not cfg or (getattr(cfg, "expert_parallel_size", 1) or 1) <= 1:
            return False
        if not self.is_fsdp_enabled:
            return False
        from axolotl.integrations.expert_parallel.checkpoint import ep_sharded_params

        return bool(ep_sharded_params(self.model))

    @contextlib.contextmanager
    def _ep_checkpoint_functions(self):
        """Route the Trainer's FSDP checkpoint save/load through the EP-aware functions.

        accelerate's FULL_STATE_DICT checkpoint gathers each expert (and expert LoRA) only
        over its own (non-ep) FSDP mesh and rank 0 writes it, so the file would hold EP
        group 0's experts and, on resume, every EP group would load group 0's experts and
        optimizer moments. The EP-aware functions gather the experts across ep on save and give each
        rank its own block back on load."""
        if not self._ep_sharded_checkpoint():
            yield
            return
        from axolotl.integrations.expert_parallel.checkpoint import (
            ep_fsdp_checkpoint_functions,
        )
        from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

        ep_group = ExpertParallelPlugin._resolve_ep_group(self.axolotl_cfg)
        with ep_fsdp_checkpoint_functions(ep_group):
            yield

    def _save_optimizer_and_scheduler(self, output_dir):
        with self._ep_checkpoint_functions():
            return super()._save_optimizer_and_scheduler(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        with self._ep_checkpoint_functions():
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_optimizer_and_scheduler(self, checkpoint):
        with self._ep_checkpoint_functions():
            return super()._load_optimizer_and_scheduler(checkpoint)

    def _load_best_model(self):
        with self._ep_checkpoint_functions():
            return super()._load_best_model()

    def _save(self, output_dir: str | None = None, state_dict=None):
        if (
            state_dict is None
            and self.accelerator.parallelism_config
            and self.accelerator.parallelism_config.dp_shard_enabled
        ):
            state_dict = self.accelerator.get_state_dict(self.model)
        self._save_fsdp2_model_only_checkpoint(output_dir, state_dict)
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
