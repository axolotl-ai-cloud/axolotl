# pylint: skip-file
import glob
import json
import logging
import os.path
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import bitsandbytes as bnb
import peft
import safetensors.torch as st
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.relora")


def reset_optimizer(optimizer: torch.optim.Optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            param_state = optimizer.state[param]
            for key in param_state:
                if "qmap" in key:
                    continue
                elif key == "step" and isinstance(param_state[key], int):
                    param_state[key] = 0
                else:
                    param_state[key] = torch.zeros_like(param_state[key])


class ReLoRACallback(TrainerCallback):
    def __init__(self, cfg: DictDefault):
        self.relora_steps = cfg.relora_steps
        self.last_full_model = cfg.base_model
        self.quantised = cfg.load_in_4bit or cfg.load_in_8bit

        assert os.path.exists(
            self.last_full_model
        ), "for ReLORA base_model must be a local path"

        self.num_lora_restarts = 0
        self.need_full_save = False

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: peft.LoraModel,
        optimizer: torch.optim.Optimizer,
        **_kwargs,
    ):
        if state.global_step > 0 and state.global_step % self.relora_steps == 0:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
            )

            with torch.no_grad():
                merge_and_save(
                    model,
                    self.last_full_model,
                    checkpoint_folder,
                    reinit=True,
                    quantized=self.quantised,
                )
            reset_optimizer(optimizer)

            self.last_full_model = checkpoint_folder
            self.num_lora_restarts += 1

        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: peft.LoraModel,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
        )
        if state.global_step >= self.relora_steps:
            if self.quantised and self.last_full_model != checkpoint_folder:
                LOG.info(f"moving last full parameter save to {checkpoint_folder}")
                chunks = glob.glob(
                    f"{self.last_full_model}/model*.safetensors"
                ) + glob.glob(f"{self.last_full_model}/model*.index.json")
                for path in chunks:
                    shutil.move(path, checkpoint_folder)
                self.last_full_model = checkpoint_folder
            else:
                model.model.save_pretrained(checkpoint_folder, save_safetensors=True)

        return control

    def on_log(
        self,
        _args: TrainingArguments,
        _state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **_kwargs,
    ):
        logs["num_lora_restarts"] = self.num_lora_restarts
        return control


class ReLoRAScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        inner_schedule: LRScheduler,
        relora_steps: int,
        warmup_steps: int,
        min_lr_scale: float = 0.001,
    ) -> None:
        self.inner_schedule = inner_schedule
        self.relora_steps = relora_steps
        self.warmup_steps = warmup_steps
        self.min_lr_scale = min_lr_scale
        super().__init__(optimizer, inner_schedule.last_epoch, inner_schedule.verbose)

    def get_lr(self) -> float:
        self.inner_schedule.last_epoch = self.last_epoch

        original = self.inner_schedule.get_lr()
        step = self.last_epoch
        if step < self.relora_steps:
            scale = 1
        else:
            cycle_t = min(1.0, (step % self.relora_steps) / self.warmup_steps)
            scale = cycle_t * (1 - self.min_lr_scale) + self.min_lr_scale
        if isinstance(original, Sequence):
            return [lr * scale for lr in original]
        else:
            return original * scale


def sharded_paths(path: str, keys: List[str]) -> Dict[str, str]:
    model_name = "model.safetensors"
    if not os.path.exists(str(Path(path) / model_name)):
        model_name = "pytorch_model.bin"

    index_path = str(Path(path) / f"{model_name}.index.json")
    if os.path.exists(index_path):
        data = json.load(open(index_path, "r"))
        return data["weight_map"]
    return {key + ".weight": model_name for key in keys}


def lora_delta_weight(layer: peft.tuners.lora.LoraLayer) -> torch.Tensor:
    if isinstance(layer, peft.tuners.lora.Linear8bitLt) or isinstance(
        layer, peft.tuners.lora.Linear4bit
    ):
        adapter = layer.active_adapter
        return (
            peft.utils.transpose(
                layer.lora_B[adapter].weight @ layer.lora_A[adapter].weight,
                getattr(layer, "fan_in_fan_out", False),
            )
            * layer.scaling[adapter]
        )
    else:
        return layer.get_delta_weight()


def merge_and_save(
    model: peft.LoraModel,
    model_src: str,
    model_dst: str,
    reinit: bool = False,
    quantized: bool = False,
):
    key_list = [key for key, _ in model.model.named_modules() if "lora" not in key]

    if not quantized:
        for key in key_list:
            try:
                _parent, target, _target_name = peft.utils._get_submodules(
                    model.model, key
                )
            except AttributeError:
                continue

            if isinstance(target, peft.tuners.lora.LoraLayer):
                update = target.get_delta_weight(target.active_adapter).detach()
                target.weight.data += update

                if reinit:
                    for adapter_name in target.lora_A:
                        target.reset_lora_parameters(adapter_name)
                    for adapter_name in target.lora_embedding_A:
                        target.reset_lora_parameters(adapter_name)
        return

    os.makedirs(model_dst, exist_ok=True)
    shard_paths = sharded_paths(model_src, key_list)

    unique_shards = list(set(shard_paths.values()))
    for shard_path in unique_shards:
        out_tensors = {}
        if shard_path.endswith(".safetensors"):
            in_tensors = st.load_file(str(Path(model_src) / shard_path))
        else:
            in_tensors = torch.load(Path(model_src) / shard_path)
            if "state_dict" in in_tensors:
                in_tensors = in_tensors["state_dict"]

        for key in key_list:
            if shard_paths[key + ".weight"] != shard_path:
                continue

            try:
                _parent, target, _target_name = peft.utils._get_submodules(
                    model.model, key
                )
            except AttributeError:
                continue

            if isinstance(target, peft.tuners.lora.LoraLayer):
                orig_weight = in_tensors[key + ".weight"]
                old_dev = target.weight.device

                update = lora_delta_weight(target).detach()
                new_weight = (orig_weight.to(old_dev) + update.to(old_dev)).cpu()
                out_tensors[key + ".weight"] = new_weight

                if reinit:
                    for adapter_name in target.lora_A:
                        target.reset_lora_parameters(adapter_name)
                    for adapter_name in target.lora_embedding_A:
                        target.reset_lora_parameters(adapter_name)

                old_dev = target.weight.device
                if isinstance(target, peft.tuners.lora.Linear4bit):
                    target.weight = bnb.nn.Params4bit(
                        new_weight,
                        requires_grad=False,
                        compress_statistics=target.weight.compress_statistics,
                        quant_type=target.weight.quant_type,
                    ).to(old_dev)
                elif isinstance(target, peft.tuners.lora.Linear8bitLt):
                    target.weight = bnb.nn.Int8Params(
                        new_weight, requires_grad=False
                    ).to(old_dev)
                else:
                    target.weight.data = new_weight.to(old_dev)

        for key in in_tensors:
            if key not in out_tensors:
                out_tensors[key] = in_tensors[key]
        del in_tensors

        out_shard_name = shard_path
        if out_shard_name.startswith("pytorch_model"):
            out_shard_name = (
                out_shard_name.replace("pytorch_model", "model").rstrip(".bin")
                + ".safetensors"
            )
        st.save_file(out_tensors, str(Path(model_dst) / out_shard_name))
        del out_tensors
        torch.cuda.empty_cache()

    if len(unique_shards) > 1:
        with open(str(Path(model_dst, "model.safetensors.index.json")), "w") as fd:
            json.dump({"metadata": {}, "weight_map": shard_paths}, fd)
