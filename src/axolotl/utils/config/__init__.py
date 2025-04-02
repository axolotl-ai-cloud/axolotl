"""Module for working with config dicts"""

import json
import logging
import os
from typing import Optional

import torch
from transformers.utils import is_torch_bf16_gpu_available
from transformers.utils.import_utils import is_torch_npu_available

from axolotl.integrations.base import PluginManager
from axolotl.integrations.config import merge_input_args
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import MULTIMODAL_AUTO_MODEL_MAPPING, load_model_config
from axolotl.utils.schemas.config import (
    AxolotlConfigWCapabilities as AxolotlConfigWCapabilitiesBase,
)
from axolotl.utils.schemas.config import AxolotlInputConfig as AxolotlInputConfigBase
from axolotl.utils.schemas.datasets import DPODataset, KTODataset, SFTDataset

LOG = logging.getLogger("axolotl")


def choose_device(cfg):
    def get_device():
        try:
            if torch.cuda.is_available():
                return f"cuda:{cfg.local_rank}"

            if torch.backends.mps.is_available():
                return "mps"

            if is_torch_npu_available():
                return f"npu:{cfg.local_rank}"

            raise SystemError("No CUDA/mps/npu device found")
        except Exception:  # pylint: disable=broad-exception-caught
            return "cpu"

    cfg.device = get_device()
    if cfg.world_size == 1:
        cfg.device_map = cfg.device_map or "auto"
    else:
        if cfg.device.startswith("cuda"):
            cfg.device_map = {"": torch.cuda.current_device()}
        elif cfg.device.startswith("npu"):
            cfg.device_map = {"npu": torch.npu.current_device()}
        else:
            cfg.device_map = {"": cfg.device}

    # in `accelerate launch`, we need to not pass through any device map and let
    # accelerate figure out which parts of the model to put on which gpu
    accelerate_vars = [var for var in os.environ if var.startswith("ACCELERATE_USE_")]
    if accelerate_vars:
        cfg.device_map = None


def resolve_dtype(cfg):
    if (
        cfg.bf16 == "auto" and not cfg.use_ray
    ):  # if we use ray we want to defer this check to the worker node
        if is_torch_bf16_gpu_available():
            LOG.debug("bf16 support detected, enabling for this configuration.")
            cfg.bf16 = True
        else:
            LOG.debug("bf16 support not detected, disabling for this configuration.")
            cfg.bf16 = False
            if cfg.fp16 is None:
                cfg.fp16 = True

    if cfg.device == "mps":
        cfg.load_in_8bit = False
        cfg.tf32 = False
        if cfg.bf16:
            cfg.fp16 = True
        cfg.bf16 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32 or False
        torch.backends.cudnn.allow_tf32 = cfg.tf32 or False
        if cfg.bf16:
            cfg.fp16 = False

    if cfg.bf16 or cfg.bfloat16:
        cfg.torch_dtype = torch.bfloat16
    elif cfg.load_in_8bit or cfg.fp16 or cfg.float16:
        cfg.torch_dtype = torch.float16
    else:
        cfg.torch_dtype = torch.float32


def normalize_config(cfg):
    # setup some derived config / hyperparams
    cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
        cfg.batch_size // cfg.micro_batch_size
    )
    cfg.batch_size = (
        cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps
    )
    if cfg.eval_batch_size is None:
        cfg.eval_batch_size = cfg.micro_batch_size
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.eval_table_size = cfg.eval_table_size or 0
    cfg.eval_max_new_tokens = cfg.eval_max_new_tokens or 128
    cfg.eval_causal_lm_metrics = cfg.eval_causal_lm_metrics or [
        "sacrebleu",
        "comet",
        "ter",
        "chrf",
    ]
    choose_device(cfg)
    cfg.ddp = cfg.ddp if cfg.ddp is not None else cfg.world_size != 1
    if cfg.ddp:
        cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        cfg.batch_size = cfg.batch_size * cfg.world_size

    if not cfg.use_ray:
        # delay resolving dtype until on worker node when launching with ray
        resolve_dtype(cfg)

    if cfg.deepspeed:
        if isinstance(cfg.deepspeed, str) and os.path.exists(cfg.deepspeed):
            ds_config_path = cfg.deepspeed
            with open(ds_config_path, encoding="utf-8") as f:
                cfg.deepspeed = json.load(f)

    if cfg.sequence_parallel_degree is None:
        cfg.sequence_parallel_degree = 1

    if cfg.saves_per_epoch:
        save_steps = 1.0 / (cfg.saves_per_epoch * cfg.num_epochs)
        if save_steps < 1.0:  # prevent saves on every step
            cfg.save_steps = save_steps
        elif save_steps > 1:
            LOG.warning(
                f"Invalid value for save_steps ({save_steps}) from saves_per_epoch and/or num_epochs. Saving at training end only."
            )
    if (cfg.val_set_size or cfg.test_datasets) and cfg.evals_per_epoch:
        eval_steps = 1.0 / (cfg.evals_per_epoch * cfg.num_epochs)
        if eval_steps < 1.0:  # prevent evals on every step
            cfg.eval_steps = eval_steps
        elif eval_steps > 1:
            LOG.warning(
                f"Invalid value for eval_steps ({eval_steps}) from evals_per_epoch and/or num_epochs. Skipping evaluations."
            )

    cfg.dataset_processes = cfg.dataset_processes or os.cpu_count()

    if not cfg.base_model_config:
        cfg.base_model_config = cfg.base_model

    model_config = load_model_config(cfg)

    cfg.tokenizer_config = (
        cfg.tokenizer_config or cfg.base_model_config or cfg.base_model
    )

    cfg.is_multimodal = (
        hasattr(model_config, "model_type")
        and model_config.model_type in MULTIMODAL_AUTO_MODEL_MAPPING
        or any(
            multimodal_name in cfg.base_model.lower()
            for multimodal_name in [
                "pixtral",
            ]
        )
        or cfg.is_multimodal
    )
    if cfg.is_multimodal:
        cfg.processor_config = (
            cfg.processor_config or cfg.base_model_config or cfg.base_model
        )

    cfg.model_config_type = model_config.model_type

    # figure out if the model is llama
    cfg.is_llama_derived_model = (
        (
            hasattr(model_config, "model_type")
            and model_config.model_type in ["llama", "mllama_text_model"]
        )
        or cfg.is_llama_derived_model
        or "llama" in cfg.base_model.lower()
        or (cfg.type_of_model and "llama" in cfg.type_of_model.lower())
    )

    # figure out if the model is falcon
    cfg.is_falcon_derived_model = (
        (
            hasattr(model_config, "model_type")
            and model_config.model_type
            in [
                "falcon",
                "RefinedWebModel",
                "RefinedWeb",
            ]
        )
        or cfg.is_falcon_derived_model
        or "falcon" in cfg.base_model.lower()
        or (cfg.type_of_model and "rwforcausallm" in cfg.type_of_model.lower())
    )

    cfg.is_mistral_derived_model = (
        (
            hasattr(model_config, "model_type")
            and model_config.model_type
            in [
                "mistral",
            ]
        )
        or cfg.is_mistral_derived_model
        or "mistral" in cfg.base_model.lower().split("/")[-1]
        or (cfg.type_of_model and "mistral" in cfg.type_of_model.lower())
    )

    cfg.is_qwen_derived_model = (
        hasattr(model_config, "model_type")
        and model_config.model_type
        in [
            "qwen",
        ]
    ) or cfg.is_qwen_derived_model

    if isinstance(cfg.pretraining_dataset, dict):
        cfg.pretraining_dataset = [cfg.pretraining_dataset]

    if (
        cfg.gradient_checkpointing
        and cfg.unfrozen_parameters is None
        and cfg.gradient_checkpointing_kwargs is None
        and cfg.rl is None
    ):
        cfg.gradient_checkpointing_kwargs = {"use_reentrant": True}

    log_gpu_memory_usage(LOG, "baseline", cfg.device)


def normalize_cfg_datasets(cfg):
    """
    helpers for mapping chat_template to various dataset configurations as necessary
    """

    if cfg.chat_template:
        if cfg.datasets:
            for idx, ds_cfg in enumerate(cfg.datasets):
                if (
                    ds_cfg.type in ["orpo.chat_template", "chat_template"]
                    and not ds_cfg.chat_template
                ):
                    LOG.info(
                        f"updating dataset {ds_cfg.path} with `chat_template: {cfg.chat_template}` to match your chat_template"
                    )
                    cfg.datasets[idx].chat_template = cfg.chat_template
                    cfg.datasets[idx].chat_template_jinja = cfg.chat_template_jinja


def validate_config(
    cfg: DictDefault,
    capabilities: Optional[dict] = None,
    env_capabilities: Optional[dict] = None,
) -> DictDefault:
    AxolotlConfigWCapabilities = AxolotlConfigWCapabilitiesBase
    AxolotlInputConfig = AxolotlInputConfigBase

    if cfg.plugins:
        (
            AxolotlConfigWCapabilities,  # pylint: disable=invalid-name
            AxolotlInputConfig,  # pylint: disable=invalid-name
        ) = merge_input_args()

    # Convert datasets to proper format if needed
    if cfg.get("datasets"):
        for idx, ds_cfg in enumerate(cfg["datasets"]):
            if cfg.get("rl") == "dpo" and not isinstance(ds_cfg, DPODataset):
                cfg["datasets"][idx] = DPODataset(**ds_cfg)
            elif cfg.get("rl") == "kto" and not isinstance(ds_cfg, KTODataset):
                cfg["datasets"][idx] = KTODataset(**dict(ds_cfg))
            elif not isinstance(ds_cfg, SFTDataset):
                cfg["datasets"][idx] = SFTDataset(**dict(ds_cfg))

    if capabilities or env_capabilities:
        if (capabilities and env_capabilities is None) or (
            env_capabilities and capabilities is None
        ):
            raise ValueError(
                "Both capabilities and env_capabilities must be provided or not provided."
            )

        return DictDefault(
            dict(
                AxolotlConfigWCapabilities(
                    **cfg.to_dict(),
                    capabilities=capabilities,
                    env_capabilities=env_capabilities,
                ).model_dump(exclude_none=True)
            )
        )

    return DictDefault(
        dict(AxolotlInputConfig(**cfg.to_dict()).model_dump(exclude_none=True))
    )


def prepare_plugins(cfg):
    """
    Prepare the plugins for the configuration
    """

    if cfg.get("plugins"):
        plugin_manager = PluginManager.get_instance()
        for plugin_name in cfg["plugins"]:
            plugin_manager.register(plugin_name)
