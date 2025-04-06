"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import importlib
import inspect
import os
import signal
import sys
import weakref
from pathlib import Path
from typing import Any, Dict

import torch
import transformers.modelcard
from accelerate.logging import get_logger
from accelerate.utils import save_fsdp_model
from datasets import Dataset
from huggingface_hub.errors import OfflineModeIsEnabled
from peft import PeftConfig, PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.contribs.lgpl import (  # pylint: disable = no-name-in-module
    fix_untrained_tokens,
)
from axolotl.core.trainer_builder import HFCausalTrainerBuilder, HFRLTrainerBuilder
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import cleanup_distributed
from axolotl.utils.freeze import freeze_layers_except
from axolotl.utils.models import load_model, load_processor, load_tokenizer
from axolotl.utils.trainer import setup_trainer

try:
    from optimum.bettertransformer import BetterTransformer
except ImportError:
    BetterTransformer = None

configure_logging()
LOG = get_logger(__name__)


def setup_model_and_tokenizer(
    cfg: DictDefault,
) -> tuple[
    PreTrainedModel, PreTrainedTokenizer, PeftConfig | None, ProcessorMixin | None
]:
    """
    Load the tokenizer, processor (for multimodal models), and model based on configuration.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.

    Returns:
        Tuple containing model, tokenizer, `peft_config` (if LoRA / QLoRA, else
            `None`), and processor (if multimodal, else `None`).
    """
    # Load tokenizer
    LOG.debug(
        f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}",
        main_process_only=True,
    )
    tokenizer = load_tokenizer(cfg)

    # Load processor for multimodal models if needed
    processor = None
    if cfg.is_multimodal:
        processor = load_processor(cfg, tokenizer)

    # Load the model and peft_config
    msg = "loading model"
    if cfg.adapter:
        msg += " and peft_config..."
    LOG.debug(msg)

    model, peft_config = load_model(cfg, tokenizer, processor=processor)
    if model.generation_config is not None:
        model.generation_config.do_sample = True

    # Apply freezing if specified
    if cfg.unfrozen_parameters:
        freeze_layers_except(model, cfg.unfrozen_parameters)

    return model, tokenizer, peft_config, processor


def setup_reference_model(
    cfg: DictDefault, tokenizer: PreTrainedTokenizer
) -> PreTrainedModel | None:
    """
    Set up the reference model for RL training if needed.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        tokenizer: The tokenizer to use for the reference model.

    Returns:
        Reference model if needed for RL training, `None` otherwise.
    """
    model_ref = None
    if cfg.rl and cfg.rl != "orpo":
        if cfg.adapter and not cfg.rl_adapter_ref_model:
            # use built-in trl autounwrap
            LOG.debug("Passing model_ref: None to RL trainer")
            model_ref = None  # explicit setting to None
        else:
            # load the model again for model_ref/baseline
            model_ref, _ = load_model(cfg, tokenizer, reference_model=True)
    return model_ref


def determine_resume_checkpoint(cfg: DictDefault) -> str | None:
    """
    Determine the checkpoint to resume from based on configuration.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.

    Returns:
        Path to the checkpoint to resume from, or `None` if not resuming.
    """
    if cfg.resume_from_checkpoint is None and cfg.auto_resume_from_checkpoints:
        possible_checkpoints = [
            str(cp) for cp in Path(cfg.output_dir).glob("checkpoint-*")
        ]
        if len(possible_checkpoints) > 0:
            sorted_paths = sorted(
                possible_checkpoints,
                key=lambda path: int(path.split("-")[-1]),
            )
            cfg.resume_from_checkpoint = sorted_paths[-1]
            LOG.info(
                f"Using Auto-resume functionality to start with checkpoint at {cfg.resume_from_checkpoint}"
            )
    return cfg.resume_from_checkpoint


def setup_signal_handler(
    cfg: DictDefault, model: PreTrainedModel, safe_serialization: bool
):
    """
    Set up signal handler for graceful termination.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        model: The model to save on termination
        safe_serialization: Whether to use safe serialization when saving
    """
    # ray workers don't have access to this signal
    if cfg.local_rank == 0 and not cfg.use_ray:

        def terminate_handler(_, __, model_weakref):
            if model_weakref() is not None:
                _model = model_weakref()
                if cfg.flash_optimum and BetterTransformer:
                    _model = BetterTransformer.reverse(_model)
                _model.save_pretrained(
                    cfg.output_dir, safe_serialization=safe_serialization
                )

            cleanup_distributed()
            sys.exit(0)

        _model_weakref = weakref.ref(model)
        signal.signal(
            signal.SIGINT,
            lambda signum, frame: terminate_handler(signum, frame, _model_weakref),
        )


def execute_training(
    cfg: DictDefault, trainer: Any, resume_from_checkpoint: str | None
):
    """
    Execute the training process with appropriate SDP kernel configurations.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        trainer: The configured trainer object.
        resume_from_checkpoint: Path to checkpoint to resume from, if applicable.
    """
    LOG.info("Starting trainer...")
    if cfg.flash_optimum:
        with torch.backends.cuda.sdp_kernel(
            # TODO configure these from the YAML w/ sdp_kernel_kwargs: ...
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
        ):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def save_trained_model(
    cfg: DictDefault,
    trainer: Any,
    model: PreTrainedModel,
    safe_serialization: bool,
):
    """
    Save the trained model according to configuration and training setup.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        trainer: The trainer object.
        model: The trained model to save.
        safe_serialization: Whether to use safe serialization.
    """
    LOG.info(f"Training completed! Saving pre-trained model to {cfg.output_dir}.")

    # Post training module hooks
    for name, module in model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(model, name)  # pylint: disable=protected-access

    # Handle FSDP state dict type
    state_dict_type = "FULL_STATE_DICT"
    if trainer.is_fsdp_enabled and str(cfg.fsdp_config.fsdp_version) != "2":
        if cfg.fsdp_final_state_dict_type:
            state_dict_type = cfg.fsdp_final_state_dict_type
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(state_dict_type)
        LOG.info(f"Set FSDP state dict type to {state_dict_type} for saving.")

    # Handle ReLoRA early return case
    if cfg.relora_steps:
        if cfg.adapter == "lora" and not (cfg.load_in_4bit or cfg.load_in_8bit):
            model = model.merge_and_unload()
        else:
            # final model weights have already been saved by `ReLoRACallback.on_train_end`
            return

    if cfg.fsdp:
        # TODO: do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
        # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple
        # processes attempt to write the same file
        if (
            state_dict_type == "SHARDED_STATE_DICT"
            and cfg.fsdp_config.fsdp_state_dict_type == "SHARDED_STATE_DICT"
        ):
            save_fsdp_model(
                trainer.accelerator.state.fsdp_plugin,
                trainer.accelerator,
                trainer.model,
                cfg.output_dir,
            )
        elif state_dict_type == "FULL_STATE_DICT":
            trainer.save_model(cfg.output_dir)
    elif cfg.deepspeed and is_deepspeed_zero3_enabled():
        # Copied over from: https://github.com/huggingface/accelerate/blob/5ae611118057232f441055f7ef9ba0b0f2b8d533/docs/source/usage_guides/deepspeed.md#saving-and-loading
        trainer.accelerator.wait_for_everyone()
        trainer.save_model(cfg.output_dir)

        # the trainer saved a model.safetensors file in the output directory,
        # but it is most likely a proxy model and if so, should be deleted
        maybe_proxy = os.path.exists(os.path.join(cfg.output_dir, "model.safetensors"))
        maybe_sharded = os.path.exists(
            os.path.join(cfg.output_dir, "model.safetensors.index.json")
        )

        if maybe_proxy and maybe_sharded:
            LOG.info(f"Deleting {os.path.join(cfg.output_dir, 'model.safetensors')}")
            LOG.info("This is a proxy model and should be deleted")
            try:
                os.remove(os.path.join(cfg.output_dir, "model.safetensors"))
            except FileNotFoundError:
                pass
    elif cfg.local_rank == 0:
        if cfg.flash_optimum and BetterTransformer:
            model = BetterTransformer.reverse(model)

        if cfg.rl and cfg.adapter and not cfg.rl_adapter_ref_model:
            trainer.model.save_pretrained(
                cfg.output_dir, safe_serialization=safe_serialization
            )
        model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)


def create_model_card(cfg: DictDefault, trainer: Trainer):
    """
    Create a model card for the trained model if needed.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        trainer: The trainer object with model card creation capabilities.
    """
    if not cfg.hub_model_id:
        # Guard since create_model_card may fail if dataset_tags is empty list
        try:
            model_card_kwarg = {
                "model_name": cfg.output_dir.lstrip("./")
                .encode("utf-8")
                .decode("utf-8")
            }

            # We check if we're using a TRL trainer; if so, `dataset_tags` is not consumed.
            rl = cfg.rl is not None or cfg.reward_model or cfg.process_reward_model
            if cfg.datasets is not None and not rl:
                dataset_tags = [
                    d["path"] for d in cfg.datasets if not Path(d["path"]).is_dir()
                ]
                dataset_tags = [d for d in dataset_tags if not d.startswith("https://")]

                if dataset_tags:
                    model_card_kwarg["dataset_tags"] = dataset_tags

            trainer.create_model_card(**model_card_kwarg)
        except (AttributeError, UnicodeDecodeError, OfflineModeIsEnabled):
            pass
    elif cfg.hub_model_id:
        # Defensively push to the hub to ensure the model card is updated
        trainer.push_to_hub()


def save_initial_configs(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    peft_config: PeftConfig | None,
    processor: ProcessorMixin | None,
):
    """
    Save initial configurations before training.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        tokenizer: The tokenizer to save.
        model: The model to save configuration for.
        peft_config: The PEFT configuration to save if applicable.
    """
    # Create output_dir if it doesn't already exist
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_dir():
        os.makedirs(cfg.output_dir, exist_ok=True)

    # Pre-save adapter config so it's available to inspect
    if peft_config:
        LOG.info(f"Pre-saving adapter config to {cfg.output_dir}...")
        peft_config.save_pretrained(cfg.output_dir)

    # Pre-save the tokenizer and model configs
    LOG.info(f"Pre-saving tokenizer to {cfg.output_dir}...")
    tokenizer.save_pretrained(str(output_dir))
    if hasattr(model, "config"):
        LOG.info(f"Pre-saving model config to {cfg.output_dir}...")
        model.config.save_pretrained(str(output_dir))

    if processor:
        LOG.info(f"Pre-saving processor to {cfg.output_dir}...")
        processor.save_pretrained(str(output_dir))


def setup_model_card(cfg: DictDefault):
    """
    Set up the Axolotl badge and add the Axolotl config to the model card if available.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
    """
    badge_markdown = """[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)"""
    transformers.modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n{badge_markdown}"

    if getattr(cfg, "axolotl_config_path"):
        raw_axolotl_cfg = Path(cfg.axolotl_config_path)
        version = importlib.metadata.version("axolotl")
        if raw_axolotl_cfg.is_file():
            transformers.modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n<details><summary>See axolotl config</summary>\n\naxolotl version: `{version}`\n```yaml\n{raw_axolotl_cfg.read_text(encoding='utf-8')}\n```\n\n</details><br>\n"


def handle_untrained_tokens_fix(
    cfg: DictDefault,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    safe_serialization: bool,
):
    """
    Apply fixes for untrained tokens if configured.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        model: The model to apply fixes to.
        tokenizer: The tokenizer for token identification.
        train_dataset: The training dataset to use.
        safe_serialization: Whether to use safe serialization when saving.
    """
    if not cfg.fix_untrained_tokens:
        return

    is_ds_zero3: bool = False
    if os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3":
        is_ds_zero3 = True

    # Check if the `token_ids_to_fix` kwarg exists in the fix_untrained_tokens args
    sig = inspect.signature(fix_untrained_tokens)

    fix_kwargs: Dict[str, Any] = {}
    # If the function has the `token_ids_to_fix` arg, and fix_untrained_tokens is a list
    if "token_ids_to_fix" in sig.parameters and isinstance(
        cfg.fix_untrained_tokens, list
    ):
        fix_kwargs["token_ids_to_fix"] = cfg.fix_untrained_tokens
    if "is_ds_zero3" in sig.parameters:
        fix_kwargs["is_ds_zero3"] = is_ds_zero3

    fix_untrained_tokens(model, tokenizer, train_dataset, **fix_kwargs)

    if cfg.local_rank == 0:
        model.save_pretrained(
            str(Path(cfg.output_dir)), safe_serialization=safe_serialization
        )


def setup_model_and_trainer(cfg: DictDefault, dataset_meta: TrainDatasetMeta) -> tuple[
    HFRLTrainerBuilder | HFCausalTrainerBuilder,
    PeftModel | PreTrainedModel,
    PreTrainedTokenizer,
    PeftConfig | None,
    ProcessorMixin | None,
]:
    """
    Load model, tokenizer, trainer, etc. Helper function to encapsulate the full
    trainer setup.

    Args:
        cfg: The configuration dictionary with training parameters.
        dataset_meta: Object with training, validation datasets and metadata.

    Returns:
        Tuple of:
            - Trainer (Causal or RLHF)
            - Model
            - Tokenizer
            - PEFT config
            - Processor
    """
    # Load tokenizer, processor and model
    model, tokenizer, peft_config, processor = setup_model_and_tokenizer(cfg)

    # Set up reference model for RL if needed
    model_ref = setup_reference_model(cfg, tokenizer)

    # Get datasets from metadata
    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

    # Set up trainer
    trainer = setup_trainer(
        cfg=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        total_num_steps=total_num_steps,
        model_ref=model_ref,
        peft_config=peft_config,
    )

    return (
        trainer,
        model,
        tokenizer,
        peft_config,
        processor,
    )


def train(
    cfg: DictDefault, dataset_meta: TrainDatasetMeta
) -> tuple[PeftModel | PreTrainedModel, PreTrainedTokenizer, Trainer]:
    """
    Train a model on the given dataset.

    Args:
        cfg: The configuration dictionary with training parameters
        dataset_meta: Object with training, validation datasets and metadata

    Returns:
        Tuple of (model, tokenizer) after training
    """
    # Setup model, tokenizer, (causal or RLHF) trainer, etc.
    (
        trainer,
        model,
        tokenizer,
        peft_config,
        processor,
    ) = setup_model_and_trainer(cfg, dataset_meta)

    # Handle untrained tokens if configured
    safe_serialization = cfg.save_safetensors is True
    train_dataset = dataset_meta.train_dataset
    handle_untrained_tokens_fix(
        cfg, model, tokenizer, train_dataset, safe_serialization
    )

    # Additional setup
    save_initial_configs(cfg, tokenizer, model, peft_config, processor)
    setup_signal_handler(cfg, model, safe_serialization)
    setup_model_card(cfg)

    # Execute the training
    resume_from_checkpoint = determine_resume_checkpoint(cfg)
    execute_training(cfg, trainer, resume_from_checkpoint)

    # Save the trained model and cleanup
    save_trained_model(cfg, trainer, model, safe_serialization)
    create_model_card(cfg, trainer)
    if not cfg.use_ray:
        cleanup_distributed()

    return model, tokenizer, trainer
