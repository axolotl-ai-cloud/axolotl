"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import shutil
import signal
import sys
import typing
import weakref
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict

import torch
import transformers.modelcard
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
from axolotl.integrations.base import PluginManager
from axolotl.loaders import ModelLoader, load_processor, load_tokenizer
from axolotl.telemetry.errors import send_errors
from axolotl.telemetry.manager import TelemetryManager
from axolotl.utils.ctx_managers.sequence_parallel import SequenceParallelContextManager
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import cleanup_distributed
from axolotl.utils.freeze import freeze_layers_except
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import RLType
from axolotl.utils.train import determine_last_checkpoint
from axolotl.utils.trainer import setup_trainer

if typing.TYPE_CHECKING:
    from axolotl.core.builders import HFCausalTrainerBuilder, HFRLTrainerBuilder

LOG = get_logger(__name__)

TELEMETRY_MANAGER = TelemetryManager.get_instance()
PLUGIN_MANAGER = PluginManager.get_instance()


def setup_model_and_tokenizer(
    cfg: DictDefault,
) -> tuple[
    PreTrainedModel, PreTrainedTokenizer, PeftConfig | None, ProcessorMixin | None
]:
    """Load the tokenizer, processor (for multimodal models), and model based on
    configuration.

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

    # Load the model
    LOG.debug("Loading model")

    model_loader = ModelLoader(cfg, tokenizer, processor=processor)
    model, peft_config = model_loader.load()
    if model.generation_config is not None:
        model.generation_config.do_sample = True

    TELEMETRY_MANAGER.send_event(
        event_type="model-load", properties=model.config.to_dict()
    )
    if peft_config:
        TELEMETRY_MANAGER.send_event(
            event_type="peft-config-load", properties=peft_config.to_dict()
        )

    # Apply freezing if specified
    if cfg.unfrozen_parameters:
        freeze_layers_except(model, cfg.unfrozen_parameters)
        if any(
            any(embed in param for embed in ["lm_head", "embed_tokens"])
            for param in cfg.unfrozen_parameters
        ):
            model.enable_input_require_grads()

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
    if cfg.rl and cfg.rl != RLType.ORPO:
        if cfg.adapter and not cfg.rl_adapter_ref_model:
            # use built-in trl autounwrap
            LOG.debug("Passing model_ref: None to RL trainer")
            model_ref = None  # explicit setting to None
        else:
            reference_model: bool = True
            if cfg.rl == RLType.GRPO and cfg.trl.beta == 0:
                reference_model = False
            # load the model again for model_ref/baseline
            model_loader = ModelLoader(cfg, tokenizer, reference_model=reference_model)
            model_ref, _ = model_loader.load()
    return model_ref


def setup_signal_handler(cfg: DictDefault, model: PreTrainedModel):
    """
    Set up signal handler for graceful termination.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        model: The model to save on termination
    """
    # ray workers don't have access to this signal
    if cfg.local_rank == 0 and not cfg.use_ray:

        def terminate_handler(_, __, model_weakref):
            if model_weakref() is not None:
                _model = model_weakref()
                _model.save_pretrained(cfg.output_dir)

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
    with ExitStack() as stack:
        # Define the context managers to use
        if cfg.flash_optimum:
            stack.enter_context(
                torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True,
                )
            )

        if cfg.context_parallel_size > 1:
            models = [trainer.model]
            if hasattr(trainer, "ref_model") and trainer.ref_model:
                models.append(trainer.ref_model)

            stack.enter_context(
                SequenceParallelContextManager(
                    models=models,
                    context_parallel_size=cfg.context_parallel_size,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    ring_attn_func=cfg.ring_attn_func,
                    heads_k_stride=cfg.heads_k_stride,
                    gather_outputs=cfg.rl is RLType.GRPO,
                    device_mesh=trainer.accelerator.torch_device_mesh,
                )
            )

        # TODO: disabling for now as not compatible with FSDP2 + torchao low bit optimizers
        # if cfg.bf16:
        #     torch.set_default_dtype(torch.bfloat16)

        LOG.info("Starting trainer...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        PLUGIN_MANAGER.post_train(cfg, trainer.model)


def save_trained_model(
    cfg: DictDefault,
    trainer: Any,
    model: PreTrainedModel,
):
    """
    Save the trained model according to configuration and training setup.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        trainer: The trainer object.
        model: The trained model to save.
    """
    LOG.info(f"Training completed! Saving trained model to {cfg.output_dir}.")

    # Post training module hooks
    for name, module in model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(model, name)

    # handle QAT
    if cfg.qat:
        from axolotl.utils.quantization import convert_qat_model

        convert_qat_model(
            model,
            quantize_embedding=cfg.qat.quantize_embedding,
        )
        LOG.info(
            "QAT usage note: please ensure you quantize your model fine-tuned using QAT by running `axolotl quantize`"
            " with the same config which you used for training."
        )
    # Handle ReLoRA early return case
    if cfg.relora:
        if cfg.adapter == "lora" and not (cfg.load_in_4bit or cfg.load_in_8bit):
            model = model.merge_and_unload()
        else:
            # final model weights have already been saved by `ReLoRACallback.on_train_end`
            return

    if trainer.is_fsdp_enabled or cfg.fsdp_config:
        if cfg.fsdp_config or cfg.fsdp:
            if cfg.fsdp_config.final_state_dict_type:
                state_dict_type = cfg.fsdp_config.final_state_dict_type
            else:
                state_dict_type = cfg.fsdp_config.state_dict_type
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type(state_dict_type)
        trainer.save_model(cfg.output_dir)  # only handles FULL_STATE_DICT
        if state_dict_type == "SHARDED_STATE_DICT":
            LOG.info(
                "The final model was saved with a sharded state dict. Please ensure you merge "
                "the sharded weights with `merge-sharded-fsdp-weights`."
            )
            checkpoint_dir = determine_last_checkpoint(cfg, update=False)
            if (
                not (Path(cfg.output_dir) / "model.safetensors.index.json").exists()
                and checkpoint_dir
            ):
                # import here to prevent circular import
                from axolotl.cli.merge_sharded_fsdp_weights import merge_fsdp_weights

                fsdp_dir = Path(checkpoint_dir) / "pytorch_model_fsdp_0"
                merged_path = str(Path(cfg.output_dir) / "merged")
                merge_fsdp_weights(
                    checkpoint_dir=str(fsdp_dir),
                    output_path=merged_path,
                )
                trainer.accelerator.wait_for_everyone()
                if trainer.accelerator.is_main_process:
                    # move all files in merged_path to cfg.output_dir
                    for merged_file in Path(merged_path).iterdir():
                        if (Path(cfg.output_dir) / merged_file.name).exists():
                            (Path(cfg.output_dir) / merged_file.name).unlink()
                        shutil.move(str(merged_file), cfg.output_dir)
                    shutil.rmtree(merged_path)  # remove what should be an empty dir
        # TODO(wing):see https://github.com/huggingface/transformers/pull/40207
        # cleanup the FSDP prefix in the model config.json
        if trainer.accelerator.is_main_process:
            with open(
                Path(cfg.output_dir) / "config.json", "r", encoding="utf-8"
            ) as config_file_io:
                # read the model config as an OrderedDict
                config = json.load(config_file_io, object_pairs_hook=OrderedDict)
                config["architectures"] = [
                    name.lstrip("FSDP") for name in config["architectures"]
                ]
            # write the updated model config back
            with open(
                os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8"
            ) as config_file_io:
                json.dump(config, config_file_io, indent=2)
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
        if cfg.rl and cfg.adapter and not cfg.rl_adapter_ref_model:
            trainer.model.save_pretrained(cfg.output_dir)

        model.save_pretrained(cfg.output_dir)

    if hasattr(cfg, "llmcompressor") and cfg.llmcompressor:
        # TODO: add integration support so this can be implemented completely within the plugin
        from axolotl.integrations.llm_compressor.utils import save_compressed_model

        save_compressed_model(
            model=model,
            output_dir=cfg.output_dir,
            trainer=trainer,
            save_compressed=cfg.llmcompressor.save_compressed,
        )

    LOG.info(f"Model successfully saved to {cfg.output_dir}")


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
    tokenizer.save_pretrained(
        str(Path(cfg.output_dir)), save_jinja_files=cfg.tokenizer_save_jinja_files
    )
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

    if cfg.axolotl_config_path:
        raw_axolotl_cfg = Path(cfg.axolotl_config_path)
        version = importlib.metadata.version("axolotl")
        if raw_axolotl_cfg.is_file():
            transformers.modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n<details><summary>See axolotl config</summary>\n\naxolotl version: `{version}`\n```yaml\n{raw_axolotl_cfg.read_text(encoding='utf-8')}\n```\n\n</details><br>\n"


def handle_untrained_tokens_fix(
    cfg: DictDefault,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
):
    """
    Apply fixes for untrained tokens if configured.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        model: The model to apply fixes to.
        tokenizer: The tokenizer for token identification.
        train_dataset: The training dataset to use.
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
        model.save_pretrained(str(Path(cfg.output_dir)))


def setup_model_and_trainer(
    cfg: DictDefault, dataset_meta: TrainDatasetMeta
) -> tuple[
    "HFRLTrainerBuilder" | "HFCausalTrainerBuilder",
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
    PLUGIN_MANAGER.post_trainer_create(cfg, trainer)

    if cfg.use_ray:
        try:
            import ray.train.huggingface.transformers

            trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
        except ImportError:
            LOG.warning(
                "The Ray integration with Hugging Face Transformers is not available. "
                "To use Ray, install the 'ray[train]' package."
            )

    return (
        trainer,
        model,
        tokenizer,
        peft_config,
        processor,
    )


@send_errors
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
    train_dataset = dataset_meta.train_dataset
    handle_untrained_tokens_fix(cfg, model, tokenizer, train_dataset)

    # Additional setup
    save_initial_configs(cfg, tokenizer, model, peft_config, processor)
    setup_signal_handler(cfg, model)
    setup_model_card(cfg)

    # Execute the training
    resume_from_checkpoint = determine_last_checkpoint(cfg)
    execute_training(cfg, trainer, resume_from_checkpoint)

    # clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save the trained model and cleanup
    save_trained_model(cfg, trainer, model)
    tokenizer.save_pretrained(
        str(Path(cfg.output_dir)), save_jinja_files=cfg.tokenizer_save_jinja_files
    )
    create_model_card(cfg, trainer)
    if not cfg.use_ray:
        cleanup_distributed()
    PLUGIN_MANAGER.post_train(cfg, model)

    return model, tokenizer, trainer
