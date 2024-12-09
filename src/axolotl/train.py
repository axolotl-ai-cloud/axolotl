"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import os
import signal
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import transformers.modelcard
from accelerate.logging import get_logger
from accelerate.utils import save_fsdp_model
from datasets import Dataset
from peft import PeftModel
from pkg_resources import get_distribution  # type: ignore
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from axolotl.common.cli import TrainerCliArgs
from axolotl.core.tokenizer_utils import fix_untrained_tokens
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.freeze import freeze_layers_except
from axolotl.utils.models import load_model, load_processor, load_tokenizer
from axolotl.utils.trainer import setup_trainer

try:
    from optimum.bettertransformer import BetterTransformer
except ImportError:
    BetterTransformer = None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = get_logger("axolotl.train")


@dataclass
class TrainDatasetMeta:
    """
    dataclass to capture the dataset specific options for training
    """

    train_dataset: Dataset
    eval_dataset: Optional[Dataset] = None
    total_num_steps: Optional[int] = None


def train(
    *, cfg: DictDefault, cli_args: TrainerCliArgs, dataset_meta: TrainDatasetMeta
) -> Tuple[Union[PeftModel, PreTrainedModel], PreTrainedTokenizer]:
    # enable expandable segments for cuda allocation to improve VRAM usage
    torch_version = torch.__version__.split(".")
    torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
    if torch_major == 2 and torch_minor >= 2:
        if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None:
            os.environ[
                "PYTORCH_CUDA_ALLOC_CONF"
            ] = "expandable_segments:True,roundup_power2_divisions:16"

    # load the tokenizer first
    LOG.debug(
        f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}",
        main_process_only=True,
    )
    tokenizer = load_tokenizer(cfg)
    processor = None
    if cfg.is_multimodal:
        processor = load_processor(cfg, tokenizer)

    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

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
    resume_from_checkpoint = cfg.resume_from_checkpoint

    # Load the model and tokenizer
    msg = "loading model"
    if cfg.adapter:
        msg += " and peft_config..."
    LOG.debug(msg)
    model, peft_config = load_model(
        cfg, tokenizer, processor=processor, inference=cli_args.inference
    )
    if model.generation_config is not None:
        model.generation_config.do_sample = True

    model_ref = None
    if cfg.rl and cfg.rl != "orpo":
        if cfg.adapter and not cfg.rl_adapter_ref_model:
            # use built-in trl autounwrap
            LOG.debug("Passing model_ref: None to RL trainer")
            model_ref = None  # explicit setting to None
        else:
            # load the model again for model_ref/baseline
            model_ref, _ = load_model(
                cfg, tokenizer, inference=cli_args.inference, reference_model=True
            )

    safe_serialization = cfg.save_safetensors is True

    if cfg.unfrozen_parameters:
        freeze_layers_except(model, cfg.unfrozen_parameters)

    trainer = setup_trainer(
        cfg,
        train_dataset,
        eval_dataset,
        (model, model_ref, peft_config),
        tokenizer,
        processor,
        total_num_steps,
    )

    if cfg.fix_untrained_tokens:
        fix_untrained_tokens(model, tokenizer, train_dataset)
        if cfg.local_rank == 0:
            model.save_pretrained(
                str(Path(cfg.output_dir)), safe_serialization=safe_serialization
            )

    # go ahead and presave, so we have the adapter config available to inspect
    if peft_config:
        LOG.info(f"Pre-saving adapter config to {cfg.output_dir}")
        peft_config.save_pretrained(cfg.output_dir)
    # additionally presave the tokenizer and model configs
    if not Path(cfg.output_dir).is_dir():
        os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer.save_pretrained(str(Path(cfg.output_dir)))
    if hasattr(model, "config"):
        model.config.save_pretrained(str(Path(cfg.output_dir)))

    # In case we want to stop early with ctrl+c, this is a nice to have to save the pretrained model
    if cfg.local_rank == 0:

        def terminate_handler(_, __, model_weakref):
            if model_weakref() is not None:
                _model = model_weakref()
                if cfg.flash_optimum and BetterTransformer:
                    _model = BetterTransformer.reverse(_model)
                _model.save_pretrained(
                    cfg.output_dir, safe_serialization=safe_serialization
                )
            sys.exit(0)

        _model_weakref = weakref.ref(model)
        signal.signal(
            signal.SIGINT,
            lambda signum, frame: terminate_handler(signum, frame, _model_weakref),
        )

    badge_markdown = """[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)"""
    transformers.modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n{badge_markdown}"

    if getattr(cfg, "axolotl_config_path"):
        raw_axolotl_cfg = Path(cfg.axolotl_config_path)
        version = get_distribution("axolotl").version
        if raw_axolotl_cfg.is_file():
            transformers.modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n<details><summary>See axolotl config</summary>\n\naxolotl version: `{version}`\n```yaml\n{raw_axolotl_cfg.read_text(encoding='utf-8')}\n```\n\n</details><br>\n"

    LOG.info("Starting trainer...")
    if cfg.group_by_length:
        LOG.info("hang tight... sorting dataset for group_by_length")

    pretrain_hooks(cfg, trainer)
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
    post_train_hooks(cfg, trainer)

    LOG.info(f"Training Completed!!! Saving pre-trained model to {cfg.output_dir}")

    # post training
    for name, module in model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(model, name)  # pylint: disable=protected-access

    state_dict_type = "FULL_STATE_DICT"
    if trainer.is_fsdp_enabled:
        if cfg.fsdp_final_state_dict_type:
            state_dict_type = cfg.fsdp_final_state_dict_type
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(state_dict_type)
        LOG.info(f"Set FSDP state dict type to {state_dict_type} for saving.")

    if cfg.relora_steps:
        if cfg.adapter == "lora" and not (cfg.load_in_4bit or cfg.load_in_8bit):
            model = model.merge_and_unload()
        else:
            # final model weights have already been saved by `ReLoRACallback.on_train_end`
            return model, tokenizer

    # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
    # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple processes attempt to write the same file
    if cfg.fsdp:
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

    if not cfg.hub_model_id:
        try:
            model_card_kwarg = {
                "model_name": cfg.output_dir.lstrip("./")
                .encode("utf-8")
                .decode("utf-8")
            }
            if cfg.datasets is not None:
                if cfg.rl is not None or cfg.reward_model:
                    dataset_tags = [
                        d["path"] for d in cfg.datasets if not Path(d["path"]).is_dir()
                    ]
                    if dataset_tags:
                        # guard as create_model_card may fail if dataset_tags is empty list
                        model_card_kwarg["dataset_name"] = dataset_tags
                else:
                    dataset_tags = [
                        d["path"] for d in cfg.datasets if not Path(d["path"]).is_dir()
                    ]
                    if dataset_tags:
                        # guard as create_model_card may fail if dataset_tags is empty list
                        model_card_kwarg["dataset_tags"] = dataset_tags

            trainer.create_model_card(**model_card_kwarg)
        except (AttributeError, UnicodeDecodeError):
            pass
    elif cfg.hub_model_id:
        # defensively push to the hub to ensure the model card is updated
        trainer.push_to_hub()

    return model, tokenizer


def pretrain_hooks(_cfg, _trainer):
    """
    Run hooks right before kicking off the training
    :param cfg:
    :param trainer:
    :return:
    """


def post_train_hooks(_cfg, _trainer):
    """
    Run hooks right after training completes
    :param cfg:
    :param trainer:
    :return:
    """
