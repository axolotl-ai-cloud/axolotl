"""Module for evaluating models."""

import os
import sys
from pathlib import Path
from typing import Tuple, Union

import torch
from accelerate.logging import get_logger
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from axolotl.common.cli import TrainerCliArgs
from axolotl.logging_config import configure_logging
from axolotl.train import TrainDatasetMeta
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_processor, load_tokenizer
from axolotl.utils.trainer import setup_trainer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = get_logger("axolotl.eval")


def evaluate(
    *, cfg: DictDefault, cli_args: TrainerCliArgs, dataset_meta: TrainDatasetMeta
) -> Tuple[Union[PeftModel, PreTrainedModel], PreTrainedTokenizer, dict]:
    """
    Evaluate a model on a dataset

    Args:
        cfg: Configuration dictionary
        cli_args: Command line arguments
        dataset_meta: Dataset metadata containing evaluation dataset

    Returns:
        Tuple containing:
        - The model (either PeftModel or PreTrainedModel)
        - The tokenizer
        - Dictionary of evaluation metrics
    """
    # Set up CUDA allocation config if using PyTorch >= 2.2
    torch_version = torch.__version__.split(".")
    torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
    if torch_major == 2 and torch_minor >= 2:
        if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None:
            os.environ[
                "PYTORCH_CUDA_ALLOC_CONF"
            ] = "expandable_segments:True,roundup_power2_divisions:16"

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

    # Get evaluation dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

    # Load model
    LOG.debug("loading model for evaluation...")
    model, _ = load_model(
        cfg, tokenizer, processor=processor, inference=cli_args.inference
    )

    # Set up trainer
    trainer = setup_trainer(
        cfg,
        train_dataset=eval_dataset,  # None # No training dataset needed for evaluation
        eval_dataset=eval_dataset,
        model=(model, None, None),  # No need for model_ref or peft_config
        tokenizer=tokenizer,
        processor=processor,
        total_num_steps=total_num_steps,
    )

    # Run evaluation
    LOG.info("Starting evaluation...")

    if cfg.flash_optimum:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
        ):
            metrics = trainer.evaluate()
    else:
        metrics = trainer.evaluate()

    # Log results
    LOG.info("Evaluation completed!")
    LOG.info("Metrics:")
    for key, value in metrics.items():
        LOG.info(f"{key}: {value}")

    # Save metrics to file if output directory is specified
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = output_dir / "eval_results.txt"
        with metrics_file.open("w", encoding="utf-8") as file:
            for key, value in metrics.items():
                file.write(f"{key} = {value}\n")

        LOG.info(f"Evaluation results saved to {metrics_file}")

    del model
    del tokenizer

    return metrics
