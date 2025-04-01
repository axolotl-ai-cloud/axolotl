"""Module for evaluating models."""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from accelerate.logging import get_logger
from datasets import Dataset
from transformers.trainer import Trainer

from axolotl.logging_config import configure_logging
from axolotl.train import TrainDatasetMeta
from axolotl.utils import set_pytorch_cuda_alloc_conf
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import cleanup_distributed
from axolotl.utils.models import load_model, load_processor, load_tokenizer
from axolotl.utils.trainer import setup_trainer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = get_logger("axolotl.evaluate")


def evaluate_dataset(
    trainer: Trainer, dataset: Dataset, dataset_type: str, flash_optimum: bool = False
) -> Optional[Dict[str, float]]:
    """Helper function to evaluate a single dataset.

    Args:
        trainer: The trainer instance.
        dataset: Dataset to evaluate.
        dataset_type: Type of dataset ('train' or 'eval').
        flash_optimum: Whether to use flash optimum.

    Returns:
        Dictionary of metrics or None if dataset is None.
    """
    if dataset is None:
        return None

    LOG.info(f"Starting {dataset_type} set evaluation...")

    if flash_optimum:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
        ):
            metrics = trainer.evaluate(dataset, metric_key_prefix=dataset_type)
    else:
        metrics = trainer.evaluate(dataset, metric_key_prefix=dataset_type)

    LOG.info(f"{dataset_type.capitalize()} set evaluation completed!")
    LOG.info(f"{dataset_type.capitalize()} Metrics:")
    for key, value in metrics.items():
        LOG.info(f"{key}: {value}")

    return metrics


def evaluate(*, cfg: DictDefault, dataset_meta: TrainDatasetMeta) -> Dict[str, float]:
    """
    Evaluate a model on training and validation datasets.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        dataset_meta: Dataset metadata containing training and evaluation datasets.

    Returns:
        Dictionary mapping metric names to their values.
    """
    # pylint: disable=duplicate-code
    # Enable expandable segments for cuda allocation to improve VRAM usage
    set_pytorch_cuda_alloc_conf()

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

    # Get datasets
    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

    # Load model
    LOG.debug("loading model for evaluation...")
    model, _ = load_model(cfg, tokenizer, processor=processor)

    # Set up trainer
    trainer = setup_trainer(
        cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=(model, None, None),  # No need for model_ref or peft_config
        tokenizer=tokenizer,
        processor=processor,
        total_num_steps=total_num_steps,
    )

    # Evaluate datasets
    all_metrics = {}
    train_metrics = evaluate_dataset(trainer, train_dataset, "train", cfg.flash_optimum)
    eval_metrics = evaluate_dataset(trainer, eval_dataset, "eval", cfg.flash_optimum)

    if train_metrics:
        all_metrics.update(train_metrics)
    if eval_metrics:
        all_metrics.update(eval_metrics)

    # Save metrics to CSV if output directory is specified and we have metrics
    if cfg.output_dir and (train_metrics or eval_metrics):
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = output_dir / "eval_summary.csv"
        with metrics_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["metric", "training", "validation"])

            # Get unique metric names (removing prefixes) from available metrics
            train_metric_names = {
                k.replace("train_", ""): k for k in (train_metrics or {})
            }
            eval_metric_names = {
                k.replace("eval_", ""): k for k in (eval_metrics or {})
            }
            all_metric_names = sorted(
                set(train_metric_names.keys()) | set(eval_metric_names.keys())
            )

            for metric_name in all_metric_names:
                train_value = (
                    train_metrics.get(train_metric_names.get(metric_name, ""), "")
                    if train_metrics
                    else ""
                )
                eval_value = (
                    eval_metrics.get(eval_metric_names.get(metric_name, ""), "")
                    if eval_metrics
                    else ""
                )
                writer.writerow([metric_name, train_value, eval_value])

        LOG.info(f"Evaluation results saved to {metrics_file}")

    del model
    del tokenizer

    cleanup_distributed()

    return all_metrics
