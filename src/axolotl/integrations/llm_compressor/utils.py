"""Utilities for llmcompressor integration with axolotl."""

from typing import Union

from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
)
from transformers import PreTrainedModel, Trainer


def save_compressed_model(
    model: PreTrainedModel,
    output_dir: Union[str, bytes],
    trainer: Trainer,
    save_compressed: bool = False,
) -> None:
    """
    Synchronize processes, apply compression hooks, and save the model.

    Args:
        model (PreTrainedModel): The model to be saved.
        output_dir (str or bytes): Path where the model files will be written.
        trainer (Trainer): Hugging Face Trainer for process synchronization.
        save_compressed (bool): Write compressed tensors if True.
    """
    trainer.accelerator.wait_for_everyone()

    # Only the main process writes the files
    if not trainer.accelerator.is_main_process:
        return

    modify_save_pretrained(model)
    model.save_pretrained(
        output_dir,
        save_compressed=save_compressed,
        skip_sparsity_compression_stats=not save_compressed,
    )
