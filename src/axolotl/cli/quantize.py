"""
CLI to post-training quantize a model using torchao
"""

from pathlib import Path
from typing import Union

from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.config import load_cfg
from axolotl.cli.utils import load_model_and_tokenizer
import logging
from axolotl.utils.quantization import quantize_model_for_ptq

LOG = logging.getLogger(__name__)


def do_quantize(
    config: Union[Path, str],
    cli_args: dict,
):
    """
    Quantizes a model's model's weights

    Args:
        config (Union[Path, str]): The path to the config file
        cli_args (dict): Additional command-line arguments
    """
    print_axolotl_text_art()

    cfg = load_cfg(config)
    model, _ = load_model_and_tokenizer(cfg=cfg)
    safe_serialization = cfg.save_safetensors is True

    if cfg.qat:
        quantize_cfg = cfg.qat
    elif cfg.quantization:
        quantize_cfg = cfg.quantization
    else:
        raise ValueError("No quantization configuration found. Please specify either qat or quantization in your config file.")

    weight_dtype = cli_args.get("weight_dtype") or quantize_cfg.weight_dtype
    activation_dtype = cli_args.get("activation_dtype") or quantize_cfg.activation_dtype
    group_size = cli_args.get("group_size") or quantize_cfg.group_size
    quantize_embedding = cli_args.get("quantize_embedding") or quantize_cfg.quantize_embedding

    quantize_model_for_ptq(model, weight_dtype, group_size, activation_dtype, quantize_embedding)

    if cfg.local_rank == 0:
        LOG.info(
            f"Quantized model saved to: {str(Path(cfg.output_dir) / 'quantized')}..."
        )
        model.save_pretrained(
            str(Path(cfg.output_dir) / "quantized"),
            safe_serialization=safe_serialization,
            progressbar=True,
        )
