"""
CLI to post-training quantize a model using torchao
"""

from pathlib import Path
from typing import Union

from transformers import AutoModelForCausalLM

from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.config import load_cfg
from axolotl.loaders import load_tokenizer
from axolotl.utils.logging import get_logger
from axolotl.utils.quantization import TorchIntDType, quantize_model_for_ptq

LOG = get_logger(__name__)


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

    if cfg.qat and cfg.quantization:
        raise ValueError(
            "QAT and quantization cannot be used together. Please specify only one of qat or quantization in your config file."
        )

    if cfg.qat:
        quantize_cfg = cfg.qat
    elif cfg.quantization:
        quantize_cfg = cfg.quantization
    else:
        raise ValueError(
            "No quantization configuration found. Please specify either qat or quantization in your config file."
        )

    model_path = cli_args.get("model_path") or cfg.output_dir
    if weight_dtype := cli_args.get("weight_dtype"):
        weight_dtype = TorchIntDType[weight_dtype]
    else:
        weight_dtype = quantize_cfg.weight_dtype
    if activation_dtype := cli_args.get("activation_dtype"):
        activation_dtype = TorchIntDType[activation_dtype]
    else:
        activation_dtype = quantize_cfg.activation_dtype
    group_size = cli_args.get("group_size") or quantize_cfg.group_size
    quantize_embedding = (
        cli_args.get("quantize_embedding") or quantize_cfg.quantize_embedding
    )
    output_dir = cli_args.get("output_dir") or cfg.output_dir

    LOG.info(f"Loading model from {model_path}...")
    tokenizer = load_tokenizer(cfg)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    LOG.info(
        f"Quantizing model with configuration: \n"
        f"\tweight_dtype: {weight_dtype}\n"
        f"\tactivation_dtype: {activation_dtype}\n"
        f"\tgroup_size: {group_size}\n"
        f"\tquantize_embedding: {quantize_embedding}"
    )

    quantize_model_for_ptq(
        model, weight_dtype, group_size, activation_dtype, quantize_embedding
    )

    LOG.info(f"Saving quantized model to: {str(Path(output_dir) / 'quantized')}...")
    model.save_pretrained(
        str(Path(output_dir) / "quantized"),
        safe_serialization=False,
        progressbar=True,
    )
    tokenizer.save_pretrained(
        str(Path(output_dir) / "quantized"),
        safe_serialization=False,
        progressbar=True,
    )
    LOG.info(f"Quantized model saved to: {str(Path(output_dir) / 'quantized')}...")
