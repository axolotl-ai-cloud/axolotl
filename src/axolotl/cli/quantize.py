"""
CLI to post-training quantize a model using torchao
"""

from pathlib import Path
from typing import Union

from transformers import AutoConfig, AutoModelForCausalLM, TorchAoConfig

from axolotl.cli.config import load_cfg
from axolotl.loaders import load_processor, load_tokenizer
from axolotl.utils.logging import get_logger
from axolotl.utils.quantization import (
    TorchAOQuantDType,
    get_quantization_config,
    quantization_config_to_str,
    quantize_model,
)

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

    model_path = cli_args.get("base_model") or cfg.output_dir
    if weight_dtype := cli_args.get("weight_dtype"):
        weight_dtype = TorchAOQuantDType.from_string(weight_dtype)
    else:
        weight_dtype = quantize_cfg.weight_dtype
    if activation_dtype := cli_args.get("activation_dtype"):
        activation_dtype = TorchAOQuantDType.from_string(activation_dtype)
    else:
        activation_dtype = quantize_cfg.activation_dtype
    group_size = cli_args.get("group_size") or quantize_cfg.group_size
    quantize_embedding = (
        cli_args.get("quantize_embedding") or quantize_cfg.quantize_embedding
    )
    output_dir = cli_args.get("output_dir") or cfg.output_dir
    hub_model_id = cli_args.get("hub_model_id") or cfg.hub_model_id

    LOG.info(f"Loading model from {model_path}.")
    tokenizer = load_tokenizer(cfg)

    processor = None
    if cfg.is_multimodal:
        processor = load_processor(cfg, tokenizer)

    config = AutoConfig.from_pretrained(model_path)
    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", dtype=torch_dtype
    )

    LOG.info(
        f"Quantizing model with configuration: \n"
        f"\tweight_dtype: {weight_dtype}\n"
        f"\tactivation_dtype: {activation_dtype}\n"
        f"\tgroup_size: {group_size}\n"
        f"\tquantize_embedding: {quantize_embedding}"
    )

    quantize_model(
        model, weight_dtype, group_size, activation_dtype, quantize_embedding
    )

    quantization_config = get_quantization_config(
        weight_dtype, activation_dtype, group_size
    )

    ao_config = TorchAoConfig(
        quant_type=quantization_config,
        include_input_output_embeddings=quantize_embedding,
    )
    model.config.quantization_config = ao_config

    LOG.info(f"Saving quantized model to: {str(Path(output_dir) / 'quantized')}.")
    model.save_pretrained(
        str(Path(output_dir) / "quantized"),
        progressbar=True,
    )
    tokenizer.save_pretrained(
        str(Path(output_dir) / "quantized"),
        progressbar=True,
        save_jinja_files=cfg.tokenizer_save_jinja_files,
    )

    if processor:
        LOG.info(f"Saving processor to: {str(Path(output_dir) / 'quantized')}.")
        processor.save_pretrained(str(Path(output_dir) / "quantized"))

    if hub_model_id:
        hub_model_id = (
            hub_model_id.rstrip("-")
            + f"-{quantization_config_to_str[type(quantization_config)]}"
        )
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        if processor:
            processor.push_to_hub(hub_model_id)
        LOG.info(f"Quantized model pushed to: {hub_model_id}.")

    LOG.info(f"Quantized model saved to: {str(Path(output_dir) / 'quantized')}.")
