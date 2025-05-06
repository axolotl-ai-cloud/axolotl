"""
CLI to post-training quantize a model using torchao
"""

from pathlib import Path
from typing import Union

from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.config import load_cfg
from axolotl.cli.utils import load_model_and_tokenizer
from axolotl.utils.logging import get_logger
from axolotl.utils.quantization import quantize_model_for_ptq

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
    model, _ = load_model_and_tokenizer(cfg=cfg)
    safe_serialization = cfg.save_safetensors is True

    weight_dtype = cli_args.get("weight_dtype") or cfg.qat.weight_dtype
    activation_dtype = cli_args.get("activation_dtype") or cfg.qat.activation_dtype
    group_size = cli_args.get("group_size") or cfg.qat.group_size

    quantize_model_for_ptq(model, weight_dtype, activation_dtype, group_size)

    if cfg.local_rank == 0:
        LOG.info(
            f"Quantized model saved to: {str(Path(cfg.output_dir) / 'quantized')}..."
        )
        model.save_pretrained(
            str(Path(cfg.output_dir) / "quantized"),
            safe_serialization=safe_serialization,
            progressbar=True,
        )
