"""CLI to convert a transformers model's attns to diff attns."""
import logging
import warnings
from pathlib import Path
from time import time
from typing import Union

import fire
import torch
import yaml
from colorama import Fore
from dotenv import load_dotenv
from transformers import HfArgumentParser

from axolotl.cli import load_cfg, print_axolotl_text_art
from axolotl.common.cli import ConvertDiffTransformerCliArgs, load_model_and_tokenizer
from axolotl.integrations.differential_transformer.convert import (
    convert_to_diff_attention,
)

LOG = logging.getLogger(__name__)


def test_inference(model, tokenizer, prompt="The quick brown fox"):
    """Run test inference and return generation time"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {
            k: v.to(device=model.device, dtype=torch.long) for k, v in inputs.items()
        }

        start = time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )
        elapsed = time() - start

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        LOG.info("Prompt: %s", prompt)
        LOG.info("Generated: %s", generated_text)
        LOG.info("Generation time: %.2fs", elapsed)

        return elapsed, generated_text

    except Exception as exc:
        LOG.error("Inference failed: %s", str(exc))
        raise


def convert_differential_transformer(cfg, cli_args, config_path):
    debug_info = {}

    # Load model and tokenizer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
        model.to(cfg.device, dtype=cfg.torch_dtype)

    # Log original model info
    LOG.info(
        "Original model config:\n\t- Hidden size: %d\n\t- Num attention heads: %d",
        model.config.hidden_size,
        model.config.num_attention_heads,
    )

    # Test original model
    if cli_args.debug:
        LOG.info("Testing original model...")
        debug_info["orig_time"], debug_info["orig_text"] = test_inference(
            model, tokenizer
        )

    # Convert attention
    LOG.info("Converting to differential attention...")
    try:
        model = convert_to_diff_attention(
            model=model,
            zero_init=cli_args.zero_init,
            sublayer_norm=cli_args.sublayer_norm,
        )
        model.to(cfg.device, dtype=cfg.torch_dtype)
    except Exception as exc:
        LOG.error(Fore.RED + "Conversion failed: %s" + Fore.RESET, str(exc))
        raise

    # Test converted model
    if cli_args.debug:
        LOG.info("Testing converted model...")
        debug_info["conv_time"], debug_info["conv_text"] = test_inference(
            model, tokenizer
        )

    # Save if requested
    if cfg.output_dir:
        # Save model and tokenizer
        LOG.info("Saving converted model to %s", cfg.output_dir)
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        # Modify config to reflect new path / differential attention
        output_config_path = Path(cfg.output_dir) / "axolotl_config.yml"
        LOG.info("Saving updated config to %s", output_config_path)

        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        data["base_model"] = cfg.output_dir
        data["diff_attention"] = True

        with open(output_config_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file)
    else:
        LOG.info("Not saving converted model to disk")
        LOG.info("Pass --output-dir path/to/save to save model")

    if cli_args.debug:
        LOG.info(
            Fore.GREEN
            + "Conversion successful!\n"
            + f"Original generation time: {debug_info['orig_time']:.2f}s\n"
            + f"Converted generation time: {debug_info['conv_time']:.2f}s"
            + Fore.RESET
        )

        if debug_info["orig_text"] == debug_info["conv_text"]:
            LOG.info(
                Fore.GREEN
                + "Generations match!\n"
                + "Model generation:\n"
                + "*" * 50
                + "\n"
                + f"{debug_info['orig_text']}\n"
                + "*" * 50
                + "\n"
                + Fore.RESET
            )
            debug_info["generations_match"] = True
        else:
            message = (
                "Generations do not match.\n"
                + "Original generation:\n"
                + "*" * 50
                + "\n"
                + f"{debug_info['orig_text']}\n"
                + "*" * 50
                + "\n"
                + "Converted generation:\n"
                + "*" * 50
                + "\n"
                + f"{debug_info['conv_text']}\n"
                + "*" * 50
                + "\n"
            )
            debug_info["generations_match"] = False

            if cli_args.zero_init and not cli_args.sublayer_norm:
                LOG.info(Fore.RED + message + Fore.RESET)
                debug_info["match_expected"] = True
            else:
                LOG.info(
                    Fore.YELLOW
                    + message
                    + "However, this is expected since --zero-init"
                    + " and --no-sublayer-norm were not passed."
                    + Fore.RESET
                )
                debug_info["match_expected"] = False

    return model, debug_info


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    print_axolotl_text_art()

    cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser(ConvertDiffTransformerCliArgs)
    cli_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    convert_differential_transformer(cfg, cli_args, config)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
