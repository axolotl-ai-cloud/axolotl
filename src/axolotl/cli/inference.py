"""CLI to run inference on a trained model."""

import importlib
import sys
from pathlib import Path
from threading import Thread
from typing import Union

import fire
import torch
import transformers
from transformers import GenerationConfig, TextIteratorStreamer, TextStreamer

from axolotl.cli.args import InferenceCliArgs
from axolotl.cli.config import load_cfg
from axolotl.cli.utils import load_model_and_tokenizer
from axolotl.cli.utils.diffusion import (
    diffusion_inference,
    launch_diffusion_gradio_ui,
)
from axolotl.integrations.base import PluginManager
from axolotl.telemetry.errors import send_errors
from axolotl.utils.chat_templates import (
    get_chat_template_from_config,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def get_multi_line_input() -> str:
    """
    Gets multi-line input from terminal.

    Returns:
        Possibly multi-line, possibly empty stdin input as a string.
    """
    print("Give me an instruction (Ctrl + D to submit): ")
    print("=" * 80)

    instruction = ""
    for line in sys.stdin:
        instruction += line

    return instruction


@send_errors
def do_inference(
    *,
    cfg: DictDefault,
    cli_args: InferenceCliArgs,
):
    """
    Runs inference on the command line in a loop. User input is accepted, a chat
    template is (optionally) applied, and the model specified in the `axolotl` config is
    used to generate completions according to a default generation config.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Inference-specific CLI arguments.
    """
    model, tokenizer, _ = load_model_and_tokenizer(cfg=cfg, inference=True)
    prompter = cli_args.prompter

    prompter_module = None
    chat_template_str = None
    if prompter:
        prompter_module = getattr(
            importlib.import_module("axolotl.prompters"), prompter
        )
    elif cfg.chat_template:
        chat_template_str = get_chat_template_from_config(
            cfg, ds_cfg=None, tokenizer=tokenizer
        )
    elif cfg.datasets and cfg.datasets[0].type == "chat_template":
        chat_template_str = get_chat_template_from_config(
            cfg=cfg, ds_cfg=cfg.datasets[0], tokenizer=tokenizer
        )

    model = model.to(cfg.device, dtype=cfg.torch_dtype)

    # Detect diffusion mode
    plugin_manager = PluginManager.get_instance()
    is_diffusion = any(
        plugin.__class__.__name__ == "DiffusionPlugin"
        for plugin in plugin_manager.plugins.values()
    )

    if is_diffusion:
        print("=" * 80)
        print("Commands:")
        print(":complete N -> completion mode with N tokens (default 64)")
        print(":mask R     -> random masking with ratio R (0.0–1.0)")

    while True:
        print("=" * 80)
        instruction = get_multi_line_input()
        if not instruction:
            return

        if prompter_module:
            prompt: str = next(
                prompter_module().build_prompt(instruction=instruction.strip("\n"))
            )
        else:
            prompt = instruction.strip()

        if chat_template_str:
            batch = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                return_tensors="pt",
                add_special_tokens=True,
                add_generation_prompt=True,
                chat_template=chat_template_str,
                tokenize=True,
                return_dict=True,
            )
        else:
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        print("=" * 80)
        model.eval()
        with torch.no_grad():
            if is_diffusion:
                diffusion_inference(
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    prompt=prompt,
                    chat_template_str=chat_template_str,
                )
                continue

            generation_config = GenerationConfig(
                repetition_penalty=1.1,
                max_new_tokens=1024,
                temperature=0.9,
                top_p=0.95,
                top_k=40,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            streamer = TextStreamer(tokenizer)
            generated = model.generate(
                inputs=batch["input_ids"].to(cfg.device),
                generation_config=generation_config,
                streamer=streamer,
            )
        print("=" * 80)
        print(tokenizer.decode(generated["sequences"].cpu().tolist()[0]))


@send_errors
def do_inference_gradio(
    *,
    cfg: DictDefault,
    cli_args: InferenceCliArgs,
):
    """
    Runs inference in a Gradio interface. User input is accepted, a chat template is
    (optionally) applied, and the model specified in the `axolotl` config is used to
    generate completions according to a default generation config.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Inference-specific CLI arguments.
    """
    import gradio as gr

    model, tokenizer, _ = load_model_and_tokenizer(cfg=cfg, inference=True)
    prompter = cli_args.prompter

    prompter_module = None
    chat_template_str = None
    if prompter:
        prompter_module = getattr(
            importlib.import_module("axolotl.prompters"), prompter
        )
    elif cfg.chat_template:
        chat_template_str = get_chat_template_from_config(
            cfg, ds_cfg=None, tokenizer=tokenizer
        )
    elif cfg.datasets and cfg.datasets[0].type == "chat_template":
        chat_template_str = get_chat_template_from_config(
            cfg=cfg, ds_cfg=cfg.datasets[0], tokenizer=tokenizer
        )

    model = model.to(cfg.device, dtype=cfg.torch_dtype)

    # Detect diffusion mode
    plugin_manager = PluginManager.get_instance()
    is_diffusion = any(
        plugin.__class__.__name__ == "DiffusionPlugin"
        for plugin in plugin_manager.plugins.values()
    )

    if is_diffusion:
        launch_diffusion_gradio_ui(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            prompter_module=prompter_module,
            chat_template_str=chat_template_str,
        )
        return

    def generate(instruction):
        if not instruction:
            return
        if prompter_module:
            prompt: str = next(
                prompter_module().build_prompt(instruction=instruction.strip("\n"))
            )
        else:
            prompt = instruction.strip()

        if chat_template_str:
            batch = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                return_tensors="pt",
                add_special_tokens=True,
                add_generation_prompt=True,
                chat_template=chat_template_str,
                tokenize=True,
                return_dict=True,
            )
        else:
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        model.eval()
        with torch.no_grad():
            generation_config = GenerationConfig(
                repetition_penalty=1.1,
                max_new_tokens=cfg.get("gradio_max_new_tokens", 1024),
                temperature=cfg.get("gradio_temperature", 0.9),
                top_p=0.95,
                top_k=40,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            streamer = TextIteratorStreamer(tokenizer)
            generation_kwargs = {
                "inputs": batch["input_ids"].to(cfg.device),
                "attention_mask": batch["attention_mask"].to(cfg.device),
                "generation_config": generation_config,
                "streamer": streamer,
            }

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            all_text = ""

            for new_text in streamer:
                all_text += new_text
                yield all_text

    demo = gr.Interface(
        fn=generate,
        inputs="textbox",
        outputs="text",
        title=cfg.get("gradio_title", "Axolotl Gradio Interface"),
    )

    demo.launch(
        footer_links=["gradio", "settings"],
        share=cfg.get("gradio_share", True),
        server_name=cfg.get("gradio_server_name", "127.0.0.1"),
        server_port=cfg.get("gradio_server_port", None),
    )


def do_cli(
    config: Union[Path, str] = Path("examples/"), gradio: bool = False, **kwargs
) -> None:
    """
    Parses axolotl config, CLI args, and calls `do_inference` or `do_inference_gradio`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """

    parsed_cfg = load_cfg(config, inference=True, rl=None, **kwargs)
    parsed_cfg.sample_packing = False
    parser = transformers.HfArgumentParser(InferenceCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if gradio:
        do_inference_gradio(cfg=parsed_cfg, cli_args=parsed_cli_args)
    else:
        do_inference(cfg=parsed_cfg, cli_args=parsed_cli_args)


if __name__ == "__main__":
    fire.Fire(do_cli)
