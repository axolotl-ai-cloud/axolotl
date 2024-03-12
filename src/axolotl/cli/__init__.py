"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import importlib
import json
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
import torch
import yaml

# add src to the pythonpath so we don't need to pip install this
from accelerate.commands.config import config_args
from art import text2art
from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
from transformers import GenerationConfig, TextIteratorStreamer, TextStreamer
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer
from axolotl.logging_config import configure_logging
from axolotl.train import TrainDatasetMeta
from axolotl.utils.config import (
    normalize_cfg_datasets,
    normalize_config,
    validate_config,
)
from axolotl.utils.data import load_prepare_dpo_datasets, prepare_dataset
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
from axolotl.utils.mlflow_ import setup_mlflow_env_vars
from axolotl.utils.models import load_tokenizer
from axolotl.utils.tokenization import check_dataset_labels
from axolotl.utils.trainer import prepare_optim_env
from axolotl.utils.wandb_ import setup_wandb_env_vars

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = logging.getLogger("axolotl.scripts")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def print_axolotl_text_art(suffix=None):
    font = "nancyj"
    ascii_text = "  axolotl"
    if suffix:
        ascii_text += f"  x  {suffix}"
    ascii_art = text2art(ascii_text, font=font)

    if is_main_process():
        print(ascii_art)


def check_remote_config(config: Union[str, Path]):
    # Check if the config is a valid HTTPS URL to a .yml or .yaml file
    if not (isinstance(config, str) and config.startswith("https://")):
        return config  # Return the original value if it's not a valid URL

    filename = os.path.basename(urlparse(config).path)
    temp_dir = tempfile.mkdtemp()

    try:
        response = requests.get(config, timeout=30)
        response.raise_for_status()  # Check for HTTP errors

        content = response.content
        try:
            # Try parsing as JSON first to catch cases where JSON content is mistakenly considered YAML
            json.loads(content)
            # Log a warning but do not raise an error; JSON is technically valid YAML - this can happen when you forget to point to a raw github link
            LOG.warning(
                f"Warning: The content of the file at {config} is JSON, which is technically valid YAML but might not be intended."
            )
        except json.JSONDecodeError:
            # If it's not valid JSON, verify it's valid YAML
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as err:
                raise ValueError(
                    f"Failed to parse the content at {config} as YAML: {err}"
                ) from err

        # Write the content to a file if it's valid YAML (or JSON treated as YAML)
        output_path = Path(temp_dir) / filename
        with open(output_path, "wb") as file:
            file.write(content)
        LOG.info(
            f"Using the following config obtained from {config}:\n\n{content.decode('utf-8')}\n"
        )
        return output_path

    except requests.RequestException as err:
        # This catches all requests-related exceptions including HTTPError
        raise RuntimeError(f"Failed to download {config}: {err}") from err
    except Exception as err:
        # Catch-all for any other exceptions
        raise err


def get_multi_line_input() -> Optional[str]:
    print("Give me an instruction (Ctrl + D to submit): ")
    instruction = ""
    for line in sys.stdin:
        instruction += line  # pylint: disable=consider-using-join
    # instruction = pathlib.Path("/proc/self/fd/0").read_text()
    return instruction


def do_merge_lora(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
):
    model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    safe_serialization = cfg.save_safetensors is True

    LOG.info("running merge of LoRA with base model")
    model = model.merge_and_unload(progressbar=True)
    try:
        model.to(dtype=cfg.torch_dtype)
    except RuntimeError:
        pass
    model.generation_config.do_sample = True

    if cfg.local_rank == 0:
        LOG.info(f"saving merged model to: {str(Path(cfg.output_dir) / 'merged')}")
        model.save_pretrained(
            str(Path(cfg.output_dir) / "merged"),
            safe_serialization=safe_serialization,
            progressbar=True,
        )
        tokenizer.save_pretrained(str(Path(cfg.output_dir) / "merged"))


def do_inference(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
):
    model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    prompter = cli_args.prompter
    default_tokens = {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

    for token, symbol in default_tokens.items():
        # If the token isn't already specified in the config, add it
        if not (cfg.special_tokens and token in cfg.special_tokens):
            tokenizer.add_special_tokens({token: symbol})

    prompter_module = None
    if prompter:
        prompter_module = getattr(
            importlib.import_module("axolotl.prompters"), prompter
        )

    model = model.to(cfg.device, dtype=cfg.torch_dtype)

    while True:
        print("=" * 80)
        # support for multiline inputs
        instruction = get_multi_line_input()
        if not instruction:
            return
        if prompter_module:
            prompt: str = next(
                prompter_module().build_prompt(instruction=instruction.strip("\n"))
            )
        else:
            prompt = instruction.strip()
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        print("=" * 40)
        model.eval()
        with torch.no_grad():
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
        print("=" * 40)
        print(tokenizer.decode(generated["sequences"].cpu().tolist()[0]))


def do_inference_gradio(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
):
    import gradio as gr

    model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    prompter = cli_args.prompter
    default_tokens = {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

    for token, symbol in default_tokens.items():
        # If the token isn't already specified in the config, add it
        if not (cfg.special_tokens and token in cfg.special_tokens):
            tokenizer.add_special_tokens({token: symbol})

    prompter_module = None
    if prompter:
        prompter_module = getattr(
            importlib.import_module("axolotl.prompters"), prompter
        )

    model = model.to(cfg.device, dtype=cfg.torch_dtype)

    def generate(instruction):
        if not instruction:
            return
        if prompter_module:
            # pylint: disable=stop-iteration-return
            prompt: str = next(
                prompter_module().build_prompt(instruction=instruction.strip("\n"))
            )
        else:
            prompt = instruction.strip()
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        model.eval()
        with torch.no_grad():
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
            streamer = TextIteratorStreamer(tokenizer)
            generation_kwargs = {
                "inputs": batch["input_ids"].to(cfg.device),
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
    demo.queue().launch(show_api=False, share=True)


def choose_config(path: Path):
    yaml_files = list(path.glob("*.yml"))

    if not yaml_files:
        raise ValueError(
            "No YAML config files found in the specified directory. Are you using a .yml extension?"
        )

    if len(yaml_files) == 1:
        print(f"Using default YAML file '{yaml_files[0]}'")
        return yaml_files[0]

    print("Choose a YAML file:")
    for idx, file in enumerate(yaml_files):
        print(f"{idx + 1}. {file}")

    chosen_file = None
    while chosen_file is None:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(yaml_files):
                chosen_file = yaml_files[choice - 1]
            else:
                print("Invalid choice. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return chosen_file


def check_not_in(list1: List[str], list2: Union[Dict[str, Any], List[str]]) -> bool:
    return not any(el in list2 for el in list1)


def load_cfg(config: Union[str, Path] = Path("examples/"), **kwargs):
    config = check_remote_config(config)
    if Path(config).is_dir():
        config = choose_config(Path(config))

    # load the config from the yaml file
    with open(config, encoding="utf-8") as file:
        cfg: DictDefault = DictDefault(yaml.safe_load(file))
    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    cfg_keys = cfg.keys()
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or not cfg.strict:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]

    cfg.axolotl_config_path = config

    try:
        device_props = torch.cuda.get_device_properties("cuda")
        gpu_version = "sm_" + str(device_props.major) + str(device_props.minor)
    except:  # pylint: disable=bare-except # noqa: E722
        gpu_version = None

    cfg = validate_config(
        cfg,
        capabilities={
            "bf16": is_torch_bf16_gpu_available(),
            "n_gpu": os.environ.get("WORLD_SIZE", 1),
            "compute_capability": gpu_version,
        },
    )

    prepare_optim_env(cfg)

    normalize_config(cfg)

    normalize_cfg_datasets(cfg)

    setup_wandb_env_vars(cfg)

    setup_mlflow_env_vars(cfg)

    return cfg


def load_datasets(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
) -> TrainDatasetMeta:
    tokenizer = load_tokenizer(cfg)

    train_dataset, eval_dataset, total_num_steps, prompters = prepare_dataset(
        cfg, tokenizer
    )

    if cli_args.debug or cfg.debug:
        LOG.info("check_dataset_labels...")
        check_dataset_labels(
            train_dataset.select(
                [
                    random.randrange(0, len(train_dataset) - 1)  # nosec
                    for _ in range(cli_args.debug_num_examples)
                ]
            ),
            tokenizer,
            num_examples=cli_args.debug_num_examples,
            text_only=cli_args.debug_text_only,
        )

        LOG.info("printing prompters...")
        for prompter in prompters:
            LOG.info(prompter)

    return TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        total_num_steps=total_num_steps,
    )


def load_rl_datasets(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,  # pylint: disable=unused-argument
) -> TrainDatasetMeta:
    train_dataset, eval_dataset = load_prepare_dpo_datasets(cfg)
    total_num_steps = int(
        math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
    )

    return TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        total_num_steps=total_num_steps,
    )


def check_accelerate_default_config():
    if Path(config_args.default_yaml_config_file).exists():
        LOG.warning(
            f"accelerate config file found at {config_args.default_yaml_config_file}. This can lead to unexpected errors"
        )


def check_user_token():
    # Skip check if HF_HUB_OFFLINE is set to True
    if os.getenv("HF_HUB_OFFLINE") == "1":
        LOG.info(
            "Skipping HuggingFace token verification because HF_HUB_OFFLINE is set to True. Only local files will be used."
        )
        return True

    # Verify if token is valid
    api = HfApi()
    try:
        user_info = api.whoami()
        return bool(user_info)
    except LocalTokenNotFoundError:
        LOG.warning(
            "Error verifying HuggingFace token. Remember to log in using `huggingface-cli login` and get your access token from https://huggingface.co/settings/tokens if you want to use gated models or datasets."
        )
        return False
