"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import importlib
import logging
import os
import random
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fire
import torch
import yaml
from transformers import GenerationConfig, TextStreamer

from axolotl.utils.data import load_prepare_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer

# add src to the pythonpath so we don't need to pip install this
from axolotl.utils.tokenization import check_dataset_labels
from axolotl.utils.trainer import setup_trainer
from axolotl.utils.validation import validate_config
from axolotl.utils.wandb import setup_wandb_env_vars

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
DEFAULT_DATASET_PREPARED_PATH = "last_run_prepared"


def choose_device(cfg):
    def get_device():
        try:
            if torch.cuda.is_available():
                return f"cuda:{cfg.local_rank}"

            if torch.backends.mps.is_available():
                return "mps"

            raise SystemError("No CUDA/mps device found")
        except Exception:  # pylint: disable=broad-exception-caught
            return "cpu"

    cfg.device = get_device()
    if cfg.device_map != "auto":
        if cfg.device.startswith("cuda"):
            cfg.device_map = {"": cfg.local_rank}
        else:
            cfg.device_map = {"": cfg.device}


def get_multi_line_input() -> Optional[str]:
    print("Give me an instruction (Ctrl + D to finish): ")
    instruction = ""
    for line in sys.stdin:
        instruction += line  # pylint: disable=consider-using-join
    # instruction = pathlib.Path("/proc/self/fd/0").read_text()
    return instruction


def do_inference(cfg, model, tokenizer, prompter="AlpacaPrompter"):
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

    if cfg.landmark_attention:
        from axolotl.monkeypatch.llama_landmark_attn import set_model_mem_id

        set_model_mem_id(model, tokenizer)
        model.set_mem_cache_args(
            max_seq_len=255, mem_freq=50, top_k=5, max_cache_size=None
        )

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


def choose_config(path: Path):
    yaml_files = list(path.glob("*.yml"))

    if not yaml_files:
        raise ValueError(
            "No YAML config files found in the specified directory. Are you using a .yml extension?"
        )

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


def train(
    config: Path = Path("configs/"),
    prepare_ds_only: bool = False,
    **kwargs,
):
    if Path(config).is_dir():
        config = choose_config(config)

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

    validate_config(cfg)

    # setup some derived config / hyperparams
    cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
        cfg.batch_size // cfg.micro_batch_size
    )
    cfg.batch_size = (
        cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps
    )
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    choose_device(cfg)
    cfg.ddp = cfg.ddp if cfg.ddp is not None else cfg.world_size != 1
    if cfg.ddp:
        cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        cfg.batch_size = cfg.batch_size * cfg.world_size

    setup_wandb_env_vars(cfg)
    if cfg.device == "mps":
        cfg.load_in_8bit = False
        cfg.tf32 = False
        if cfg.bf16:
            cfg.fp16 = True
        cfg.bf16 = False

    if cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # load the tokenizer first
    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    logging.info(f"loading tokenizer... {tokenizer_config}")
    tokenizer = load_tokenizer(tokenizer_config, cfg.tokenizer_type, cfg)

    if (
        check_not_in(["shard", "merge_lora"], kwargs) and not cfg.inference
    ):  # don't need to load dataset for these
        train_dataset, eval_dataset = load_prepare_datasets(
            tokenizer, cfg, DEFAULT_DATASET_PREPARED_PATH
        )

    if cfg.debug or "debug" in kwargs:
        logging.info("check_dataset_labels...")
        check_dataset_labels(
            train_dataset.select(
                [random.randrange(0, len(train_dataset) - 1) for _ in range(5)]  # nosec
            ),
            tokenizer,
        )

    if prepare_ds_only:
        logging.info("Finished preparing dataset. Exiting...")
        return

    # Load the model and tokenizer
    logging.info("loading model and peft_config...")
    model, peft_config = load_model(
        cfg.base_model,
        cfg.base_model_config,
        cfg.model_type,
        tokenizer,
        cfg,
        adapter=cfg.adapter,
    )

    if "merge_lora" in kwargs and cfg.adapter is not None:
        logging.info("running merge of LoRA with base model")
        model = model.merge_and_unload()
        model.to(dtype=torch.float16)

        if cfg.local_rank == 0:
            logging.info("saving merged model")
            model.save_pretrained(str(Path(cfg.output_dir) / "merged"))
        return

    if cfg.inference:
        logging.info("calling do_inference function")
        inf_kwargs: Dict[str, Any] = {}
        if "prompter" in kwargs:
            if kwargs["prompter"] == "None":
                inf_kwargs["prompter"] = None
            else:
                inf_kwargs["prompter"] = kwargs["prompter"]
        do_inference(cfg, model, tokenizer, **inf_kwargs)
        return

    if "shard" in kwargs:
        model.save_pretrained(cfg.output_dir)
        return

    trainer = setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer)

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        logging.info("Compiling torch model")
        model = torch.compile(model)

    # go ahead and presave, so we have the adapter config available to inspect
    if peft_config:
        logging.info(f"Pre-saving adapter config to {cfg.output_dir}")
        peft_config.save_pretrained(cfg.output_dir)

    # In case we want to stop early with ctrl+c, this is a nice to have to save the pretrained model
    if cfg.local_rank == 0:
        signal.signal(
            signal.SIGINT,
            lambda signal, frame: (
                model.save_pretrained(cfg.output_dir),
                sys.exit(0),
            ),
        )

    logging.info("Starting trainer...")
    if cfg.group_by_length:
        logging.info("hang tight... sorting dataset for group_by_length")
    resume_from_checkpoint = cfg.resume_from_checkpoint
    if cfg.resume_from_checkpoint is None and cfg.auto_resume_from_checkpoints:
        possible_checkpoints = [
            str(cp) for cp in Path(cfg.output_dir).glob("checkpoint-*")
        ]
        if len(possible_checkpoints) > 0:
            sorted_paths = sorted(
                possible_checkpoints,
                key=lambda path: int(path.split("-")[-1]),
            )
            resume_from_checkpoint = sorted_paths[-1]
            logging.info(
                f"Using Auto-resume functionality to start with checkpoint at {resume_from_checkpoint}"
            )

    if not Path(cfg.output_dir).is_dir():
        os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logging.info(f"Training Completed!!! Saving pre-trained model to {cfg.output_dir}")

    # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
    # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple processes attempt to write the same file
    if cfg.local_rank == 0:
        model.save_pretrained(cfg.output_dir)

    # trainer.save_model(cfg.output_dir)  # TODO this may be needed for deepspeed to work? need to review another time


if __name__ == "__main__":
    fire.Fire(train)
