import importlib
import logging
import os
import random
import signal
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import fire
import torch
import yaml
from attrdict import AttrDefault

# add src to the pythonpath so we don't need to pip install this
from axolotl.utils.tokenization import check_dataset_labels
from axolotl.utils.validation import validate_config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from axolotl.utils.data import load_prepare_datasets
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.trainer import setup_trainer
from axolotl.utils.wandb import setup_wandb_env_vars

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
DEFAULT_DATASET_PREPARED_PATH = "last_run_prepared"


def choose_device(cfg):
    def get_device():
        if torch.cuda.is_available():
            return f"cuda:{cfg.local_rank}"
        else:
            try:
                if torch.backends.mps.is_available():
                    return "mps"
            except:
                return "cpu"

    cfg.device = get_device()
    if cfg.device == "cuda":
        cfg.device_map = {"": cfg.local_rank}
    else:
        cfg.device_map = {"": cfg.device}


def get_multi_line_input() -> Optional[str]:
    print("Give me an instruction (Ctrl + D to finish): ")
    instruction = ""
    for line in sys.stdin:
        instruction += line
    # instruction = pathlib.Path("/proc/self/fd/0").read_text()
    return instruction


def do_inference(cfg, model, tokenizer, prompter="AlpacaPrompter"):
    tokenizer.add_special_tokens({"unk_token": "<unk>"})
    tokenizer.add_special_tokens({"bos_token": "<s>"})
    tokenizer.add_special_tokens({"eos_token": "</s>"})

    prompter_module = getattr(importlib.import_module("axolotl.prompters"), prompter)

    while True:
        # support for multiline inputs
        instruction = get_multi_line_input()
        if not instruction:
            return
        prompt: str = next(prompter_module().build_prompt(instruction=instruction))
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        model.eval()
        with torch.no_grad():
            # gc = GenerationConfig()  # TODO swap out and use this
            generated = model.generate(
                inputs=batch["input_ids"].to(cfg.device),
                do_sample=True,
                use_cache=True,
                repetition_penalty=1.1,
                max_new_tokens=100,
                temperature=0.9,
                top_p=0.95,
                top_k=40,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
        print(tokenizer.decode(generated["sequences"].cpu().tolist()[0]))


def choose_config(path: Path):
    yaml_files = [file for file in path.glob("*.yml")]

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
    with open(config, "r") as f:
        cfg: AttrDefault = AttrDefault(lambda: None, yaml.load(f, Loader=yaml.Loader))
    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    cfg_keys = dict(cfg).keys()
    for k in kwargs:
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or cfg.strict is False:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]

    # setup some derived config / hyperparams
    cfg.gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    choose_device(cfg)
    cfg.ddp = cfg.ddp if cfg.ddp is not None else cfg.world_size != 1
    if cfg.ddp:
        cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        cfg.gradient_accumulation_steps = (
            cfg.gradient_accumulation_steps // cfg.world_size
        )
    setup_wandb_env_vars(cfg)
    if cfg.device == "mps":
        cfg.load_in_8bit = False
        cfg.tf32 = False
        if cfg.bf16:
            cfg.fp16 = True
        cfg.bf16 = False

    validate_config(cfg)

    # load the tokenizer first
    logging.info("loading tokenizer...")
    tokenizer = load_tokenizer(
        cfg.base_model_config,
        cfg.tokenizer_type,
        cfg
    )

    if check_not_in(["inference", "shard", "merge_lora"], kwargs):  # don't need to load dataset for these
        train_dataset, eval_dataset = load_prepare_datasets(
            tokenizer, cfg, DEFAULT_DATASET_PREPARED_PATH
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
        inference=("inference" in kwargs),
    )

    if "merge_lora" in kwargs and cfg.adapter is not None:
        logging.info("running merge of LoRA with base model")
        model = model.merge_and_unload()
        model.to(dtype=torch.float16)

        if cfg.local_rank == 0:
            logging.info("saving merged model")
            model.save_pretrained(str(Path(cfg.output_dir) / "merged"))
        return

    if "inference" in kwargs:
        logging.info("calling do_inference function")
        do_inference(cfg, model, tokenizer)
        return

    if "shard" in kwargs:
        model.save_pretrained(cfg.output_dir)
        return

    if cfg.debug:
        logging.info("check_dataset_labels...")
        check_dataset_labels(
            train_dataset.select(
                [random.randrange(0, len(train_dataset) - 1) for i in range(5)]
            ),
            tokenizer,
        )

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
            lambda signal, frame: (model.save_pretrained(cfg.output_dir), exit(0)),
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
                possible_checkpoints, key=lambda path: int(path.split("-")[-1])
            )
            resume_from_checkpoint = sorted_paths[-1]
            logging.info(
                f"Using Auto-resume functionality to start with checkpoint at {resume_from_checkpoint}"
            )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logging.info(f"Training Completed!!! Saving pre-trained model to {cfg.output_dir}")

    # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
    # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple processes attempt to write the same file
    if cfg.local_rank == 0:
        model.save_pretrained(cfg.output_dir)
    # trainer.save_model(cfg.output_dir)  # TODO this may be needed for deepspeed to work? need to review another time


if __name__ == "__main__":
    fire.Fire(train)
