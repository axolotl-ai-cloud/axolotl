import logging
import math
import os
import random
import signal
import sys
from pathlib import Path

import bitsandbytes as bnb
import fire
import torch
import transformers
import yaml
from attrdict import AttrDefault
from datasets import load_dataset, IterableDataset, Dataset, load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel,
)
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    EarlyStoppingCallback,
)

# add src to the pythonpath so we don't need to pip install this
from transformers.trainer_pt_utils import get_parameter_names

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from axolotl.datasets import TokenizedPromptDataset, ConstantLengthDataset
from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    ShareGPTPromptTokenizingStrategy,
    LLAMA_DEFAULT_PAD_TOKEN,
    GPTeacherPromptTokenizingStrategy,
)
from axolotl.prompters import AlpacaPrompter, GPTeacherPrompter, ShareGPTPrompter

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
DEFAULT_DATASET_PREPARED_PATH = "data/last_run"


def setup_wandb_env_vars(cfg):
    if len(cfg.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        cfg.use_wandb = True
        if cfg.wandb_watch and len(cfg.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = cfg.wandb_watch
        if cfg.wandb_log_model and len(cfg.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = cfg.wandb_log_model
        if cfg.wandb_run_id and len(cfg.wandb_run_id) > 0:
            os.environ["WANDB_RUN_ID"] = cfg.wandb_run_id


def load_model(base_model, model_type, tokenizer_type, cfg, adapter="lora", inference: bool=False):
    if adapter != "lora":
        raise NotImplementedError(f"{adapter} peft adapter not available")
    if "llama" in base_model:
        if cfg.device not in ["mps", "cpu"] and inference is False:
            from axolotl.flash_attn import replace_llama_attn_with_flash_attn
            replace_llama_attn_with_flash_attn()

    torch_dtype = torch.float16 if cfg.load_in_8bit or cfg.fp16 else torch.float32,
    try:
        if "llama" in base_model:
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=cfg.load_in_8bit,
                torch_dtype=torch_dtype,
                device_map=cfg.device_map,
            )
        else:
            model = getattr(transformers, model_type).from_pretrained(
                base_model,
                load_in_8bit=cfg.load_in_8bit,
                torch_dtype=torch_dtype,
                device_map=cfg.device_map,
            )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=cfg.load_in_8bit,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
        )

    try:
        if "llama" in base_model:
            tokenizer = LlamaTokenizer.from_pretrained(model)
        else:
            tokenizer = getattr(transformers, tokenizer_type).from_pretrained(model)
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.__class__.__name__ in ["LlamaTokenizer", "LlamaTokenizerFast"]:
        tokenizer.pad_token = LLAMA_DEFAULT_PAD_TOKEN

    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg.load_in_8bit:
        model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        model = PeftModel.from_pretrained(model, cfg.lora_model_dir, device_map = cfg.device_map, torch_dtype=torch.float16)
    else:
        model = get_peft_model(model, lora_config)

    if cfg.ddp:
        model.to(f"cuda:{cfg.local_rank}")

    # TODO resume_from_checkpoint handling
    model.print_trainable_parameters()
    return model, tokenizer, lora_config


def choose_device(cfg):
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
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


def check_dataset_labels(dataset, tokenizer):
    from termcolor import colored

    # the dataset is already shuffled, so let's just check the first 5 elements
    for idx in range(5):
        # Get the input_ids, labels, and attention_mask from the dataset
        input_ids = dataset[idx]["input_ids"]
        labels = dataset[idx]["labels"]
        attention_mask = dataset[idx]["attention_mask"]

        # You can compare the input_ids and labels element-wise
        # Remember to ignore positions with IGNORE_TOKEN_ID (if you use it) or attention_mask equal to 0
        colored_tokens = []
        for i, (input_id, label_id, mask) in enumerate(
            zip(input_ids, labels, attention_mask)
        ):
            decoded_input_token = tokenizer.decode(input_id)
            # Choose the color based on whether the label has the ignore value or not
            color = (
                "red" if label_id == -100 else ("yellow" if label_id == 0 else "green")
            )
            colored_token = colored(decoded_input_token, color) + colored(
                f"({label_id}, {mask})", "white"
            )
            colored_tokens.append(colored_token)

        logging.info(" ".join(colored_tokens))
        logging.info("\n\n\n")


def do_inference(cfg, model, tokenizer):
    instruction = "Tell me a joke about dromedaries."
    input = ""
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n".format(instruction=instruction, input=input)
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    model.eval()
    with torch.no_grad():
        generated = model.generate(inputs=batch["input_ids"],
                                   do_sample=True, use_cache=True,
                                   repetition_penalty=1.1,
                                   max_new_tokens=100,
                                   temperature=0.9,
                                   top_p=0.95,
                                   top_k=40,
                                   return_dict_in_generate=True,
                                   output_attentions=False,
                                   output_hidden_states=False,
                                   output_scores=False)
    print(tokenizer.decode(generated['sequences'].cpu().tolist()[0]))


def choose_config(path: Path):
    yaml_files = [file for file in path.glob("*.yml")]

    if not yaml_files:
        raise ValueError("No YAML config files found in the specified directory. Are you using a .yml extension?")

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


def setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer):
    total_num_steps = int(
        math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
    )
    warmup_steps = min(int(0.03 * total_num_steps), 100)
    logging_steps = min(int(0.005 * total_num_steps), 10)
    save_steps = eval_steps = min(int(0.05 * total_num_steps), 200)

    training_arguments_kwargs = {}
    if cfg.bf16 == "full":
        training_arguments_kwargs["bf16_full_eval"] = True
    else:
        training_arguments_kwargs["bf16"] = cfg.bf16
    training_arguments_kwargs["tf32"] = cfg.tf32
    training_arguments_kwargs["warmup_steps"] = warmup_steps
    training_arguments_kwargs["logging_steps"] = logging_steps

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        evaluation_strategy="steps" if cfg.val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if cfg.val_set_size > 0 else None,
        save_steps=save_steps,
        output_dir=cfg.output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if cfg.val_set_size > 0 else False,
        ddp_find_unused_parameters=False if cfg.ddp else None,
        group_by_length=cfg.group_by_length,
        report_to="wandb" if cfg.use_wandb else None,
        run_name=cfg.wandb_run_id if cfg.use_wandb else None,
        gradient_checkpointing=cfg.gradient_checkpointing,
        **training_arguments_kwargs,
    )

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )

    # TODO optionally use torch.optim.OneCycleLR
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        adam_bnb_optim,
        training_args.warmup_steps,
        total_num_steps,
    )

    trainer_kwargs = {}
    if cfg.early_stopping_patience:
        early_stop_cb = EarlyStoppingCallback(
            cfg.early_stopping_patience,
        )
        trainer_kwargs["callbacks"] = [early_stop_cb]

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        optimizers=(adam_bnb_optim, lr_scheduler),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        **trainer_kwargs,
    )

    return trainer

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
        if k in cfg_keys:
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
    cfg.ddp = cfg.world_size != 1
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

    # Load the model and tokenizer
    model, tokenizer, lora_config = load_model(
        cfg.base_model, cfg.model_type, cfg.tokenizer_type, cfg, adapter=cfg.adapter, inference=("inference" in kwargs)
    )

    if "inference" in kwargs:
        do_inference(cfg, model, tokenizer)
        return

    if cfg.dataset_prepared_path and any(Path(cfg.dataset_prepared_path).glob("*")):
        logging.info("Loading prepared dataset from disk...")
        dataset = load_from_disk(cfg.dataset_prepared_path)
        logging.info("Prepared dataset loaded from disk...")
    else:
        datasets = []
        for d in cfg.datasets:
            if Path(d.path).exists():
                ds: IterableDataset = load_dataset(
                    "json", data_files=d.path, streaming=True, split=None
                )
            # elif d.name and d.path:
            #     # TODO load from huggingface hub, but it only seems to support arrow or parquet atm
            #     ds = load_dataset(d.path, split=None, data_files=d.name)
            else:
                raise Exception("unhandled dataset load")

            if d.type == "alpaca":
                ds_strategy = AlpacaPromptTokenizingStrategy(
                    AlpacaPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "gpteacher":
                ds_strategy = GPTeacherPromptTokenizingStrategy(
                    GPTeacherPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "sharegpt":
                ds_strategy = ShareGPTPromptTokenizingStrategy(
                    ShareGPTPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
        constant_len_dataset = ConstantLengthDataset(
            tokenizer, datasets, seq_length=cfg.sequence_len
        )
        dataset = Dataset.from_list(
            [_ for _ in constant_len_dataset]
        ).train_test_split(test_size=cfg.val_set_size, shuffle=True, seed=42)

        if cfg.local_rank == 0:
            logging.info("Saving prepared dataset to disk...")
            if cfg.dataset_prepared_path:
                dataset.save_to_disk(cfg.dataset_prepared_path)
            else:
                dataset.save_to_disk(DEFAULT_DATASET_PREPARED_PATH)

        if prepare_ds_only:
            logging.info("Finished preparing dataset. Exiting...")
            return

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if cfg.debug:
        check_dataset_labels(
            train_dataset.select([random.randrange(0, len(train_dataset) - 1)]),
            tokenizer,
        )

    trainer = setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer)

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        logging.info("Compiling torch model")
        model = torch.compile(model)

    # go ahead and presave, so we have the adapter config available to inspect
    logging.info(f"Pre-saving adapter config to {cfg.output_dir}")
    lora_config.save_pretrained(cfg.output_dir)

    # In case we want to stop early with ctrl+c, this is a nice to have to save the pretrained model
    if cfg.local_rank == 0:
        signal.signal(
            signal.SIGINT,
            lambda signal, frame: (model.save_pretrained(cfg.output_dir), exit(0)),
        )

    logging.info("Starting trainer...")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    if cfg.local_rank == 0:
        # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
        logging.info(f"Training Completed!!! Saving pre-trained model to {cfg.output_dir}")
        model.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    fire.Fire(train)
