# import logging
# import os
# import random
# import signal
# import sys
# from pathlib import Path

# import fire
# import torch
# import yaml
# from addict import Dict

# from peft import set_peft_model_state_dict, get_peft_model_state_dict

# # add src to the pythonpath so we don't need to pip install this
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# src_dir = os.path.join(project_root, "src")
# sys.path.insert(0, src_dir)

# from axolotl.utils.data import load_prepare_datasets
# from axolotl.utils.models import load_model
# from axolotl.utils.trainer import setup_trainer
# from axolotl.utils.wandb import setup_wandb_env_vars

# logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# def choose_device(cfg):
#     def get_device():
#         if torch.cuda.is_available():
#             return "cuda"
#         else:
#             try:
#                 if torch.backends.mps.is_available():
#                     return "mps"
#             except:
#                 return "cpu"

#     cfg.device = get_device()
#     if cfg.device == "cuda":
#         cfg.device_map = {"": cfg.local_rank}
#     else:
#         cfg.device_map = {"": cfg.device}


# def choose_config(path: Path):
#     yaml_files = [file for file in path.glob("*.yml")]

#     if not yaml_files:
#         raise ValueError(
#             "No YAML config files found in the specified directory. Are you using a .yml extension?"
#         )

#     print("Choose a YAML file:")
#     for idx, file in enumerate(yaml_files):
#         print(f"{idx + 1}. {file}")

#     chosen_file = None
#     while chosen_file is None:
#         try:
#             choice = int(input("Enter the number of your choice: "))
#             if 1 <= choice <= len(yaml_files):
#                 chosen_file = yaml_files[choice - 1]
#             else:
#                 print("Invalid choice. Please choose a number from the list.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")

#     return chosen_file


# def save_latest_checkpoint_as_lora(
#     config: Path = Path("configs/"),
#     prepare_ds_only: bool = False,
#     **kwargs,
# ):
#     if Path(config).is_dir():
#         config = choose_config(config)

#     # load the config from the yaml file
#     with open(config, "r") as f:
#         cfg: Dict = Dict(lambda: None, yaml.load(f, Loader=yaml.Loader))
#     # if there are any options passed in the cli, if it is something that seems valid from the yaml,
#     # then overwrite the value
#     cfg_keys = dict(cfg).keys()
#     for k in kwargs:
#         if k in cfg_keys:
#             # handle booleans
#             if isinstance(cfg[k], bool):
#                 cfg[k] = bool(kwargs[k])
#             else:
#                 cfg[k] = kwargs[k]

#     # setup some derived config / hyperparams
#     cfg.gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
#     cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
#     cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     assert cfg.local_rank == 0, "Run this with only one device!"

#     choose_device(cfg)
#     cfg.ddp = False

#     if cfg.device == "mps":
#         cfg.load_in_8bit = False
#         cfg.tf32 = False
#         if cfg.bf16:
#             cfg.fp16 = True
#         cfg.bf16 = False

#     # Load the model and tokenizer
#     logging.info("loading model, tokenizer, and lora_config...")
#     model, tokenizer, lora_config = load_model(
#         cfg.base_model,
#         cfg.base_model_config,
#         cfg.model_type,
#         cfg.tokenizer_type,
#         cfg,
#         adapter=cfg.adapter,
#         inference=True,
#     )

#     model.config.use_cache = False

#     if torch.__version__ >= "2" and sys.platform != "win32":
#         logging.info("Compiling torch model")
#         model = torch.compile(model)

#     possible_checkpoints = [str(cp) for cp in Path(cfg.output_dir).glob("checkpoint-*")]
#     if len(possible_checkpoints) > 0:
#         sorted_paths = sorted(
#             possible_checkpoints, key=lambda path: int(path.split("-")[-1])
#         )
#         resume_from_checkpoint = sorted_paths[-1]
#     else:
#         raise FileNotFoundError("Checkpoints folder not found")

#     pytorch_bin_path = os.path.join(resume_from_checkpoint, "pytorch_model.bin")

#     assert os.path.exists(pytorch_bin_path), "Bin not found"

#     logging.info(f"Loading {pytorch_bin_path}")
#     adapters_weights = torch.load(pytorch_bin_path, map_location="cpu")

#     # d = get_peft_model_state_dict(model)
#     print(model.load_state_dict(adapters_weights))
#     # with open('b.log', "w") as f:
#     #     f.write(str(d.keys()))
#     assert False

#     print((adapters_weights.keys()))
#     with open("a.log", "w") as f:
#         f.write(str(adapters_weights.keys()))
#     assert False

#     logging.info("Setting peft model state dict")
#     set_peft_model_state_dict(model, adapters_weights)

#     logging.info(f"Set Completed!!! Saving pre-trained model to {cfg.output_dir}")
#     model.save_pretrained(cfg.output_dir)


# if __name__ == "__main__":
#     fire.Fire(save_latest_checkpoint_as_lora)
