"""Wizard for creating yaml configs."""

import click
import torch
import yaml
from packaging import version
from transformers.training_args import OptimizerNames

from axolotl.cli.art import print_axolotl_text_art
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model_config
from axolotl.utils.schemas.enums import CustomSupportedOptimizers


def do_wizard():
    print_axolotl_text_art()

    # Ask where to save the config
    cfg = DictDefault({})
    config_path = click.prompt(
        "Where do you want to save the config?", type=str, default="config.yaml"
    )

    # Ask base model
    base_model = click.prompt("What base model do you want to use?", type=str)
    cfg["base_model"] = base_model.strip()

    # Ask whether want to enable Vision model
    # TODO: check if model has vision layers instead of asking user
    train_vision_model = click.confirm(
        "If this model has vision layers, do you want to train them?", default=False
    )

    if train_vision_model:
        cfg["processor_type"] = "AutoProcessor"
        cfg["skip_prepare_dataset"] = True
        cfg["remove_unused_columns"] = False
        cfg["sample_packing"] = False

    # Ask whether they want to set any advanced model features (custom tokenizer, custom config, etc)
    advanced_model_features = click.confirm(
        "Do you want to set any advanced model features? (custom tokenizer, custom config, remote code etc)",
        default=False,
    )

    if advanced_model_features:
        # Ask whether they want to use a custom config
        base_model_config = click.prompt(
            "What model config do you want to use? (leave blank for default)",
            type=str,
            default="",
        )

        if base_model_config:
            cfg["base_model_config"] = base_model_config

        # Ask whether they want to use a specific revision of the model
        revision_of_model = click.prompt(
            "What revision of the model do you want to use? (leave blank for default)",
            type=str,
            default="",
        )

        if revision_of_model:
            cfg["revision_of_model"] = revision_of_model

        # Ask whether they want to use a custom tokenizer
        tokenizer_config = click.prompt(
            "What tokenizer do you want to use? (leave blank for default)",
            type=str,
            default="",
        )

        if tokenizer_config:
            cfg["tokenizer_config"] = tokenizer_config

        # Ask whether they want to use remote code
        trust_remote_code = click.confirm(
            "Do you want to use remote code?", default=False
        )

        if trust_remote_code:
            cfg["trust_remote_code"] = trust_remote_code

        # Whether to resize token embeddings
        resize_token_embeddings_to_32x = click.confirm(
            "Do you want to resize token embeddings to 32x?", default=False
        )

        if resize_token_embeddings_to_32x:
            cfg["resize_token_embeddings_to_32x"] = resize_token_embeddings_to_32x

        # Whether to shrink embeddings to len(tokenizer)
        shrink_embeddings = click.confirm(
            "Do you want to shrink embeddings to len(tokenizer)?", default=False
        )

        if shrink_embeddings:
            cfg["shrink_embeddings"] = shrink_embeddings

        # Whether to skip upcast embeddings
        embeddings_skip_upcast = click.confirm(
            "Do you want to skip upcast embeddings?", default=False
        )

        if embeddings_skip_upcast:
            cfg["embeddings_skip_upcast"] = embeddings_skip_upcast

        # Whether to random init weights
        random_init_weights = click.confirm(
            "Do you want to random init weights?", default=False
        )

        if random_init_weights:
            cfg["random_init_weights"] = random_init_weights

    # Get model type
    config = load_model_config(cfg)
    model_type = config.model_type

    # Ask sequence length
    sequence_length = click.prompt("What sequence length do you want to use?", type=int)
    cfg["sequence_length"] = sequence_length

    # Whether to turn on sample packing
    if cfg["sample_packing"] is None:
        cfg["sample_packing"] = click.confirm(
            "Do you want to turn on sample packing? This will speed up training by packing multiple samples into a single batch.",
            default=True,
        )

        if cfg["sample_packing"]:
            cfg["pad_to_sequence_len"] = True

        # Whether to turn off eval sample packing
        no_eval_sample_packing = click.confirm(
            "Do you want to turn off eval sample packing? This will slow down evaluation but is recommended if you are using a small validation set.",
            default=False,
        )

        if no_eval_sample_packing:
            cfg["eval_sample_packing"] = False

    # Hardware check
    try:
        is_ampere_or_newer = torch.cuda.get_device_capability()[0] >= 8
    except RuntimeError:
        is_ampere_or_newer = False
    except AssertionError:  # this is raised if no cuda is available
        is_ampere_or_newer = False

    # Get num gpus
    try:
        num_gpus = torch.cuda.device_count()
    except RuntimeError:
        num_gpus = 0

    # Get torch version
    torch_version = str(torch.__version__).split("+", maxsplit=1)[0]

    is_torch_2_6_or_newer = version.parse(torch_version) >= version.parse("2.6.0")

    # Whether to turn on attention
    opt = ["xformers", "sdp"]

    if is_ampere_or_newer:
        opt.append("flash")

    if is_torch_2_6_or_newer:
        opt.append("flex")

    if cfg["sample_packing"]:
        if "flash" in opt:
            default_opt = "flash"
        elif "flex" in opt:
            default_opt = "flex"
        else:
            default_opt = opt[0]

        attention = click.prompt(
            "Which attention backend do you want to use? Sample packing requires an attention backend to be set.",
            type=click.Choice(opt),
            default=default_opt,
        )
    else:
        # non-sample packing supports no attention and S2
        opt.extend(["none", "s2"])

        attention = click.prompt(
            "Which attention backend do you want to use?",
            type=click.Choice(opt),
            default="none",
        )

        if attention == "none":
            attention = None

    # TODO: if xformers, check if FA is installed
    # TODO: flex doc mentioned requiring seq len to be divisible by 128. Unclear if limitation still exists

    # TODO: requires #2489
    cfg["attention"] = attention

    # Whether to turn on gradient checkpointing
    # TODO: need to wait for offload_disk PR to be merged
    gradient_checkpointing = click.prompt(
        "Which gradient checkpointing strategy do you want to use?",
        type=click.Choice(["none", "true", "offload", "offload_disk"]),
        default="true",
    )

    if gradient_checkpointing == "none":
        gradient_checkpointing = False
    elif gradient_checkpointing == "true":
        gradient_checkpointing = True

    # Ask whether to set use_reentrant
    # TODO: get correct defaults based on SFT/RL mode and single/multigpu
    # use_reentrant = click.confirm(
    #     "Do you want to set use_reentrant?",
    #     default=True,
    # )

    # if use_reentrant:
    #     cfg["use_reentrant"] = use_reentrant

    # Optimizer
    cfg["optimizer"] = click.prompt(
        "Which optimizer do you want to use?",
        type=click.Choice((OptimizerNames | CustomSupportedOptimizers)),
        default=OptimizerNames.ADAMW_TORCH_FUSED,
    )

    cfg["lr_scheduler"] = click.prompt(
        "Which learning rate scheduler do you want to use?",
        type=click.Choice(
            [
                "cosine",
                "one_cycle",
                "rex",
                "log_sweep",
                "linear",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
                "inverse_sqrt",
                "reduce_lr_on_plateau",
                "cosine_with_min_lr",
                "warmup_stable_decay",
            ]
        ),
        default="cosine",
    )

    # Plugins

    cfg["plugins"] = []

    # Whether to turn on cut cross entropy
    if is_ampere_or_newer:
        # Note: This may error if users don't have CCE installed
        from axolotl.integrations.cut_cross_entropy.monkeypatch.patch import (
            CUT_CROSS_ENTROPY_MODEL_MAPPING,
        )

        if model_type in CUT_CROSS_ENTROPY_MODEL_MAPPING:
            cut_cross_entropy = click.confirm(
                "Do you want to turn on cut cross entropy? This will save VRAM if the model has a large vocab size.",
                default=True,
            )

            if cut_cross_entropy:
                cfg["plugins"].append(
                    "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"
                )

                cfg["cut_cross_entropy"] = True

    use_liger_kernel = click.confirm(
        "Do you want to use the liger kernel? This will speed up training and save VRAM.",
        default=True,
    )

    if use_liger_kernel:
        cfg["plugins"].append("axolotl.integrations.liger.LigerPlugin")

        cfg["liger_rope"] = click.confirm(
            "Do you want to enable liger rope?",
            default=True,
        )

        cfg["liger_rms_norm"] = click.confirm(
            "Do you want to enable liger rms norm?",
            default=True,
        )

        cfg["liger_glu_activation"] = click.confirm(
            "Do you want to enable liger glu activation?",
            default=True,
        )

        cfg["liger_layer_norm"] = click.confirm(
            "Do you want to enable liger layer norm?",
            default=True,
        )

        if cfg["cut_cross_entropy"] is not True:
            cfg["liger_fused_linear_cross_entropy"] = click.confirm(
                "Do you want to enable liger fused linear cross entropy?",
                default=True,
            )

    # TODO: lora kernels (but they auto enable via validator already)

    # TODO: is there incompat between torch compile and liger?
    cfg["torch_compile"] = click.confirm(
        "Do you want to enable torch compile?",
        default=True,
    )

    # Multi-gpu
    if num_gpus > 1:
        # Ask whether to use DDP/Deepspeed/FSDP
        multi_gpu_mode = click.prompt(
            "Which multi-gpu mode do you want to use?",
            type=click.Choice(["ddp", "deepspeed", "fsdp"]),
            default="ddp",
        )

        if multi_gpu_mode == "deepspeed":
            # Ask which deepspeed config to use
            cfg["deepspeed"] = click.prompt(
                "Which deepspeed config do you want to use? The higher the number, the more VRAM you will save, but the slower it will run.",
                type=click.Choice(
                    [
                        "zero1.json",
                        "zero1_torch_compile.json",
                        "zero2.json",
                        "zero3.json",
                        "zero3_bf16.json",
                        "zero3_bf16_cpuoffload_all.json",
                        "zero3_bf16_cpuoffload_params.json",
                    ]
                ),
                default="zero1.json",
            )
        elif multi_gpu_mode == "fsdp":
            fsdp_version = click.prompt(
                "Which fsdp version do you want to use?",
                type=click.Choice([1, 2]),
                default=2,
            )

            # TODO: Handle FSDP config

            if fsdp_version == 1:
                cfg["fsdp"] = ["full_shard", "auto_wrap"]

                # Ask which state dict type to use
                fsdp_state_dict_type = click.prompt(
                    "Which fsdp state dict type do you want to use?",
                    type=click.Choice(["FULL_STATE_DICT", "SHARDED_STATE_DICT"]),
                    default="FULL_STATE_DICT",
                )

                fsdp_offload_params = click.confirm(
                    "Do you want to offload parameters?",
                    default=True,
                )

                # TODO: can we load the model class and auto pull a default for this?
                fsdp_transformer_layer_cls_to_wrap = click.prompt(
                    "Which transformer layer class to wrap? It is usually the Decoder layer class.",
                    type=str,
                )

                # TODO: add other options

                cfg["fsdp_config"] = {
                    "state_dict_type": fsdp_state_dict_type,
                    "offload_params": fsdp_offload_params,
                    "transformer_layer_cls_to_wrap": fsdp_transformer_layer_cls_to_wrap,
                }

            elif fsdp_version == 2:
                raise NotImplementedError()

    # Training mode (sft or rl)
    training_mode = click.prompt(
        "Which training mode do you want to use?",
        type=click.Choice(["sft", "rl"]),
        default="sft",
    )

    if training_mode == "rl":
        cfg["rl"] = click.prompt(
            "Which rl mode do you want to use?",
            type=click.Choice(["dpo", "ipo", "orpo", "kto", "grpo", "simpo"]),
        )

        # TODO: handle RL options

    # Whether to use adapter

    # Get batch/grad accu

    # Get learning rate

    # Get weight decay

    # Get max grad norm

    # Get num train epochs

    # Get warmup ratio

    # Get save ratio

    # Get eval ratio

    # Get dataset config

    # Load metric tracker

    # Save config to yaml
    # TODO: improve output yaml formatting. Need to add comments to help separate sections
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, sort_keys=False)
