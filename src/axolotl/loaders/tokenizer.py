"""Tokenizer loading functionality and associated utils"""

import json
import os

import transformers
from transformers import (
    AddedToken,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from axolotl.integrations.base import PluginManager
from axolotl.loaders.utils import get_linear_embedding_layers, load_model_config
from axolotl.prompt_tokenizers import LLAMA_DEFAULT_EOS_TOKEN
from axolotl.telemetry.errors import send_errors
from axolotl.utils.chat_templates import get_chat_template_from_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import (
    barrier,
    is_local_main_process,
    is_main_process,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
PLUGIN_MANAGER = PluginManager.get_instance()


def modify_tokenizer_files(
    tokenizer_path: str, token_mappings: dict[int, str], output_dir: str
) -> str:
    """
    Modify tokenizer files to replace added_tokens strings, save to output directory,
    and return the path to the modified tokenizer.

    This only works with reserved tokens that were added to the tokenizer, not tokens
    already part of the vocab.

    Args:
        tokenizer_path: Path or name of the original tokenizer
        token_mappings: Dict mapping {token_id (int): new_token_string}
        output_dir: Directory to save the modified tokenizer

    Returns:
        Path to the modified tokenizer directory

    Ref: https://github.com/huggingface/transformers/issues/27974#issuecomment-1854188941
    """
    # Create the tokenizer directory in output_dir if it doesn't exist
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    if is_local_main_process():
        # Load the tokenizer
        temp_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        # Save the tokenizer to the output directory
        temp_tokenizer.save_pretrained(tokenizer_dir)

        # Get the token IDs and map them to their new values
        token_id_mappings = {
            int(token_id): new_value for token_id, new_value in token_mappings.items()
        }

        # 1. Update tokenizer_config.json - added_tokens_decoder
        config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Update added_tokens_decoder
            if "added_tokens_decoder" in config_data:
                for token_id, new_value in token_id_mappings.items():
                    token_id_str = str(token_id)
                    if token_id_str in config_data["added_tokens_decoder"]:
                        config_data["added_tokens_decoder"][token_id_str]["content"] = (
                            new_value
                        )
                    else:
                        raise ValueError(
                            f"Token ID {token_id_str} not found in added_tokens_decoder"
                        )

            # Write the updated config back
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)

        # 2. Update tokenizer.json - added_tokens
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)

            # Update added_tokens
            if "added_tokens" in tokenizer_data:
                for token_id, new_value in token_id_mappings.items():
                    for i, token_entry in enumerate(tokenizer_data["added_tokens"]):
                        if token_entry["id"] == token_id:
                            tokenizer_data["added_tokens"][i]["content"] = new_value
                            break
                    else:
                        # Reaching this section means the token_id was not found in tokenizer.json added_tokens
                        raise ValueError(
                            f"Token ID {token_id} not found in added_tokens"
                        )
            if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                for token_id, new_value in token_id_mappings.items():
                    for entry_val, entry_id in tokenizer_data["model"]["vocab"].items():
                        if entry_id == token_id:
                            del tokenizer_data["model"]["vocab"][entry_val]
                            tokenizer_data["model"]["vocab"][new_value] = token_id
                            break

            # Write the updated tokenizer data back
            with open(tokenizer_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_data, f, indent=2)

    barrier()
    return tokenizer_dir


@send_errors
def load_tokenizer(cfg: DictDefault) -> PreTrainedTokenizer:
    """Load and configure the tokenizer based on the provided config."""

    # Apply patches that need to be in place before tokenizer loading
    from axolotl.loaders.patch_manager import PatchManager

    PatchManager.apply_pre_tokenizer_load_patches(cfg)

    def _load_mistral_common_tokenizer(cfg: DictDefault):
        """Load mistral-common tokenizer"""
        from axolotl.utils.mistral import HFMistralTokenizer

        # Load the HF-compatible wrapper around MistralTokenizer
        tokenizer = HFMistralTokenizer.from_pretrained(cfg.tokenizer_config)

        return tokenizer

    if cfg.tokenizer_use_mistral_common:
        return _load_mistral_common_tokenizer(cfg)

    model_config = load_model_config(cfg)
    tokenizer_kwargs = {}
    use_fast = True  # this is the default

    if cfg.tokenizer_use_fast is not None:
        use_fast = cfg.tokenizer_use_fast
    if cfg.tokenizer_legacy is not None:
        # True is the default w/ https://github.com/huggingface/transformers/pull/25224
        tokenizer_kwargs["legacy"] = cfg.tokenizer_legacy

    tokenizer_cls = AutoTokenizer
    if cfg.tokenizer_type:
        tokenizer_cls = getattr(transformers, cfg.tokenizer_type)

    # Set base tokenizer path
    tokenizer_path = cfg.tokenizer_config

    # Apply token string overrides if specified
    if cfg.added_tokens_overrides:
        # Modify tokenizer files and get path to modified tokenizer
        tokenizer_path = modify_tokenizer_files(
            tokenizer_path, cfg.added_tokens_overrides, output_dir=cfg.output_dir
        )

    tokenizer = tokenizer_cls.from_pretrained(
        tokenizer_path,
        trust_remote_code=cfg.trust_remote_code or False,
        use_fast=use_fast,
        **tokenizer_kwargs,
    )

    if (
        tokenizer.__class__.__name__
        in [
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
            "CodeLlamaTokenizerFast",
        ]
        and hasattr(tokenizer, "pad_token")
        and not tokenizer.pad_token
    ):
        # set a pad_token, but use eos_token so we don't add a new token
        tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Mistral's official FA implementation requires left padding
    if cfg.is_mistral_derived_model and cfg.flash_attention and not cfg.sample_packing:
        tokenizer.padding_side = "left"

    # Qwen base only has single token, so we need to set the special tokens
    # the following check is for Qwen1 base models
    if cfg.is_qwen_derived_model and hasattr(tokenizer, "eod_id"):
        token_ids = ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]
        for attr_name in token_ids:
            if getattr(tokenizer, attr_name) is None:
                setattr(tokenizer, attr_name, tokenizer.eod_id)

        token_names = ["bos_token", "eos_token", "pad_token", "unk_token"]
        for attr_name in token_names:
            if getattr(tokenizer, attr_name) is None:
                setattr(tokenizer, attr_name, "<|endoftext|>")

    additional_special_tokens = None
    if cfg.special_tokens:
        special_tokens = cfg.special_tokens.to_dict()
        additional_special_tokens = special_tokens.pop(
            "additional_special_tokens", None
        )
        lora_modules_to_save = get_linear_embedding_layers(model_config.model_type)
        for k, val in special_tokens.items():
            # check if new special token is not already in tokenizer and
            # is adapter training to make sure lora_modules_to_save is set

            if (
                (getattr(tokenizer, k) is None or getattr(tokenizer, k) != val)
                and (len(tokenizer.encode(val, add_special_tokens=False)) > 2)
                and cfg.adapter
                and (
                    not cfg.lora_modules_to_save
                    or not all(
                        x in cfg.lora_modules_to_save for x in lora_modules_to_save
                    )
                )
                and k != "pad_token"
            ):
                lora_modules_to_save_str = ", ".join(
                    [f"`{x}`" for x in lora_modules_to_save]
                )
                raise ValueError(
                    f"Please set lora_modules_to_save to [{lora_modules_to_save_str}] "
                    "when using an adapter and changing the special tokens."
                )

            tokenizer.add_special_tokens(
                {k: AddedToken(val, rstrip=False, lstrip=False, normalized=False)}
            )

        # If we add bos_token and eos_token, we need to update the post processor to
        # handle them correctly.
        # https://github.com/huggingface/transformers/pull/24132
        bos_or_eos_in_special_tokens = (
            "bos_token" in cfg.special_tokens and "eos_token" in cfg.special_tokens
        )
        if (
            tokenizer.__class__.__name__
            in (
                "LlamaTokenizerFast",
                "CodeLlamaTokenizerFast",
            )
            and bos_or_eos_in_special_tokens
        ):
            tokenizer.update_post_processor()

    if cfg.tokens:
        tokenizer.add_tokens(
            [
                AddedToken(token, rstrip=False, lstrip=False, normalized=False)
                for token in cfg.tokens
            ]
        )

    # Additional special tokens are a List, and need to be treated differently than regular special
    # tokens. We add them after we have called `add_tokens` in case these additional special tokens
    # are new tokens.
    #
    # Usage:
    #
    # ```py
    # special_tokens:
    #   additional_special_tokens: ["<|im_start|>", "<|im_end|>"]
    # ```
    if additional_special_tokens is not None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}
        )

    if is_main_process():
        LOG.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
        LOG.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
        LOG.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
        LOG.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    if cfg.chat_template:
        chat_template_string = get_chat_template_from_config(
            cfg=cfg,
            tokenizer=tokenizer,
        )
        if cfg.default_system_message and cfg.chat_template == "chatml":
            chat_template_string = chat_template_string.replace(
                "You are a helpful assistant.", cfg.default_system_message
            )

        tokenizer.chat_template = chat_template_string
    elif getattr(tokenizer, "chat_template", None) is None:
        LOG.info(
            "No Chat template selected. Consider adding a chat template for easier inference."
        )

    # make the tokenizer.pad call quieter 🤐
    if hasattr(tokenizer, "deprecation_warnings"):
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    return tokenizer
