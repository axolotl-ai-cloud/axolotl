"""Tokenizer loading functionality and associated utils."""

import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
import transformers
from huggingface_hub import hf_hub_download
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import SpecialTokens
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import AddedToken, AutoTokenizer

from axolotl.integrations.base import PluginManager
from axolotl.loaders.utils import get_linear_embedding_layers, load_model_config
from axolotl.prompt_tokenizers import LLAMA_DEFAULT_EOS_TOKEN
from axolotl.utils.chat_templates import get_chat_template_from_config
from axolotl.utils.distributed import (
    barrier,
    is_local_main_process,
    is_main_process,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
PLUGIN_MANAGER = PluginManager.get_instance()

# Constants
LLAMA_TOKENIZER_CLASSES = {
    "LlamaTokenizer",
    "LlamaTokenizerFast",
    "CodeLlamaTokenizer",
    "CodeLlamaTokenizerFast",
}
FAST_LLAMA_TOKENIZER_CLASSES = {"LlamaTokenizerFast", "CodeLlamaTokenizerFast"}
MISTRAL_MODEL_TYPES = {"mistral", "mistral3"}

QWEN_DEFAULT_TOKEN = "<|endoftext|>"  # nosec B105
GPTNEOX_PAD_TOKEN = "[PAD]"  # nosec B105
CHATML_DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


class MistralTokenizerWrapper:
    """
    Wrapper to make MistralTokenizer compatible with Hugging Face tokenizer
    interface. This provides a bridge between Mistral's native tokenizer and axolotl's
    expectations.
    """

    def __init__(
        self,
        mistral_tokenizer: MistralTokenizer,
        model_id: str,
        system_prompt: str | None = None,
    ):
        self.mistral_tokenizer = mistral_tokenizer
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.padding_side = "right"  # Default padding side
        self.chat_template = None

    # pylint: disable=unused-argument
    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """Encode text to token IDs"""
        # For simple string encoding, create a user message
        messages = []
        if self.system_prompt and add_special_tokens:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(UserMessage(content=text))

        tokenized = self.mistral_tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=messages)
        )
        return tokenized.tokens

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,  # pylint: disable=unused-argument
    ) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.mistral_tokenizer.decode(token_ids)

    def __call__(self, text: str, **kwargs):
        """Make the tokenizer callable like HF tokenizers"""
        tokens = self.encode(text, **kwargs)
        return {"input_ids": torch.tensor([tokens])}

    @property
    def special_tokens_reverse_vocab(self):
        # pylint: disable=protected-access
        return (
            self.mistral_tokenizer.instruct_tokenizer.tokenizer._special_tokens_reverse_vocab
        )

    @property
    def eos_token(self):
        return SpecialTokens.eos

    @property
    def bos_token(self):
        return SpecialTokens.bos

    @property
    def pad_token(self):
        return self.eos_token  # Use EOS as pad token

    @property
    def unk_token(self):
        return SpecialTokens.unk

    @property
    def eos_token_id(self):
        return self.special_tokens_reverse_vocab[self.eos_token]

    @property
    def bos_token_id(self):
        return self.special_tokens_reverse_vocab[self.bos_token]

    @property
    def pad_token_id(self):
        return self.special_tokens_reverse_vocab[self.pad_token]

    @property
    def unk_token_id(self):
        return self.special_tokens_reverse_vocab[self.unk_token]

    # pylint: disable=unused-argument
    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        """Placeholder for special token addition - Mistral tokenizer handles this internally"""
        LOG.warning(
            "add_special_tokens called on MistralTokenizer wrapper - this is handled internally"
        )
        return 0

    # pylint: disable=unused-argument
    def add_tokens(self, tokens) -> int:
        """Placeholder for token addition - Mistral tokenizer handles this internally"""
        LOG.warning(
            "add_tokens called on MistralTokenizer wrapper - this is handled internally"
        )
        return 0


class TokenizerFileModifier:
    """Handles modification of tokenizer files for token overrides."""

    def __init__(
        self, tokenizer_path: str, token_mappings: Dict[int, str], output_dir: str
    ):
        self.tokenizer_path = tokenizer_path
        self.token_mappings = token_mappings
        self.output_dir = output_dir
        self.tokenizer_dir = os.path.join(output_dir, "tokenizer")

    def modify_and_save(self) -> str:
        """Modify tokenizer files and return path to modified tokenizer."""
        os.makedirs(self.tokenizer_dir, exist_ok=True)

        if is_local_main_process():
            self._perform_modifications()
        barrier()

        return self.tokenizer_dir

    def _perform_modifications(self):
        """Perform the actual file modifications."""
        # Load and save tokenizer to output directory
        temp_tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, use_fast=True
        )
        temp_tokenizer.save_pretrained(self.tokenizer_dir)

        # Convert token mappings to proper format
        token_id_mappings = {
            int(token_id): new_value
            for token_id, new_value in self.token_mappings.items()
        }

        # Update both tokenizer files
        self._update_tokenizer_config(token_id_mappings)
        self._update_tokenizer_json(token_id_mappings)

    def _update_tokenizer_config(self, token_id_mappings: Dict[int, str]):
        """Update tokenizer_config.json with new token mappings."""
        config_path = os.path.join(self.tokenizer_dir, "tokenizer_config.json")
        if not os.path.exists(config_path):
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if "added_tokens_decoder" in config_data:
            self._update_added_tokens_decoder(config_data, token_id_mappings)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    def _update_added_tokens_decoder(
        self, config_data: Dict, token_id_mappings: Dict[int, str]
    ):
        """Update the added_tokens_decoder section."""
        for token_id, new_value in token_id_mappings.items():
            token_id_str = str(token_id)
            if token_id_str in config_data["added_tokens_decoder"]:
                config_data["added_tokens_decoder"][token_id_str]["content"] = new_value
            else:
                raise ValueError(
                    f"Token ID {token_id_str} not found in added_tokens_decoder"
                )

    def _update_tokenizer_json(self, token_id_mappings: Dict[int, str]):
        """Update tokenizer.json with new token mappings."""
        tokenizer_json_path = os.path.join(self.tokenizer_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_json_path):
            return

        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        self._update_added_tokens_list(tokenizer_data, token_id_mappings)
        self._update_vocab_mappings(tokenizer_data, token_id_mappings)

        with open(tokenizer_json_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, indent=2)

    def _update_added_tokens_list(
        self, tokenizer_data: Dict, token_id_mappings: Dict[int, str]
    ):
        """Update the added_tokens list in tokenizer.json."""
        if "added_tokens" not in tokenizer_data:
            return

        for token_id, new_value in token_id_mappings.items():
            for i, token_entry in enumerate(tokenizer_data["added_tokens"]):
                if token_entry["id"] == token_id:
                    tokenizer_data["added_tokens"][i]["content"] = new_value
                    break
            else:
                raise ValueError(f"Token ID {token_id} not found in added_tokens")

    def _update_vocab_mappings(
        self, tokenizer_data: Dict, token_id_mappings: Dict[int, str]
    ):
        """Update vocab mappings in tokenizer.json."""
        if not (tokenizer_data.get("model") and tokenizer_data["model"].get("vocab")):
            return

        vocab = tokenizer_data["model"]["vocab"]
        for token_id, new_value in token_id_mappings.items():
            # Find and update the vocab entry
            for entry_val, entry_id in list(vocab.items()):
                if entry_id == token_id:
                    del vocab[entry_val]
                    vocab[new_value] = token_id
                    break


class TokenizerConfiguration:
    """Handles tokenizer configuration and initialization."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_config = load_model_config(cfg)

    def should_use_mistral_tokenizer(self) -> bool:
        """Determine if Mistral tokenizer should be used."""
        # Explicit configuration
        return self.model_config.model_type in MISTRAL_MODEL_TYPES

    def load_mistral_tokenizer(self) -> MistralTokenizerWrapper:
        """Load Mistral tokenizer from model configuration."""
        model_id = getattr(self.cfg, "model_name_or_path", None) or getattr(
            self.cfg, "base_model", None
        )
        if not model_id:
            raise ValueError(
                "model_name_or_path or base_model must be specified for Mistral tokenizer"
            )

        try:
            # Download the tekken.json file for the tokenizer
            tekken_file = hf_hub_download(repo_id=model_id, filename="tekken.json")

            # Load the Mistral tokenizer
            mistral_tokenizer = MistralTokenizer.from_file(tekken_file)

            # Wrap it for compatibility
            wrapped_tokenizer = MistralTokenizerWrapper(mistral_tokenizer, model_id)

            LOG.info(f"Loaded Mistral tokenizer for model: {model_id}")
            return wrapped_tokenizer

        except Exception as e:
            LOG.error(f"Failed to load Mistral tokenizer: {e}")
            raise

    def get_tokenizer_class(self):
        """Get the appropriate tokenizer class."""
        if self.cfg.tokenizer_type:
            return getattr(transformers, self.cfg.tokenizer_type)
        return AutoTokenizer

    def get_tokenizer_kwargs(self) -> Dict[str, Any]:
        """Build tokenizer initialization kwargs."""
        kwargs = {}
        if self.cfg.tokenizer_legacy is not None:
            kwargs["legacy"] = self.cfg.tokenizer_legacy
        return kwargs

    def get_tokenizer_path(self) -> str:
        """Get the tokenizer path, applying overrides if needed."""
        tokenizer_path = self.cfg.tokenizer_config

        if self.cfg.added_tokens_overrides:
            modifier = TokenizerFileModifier(
                tokenizer_path, self.cfg.added_tokens_overrides, self.cfg.output_dir
            )
            tokenizer_path = modifier.modify_and_save()

        return tokenizer_path

    def should_use_fast_tokenizer(self) -> bool:
        """Determine if fast tokenizer should be used."""
        return (
            self.cfg.tokenizer_use_fast
            if self.cfg.tokenizer_use_fast is not None
            else True
        )


class TokenizerPostProcessor:
    """Handles post-processing configuration of loaded tokenizers."""

    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model_config = load_model_config(cfg)

    def apply_all_configurations(self):
        """Apply all post-processing configurations to the tokenizer."""
        # Skip most configurations for Mistral wrapper
        if isinstance(self.tokenizer, MistralTokenizerWrapper):
            self._configure_mistral_wrapper()
            return

        self._configure_padding_token()
        self._configure_gptneox_settings()
        self._configure_mistral_padding()
        self._configure_qwen_tokens()
        self._add_special_tokens()
        self._add_regular_tokens()
        self._configure_chat_template()

    def _configure_mistral_wrapper(self):
        """Apply limited configurations for Mistral wrapper."""
        # Set padding side if needed
        if (
            self.cfg.is_mistral_derived_model
            and self.cfg.flash_attention
            and not self.cfg.sample_packing
        ):
            self.tokenizer.padding_side = "left"

        # Configure chat template for Mistral
        self._configure_chat_template()

    def _configure_padding_token(self):
        """Configure padding token for Llama-based tokenizers."""
        if (
            self.tokenizer.__class__.__name__ in LLAMA_TOKENIZER_CLASSES
            and hasattr(self.tokenizer, "pad_token")
            and not self.tokenizer.pad_token
        ):
            self.tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

    def _configure_gptneox_settings(self):
        """Configure GPTNeoX-specific settings."""
        if self.tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
            self.tokenizer.add_special_tokens({"pad_token": GPTNEOX_PAD_TOKEN})
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _configure_mistral_padding(self):
        """Configure left padding for Mistral models with Flash Attention."""
        if (
            self.cfg.is_mistral_derived_model
            and self.cfg.flash_attention
            and not self.cfg.sample_packing
        ):
            self.tokenizer.padding_side = "left"

    def _configure_qwen_tokens(self):
        """Configure special tokens for Qwen models."""
        if not self.cfg.is_qwen_derived_model:
            return

        # Set token IDs
        token_id_attributes = [
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        ]
        for attr_name in token_id_attributes:
            if getattr(self.tokenizer, attr_name) is None:
                setattr(self.tokenizer, attr_name, self.tokenizer.eod_id)

        # Set token strings
        token_name_attributes = ["bos_token", "eos_token", "pad_token", "unk_token"]
        for attr_name in token_name_attributes:
            if getattr(self.tokenizer, attr_name) is None:
                setattr(self.tokenizer, attr_name, QWEN_DEFAULT_TOKEN)

    def _add_special_tokens(self):
        """Add special tokens from configuration."""
        if not self.cfg.special_tokens:
            return

        special_tokens_dict = self.cfg.special_tokens.to_dict()
        additional_special_tokens = special_tokens_dict.pop(
            "additional_special_tokens", None
        )

        self._validate_and_add_special_tokens(special_tokens_dict)
        self._update_post_processor_if_needed(special_tokens_dict)
        self._add_additional_special_tokens_if_present(additional_special_tokens)

    def _validate_and_add_special_tokens(self, special_tokens: Dict[str, str]):
        """Validate special tokens for adapter training and add them."""
        lora_modules_to_save = get_linear_embedding_layers(self.model_config.model_type)

        for key, value in special_tokens.items():
            self._validate_token_for_adapter(key, value, lora_modules_to_save)
            self.tokenizer.add_special_tokens(
                {key: AddedToken(value, rstrip=False, lstrip=False, normalized=False)}
            )

    def _validate_token_for_adapter(
        self, key: str, value: str, lora_modules_to_save: List[str]
    ):
        """Validate a single token for adapter training requirements."""
        if not self._should_validate_token_for_adapter(
            key, value, lora_modules_to_save
        ):
            return

        modules_str = ", ".join(f"`{x}`" for x in lora_modules_to_save)
        raise ValueError(
            f"Please set lora_modules_to_save to [{modules_str}] "
            f"when using an adapter and changing the special tokens."
        )

    def _should_validate_token_for_adapter(
        self, key: str, value: str, lora_modules_to_save: List[str]
    ) -> bool:
        """Check if token should be validated for adapter configuration."""
        if key == "pad_token" or not self.cfg.adapter:
            return False

        current_token = getattr(self.tokenizer, key)
        token_changed = current_token is None or current_token != value
        token_is_multi_char = (
            len(self.tokenizer.encode(value, add_special_tokens=False)) > 2
        )
        lora_modules_missing = not self.cfg.lora_modules_to_save or not all(
            x in self.cfg.lora_modules_to_save for x in lora_modules_to_save
        )

        return token_changed and token_is_multi_char and lora_modules_missing

    def _update_post_processor_if_needed(self, special_tokens: Dict[str, str]):
        """Update post processor for Llama tokenizers when BOS/EOS tokens are added."""
        has_bos_and_eos = (
            "bos_token" in special_tokens and "eos_token" in special_tokens
        )
        is_fast_llama = (
            self.tokenizer.__class__.__name__ in FAST_LLAMA_TOKENIZER_CLASSES
        )

        if is_fast_llama and has_bos_and_eos:
            self.tokenizer.update_post_processor()

    def _add_additional_special_tokens_if_present(
        self, additional_special_tokens: Optional[List[str]]
    ):
        """Add additional special tokens if they exist."""
        if additional_special_tokens is not None:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": additional_special_tokens}
            )

    def _add_regular_tokens(self):
        """Add regular (non-special) tokens from configuration."""
        if self.cfg.tokens:
            self.tokenizer.add_tokens(
                [
                    AddedToken(token, rstrip=False, lstrip=False, normalized=False)
                    for token in self.cfg.tokens
                ]
            )

    def _configure_chat_template(self):
        """Configure chat template if specified."""
        if not self.cfg.chat_template:
            LOG.info(
                "No Chat template selected. Consider adding a chat template for easier inference."
            )
            return

        chat_template_string = get_chat_template_from_config(
            cfg=self.cfg,
            tokenizer=self.tokenizer,
        )

        if self._should_replace_default_system_message():
            chat_template_string = chat_template_string.replace(
                CHATML_DEFAULT_SYSTEM_MESSAGE, self.cfg.default_system_message
            )

        self.tokenizer.chat_template = chat_template_string

    def _should_replace_default_system_message(self) -> bool:
        """Check if default system message should be replaced."""
        return self.cfg.default_system_message and self.cfg.chat_template == "chatml"


def load_tokenizer(cfg):
    """Load and configure the tokenizer based on the provided config.

    This function handles the complete tokenizer loading pipeline:
        - Check if Mistral tokenizer should be used
        - Configure tokenizer parameters and get the appropriate class
        - Handle token file modifications if needed
        - Initialize the tokenizer with the correct parameters
        - Apply all post-processing configurations (padding, special tokens, etc.)
        - Set up chat templates and logging

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.

    Returns:
        Fully configured tokenizer instance.
    """
    # Configure tokenizer parameters
    config = TokenizerConfiguration(cfg)

    # Check if we should use Mistral tokenizer
    if config.should_use_mistral_tokenizer():
        tokenizer = config.load_mistral_tokenizer()
    else:
        # Standard tokenizer loading
        tokenizer_cls = config.get_tokenizer_class()
        tokenizer_path = config.get_tokenizer_path()
        use_fast = config.should_use_fast_tokenizer()
        tokenizer_kwargs = config.get_tokenizer_kwargs()

        # Initialize the tokenizer
        tokenizer = tokenizer_cls.from_pretrained(
            tokenizer_path,
            trust_remote_code=cfg.trust_remote_code or False,
            use_fast=use_fast,
            **tokenizer_kwargs,
        )

    # Apply all post-processing configurations
    post_processor = TokenizerPostProcessor(tokenizer, cfg)
    post_processor.apply_all_configurations()

    if is_main_process(use_environ=True):
        LOG.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
        LOG.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
        LOG.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
        LOG.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    return tokenizer
