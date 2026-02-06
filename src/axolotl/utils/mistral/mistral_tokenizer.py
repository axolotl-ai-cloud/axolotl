"""Wrapper for MistralTokenizer from mistral-common"""

import os
from typing import Optional

import numpy as np
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub
from torch import Tensor
from transformers.tokenization_mistral_common import MistralCommonBackend
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER


class HFMistralTokenizer(MistralCommonBackend):
    """
    Wraps mistral_common.tokens.tokenizers.mistral.MistralTokenizer
    and exposes HuggingFace API for special tokens.
    """

    def __init__(self, name_or_path: str, **kwargs):
        """
        Args:
            name_or_path: The name or path to the tokenizer files or the repo id.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        kwargs.pop("mode", None)

        mode = ValidationMode.finetuning
        super().__init__(**kwargs, mode=mode)

        self._name_or_path = name_or_path

        # set mode as is not set upstream
        self._set_mode(mode)

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, name_or_path: str) -> None:
        self._name_or_path = name_or_path

    @property
    def chat_template(self) -> str | None:
        """Chat template is not supported. Dummy method to satisfy HuggingFace API."""
        return "[This is a dummy chat template]"

    @chat_template.setter
    def chat_template(self, chat_template: str | None) -> None:
        pass

    def _set_mode(self, mode: ValidationMode):
        """Set the mode of the MistralRequestValidator.

        Args:
            mode: The mode to set.

        Raises:
            RuntimeError: If the MistralRequestValidator does not have a _mode attribute.
        """
        # Check if MistralRequestValidator has a _mode attribute.
        # This is a private API and may change in the future.

        from mistral_common.protocol.instruct.validator import MistralRequestValidator

        if not (
            hasattr(self.tokenizer, "_chat_completion_request_validator")
            and isinstance(
                self.tokenizer._chat_completion_request_validator,
                MistralRequestValidator,
            )
            and hasattr(self.tokenizer._chat_completion_request_validator, "_mode")
        ):
            raise RuntimeError(
                f"Unable to switch mistral tokenizer to {mode.value} mode - "
                "private API `_chat_completion_request_validator._mode` missing."
            )

        self.tokenizer._chat_completion_request_validator._mode = mode

    def apply_chat_template(  # type: ignore
        self,
        conversation: list[dict] | list[list[dict]],
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str | list[int]:
        """Patched fn to handle setting serving mode, continue_final_message, remove chat_template and add_generation_prompt kwarg"""

        # pop unnecessary kwarg for mistral
        kwargs.pop("real_last_index", None)

        try:
            if add_generation_prompt:
                self._set_mode(ValidationMode.serving)
                kwargs["continue_final_message"] = True

            out = super().apply_chat_template(conversation, **kwargs)

            return out  # type: ignore

        finally:
            if add_generation_prompt:
                self._set_mode(ValidationMode.finetuning)

    def decode(  # type: ignore
        self,
        token_ids: int | list[int] | np.ndarray | Tensor,
        **kwargs,
    ) -> str:
        """
        Decode token_ids into str.

        This overrides upstream.decode to convert int to list[int]
        """

        if isinstance(token_ids, int):
            token_ids = [token_ids]

        return super().decode(token_ids, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *init_inputs,
        mode: ValidationMode = ValidationMode.test,
        cache_dir: Optional[str | os.PathLike] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str | bool] = None,
        revision: str = "main",
        model_max_length: int = VERY_LARGE_INTEGER,
        padding_side: str = "left",
        truncation_side: str = "right",
        model_input_names: Optional[list[str]] = None,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        r"""
        Patched fn to pass `name_or_path` and remove extra kwargs.

        Instantiate a `MistralCommonBackend` from a predefined
        tokenizer.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing the tokenizer config, for instance saved
                  using the [`MistralCommonBackend.tokenization_mistral_common.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
            mode (`ValidationMode`, *optional*, defaults to `ValidationMode.test`):
                Validation mode for the `MistralTokenizer` tokenizer.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
                exist.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `hf auth login` (stored in `~/.huggingface`).
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only rely on local files and not to attempt to download any files.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            padding_side (`str`, *optional*, defaults to `"left"`):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            truncation_side (`str`, *optional*, defaults to `"right"`):
                The side on which the model should have truncation applied. Should be selected between ['right', 'left'].
            model_input_names (`List[string]`, *optional*):
                The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
                `"attention_mask"`). Default value is picked from the class attribute of the same name.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not the model should cleanup the spaces that were added when splitting the input text during the
                tokenization process.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonBackend.from_pretrained`.
                Will raise an error if used.
        """
        if init_inputs:
            raise ValueError(
                "`init_inputs` are not supported by `MistralCommonBackend.from_pretrained`."
            )

        # Delete trust_remote_code as it does nothing
        kwargs.pop("trust_remote_code", None)

        # Delete tokenizer as it does nothing
        kwargs.pop("tokenizer", None)

        # Handle kwargs and AutoTokenizer case
        if kwargs and not kwargs.keys() == {"_from_auto"}:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.from_pretrained`."
            )

        if not os.path.isfile(pretrained_model_name_or_path):
            tokenizer_path = download_tokenizer_from_hf_hub(
                repo_id=str(pretrained_model_name_or_path),
                cache_dir=str(cache_dir),
                token=token,
                revision=revision,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        else:
            tokenizer_path = str(pretrained_model_name_or_path)

        return cls(
            name_or_path=str(pretrained_model_name_or_path),
            tokenizer_path=tokenizer_path,
            mode=mode,
            model_max_length=model_max_length,
            padding_side=padding_side,
            truncation_side=truncation_side,
            model_input_names=model_input_names,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def save_pretrained(self, *args, **kwargs) -> tuple[str, ...]:
        """
        Patches to remove save_jinja_files from being passed onwards.
        """
        kwargs.pop("save_jinja_files", None)
        return super().save_pretrained(*args, **kwargs)
