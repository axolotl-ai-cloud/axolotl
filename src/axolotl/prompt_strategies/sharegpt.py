"""Module containing the SimpleShareGPTPromptTokenizingStrategy class"""
import copy
import logging
from typing import Any, Dict, Optional

from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

from axolotl.prompt_tokenizers import (
    InvalidDataException,
    ShareGPTPromptTokenizingStrategy,
    parse_tokenized_to_result,
    tokenize_prompt_default,
)
from axolotl.prompters import (
    IGNORE_TOKEN_ID,
    ShareGPTPrompterV2,
    ShareGPTPrompterV2MultiRole,
)

LOG = logging.getLogger("axolotl")


def register_chatml_template(system_message=None):
    system_message = system_message or "You are a helpful assistant."
    register_conv_template(
        Conversation(
            name="chatml",
            system_template="<|im_start|>system\n{system_message}",
            system_message=system_message,
            roles=["<|im_start|>user", "<|im_start|>assistant"],
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
        )
    )


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    conversation = (
        ds_cfg["conversation"] if ds_cfg and "conversation" in ds_cfg else None
    )
    field_human = ds_cfg["field_human"] if ds_cfg and "field_human" in ds_cfg else None
    field_model = ds_cfg["field_model"] if ds_cfg and "field_model" in ds_cfg else None
    strategy = SimpleShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(
            conversation=conversation,
            role_key_model=field_model,
            role_key_human=field_human,
        ),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    if ds_cfg and "strict" in ds_cfg:
        strategy.strict = ds_cfg["strict"]
    return strategy


def load_ultrachat(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    conversation = (
        ds_cfg["conversation"] if ds_cfg and "conversation" in ds_cfg else None
    )
    strategy = UltrachatShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(
            conversation=conversation,
        ),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    if ds_cfg and "strict" in ds_cfg:
        strategy.strict = ds_cfg["strict"]
    return strategy


def load_role(tokenizer, cfg):
    return SimpleRoleShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_guanaco(tokenizer, cfg):
    return GuanacoShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_multirole(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    conversation = (
        ds_cfg["conversation"] if ds_cfg and "conversation" in ds_cfg else None
    )
    field_human = ds_cfg["field_human"] if ds_cfg and "field_human" in ds_cfg else None
    field_model = ds_cfg["field_model"] if ds_cfg and "field_model" in ds_cfg else None
    strategy = MultiRoleShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2MultiRole(
            conversation=conversation,
            role_key_model=field_model,
            role_key_human=field_human,
        ),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    if ds_cfg and "strict" in ds_cfg:
        strategy.strict = ds_cfg["strict"]

    return strategy


class SimpleShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    basic sharegpt strategy to grab conversations from the sample row
    """

    _strict = True

    @property
    def strict(self):
        return self._strict

    @strict.setter
    def strict(self, strict):
        self._strict = strict

    def get_conversation_thread(self, prompt):
        conversations = prompt["conversations"]
        if self.strict:
            return conversations
        # remap roles - allow for assistant turn
        role_map = {"human": "human", "assistant": "gpt", "gpt": "gpt"}
        turns = [
            {"from": role_map[t["from"]], "value": t["value"]} for t in conversations
        ]
        return turns


class SimpleRoleShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    basic sharegpt strategy to grab conversations from the sample row, but uses role instead of from
    """

    def get_conversation_thread(self, prompt):
        conversations = prompt["conversations"]
        # remap role: prompter/assistant, text: ... => from: human/gpt, value: ...
        turns = [{"from": t["role"], "value": t["value"]} for t in conversations]
        return turns


class GuanacoShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    sharegpt strategy that remaps oasst data to sharegpt format
    """

    def get_conversation_thread(self, prompt):
        conversations = prompt["conversations"]
        # remap role: prompter/assistant, text: ... => from: human/gpt, value: ...
        role_map = {"prompter": "human", "assistant": "gpt"}
        turns = [
            {"from": role_map[t["role"]], "value": t["text"]} for t in conversations
        ]
        return turns


class UltrachatShareGPTPromptTokenizingStrategy(SimpleShareGPTPromptTokenizingStrategy):
    """
    sharegpt strategy that remaps ultrachat data to sharegpt format
    """

    def get_conversation_thread(self, prompt):
        conversations = prompt["messages"]
        role_map = {"user": "human", "assistant": "gpt"}
        turns = [
            {"from": role_map[t["role"]], "value": t["content"]} for t in conversations
        ]
        return turns


class MultiRoleShareGPTPromptTokenizingStrategy(SimpleShareGPTPromptTokenizingStrategy):
    """
    sharegpt strategy for support of multi-role
    """

    def tokenize_prompt(self, prompt):
        # Initial values. We will append to these as we go through the conversation.
        result, current_len = tokenize_prompt_default()
        conversation: Conversation = (
            self.prompter._conversation.copy()  # pylint: disable=protected-access
        )
        user, assistant = conversation.roles

        input_roles = {
            "human",
            "funcresponse",
            "funccaller",
            "tool",
            "tool_response",
            user,
        }
        output_roles = {"gpt", "tool_caller", assistant}

        # support for custom roles from the dataset, only useful for vicuna style prompts/roles
        role_remap = []
        if (
            conversation.name == "vicuna_v1.1"
            and "roles" in prompt
            and len(prompt["roles"]) >= 2
        ):
            role_remap = [
                {"from": conversation.roles[0], "to": prompt["roles"][0]},
                {"from": conversation.roles[1], "to": prompt["roles"][1]},
            ]

        try:
            for _, part in enumerate(
                self.prompter.build_prompt(self.get_conversation_thread(prompt))
            ):
                if not isinstance(part, tuple):
                    LOG.warning(f"expected tuple, got {part}")
                    continue

                role, content = part

                # Uses "in" because role contains extra characters
                input_turn = any(r in role.lower() for r in input_roles)
                output_turn = any(r in role.lower() for r in output_roles)

                if input_turn:
                    role = (
                        role.replace(role_remap[0]["from"], role_remap[0]["to"])
                        if role_remap
                        else role
                    )
                    turn = role + content
                    # this is still the user query, we should
                    if not content.strip():
                        LOG.warning(f"user turn has empty text: {prompt}")
                    res = self._tokenize(
                        turn,
                        add_eos_token=False,
                        strip_bos_token=True,
                    )
                    if self.train_on_inputs:
                        labels = copy.deepcopy(res["input_ids"])
                    else:
                        # everything from this is masked out from the labels
                        labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
                elif output_turn:
                    role = (
                        role.replace(role_remap[1]["from"], role_remap[1]["to"])
                        if role_remap
                        else role
                    )
                    turn = role + content
                    # this should be the assistant response, should end with an eos token
                    if not content.strip():
                        LOG.warning(f"assistant turn has empty text: {prompt}")
                    add_eos_token = not (
                        conversation.name == "chatml"
                        and conversation.sep == self.tokenizer.eos_token
                    )
                    res = self._tokenize(
                        turn,
                        add_eos_token=add_eos_token,
                        strip_bos_token=True,
                    )
                    role_res = self._tokenize(
                        role.rstrip(),
                        add_eos_token=False,
                        strip_bos_token=True,
                    )
                    labels = copy.deepcopy(res["input_ids"])
                    if not self.train_on_inputs:
                        # mask out role tokens from the labels
                        len_role = len(role_res["input_ids"])
                        labels[:len_role] = [IGNORE_TOKEN_ID] * min(
                            len_role, len(labels)
                        )
                elif role == "":
                    turn = content
                    # this is only ever the first part, should include the bos token and the user query
                    res = self._tokenize(
                        turn, add_eos_token=False, strip_bos_token=False
                    )
                    if self.train_on_inputs:
                        labels = copy.deepcopy(res["input_ids"])
                    else:
                        # everything from this is masked out from the labels
                        labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
                else:
                    LOG.warning(f"unhandled role: {role}")
                    continue

                # pylint: disable=duplicate-code
                result, current_len = parse_tokenized_to_result(
                    result,
                    current_len,
                    res,
                    labels,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            return result
        except (KeyError, AssertionError, IndexError) as err:
            raise InvalidDataException(str(err)) from err
