"""
HF Chat Templates prompt strategy
"""

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Set, Union

from pydantic import BaseModel
from transformers import ProcessorMixin

from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer
from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import IGNORE_TOKEN_ID, Prompter
from axolotl.utils.chat_templates import get_chat_template_from_config
from axolotl.utils.dict import remove_none_values
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.datasets import DatasetConfig

if TYPE_CHECKING:
    from axolotl.utils.mistral import HFMistralTokenizer

# Configure the logger
LOG = get_logger(__name__)
LOG.setLevel("INFO")


class ChatTemplatePrompter(Prompter):
    """Prompter for HF chat templates"""

    def __init__(
        self,
        tokenizer,
        chat_template: str,
        processor=None,
        max_length=2048,
        message_property_mappings: dict[str, str] | None = None,
        message_field_training: str | None = None,
        message_field_training_detail: str | None = None,
        field_messages: str = "messages",
        field_system: str = "system",
        field_tools: str = "tools",
        field_thinking: str = "reasoning_content",
        roles: dict[str, list[str]] | None = None,
        template_thinking_key: str | None = "reasoning_content",
        chat_template_kwargs: dict[str, Any] | None = None,
        drop_system_message: bool = False,
    ):
        # check if message_property_mappings is None or empty dict
        if message_property_mappings is None or (not message_property_mappings):
            message_property_mappings = {
                "role": "role",
                "content": "content",
            }
            if template_thinking_key and field_thinking:
                message_property_mappings[template_thinking_key] = field_thinking

        if roles:
            self.roles = {s: t for t, sources in roles.items() for s in sources}
        else:
            self.roles = {
                "human": "user",
                "user": "user",
                "assistant": "assistant",
                "gpt": "assistant",
                "system": "system",
                "tool": "tool",
            }

        self._chat_template_msg_variables = self.get_chat_template_msg_variables(
            chat_template, field_messages
        )
        self.message_property_mappings = message_property_mappings
        self.message_field_training = message_field_training
        self.message_field_training_detail = message_field_training_detail
        self.field_messages = field_messages
        self.field_system = field_system
        self.field_tools = field_tools
        self.field_thinking = field_thinking
        self.tokenizer = tokenizer
        self.processor: ProcessorMixin | None = processor
        self.chat_template = chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.template_thinking_key: str = template_thinking_key or "reasoning_content"
        self.max_length = max_length
        self.drop_system_message = drop_system_message

    @property
    def chat_template_msg_variables(self) -> Set[str]:
        return self._chat_template_msg_variables

    def build_prompt(
        self,
        conversation: list[dict],
        add_generation_prompt=False,
        images=None,
        tools=None,
        real_last_index=None,
    ):
        """
        Build a prompt from a conversation.

        Args:
            conversation: A list of messages.
            add_generation_prompt: Whether to add a generation prompt.
            images: A list of images. (optional)
            tools: A list of tools. (optional)
        """
        chat_template_kwargs = {
            "chat_template": self.chat_template,
            "add_generation_prompt": add_generation_prompt,
            **self.chat_template_kwargs,
        }

        if tools:
            chat_template_kwargs["tools"] = tools

        if real_last_index:
            chat_template_kwargs["real_last_index"] = real_last_index

        if self.processor:
            if not callable(self.processor):
                raise TypeError("Processor must be callable")

            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                **chat_template_kwargs,
            )
            batch = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
            )
            if hasattr(batch, "to_dict"):
                batch = batch.to_dict()
            else:
                batch = dict(batch)

            # workaround since processor works in batches instead of single examples
            out = {}
            for k, val in batch.items():
                if hasattr(val, "tolist"):
                    out[k] = (
                        val.tolist() if k == "pixel_values" else val.squeeze(0).tolist()
                    )
                else:
                    out[k] = val
            return out

        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=False,
            **chat_template_kwargs,
        )

    def get_offsets_for_train_detail(
        self, text: str, train_details: List[Dict], mask_untrainable: bool = True
    ) -> List[int]:
        tokenized_output = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        tokens = tokenized_output.tokens()
        token_offsets = tokenized_output["offset_mapping"]

        LOG.debug(f"Tokenizing text: {text}")
        LOG.debug(f"Tokens: {tokens}")
        # Adjust the end offsets. For some reason by default they are set to the same value as the start offsets.
        for i in range(len(token_offsets) - 1):
            token_offsets[i] = (token_offsets[i][0], token_offsets[i + 1][0] - 1)
        # Ensure the last token's end offset is set correctly
        token_offsets[-1] = (token_offsets[-1][0], len(text) - 1)
        LOG.debug(f"Token offsets: {token_offsets}")

        # Initialize all offsets as IGNORE_TOKEN_ID (not trained)
        result = [IGNORE_TOKEN_ID] * len(token_offsets)

        # Adjust train_details to align with token boundaries
        adjusted_train_details = self.adjust_train_details(train_details, token_offsets)

        for idx, (start, end) in enumerate(token_offsets):
            for detail in adjusted_train_details:
                # Check if the token is completely within the detail's range
                if start >= detail["begin_offset"] and end <= detail["end_offset"]:
                    if detail["train"] or not mask_untrainable:
                        result[idx] = start
                        LOG.debug(f"Token {idx} ({tokens[idx]}) marked for training")
                    else:
                        LOG.debug(
                            f"Token {idx} ({tokens[idx]}) marked as non-trainable"
                        )
                elif start < detail["end_offset"] and end > detail["begin_offset"]:
                    # Token partially overlaps with detail, always mark as non-trainable
                    LOG.debug(
                        f"Token {idx} ({tokens[idx]}) partially overlaps detail, marked as non-trainable"
                    )

        LOG.debug(f"Final result: {result}")
        return result

    def adjust_train_details(
        self, train_details: List[Dict], token_offsets: List[tuple]
    ) -> List[Dict]:
        adjusted_details = []
        for detail in train_details:
            begin_offset = detail["begin_offset"]
            end_offset = detail["end_offset"]

            # Find the first token that starts after or at the begin_offset
            begin_token = next(
                (
                    i
                    for i, (t_start, t_end) in enumerate(token_offsets)
                    if t_start >= begin_offset
                ),
                len(token_offsets),
            )
            if begin_token > 0 and token_offsets[begin_token - 1][1] > begin_offset:
                begin_token -= 1

            # Find the last token that ends before or at the end_offset
            end_token = next(
                (
                    i
                    for i in range(len(token_offsets) - 1, -1, -1)
                    if token_offsets[i][1] <= end_offset
                ),
                -1,
            )
            if (
                end_token < len(token_offsets) - 1
                and token_offsets[end_token + 1][0] < end_offset
            ):
                end_token += 1

            if begin_token <= end_token:
                adjusted_begin = token_offsets[begin_token][0]
                adjusted_end = token_offsets[end_token][1]

                if adjusted_begin != begin_offset or adjusted_end != end_offset:
                    LOG.warning(
                        f"Adjusting detail offsets: ({begin_offset}, {end_offset}) -> ({adjusted_begin}, {adjusted_end})"
                    )

                adjusted_details.append(
                    {
                        "begin_offset": adjusted_begin,
                        "end_offset": adjusted_end,
                        "train": detail["train"],
                    }
                )
            else:
                LOG.warning(
                    f"Could not adjust detail offsets: ({begin_offset}, {end_offset}). Skipping this detail."
                )

        return adjusted_details

    def get_chat_template_msg_variables(
        self, chat_template: str, field_messages: str
    ) -> Set[str]:
        template_analyzer = JinjaTemplateAnalyzer(chat_template)
        return template_analyzer.get_message_vars(field_messages)


class ChatTemplateStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for instruction-based prompts.
    """

    def __init__(
        self,
        prompter: "ChatTemplatePrompter",
        tokenizer,
        train_on_inputs: bool,
        sequence_len: int,
        roles_to_train: list[str] | None = None,
        train_on_eos: str | None = None,
        train_on_eot: str | None = None,
        eot_tokens: list[str] | None = None,
        split_thinking: bool | None = False,
    ):
        super().__init__(prompter, tokenizer, train_on_inputs, sequence_len)
        self.prompter: ChatTemplatePrompter = prompter

        self.roles_to_train = []
        if roles_to_train:
            # map roles if exist in prompter.roles else use the role as is
            self.roles_to_train = [
                prompter.roles.get(role, role) for role in roles_to_train
            ]

        self.train_on_eos = train_on_eos
        # Backward compatibility, load from train_on_eos
        self.train_on_eot = train_on_eot if train_on_eot is not None else train_on_eos

        # Default to eos_token if eot_tokens not provided
        self.eot_tokens = []
        if eot_tokens is not None:
            self.eot_tokens = eot_tokens
        elif (
            hasattr(self.tokenizer, "eos_token")
            and self.tokenizer.eos_token is not None
        ):
            self.eot_tokens = [self.tokenizer.eos_token]

        self.split_thinking = split_thinking

        self.images = "images"

        LOG.debug(
            f"The chat template uses the following properites on the message: {self.prompter.chat_template_msg_variables}"
        )

        self._validate_eot_and_eos_tokens()

    def _validate_eot_and_eos_tokens(self):
        """
        - Validates that EOT tokens (or eos_token) are in the chat_template
        - Checks if EOT tokens are encoded as multiple tokens in the tokenizer.
        - Checks for potential conflicts between train_on_eos and train_on_eot.
        """
        if self.prompter.chat_template is None:
            # Usually this should not happen
            LOG.warning(
                "No chat template provided, skipping EOT and EOS token validation"
            )
            return

        # If the EOT token is the same as the EOS token, we need to check differently
        if len(self.eot_tokens) == 1 and self.eot_tokens[0] == self.tokenizer.eos_token:
            # Check if the eos_token is in the chat_template or as a variable `eos_token`
            # Note: we check for `eos_token` in the string, but it could possibly not be a variable
            if (
                self.tokenizer.eos_token not in self.prompter.chat_template
                and "eos_token" not in self.prompter.chat_template
            ):
                LOG.warning(
                    f"EOS token '{self.tokenizer.eos_token}' not found in chat_template. Please check if your template/EOS token is correct."
                )
            return

        # Create a new list to store tokens that should be kept
        valid_eot_tokens = []
        for token in self.eot_tokens:
            # Check if EOT token is in the chat_template
            if token not in self.prompter.chat_template:
                LOG.warning(f"EOT token '{token}' not found in chat_template.")
                # Don't add to the valid tokens list
                continue

            valid_eot_tokens.append(token)

        # Replace the original list with the filtered one
        self.eot_tokens = valid_eot_tokens

        for token in self.eot_tokens:
            # If token in template, check if EOT token is in tokenizer and not encoded as multiple tokens
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids:
                raise ValueError(
                    "EOT token encoding failed. Please check if the token is valid and can be encoded."
                )
            if token_ids and len(token_ids) > 1:
                raise ValueError(
                    f"EOT token '{token}' is encoded as multiple tokens: {token_ids}. Please add it under `tokens: ` in the config "
                    "or (recommended) override unused added_tokens via `added_tokens_overrides: `."
                )

        # If eos_token is in eot_tokens and conflict between train_on_eos and train_on_eot, raise an error
        if (
            self.tokenizer.eos_token in self.eot_tokens
            and self.train_on_eos != self.train_on_eot
        ):
            raise ValueError(
                "Conflict between train_on_eos and train_on_eot. eos_token is in eot_tokens and train_on_eos != train_on_eot"
                f"train_on_eos: {self.train_on_eos}, train_on_eot: {self.train_on_eot}"
                f"eot_tokens: {self.eot_tokens}"
                f"eos_token: {self.tokenizer.eos_token}"
            )

    @property
    def supports_batched(self) -> bool:
        # Let calling code know we can handle lists of examples
        return True

    def is_prompt_batched(self, prompt: dict[str, Any]) -> bool:
        try:
            return all(isinstance(v, list) for v in prompt.values()) and all(
                isinstance(v, list) for v in prompt[self.prompter.field_messages]
            )
        except KeyError:
            return False

    def tokenize_prompt(self, prompt: dict[str, Any]):
        """
        Public method that can handle either a single prompt or a batch of prompts.
        """

        prompt = remove_none_values(prompt)

        if not self.is_prompt_batched(prompt) or not self.supports_batched:
            return self._tokenize_single_prompt(prompt)

        res = defaultdict(lambda: [])
        feature_names = list(prompt.keys())

        # Process each prompt individually
        for row in zip(*prompt.values(), strict=False):
            tokenized_prompt = self._tokenize_single_prompt(
                dict(zip(feature_names, row, strict=False))
            )
            for key, val in tokenized_prompt.items():
                res[key].append(val)

        # If there are no examples left, return an empty dictionary
        if not res:
            return {}

        return dict(res)

    def _tokenize_single_prompt(self, prompt: dict) -> Dict[str, List[int]]:
        # Old simple legacy behavior that works reliably.
        if (
            not self.roles_to_train
            and not self.train_on_eos
            and not self.train_on_eot
            and not self.prompter.message_field_training  # type: ignore
            and not self.prompter.message_field_training_detail  # type: ignore
        ):
            turns = self.get_conversation_thread(prompt)
            images = self._get_images(prompt)
            prompt_ids = self.prompter.build_prompt(  # type: ignore
                turns[:-1],
                add_generation_prompt=True,
                images=images,
            )
            tokenized_res = self.prompter.build_prompt(turns, images=images)  # type: ignore
            tokenized_prompt = {}
            if isinstance(tokenized_res, list):
                input_ids = prompt_ids + tokenized_res[len(prompt_ids) :]
                tokenized_prompt["input_ids"] = input_ids
                tokenized_prompt["attention_mask"] = [1] * len(input_ids)
            else:
                input_ids = tokenized_res["input_ids"]
                tokenized_prompt = dict(tokenized_res)

            if not self.train_on_inputs:
                if isinstance(prompt_ids, dict):
                    user_prompt_len = len(prompt_ids["input_ids"])
                else:
                    user_prompt_len = len(prompt_ids)
                labels = [-100] * user_prompt_len + input_ids[user_prompt_len:]
            else:
                labels = input_ids

            tokenized_prompt["labels"] = labels

            return tokenized_prompt

        turns = self.get_conversation_thread(prompt)
        tools = self._get_tools(prompt)
        input_ids = self.prompter.build_prompt(turns, tools=tools)  # type: ignore
        labels = [IGNORE_TOKEN_ID] * len(input_ids)

        last_eos_idx = -1
        last_eot_idx = -1
        for index, turn in enumerate(turns):
            role = turn.get("role")
            content = turn.get("content")
            train_turn = turn.get("training")
            train_detail = turn.get("training_detail")

            LOG.debug(
                f"Processing turn {index}: role={role}, content={content}, train_turn={train_turn}, train_detail={train_detail}"
            )

            should_train = None
            if train_turn is not None:
                should_train = train_turn
            elif train_detail is not None:
                should_train = bool(train_detail)
            else:
                should_train = self.train_on_inputs or role in self.roles_to_train

            LOG.debug(f"Should train: {should_train}")

            # turn not trainable, skip having to find the turn indices
            # unless last turn and train_on_eos/train_on_eot is all
            if not should_train and (
                self.train_on_eos != "all" and self.train_on_eot != "all"
            ):
                if index == len(turns) - 1:
                    LOG.warning(
                        "Last turn is not trainable, skipping having to find the turn indices. "
                        "This may cause incorrect last EOT/EOS token to be unmasked."
                        "This is likely a dataset design issue. Please ensure last turn is trainable."
                    )

                continue

            turn_start_idx, turn_end_idx = self.find_turn(
                turns=turns, turn_idx=index, tools=tools
            )

            LOG.debug(f"Turn indices: start={turn_start_idx}, end={turn_end_idx}")

            if should_train and turn_start_idx != -1 and turn_end_idx != -1:
                if train_detail:
                    # Block multi-content for now
                    if not isinstance(content, str):
                        raise ValueError(
                            "`train_detail` is not supported when `content` is not a string."
                        )

                    token_offsets = self.prompter.get_offsets_for_train_detail(  # type: ignore
                        content, train_detail
                    )
                    LOG.debug(f"Token offsets: {token_offsets}")
                    for i, offset in enumerate(token_offsets):
                        if offset != IGNORE_TOKEN_ID and turn_start_idx + i < len(
                            input_ids
                        ):
                            labels[turn_start_idx + i] = input_ids[turn_start_idx + i]
                            LOG.debug(
                                f"Label set at index {turn_start_idx + i}: {input_ids[turn_start_idx + i]}"
                            )
                else:
                    labels[turn_start_idx:turn_end_idx] = input_ids[
                        turn_start_idx:turn_end_idx
                    ]
                    LOG.debug(
                        f"Set labels for training from {turn_start_idx} to {turn_end_idx}"
                    )

                LOG.debug(f"Labels after processing turn {index}: {labels}")

            # Handle special tokens (EOT and EOS)
            for token_type, find_func, train_option in [
                ("EOT", self.find_first_eot_token, self.train_on_eot),
                ("EOS", self.find_first_eos_token, self.train_on_eos),
            ]:
                token_idx = find_func(input_ids, start_idx=turn_end_idx)

                if (
                    token_idx != -1 and abs(token_idx - turn_end_idx) <= 3
                ):  # Allow for some template padding
                    # Update the last token index
                    if token_type == "EOT":  # nosec B105
                        last_eot_idx = token_idx
                    else:
                        last_eos_idx = token_idx

                    # Set labels if needed for this turn
                    if train_option == "all" or (
                        train_option == "turn" and should_train
                    ):
                        labels[token_idx] = input_ids[token_idx]
                        LOG.debug(
                            f"{token_type} token set for training at index {token_idx}"
                        )
                else:
                    LOG.debug(
                        f"{token_type} token missing after turn {turn}. {token_type.lower()}_idx: {token_idx}, turn_end_idx: {turn_end_idx}"
                    )

        # Handle 'last' option for special tokens
        for token_type, last_idx, train_option in [
            ("EOT", last_eot_idx, self.train_on_eot),
            ("EOS", last_eos_idx, self.train_on_eos),
        ]:
            if train_option == "last" and last_idx != -1:
                labels[last_idx] = input_ids[last_idx]
                LOG.debug(
                    f"Last {token_type} token set for training at index {last_idx}"
                )

        LOG.debug(f"Final labels: {labels}")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    def find_first_eos_token(self, input_ids, start_idx):
        eos_token_id = self.tokenizer.eos_token_id
        for i in range(start_idx, len(input_ids)):
            if input_ids[i] == eos_token_id:
                return i
        return -1

    def find_first_eot_token(self, input_ids, start_idx):
        """Find the first EOT token in the input_ids starting from start_idx."""
        # Get token IDs for all EOT tokens
        eot_token_ids = []
        for token in self.eot_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"EOT token '{token}' is encoded as multiple tokens: {token_ids}. Please add it under `tokens: ` in the config."
                )

            eot_token_ids.append(token_ids[0])  # Use the last token ID if multiple

        # Search for any of the EOT token IDs
        for i in range(start_idx, len(input_ids)):
            if input_ids[i] in eot_token_ids:
                return i
        return -1

    def find_turn(
        self, turns: list[dict], turn_idx: int, tools: list[dict] | None = None
    ):
        """
        Locate the starting and ending indices of the specified turn in a conversation.
        """

        if turn_idx >= len(turns):
            raise ValueError(f"Turn index {turn_idx} out of range")

        # mistral/gemma3 does not output message if it contains only system message
        if (
            turn_idx == 0
            and turns[0].get("role") == "system"
            and ("mistral" in self.tokenizer.name_or_path.lower())
        ):
            return -1, -1

        empty_turn = {
            "role": turns[turn_idx].get("role"),
            "content": "[[dummy_message]]",
        }

        # Create conversation versions
        turns_with_empty = turns[:turn_idx] + [empty_turn]
        turns_with_content = turns[: turn_idx + 1]

        real_last_index = len(turns) - 1

        # Generate the conversation up to the turn, with final turn replaced with dummy content
        dummy_ids = self.prompter.build_prompt(
            turns_with_empty, tools=tools, real_last_index=real_last_index
        )  # type: ignore

        # Generate the conversation up to the turn, with final turn included
        full_ids = self.prompter.build_prompt(
            turns_with_content, tools=tools, real_last_index=real_last_index
        )  # type: ignore

        if not full_ids or not dummy_ids:
            LOG.warning(f"Empty template generated for turn {turn_idx}")
            return -1, -1

        # Find first difference (start of content)
        start_idx = None
        min_len = min(len(dummy_ids), len(full_ids))
        for i in range(min_len):
            if dummy_ids[i] != full_ids[i]:
                start_idx = i
                break

        if start_idx is None:
            LOG.warning(f"Could not find content start boundary for turn {turn_idx}")
            return -1, -1

        # Find last difference (end of content)
        end_idx = None
        for i in range(min_len):
            dummy_pos = len(dummy_ids) - 1 - i
            full_pos = len(full_ids) - 1 - i
            if dummy_ids[dummy_pos] != full_ids[full_pos]:
                end_idx = full_pos + 1  # Add one to include the last token when slice
                break

        if end_idx is None:
            LOG.warning(f"Could not find content end boundary for turn {turn_idx}")
            return -1, -1

        if end_idx < start_idx:
            LOG.warning(
                f"Content end boundary is before start boundary for turn {turn_idx}"
            )
            return -1, -1

        if end_idx == start_idx:
            LOG.warning(
                f"Content end boundary is the same as start boundary for turn {turn_idx}. This is likely an empty turn."
            )
            return -1, -1

        LOG.debug(f"Content boundaries: {start_idx}, {end_idx}")
        LOG.debug(
            f"Content tokens: {self.tokenizer.convert_ids_to_tokens(full_ids[start_idx:end_idx])}"
        )

        return start_idx, end_idx

    def get_conversation_thread(self, prompt):
        turns = []

        messages = self._get_messages(prompt)

        possible_sys_turn = self.transform_message(messages[0])

        if (
            possible_sys_turn["role"] != "system"
            and self.prompter.field_system in prompt
        ):
            turn = {"role": "system", "content": prompt[self.prompter.field_system]}
            turns.append(turn)

        for message in messages:
            transformed_message = self.transform_message(message)

            turn = transformed_message

            training = message.get(self.prompter.message_field_training)
            training_detail = message.get(self.prompter.message_field_training_detail)
            if training is not None:
                turn["training"] = training
            if training_detail is not None:
                turn["training_detail"] = training_detail

            turns.append(turn)

        if self.prompter.drop_system_message and turns[0]["role"] == "system":
            turns = turns[1:]

        return turns

    def transform_message(self, message: dict) -> dict:
        # Build the initial transformed message from the mappings
        transformed_message = {}
        for key, value in self.prompter.message_property_mappings.items():
            if message.get(value) is not None:
                transformed_message[key] = message[value]
            else:
                LOG.debug(
                    f"Could not find value for property {value} in message: {message}"
                )

        # Map the role if necessary
        if "role" in transformed_message:
            transformed_message["role"] = self.prompter.roles.get(
                transformed_message["role"], transformed_message["role"]
            )

        # TODO handle reasoning_content with split_thinking
        # if the role is assistant that we want to use reasoning_content
        if self.split_thinking and transformed_message["role"] == "assistant":
            content = transformed_message["content"]
            thinking_pairs = [
                ("<think>", "</think>"),
                ("<reasoning>", "</reasoning>"),
                ("<|begin_of_thought|>", "<|end_of_thought|>"),
            ]
            content_pairs = [("<|begin_of_solution|>", "<|end_of_solution|>")]
            for tpair in thinking_pairs:
                # check if the thinking pair is in the content
                if tpair[0] in content and tpair[1] in content:
                    # find the start and end index of the thinking pair
                    t_start_idx = content.find(tpair[0])
                    t_end_idx = content.find(tpair[1])

                    # get the thinking content
                    thinking_content = content[t_start_idx + len(tpair[0]) : t_end_idx]
                    transformed_message[self.prompter.template_thinking_key] = (
                        thinking_content.strip()
                    )

                    # take remainder of the content
                    # strip whitespace from beginning of the remainder (thinking tokens)
                    remainder = content[t_end_idx + len(tpair[1]) :].lstrip()

                    # check if the content pair is in the remainder
                    cpair_found = False
                    for cpair in content_pairs:
                        if cpair[0] in remainder and cpair[1] in remainder:
                            # find the start and end index of the content pair
                            c_start_idx = remainder.find(cpair[0])
                            c_end_idx = remainder.find(cpair[1])

                            # get the content content
                            content_content = remainder[
                                c_start_idx + len(cpair[0]) : c_end_idx
                            ]
                            transformed_message["content"] = content_content.strip()
                            cpair_found = True
                            break

                    # else, the content is the remainder
                    if not cpair_found:
                        transformed_message["content"] = remainder
                    break

        # Determine which keys in the original message were not mapped
        mapped_values = set(self.prompter.message_property_mappings.values())
        remaining_keys = set(message) - mapped_values

        # Keep only the properties defined in the chat template
        # and not already mapped
        for key in self.prompter.chat_template_msg_variables:
            if key in remaining_keys:
                val = message.get(key)
                if val is not None:
                    transformed_message[key] = val

        if "tool_calls" in transformed_message and transformed_message["tool_calls"]:
            for tool_call in transformed_message["tool_calls"]:
                if "function" in tool_call and "arguments" in tool_call["function"]:
                    args = tool_call["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            tool_call["function"]["arguments"] = json.loads(args)
                        except json.JSONDecodeError as e:
                            LOG.error(
                                f"Error parsing tool_calls arguments as JSON. "
                                f"Function: {tool_call.get('function', {}).get('name', 'unknown')}, "
                                f"Arguments string: {args!r}, "
                                f"Error: {e}"
                            )
                            raise

        return transformed_message

    def _get_images(self, prompt):
        return prompt.get(self.images, None)

    def _get_tools(self, prompt) -> list[dict] | None:
        """Get tools from prompt if available."""
        tools = prompt.get(self.prompter.field_tools, None)
        if tools is None:
            return None

        if isinstance(tools, list):
            # Process each tool to handle JSON string parameters
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    function = tool["function"]
                    if "parameters" in function:
                        params = function["parameters"]
                        if isinstance(params, str):
                            try:
                                function["parameters"] = json.loads(params)
                            except json.JSONDecodeError as e:
                                LOG.error(
                                    f"Error parsing tool parameters as JSON. "
                                    f"Function: {function.get('name', 'unknown')}, "
                                    f"Parameters string: {params!r}, "
                                    f"Error: {e}"
                                )
                                raise
            return tools

        raise ValueError(
            "Unknown tools format. Please convert it into a list[dict].\n"
            f"Current format: {type(tools)}"
        )

    def _get_messages(self, prompt):
        messages = prompt.get(self.prompter.field_messages, None)
        if messages is None:
            raise ValueError("Messages is null. Please check `field_messages`.")

        if isinstance(messages, list):
            return messages

        raise ValueError(
            "Unknown messages format. Please convert it into a list[dict].\n"
            f"Current format: {type(messages)}"
        )


class MistralStrategy(ChatTemplateStrategy):
    """
    Mistral strategy for chat template.
    """

    def __init__(
        self,
        prompter: "ChatTemplatePrompter",
        tokenizer: "HFMistralTokenizer",
        train_on_inputs: bool,
        sequence_len: int,
        roles_to_train: list[str] | None = None,
        train_on_eos: str | None = None,
        train_on_eot: str | None = None,
        eot_tokens: list[str] | None = None,
        split_thinking: bool | None = False,
    ):
        # Call the parent's parent __init__ (PromptTokenizingStrategy) to skip ChatTemplateStrategy's validation

        PromptTokenizingStrategy.__init__(
            self, prompter, tokenizer, train_on_inputs, sequence_len
        )
        self.prompter: ChatTemplatePrompter = prompter

        self.roles_to_train = []
        if roles_to_train:
            # map roles if exist in prompter.roles else use the role as is
            self.roles_to_train = [
                prompter.roles.get(role, role) for role in roles_to_train
            ]

        self.train_on_eos = train_on_eos
        # Backward compatibility, load from train_on_eos
        self.train_on_eot = train_on_eot if train_on_eot is not None else train_on_eos

        # Default to eos_token if eot_tokens not provided
        self.eot_tokens = []
        if eot_tokens is not None:
            self.eot_tokens = eot_tokens
        else:
            # set eot_tokens to the eos_token
            self.eot_tokens = [self.tokenizer.eos_token]

        self.split_thinking = split_thinking

        self.images = "images"

        LOG.debug(
            f"The chat template uses the following properites on the message: {self.prompter.chat_template_msg_variables}"
        )

        # Skip the validation that ChatTemplateStrategy calls
        # TODO: address this in the future with mistral-specific checks
        # self._validate_eot_and_eos_tokens()

    def find_first_eot_token(self, input_ids, start_idx):
        """Find the first EOT token in the input_ids starting from start_idx."""
        # mistral-common tokenizer does not support eot_tokens
        return self.find_first_eos_token(input_ids, start_idx)


class MistralPrompter(ChatTemplatePrompter):
    """
    Mistral prompter for chat template.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._chat_template_msg_variables = set(["tool_call_id", "name", "tool_calls"])


class StrategyLoader:
    """
    Load chat template strategy based on configuration.
    """

    def _get_strategy_cls(self, cfg):
        if cfg.tokenizer_use_mistral_common:
            return MistralStrategy

        return ChatTemplateStrategy

    def _get_prompter_cls(self, cfg):
        if cfg.tokenizer_use_mistral_common:
            return MistralPrompter

        return ChatTemplatePrompter

    def _get_strategy_params(self, cfg, ds_cfg: Dict[str, Any]):
        return {
            "train_on_inputs": cfg.train_on_inputs,
            "sequence_len": cfg.sequence_len,
            "roles_to_train": ds_cfg.get("roles_to_train", ["assistant"]),
            "train_on_eos": ds_cfg.get("train_on_eos", "turn"),
            "train_on_eot": ds_cfg.get("train_on_eot", None),
            "eot_tokens": cfg.get("eot_tokens", None),  # loads from cfg, not ds_cfg
            "split_thinking": ds_cfg.get("split_thinking", False),
        }

    def __call__(
        self,
        tokenizer,
        cfg,
        ds_cfg: Union[Dict[str, Any], DatasetConfig] | None = None,
        processor=None,
    ):
        if ds_cfg is None:
            dataset_config = {}
        elif isinstance(ds_cfg, BaseModel):
            dataset_config = ds_cfg.model_dump()
        else:
            dataset_config = ds_cfg

        if cfg.tokenizer_use_mistral_common:
            # mistral-common does not use this, so we pass an empty string
            chat_template_string = ""
        else:
            chat_template_string = get_chat_template_from_config(
                cfg=cfg, ds_cfg=dataset_config, tokenizer=tokenizer
            )

        LOG.info(f"Using chat template:\n---\n{chat_template_string!s}\n---")

        prompter_params = {
            "tokenizer": tokenizer,
            "chat_template": chat_template_string,
            "chat_template_kwargs": cfg.get("chat_template_kwargs", {}),
            "message_property_mappings": dataset_config.get(
                "message_property_mappings", {}
            ),
            "message_field_training": dataset_config.get(
                "message_field_training", None
            ),
            "message_field_training_detail": dataset_config.get(
                "message_field_training_detail",
                None,
            ),
            "field_messages": dataset_config.get("field_messages", "messages"),
            "field_thinking": dataset_config.get("field_thinking", "reasoning_content"),
            "template_thinking_key": dataset_config.get(
                "template_thinking_key", "reasoning_content"
            ),
            "roles": dataset_config.get("roles"),
            "drop_system_message": dataset_config.get("drop_system_message", False),
            # we need to add one for detecting sequences with exceeding the `sequence_len` limit.
            "max_length": cfg.sequence_len + 1,
            "processor": processor,
        }

        strategy_params = self._get_strategy_params(cfg, dataset_config)
        strategy_cls = self._get_strategy_cls(cfg)
        prompter_cls = self._get_prompter_cls(cfg)

        strategy = strategy_cls(
            prompter_cls(**prompter_params),
            tokenizer=tokenizer,
            **strategy_params,
        )

        return strategy


load = StrategyLoader()
