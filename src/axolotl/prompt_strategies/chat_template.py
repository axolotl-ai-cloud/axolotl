"""
HF Chat Templates prompt strategy
"""

import logging
from typing import Any, Dict, List, Optional

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import IGNORE_TOKEN_ID, Prompter
from axolotl.utils.chat_templates import chat_templates

# Configure the logger
LOG = logging.getLogger("axolotl")
LOG.setLevel(logging.INFO)


class ChatTemplatePrompter(Prompter):
    """Prompter for HF chat templates"""

    def __init__(
        self,
        tokenizer,
        chat_template=None,
        max_length=2048,
        message_field_role: str = "from",
        message_field_content: str = "value",
        message_field_training: Optional[str] = None,
        message_field_training_detail: Optional[str] = None,
        roles: Optional[Dict[str, List[str]]] = None,
        drop_system_message: bool = False,
    ):
        if roles:
            self.roles = {s: t for t, sources in roles.items() for s in sources}
        else:
            self.roles = {
                "human": "user",
                "user": "user",
                "assistant": "assistant",
                "gpt": "assistant",
                "system": "system",
            }
        self.message_field_role = message_field_role
        self.message_field_content = message_field_content
        self.message_field_training = message_field_training
        self.message_field_training_detail = message_field_training_detail
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length
        self.drop_system_message = drop_system_message

    def build_prompt(self, conversation, add_generation_prompt=False):
        turns = [
            {
                "role": self.roles[t[self.message_field_role]],
                "content": t[self.message_field_content],
                "training": t.get(self.message_field_training, None),
            }
            for t in conversation
        ]

        if self.drop_system_message and turns[0]["role"] == "system":
            turns = turns[1:]

        return self.tokenizer.apply_chat_template(
            turns,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.chat_template,
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


class ChatTemplateStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for instruction-based prompts.
    """

    _messages = "conversations"

    def __init__(
        self,
        prompter,
        tokenizer,
        train_on_inputs,
        sequence_len,
        roles_to_train=None,
        train_on_eos=None,
    ):
        super().__init__(prompter, tokenizer, train_on_inputs, sequence_len)
        self.roles_to_train = roles_to_train if roles_to_train is not None else []
        self.train_on_eos = train_on_eos

    @property
    def messages(self):
        return self._messages

    @messages.setter
    def messages(self, messages):
        self._messages = messages

    def tokenize_prompt(self, prompt):
        # Old simple legacy behavior that works reliably.
        if (
            not self.roles_to_train
            and not self.train_on_eos
            and not self.prompter.message_field_training
            and not self.prompter.message_field_training_detail
        ):
            turns = self.get_conversation_thread(prompt)
            prompt_ids = self.prompter.build_prompt(
                turns[:-1], add_generation_prompt=True
            )
            input_ids = self.prompter.build_prompt(turns)

            if not self.train_on_inputs:
                user_prompt_len = len(prompt_ids)
                labels = [-100] * user_prompt_len + input_ids[user_prompt_len:]
            else:
                labels = input_ids

            tokenized_prompt = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            }

            return tokenized_prompt
        LOG.info(self.roles_to_train)
        LOG.info(self.train_on_eos)
        LOG.info(self.prompter.message_field_training)
        LOG.info(self.prompter.message_field_training_detail)

        turns = prompt[self.messages]
        input_ids = self.prompter.build_prompt(turns)
        labels = [IGNORE_TOKEN_ID] * len(input_ids)

        last_eos_idx = -1
        for index, turn in enumerate(turns):
            role = turn.get(self.prompter.message_field_role)
            content = turn.get(self.prompter.message_field_content)
            train_turn = turn.get(self.prompter.message_field_training)
            train_detail = turn.get(self.prompter.message_field_training_detail)

            LOG.debug(
                f"Processing turn {index}: role={role}, content={content}, train_turn={train_turn}, train_detail={train_detail}"
            )

            should_train = (
                train_turn
                if train_turn is not None
                else (
                    bool(train_detail is not None)
                    if train_detail is not None
                    else self.train_on_inputs or role in self.roles_to_train
                )
            )

            LOG.debug(f"Should train: {should_train}")

            turn_start_idx, turn_end_idx = self.find_turn(
                conversation_ids=input_ids, turn=index, turn_content=turn
            )

            LOG.debug(f"Turn indices: start={turn_start_idx}, end={turn_end_idx}")

            if should_train and turn_start_idx != -1 and turn_end_idx != -1:
                if train_detail:
                    token_offsets = self.prompter.get_offsets_for_train_detail(
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
                    LOG.debug(f"Labels set for range {turn_start_idx}:{turn_end_idx}")

                LOG.debug(f"Labels after processing turn {index}: {labels}")

            # Handle EOS token
            eos_idx = self.find_eos_token(input_ids, turn_end_idx)
            if eos_idx == turn_end_idx:
                last_eos_idx = eos_idx
                if self.train_on_eos == "all" or (
                    self.train_on_eos == "turn" and should_train
                ):
                    labels[eos_idx] = input_ids[eos_idx]
                    LOG.debug(f"EOS token set for training at index {eos_idx}")
            else:
                LOG.debug(
                    f"EOS token missing after turn {turn}. eos_idx: {eos_idx}, turn_end_idx: {turn_end_idx}"
                )

        # Handle 'last' option for train_on_eos
        if self.train_on_eos == "last" and last_eos_idx != -1:
            labels[last_eos_idx] = input_ids[last_eos_idx]
            LOG.debug(f"Last EOS token set for training at index {last_eos_idx}")

        LOG.debug(f"Final labels: {labels}")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    def find_eos_token(self, input_ids, start_idx):
        eos_token_id = self.tokenizer.eos_token_id
        for i in range(start_idx, len(input_ids)):
            if input_ids[i] == eos_token_id:
                return i
        return -1

    def find_turn(self, conversation_ids, turn, turn_content):
        """
        Locate the starting and ending indices of the specified turn in a conversation.

        Args:
            conversation_ids (list[int]): Token IDs representing the conversation.
            turn (int): The turn number to locate (based on EOS tokens).
            turn_content (str): String containing the content of the turn.

        Returns:
            tuple: (start_idx, end_idx) indices of the start and end of the turn content.
                   Returns (-1, -1) if the turn content is not found.
        """
        content = turn_content.get(self.prompter.message_field_content, "")
        content_ids = self.tokenizer.encode(content, add_special_tokens=False)

        eos_token_id = self.tokenizer.eos_token_id
        eos_count = 0
        start_search_idx = 0

        # Locate the starting index after the specified number of EOS tokens
        for i, token_id in enumerate(conversation_ids):
            if token_id == eos_token_id:
                eos_count += 1
                if eos_count == turn:
                    start_search_idx = (
                        i + 1
                    )  # Start searching after the specified turn's EOS token
                    break

        # Find the start index of the content within the conversation
        start_idx = -1
        for i in range(start_search_idx, len(conversation_ids) - len(content_ids) + 1):
            if conversation_ids[i : i + len(content_ids)] == content_ids:
                start_idx = i
                break

        if start_idx != -1:
            end_idx = start_idx + len(content_ids)
        else:
            end_idx = -1

        return start_idx, end_idx

    def get_conversation_thread(self, prompt):
        return prompt[self.messages]


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    ds_cfg = ds_cfg or {}

    prompter_params = {
        "tokenizer": tokenizer,
        "chat_template": chat_templates(ds_cfg.get("chat_template", "chatml")),
        "message_field_role": ds_cfg.get("message_field_role", "role"),
        "message_field_content": ds_cfg.get("message_field_content", "content"),
        "message_field_training": ds_cfg.get("message_field_training", None),
        "message_field_training_detail": ds_cfg.get(
            "message_field_training_detail",
            None,
        ),
        "roles": ds_cfg.get("roles"),
        "drop_system_message": ds_cfg.get("drop_system_message", False),
        # we need to add one for detecting sequences with exceeding the `sequence_len` limit.
        "max_length": cfg.sequence_len + 1,
    }

    strategy_params = {
        "train_on_inputs": cfg.train_on_inputs,
        "sequence_len": cfg.sequence_len,
        "roles_to_train": ds_cfg.get("roles_to_train", []),
        "train_on_eos": ds_cfg.get("train_on_eos", None),
    }

    strategy = ChatTemplateStrategy(
        ChatTemplatePrompter(**prompter_params), tokenizer=tokenizer, **strategy_params
    )

    if "field_messages" in ds_cfg and hasattr(strategy, "messages"):
        strategy.messages = ds_cfg["field_messages"]

    return strategy
