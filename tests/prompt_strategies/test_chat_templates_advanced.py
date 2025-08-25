"""
tests for chat_template prompt strategy
"""

from copy import deepcopy

import pytest
from datasets import Dataset
from tokenizers import AddedToken
from transformers import PreTrainedTokenizer

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.prompters import IGNORE_TOKEN_ID
from axolotl.utils.chat_templates import get_chat_template
from axolotl.utils.logging import get_logger

from tests.hf_offline_utils import enable_hf_offline

LOG = get_logger(__name__)

PARAMETRIZE_KEYS = "tokenizer, chat_template, chat_template_jinja, eos_token"
PARAMETRIZE_PARAMS = [
    ("llama3_tokenizer", "llama3", None, None),
    ("llama3_tokenizer", "chatml", None, "<|im_end|>"),
    (
        "mistralv03_tokenizer",
        "jinja",
        "mistralv03_tokenizer_chat_template_jinja",
        "[/INST]",
    ),
    (
        "gemma2_tokenizer",
        "jinja",
        "gemma2_tokenizer_chat_template_jinja",
        "<end_of_turn>",
    ),
    ("phi35_tokenizer", "phi_35", None, "<|end|>"),
    ("phi4_tokenizer", "phi_4", None, "<|im_end|>"),
]


@pytest.mark.parametrize(
    PARAMETRIZE_KEYS,
    PARAMETRIZE_PARAMS,
)
class TestChatTemplateConfigurations:
    """
    Test class for various configurations of ChatTemplateStrategy.
    """

    @staticmethod
    def setup_tokenizer(
        tokenizer_name,
        chat_template,
        chat_template_jinja=None,
        eos_token=None,
        request=None,
        eot_token=None,
    ) -> tuple[PreTrainedTokenizer, str]:
        """
        Helper function to set up the tokenizer and chat template for the test.
        """
        tokenizer = deepcopy(request.getfixturevalue(tokenizer_name))
        if chat_template == "jinja":
            chat_template_jinja = request.getfixturevalue(chat_template_jinja)
        if eos_token:
            tokenizer.add_special_tokens(
                {
                    "eos_token": AddedToken(
                        eos_token, rstrip=False, lstrip=False, normalized=False
                    )
                }
            )
            if tokenizer.__class__.__name__ in (
                "LlamaTokenizerFast",
                "CodeLlamaTokenizerFast",
            ):
                tokenizer.update_post_processor()

        if eot_token:
            tokenizer.add_special_tokens({"additional_special_tokens": [eot_token]})

        return tokenizer, chat_template_jinja

    def _should_skip_turn(self, tokenizer, turn, turn_idx, start_idx, end_idx):
        """Helper method to determine if a turn should be skipped in testing.
        This is used to skip system messages for Mistral as the template does not output them without more turns.
        """
        if (
            turn_idx == 0
            and turn.get("from") in ["system", "context"]
            and ("mistral" in tokenizer.name_or_path.lower())
        ):
            assert start_idx == -1 and end_idx == -1, (
                "Expected system message to be skipped"
            )
            return True
        return False

    @enable_hf_offline
    def test_train_on_inputs_true(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing with train_on_inputs=True")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=True,
            sequence_len=512,
            roles_to_train=["assistant"],
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        turns = strategy.get_conversation_thread(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Verify assistant responses are labeled
        for i, turn in enumerate(basic_dataset[0]["conversations"]):
            start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)

            if self._should_skip_turn(tokenizer, turn, i, start_idx, end_idx):
                continue

            decoded_response = tokenizer.decode(input_ids[start_idx:end_idx])
            response = turn["value"]

            assert response in decoded_response, (
                f"Response {response} not found in index {start_idx}:{end_idx} "
                f"decoded:{decoded_response}"
            )

            assert all(
                label != IGNORE_TOKEN_ID for label in labels[start_idx:end_idx]
            ), (
                f"Expected labels for input '{response}' to be ignored, but got {labels[start_idx:end_idx]}"
            )

        LOG.debug("Full labels: %s", labels)
        LOG.debug("Full input_ids: %s", input_ids)

    def test_train_on_inputs_false(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing with train_on_inputs=False, on assistant only")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        turns = strategy.get_conversation_thread(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Process all turns and verify correct labeling based on role
        for i, turn in enumerate(basic_dataset[0]["conversations"]):
            start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)

            if self._should_skip_turn(tokenizer, turn, i, start_idx, end_idx):
                continue

            decoded_response = tokenizer.decode(input_ids[start_idx:end_idx])
            response = turn["value"]

            assert response in decoded_response, (
                f"Response {response} not found in index {start_idx}:{end_idx} "
                f"decoded:{decoded_response}"
            )

            # Verify that assistant responses are labeled and other inputs are not
            is_assistant = turn["from"] == "assistant"
            if is_assistant:
                assert all(
                    label != IGNORE_TOKEN_ID for label in labels[start_idx:end_idx]
                ), (
                    f"Expected labels for assistant response '{response}' to be set, but got {labels[start_idx:end_idx]}"
                )
            else:
                assert all(
                    label == IGNORE_TOKEN_ID for label in labels[start_idx:end_idx]
                ), (
                    f"Expected labels for human input '{response}' to be IGNORE_TOKEN_ID, but got {labels[start_idx:end_idx]}"
                )

    def test_roles_to_train_human_assistant_only(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing roles_to_train with human assistant only")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant", "human"],
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        turns = strategy.get_conversation_thread(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Process all turns and verify correct labeling based on role
        for i, turn in enumerate(basic_dataset[0]["conversations"]):
            start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)

            if self._should_skip_turn(tokenizer, turn, i, start_idx, end_idx):
                continue

            decoded_response = tokenizer.decode(input_ids[start_idx:end_idx])
            response = turn["value"]

            assert response in decoded_response, (
                f"Response {response} not found in index {start_idx}:{end_idx} "
                f"decoded:{decoded_response}"
            )

            # Verify that non-system responses are labeled and system are not
            should_be_labelled = turn["from"] != "system"
            if should_be_labelled:
                assert all(
                    label != IGNORE_TOKEN_ID for label in labels[start_idx:end_idx]
                ), (
                    f"Expected labels for assistant response '{response}' to be set, but got {labels[start_idx:end_idx]}"
                )
            else:
                assert all(
                    label == IGNORE_TOKEN_ID for label in labels[start_idx:end_idx]
                ), (
                    f"Expected labels for human input '{response}' to be IGNORE_TOKEN_ID, but got {labels[start_idx:end_idx]}"
                )

    def test_roles_to_train_all(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing roles_to_train with all roles")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=True,
            sequence_len=512,
            roles_to_train=["human", "assistant"],
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        turns = strategy.get_conversation_thread(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Verify that all responses are labeled (except for special tokens)
        for i, turn in enumerate(basic_dataset[0]["conversations"]):
            response = turn["value"]

            start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)

            if self._should_skip_turn(tokenizer, turn, i, start_idx, end_idx):
                continue

            decoded_response = tokenizer.decode(input_ids[start_idx:end_idx])
            assert response in decoded_response, (
                f"Response {response} not found in index {start_idx}:{end_idx} decoded:{decoded_response}"
            )

            assert all(
                label != IGNORE_TOKEN_ID for label in labels[start_idx:end_idx]
            ), (
                f"Expected labels for response '{response}' to be set, but got {labels[start_idx:end_idx]}"
            )

    def test_empty_roles_to_train(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing with empty roles_to_train")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=[],
            train_on_eos="none",  # Add this line
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]

        # Verify that no labels are set when roles_to_train is empty
        LOG.debug("Full labels: %s", labels)
        assert all(label == IGNORE_TOKEN_ID for label in labels), (
            "Expected all labels to be IGNORE_TOKEN_ID when roles_to_train is empty"
        )

    def test_train_on_eos_all(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing with train_on_eos='all'")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="all",
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        eos_token_id = tokenizer.eos_token_id
        eos_indices = [
            i for i, token_id in enumerate(input_ids) if token_id == eos_token_id
        ]

        assert len(eos_indices) > 0, "Expected at least one EOS token in the input"
        for eos_idx in eos_indices:
            assert labels[eos_idx] != IGNORE_TOKEN_ID, (
                f"Expected EOS token at index {eos_idx} to be labeled"
            )

    def test_train_on_eos_turn(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing with train_on_eos='turn'")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="turn",
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        turns = strategy.get_conversation_thread(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        eos_token_id = tokenizer.eos_token_id
        # Process all turns and verify EOS token labeling
        for i, turn in enumerate(basic_dataset[0]["conversations"]):
            start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)

            if self._should_skip_turn(tokenizer, turn, i, start_idx, end_idx):
                continue

            decoded_response = tokenizer.decode(input_ids[start_idx:end_idx])
            response = turn["value"]

            assert response in decoded_response, (
                f"Response {response} not found in index {start_idx}:{end_idx} "
                f"decoded:{decoded_response}"
            )

            # Find the EOS token after this turn
            eos_idx = end_idx
            while eos_idx < len(input_ids) and input_ids[eos_idx] != eos_token_id:
                eos_idx += 1

            assert eos_idx < len(input_ids), (
                f"Could not find EOS token after '{response}'"
            )

            LOG.debug(
                f"Turn {i}: role={turn['from']}, content='{turn['value']}', start_idx={start_idx}, end_idx={end_idx}, eos_idx={eos_idx}"
            )

            LOG.debug(
                f"Labels for turn {i}: {labels[start_idx:end_idx]}, EOS label: {labels[eos_idx]}"
            )

            # Verify EOS token labeling based on role
            is_assistant = turn["from"] == "assistant"
            if is_assistant:
                assert labels[eos_idx] != IGNORE_TOKEN_ID, (
                    f"Expected EOS token after assistant response '{response}' to be labeled"
                )
            else:
                assert labels[eos_idx] == IGNORE_TOKEN_ID, (
                    f"Expected EOS token after non-assistant input '{response}' to not be labeled"
                )

    def test_train_on_eos_last(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing with train_on_eos='last'")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="last",
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        eos_token_id = tokenizer.eos_token_id
        eos_indices = [
            i for i, token_id in enumerate(input_ids) if token_id == eos_token_id
        ]

        assert len(eos_indices) > 0, "Expected at least one EOS token in the input"
        last_eos_idx = eos_indices[-1]

        # Check that only the last EOS token is labeled
        for idx in eos_indices[:-1]:
            assert labels[idx] == IGNORE_TOKEN_ID, (
                f"Expected EOS token at index {idx} to not be labeled"
            )
        assert labels[last_eos_idx] != IGNORE_TOKEN_ID, (
            f"Expected last EOS token at index {last_eos_idx} to be labeled"
        )

    def test_train_on_eos_none(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing with train_on_eos='none'")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="none",
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        eos_token_id = tokenizer.eos_token_id
        eos_indices = [
            i for i, token_id in enumerate(input_ids) if token_id == eos_token_id
        ]

        assert len(eos_indices) > 0, "Expected at least one EOS token in the input"
        for eos_idx in eos_indices:
            assert labels[eos_idx] == IGNORE_TOKEN_ID, (
                f"Expected EOS token at index {eos_idx} to not be labeled"
            )

    def test_drop_system_message(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        LOG.info("Testing with drop_system_message=True")
        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                drop_system_message=True,
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        input_ids = res["input_ids"]

        # Check if system message is not present in input_ids
        system_message = "You are an AI assistant."
        decoded_message = tokenizer.decode(input_ids)
        assert system_message not in decoded_message, (
            "Expected system message to be dropped"
        )

    def test_custom_roles(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        request,
    ):
        LOG.info("Testing with custom roles mapping")
        custom_roles = {
            "user": ["human", "user"],
            "assistant": ["ai", "assistant"],
            "system": ["context"],
        }
        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                roles=custom_roles,
                message_property_mappings={"role": "from", "content": "value"},
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["ai"],
        )

        # Create a new dataset with modified role names
        modified_conversations = [
            {"from": "context", "value": "You are an AI assistant."},
            {"from": "human", "value": "Hello"},
            {"from": "ai", "value": "Hi there!"},
            {"from": "human", "value": "How are you?"},
            {"from": "ai", "value": "I'm doing well, thank you!"},
        ]

        modified_dataset = Dataset.from_dict({"messages": [modified_conversations]})

        res = strategy.tokenize_prompt(modified_dataset[0])
        turns = strategy.get_conversation_thread(modified_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Process all turns and verify labeling
        for i, turn in enumerate(modified_dataset[0]["messages"]):
            start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)

            if self._should_skip_turn(tokenizer, turn, i, start_idx, end_idx):
                continue

            decoded_response = tokenizer.decode(input_ids[start_idx:end_idx])
            response = turn["value"]

            assert response in decoded_response, (
                f"Response {response} not found in index {start_idx}:{end_idx} "
                f"decoded:{decoded_response}"
            )

            # Check if responses are labeled correctly based on role
            is_ai = turn["from"] == "ai"
            if is_ai:
                assert all(
                    label != IGNORE_TOKEN_ID for label in labels[start_idx:end_idx]
                ), f"Expected labels for AI response '{response}' to be set"
            else:
                assert all(
                    label == IGNORE_TOKEN_ID for label in labels[start_idx:end_idx]
                ), (
                    f"Expected labels for non-AI message '{response}' to be IGNORE_TOKEN_ID"
                )

    def test_message_field_training(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        request,
    ):
        LOG.info("Testing with message_field_training")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_field_training="train",
                message_field_training_detail="train_detail",
                message_property_mappings={"role": "from", "content": "value"},
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=[],
        )

        # Create a new dataset with the train and train_detail fields
        modified_conversation = [
            {"from": "system", "value": "You are an AI assistant.", "train": False},
            {"from": "human", "value": "Hello", "train": False},
            {"from": "assistant", "value": "Hello", "train": True},
            {"from": "human", "value": "How are you?", "train": True},
            {
                "from": "assistant",
                "value": "I'm doing very well, thank you!",
                "train_detail": [
                    {"begin_offset": 0, "end_offset": 8, "train": False},
                    {"begin_offset": 9, "end_offset": 18, "train": True},
                    {"begin_offset": 19, "end_offset": 30, "train": False},
                ],
            },
            {
                "from": "human",
                "value": "I'm doing very well, thank you!",
                "train": False,
            },
            {"from": "assistant", "value": "Hi there!", "train": True},
        ]

        modified_dataset = Dataset.from_dict({"messages": [modified_conversation]})

        res = strategy.tokenize_prompt(modified_dataset[0])
        turns = strategy.get_conversation_thread(modified_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        def verify_labels(labels_span, should_train, context_message):
            """Helper to verify if a span of labels matches expected training state"""
            if should_train:
                assert all(label != IGNORE_TOKEN_ID for label in labels_span), (
                    f"Expected all labels for {context_message} to be set, but got {labels_span}"
                )
            else:
                assert all(label == IGNORE_TOKEN_ID for label in labels_span), (
                    f"Expected all labels for {context_message} to be {IGNORE_TOKEN_ID}, but got {labels_span}"
                )

        # Process all turns and verify labeling
        for i, turn in enumerate(modified_dataset[0]["messages"]):
            start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)

            if self._should_skip_turn(tokenizer, turn, i, start_idx, end_idx):
                continue

            decoded_response = tokenizer.decode(input_ids[start_idx:end_idx])
            response = turn["value"]

            assert response in decoded_response, (
                f"Response {response} not found in index {start_idx}:{end_idx} "
                f"decoded:{decoded_response}"
            )

            LOG.debug(
                f"Processing turn {i}: role={turn['from']}, content='{turn['value']}', "
                f"start_idx={start_idx}, end_idx={end_idx}"
            )

            if turn.get("train_detail", None) is not None:
                # Handle detailed token-level training control
                tokenized_output = tokenizer(
                    turn["value"], return_offsets_mapping=True, add_special_tokens=False
                )
                assert tokenized_output["input_ids"] == input_ids[start_idx:end_idx], (
                    f"Tokenized input mismatch for turn: {turn['value']}\n"
                    f"Expected: {input_ids[start_idx:end_idx]}\nActual: {tokenized_output['input_ids']}\n"
                    f"This will likely be a mismatch between template content and encoded content"
                )

                token_offsets = tokenized_output["offset_mapping"]

                # Adjust token offsets
                for j in range(len(token_offsets) - 1):
                    token_offsets[j] = (
                        token_offsets[j][0],
                        token_offsets[j + 1][0] - 1,
                    )
                token_offsets[-1] = (token_offsets[-1][0], len(turn["value"]) - 1)

                adjusted_train_details = strategy.prompter.adjust_train_details(
                    turn["train_detail"], token_offsets
                )

                LOG.debug(f"Original train_details: {turn['train_detail']}")
                LOG.debug(f"Adjusted train_details: {adjusted_train_details}")

                # Get and verify token offsets
                turn_tokens = input_ids[start_idx:end_idx]
                token_offsets_unmasked = strategy.prompter.get_offsets_for_train_detail(
                    text=turn["value"],
                    train_details=adjusted_train_details,
                    mask_untrainable=False,
                )

                for i, offset in enumerate(token_offsets_unmasked):
                    assert token_offsets[i][0] == offset, (
                        f"Token start offsets mismatch for turn: {turn['value']}\n"
                        f"Expected: {token_offsets[i][0]}\nActual: {offset}"
                    )

                token_offsets_masked = strategy.prompter.get_offsets_for_train_detail(
                    text=turn["value"],
                    train_details=adjusted_train_details,
                    mask_untrainable=True,
                )
                LOG.debug(f"Token offsets: {token_offsets_masked}")

                # Verify expected labels against actual labels
                expected_labels = [IGNORE_TOKEN_ID] * len(turn_tokens)
                for i, offset in enumerate(token_offsets_masked):
                    if offset != IGNORE_TOKEN_ID:
                        expected_labels[i] = turn_tokens[i]
                actual_labels = labels[
                    start_idx : start_idx + len(token_offsets_masked)
                ]
                assert actual_labels == expected_labels, (
                    f"Labels mismatch for turn: {turn['value']}\nExpected: {expected_labels}\nActual: {actual_labels}"
                )

                # Verify each detail section
                for detail in adjusted_train_details:
                    detail_start = start_idx + next(
                        j
                        for j, offset in enumerate(token_offsets_unmasked)
                        if offset >= detail["begin_offset"]
                    )
                    detail_end = start_idx + next(
                        (
                            j
                            for j, offset in enumerate(token_offsets_unmasked)
                            if offset > detail["end_offset"]
                        ),
                        len(token_offsets),
                    )

                    detail_text = turn["value"][
                        detail["begin_offset"] : detail["end_offset"] + 1
                    ]
                    detail_labels = labels[detail_start:detail_end]

                    context = (
                        f"detail (ind {detail_start}:{detail_end}): '{detail_text}'\n"
                        f"decoded: '{tokenizer.decode(input_ids[detail_start:detail_end])}')"
                    )
                    verify_labels(detail_labels, detail["train"], context)
            else:
                # Handle regular turn-level training control
                should_train = turn.get("train", False)
                turn_labels = labels[start_idx:end_idx]
                context = (
                    f"turn (ind {start_idx}:{end_idx}): '{turn['value']}'\n"
                    f"decoded: '{decoded_response}')"
                )
                verify_labels(turn_labels, should_train, context)

        LOG.debug(f"Final labels: {labels}")
        LOG.debug(f"Final input_ids: {input_ids}")

    def test_get_chat_template_variables(
        self, tokenizer, chat_template, chat_template_jinja, eos_token, request
    ):
        LOG.info("Testing get_chat_template_variables")

        actual_tokenizer, actual_jinja_template = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        prompter = ChatTemplatePrompter(
            actual_tokenizer,
            chat_template=get_chat_template(
                chat_template, jinja_template=actual_jinja_template
            ),
            message_property_mappings={"from": "role", "value": "content"},
        )

        variables = prompter.get_chat_template_msg_variables(
            (
                actual_jinja_template
                if actual_jinja_template
                else actual_tokenizer.get_chat_template()
            ),
            "messages",
        )

        # Special case for Mistral with additional tool variables
        if chat_template == "jinja" and tokenizer == "mistralv03_tokenizer":
            expected_variables = {"role", "content", "tool_call_id", "tool_calls"}
        # Most chat templates use the standard role and content variables
        elif chat_template in ["llama3", "chatml", "phi_35", "phi_4"] or (
            chat_template == "jinja" and tokenizer == "gemma2_tokenizer"
        ):
            expected_variables = {"role", "content"}
        else:
            LOG.warning(
                f"Unsupported chat template: {chat_template} with {chat_template_jinja}"
            )
            raise ValueError(
                f"Unsupported chat template: {chat_template} with {chat_template_jinja}"
            )

        assert variables == expected_variables, (
            f"Expected variables: {expected_variables} from {tokenizer}/{chat_template}\n"
            f"Got: {variables}\n"
            f"Chat template: {actual_jinja_template}"
        )

    def test_eot_tokens_conflict_with_eos_token(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        """Test that an error is raised when eot_tokens contains eos_token and train_on_eot/train_on_eos conflict"""
        LOG.info(
            "Testing conflict between eot_tokens containing eos_token and train_on_eot/train_on_eos mismatch"
        )

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        # Create a situation where eot_tokens contains eos_token
        eot_tokens = [
            tokenizer.eos_token,
            "[/INST]",
        ]  # Deliberately including eos_token

        # Create conflicting train_on_eos and train_on_eot settings
        with pytest.raises(
            ValueError,
            match=".*eos_token is in eot_tokens and train_on_eos != train_on_eot.*",
        ):
            ChatTemplateStrategy(
                ChatTemplatePrompter(
                    tokenizer,
                    chat_template=get_chat_template(
                        chat_template, jinja_template=chat_template_jinja
                    ),
                    message_property_mappings={"role": "from", "content": "value"},
                    field_messages="conversations",
                ),
                tokenizer=tokenizer,
                train_on_inputs=False,
                sequence_len=512,
                roles_to_train=["assistant"],
                train_on_eos="none",  # Setting to none
                train_on_eot="turn",  # Different from train_on_eos
                eot_tokens=eot_tokens,
            )

    def test_eot_token_backward_compatibility(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        """Test that eot_tokens inherits from eos_token when not specified"""
        LOG.info("Testing backward compatibility that eot_token inherits eos_token")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="turn",  # Setting train_on_eos to "turn"
        )

        # In backward compatibility mode, eot_tokens should be derived from eos_token
        assert strategy.eot_tokens == [tokenizer.eos_token], (
            f"Expected eot_tokens to inherit from eos_token, got {strategy.eot_tokens}"
        )
        assert strategy.train_on_eot == "turn", (
            f"Expected train_on_eot to inherit from train_on_eos, got {strategy.train_on_eot}"
        )

    def test_token_not_in_template(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        """Test runs even when tokens are not found in the template"""
        LOG.info("Testing runs even when tokens are not found in template")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        # Create a non-existent token that definitely won't be in the template
        non_existent_token = "[DEFINITELY_NOT_IN_TEMPLATE]"
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [non_existent_token]}
        )

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template(
                    chat_template, jinja_template=chat_template_jinja
                ),
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            eot_tokens=[non_existent_token],
        )

        # Force template check by calling tokenize_prompt
        strategy.tokenize_prompt(basic_dataset[0])

        # We can also check that a warning was logged, but there's
        # caplog conflicts when running with other tests
        # assert any(
        #     "not found in chat_template" in record.message for record in self._caplog.records
        # ), "Expected warning about token not found in template was not logged"

    def test_custom_eot_tokens(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        """Test with custom EOT tokens to ensure proper masking and training"""
        LOG.info("Testing with custom EOT tokens")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, None, request
        )

        # Add custom EOT tokens to the tokenizer
        custom_eot = "[EOT]"
        tokenizer.add_special_tokens({"additional_special_tokens": [custom_eot]})

        # Create a custom chat template that uses our EOT token
        custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}[EOT]{% endif %}{% endfor %}"""

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=custom_template,
                message_property_mappings={"role": "from", "content": "value"},
                field_messages="conversations",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eot="turn",  # Train on EOT token after each turn
            eot_tokens=[custom_eot],
        )

        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Find indices of the EOT token
        eot_token_id = tokenizer.convert_tokens_to_ids(custom_eot)
        eot_indices = [
            i for i, token_id in enumerate(input_ids) if token_id == eot_token_id
        ]

        assert len(eot_indices) > 0, "Expected at least one EOT token in the input"

        # Verify labeling for EOT tokens based on role
        turns = strategy.get_conversation_thread(basic_dataset[0])
        assistant_turn_indices = []
        non_assistant_turn_indices = []

        for i, turn in enumerate(basic_dataset[0]["conversations"]):
            start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)
            if start_idx != -1 and end_idx != -1:  # If turn is found
                if turn["from"] == "assistant":
                    assistant_turn_indices.append((start_idx, end_idx))
                else:
                    non_assistant_turn_indices.append((start_idx, end_idx))

        # Check EOT tokens after assistant turns are labeled
        for eot_idx in eot_indices:
            is_after_assistant = any(
                start_idx <= eot_idx <= end_idx + 1  # +1 to include the EOT token
                for start_idx, end_idx in assistant_turn_indices
            )

            if is_after_assistant:
                assert labels[eot_idx] != IGNORE_TOKEN_ID, (
                    f"Expected EOT token after assistant turn at index {eot_idx} to be labeled"
                )
            else:
                assert labels[eot_idx] == IGNORE_TOKEN_ID, (
                    f"Expected EOT token not after assistant turn at index {eot_idx} to not be labeled"
                )

    def test_multiple_train_on_eot_settings(
        self,
        tokenizer,
        chat_template,
        chat_template_jinja,
        eos_token,
        basic_dataset,
        request,
    ):
        """Test different train_on_eot settings"""
        LOG.info("Testing different train_on_eot settings")

        tokenizer, chat_template_jinja = self.setup_tokenizer(
            tokenizer, chat_template, chat_template_jinja, eos_token, request
        )

        # Create a list to test different train_on_eot settings
        test_settings = [
            ("none", lambda idx, is_assistant: False),  # Never train on EOT
            ("all", lambda idx, is_assistant: True),  # Always train on EOT
            (
                "turn",
                lambda idx, is_assistant: is_assistant,
            ),  # Train on EOT after assistant turns
            ("last", lambda idx, is_last: is_last),  # Only train on last EOT
        ]

        for setting, expected_train_func in test_settings:
            LOG.info(f"Testing train_on_eot='{setting}'")

            strategy = ChatTemplateStrategy(
                ChatTemplatePrompter(
                    tokenizer,
                    chat_template=get_chat_template(
                        chat_template, jinja_template=chat_template_jinja
                    ),
                    message_property_mappings={"role": "from", "content": "value"},
                    field_messages="conversations",
                ),
                tokenizer=tokenizer,
                train_on_inputs=False,
                sequence_len=512,
                roles_to_train=["assistant"],
                train_on_eot=setting,
                eot_tokens=[
                    tokenizer.eos_token
                ],  # Use eos_token as the EOT token for simplicity
            )

            res = strategy.tokenize_prompt(basic_dataset[0])
            turns = strategy.get_conversation_thread(basic_dataset[0])
            labels = res["labels"]
            input_ids = res["input_ids"]

            eos_token_id = tokenizer.eos_token_id
            eos_indices = [
                i for i, token_id in enumerate(input_ids) if token_id == eos_token_id
            ]

            assert len(eos_indices) > 0, (
                "Expected at least one EOS/EOT token in the input"
            )

            # Check labeling for each EOS/EOT token
            for idx, eos_idx in enumerate(eos_indices):
                # Find which turn this EOS token belongs to
                preceding_turn = None
                for i, turn in enumerate(basic_dataset[0]["conversations"]):
                    start_idx, end_idx = strategy.find_turn(turns=turns, turn_idx=i)
                    if (
                        start_idx != -1
                        and end_idx != -1
                        and start_idx <= eos_idx <= end_idx + 1
                    ):
                        preceding_turn = turn
                        break

                is_assistant = (
                    preceding_turn is not None and preceding_turn["from"] == "assistant"
                )
                is_last = idx == len(eos_indices) - 1

                expected_label = not expected_train_func(
                    idx, is_assistant if setting != "last" else is_last
                )

                if expected_label:
                    assert labels[eos_idx] == IGNORE_TOKEN_ID, (
                        f"Expected EOT token at index {eos_idx} to not be labeled with train_on_eot='{setting}'"
                    )
                else:
                    assert labels[eos_idx] != IGNORE_TOKEN_ID, (
                        f"Expected EOT token at index {eos_idx} to be labeled with train_on_eot='{setting}'"
                    )


class TestChatTemplateToolCalling:
    """
    Test class for tool calling functionality with chat templates.
    """

    def test_tool_calling_with_llama4_template(
        self,
        llama3_tokenizer,
    ):
        LOG.info("Testing tool calling with llama3 tokenizer and llama4 chat template")

        # Create tool calling dataset
        tool_calling_dataset = [
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "xml_escape",
                            "description": 'Replaces any "<", ">", or "&" characters in the input string with their corresponding XML entities.',
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "s": {
                                        "type": "string",
                                        "description": "The input string to be XML-escaped.",
                                    }
                                },
                                "required": ["s"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "multiples",
                            "description": "Generates a list of all the multiples of a number that are less than a given limit.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "number": {
                                        "type": "integer",
                                        "description": "The number to find multiples of.",
                                    },
                                    "limit": {
                                        "type": "integer",
                                        "description": "The upper limit for the multiples.",
                                    },
                                },
                                "required": ["number", "limit"],
                            },
                        },
                    },
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "Can you help me find multiples of 5 that are less than 20?",
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "multiples",
                                    "arguments": {
                                        "number": 5,
                                        "limit": 20,
                                    },
                                },
                            }
                        ],
                    },
                    {"role": "tool", "name": "multiples", "content": "5,10,15"},
                    {
                        "role": "assistant",
                        "content": "The multiples of 5 less than 20 are: 5, 10, and 15.",
                    },
                ],
            }
        ]

        # Setup tokenizer with llama4 chat template
        tokenizer = deepcopy(llama3_tokenizer)

        # Add EOS token to the tokenizer
        eot_token = "<|eot_id|>"
        tokenizer.add_special_tokens({"additional_special_tokens": [eot_token]})

        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                tokenizer,
                chat_template=get_chat_template("llama4"),
                message_property_mappings={"role": "role", "content": "content"},
                field_messages="messages",
                field_tools="tools",
            ),
            tokenizer=tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            eot_tokens=[eot_token],
        )

        res = strategy.tokenize_prompt(tool_calling_dataset[0])
        input_ids = res["input_ids"]
        labels = res["labels"]

        # Verify that the input_ids contain expected tokens
        assert len(input_ids) > 0, "Input IDs should not be empty"
        assert len(labels) == len(input_ids), "Labels should match input_ids length"

        # Decode the full conversation to verify structure
        decoded_conversation = tokenizer.decode(input_ids)

        # Verify tool calling structure is present in the decoded conversation
        assert '"type": "function",' in decoded_conversation, (
            "Tool type function should be in conversation"
        )
        assert '"name": "multiples",' in decoded_conversation, (
            "Tool function name should be in conversation"
        )

        assert (
            '<|python_start|><|python_end|>{"name": "multiples", "parameters": {"number": 5, "limit": 20}}<|eot|>'
            in decoded_conversation
        ), "Assistant tool call should be in conversation"
        assert "<|header_start|>ipython<|header_end|>" in decoded_conversation, (
            "IPython header should be in conversation"
        )
        assert '"5,10,15"' in decoded_conversation, (
            "Tool response should be in conversation"
        )

        # Get conversation turns to verify labeling
        turns = strategy.get_conversation_thread(tool_calling_dataset[0])
        tools = strategy._get_tools(tool_calling_dataset[0])

        # Check that assistant responses are properly labeled
        for i, turn in enumerate(tool_calling_dataset[0]["messages"]):
            if turn["role"] == "assistant":
                start_idx, end_idx = strategy.find_turn(
                    turns=turns, turn_idx=i, tools=tools
                )

                assert start_idx != -1 and end_idx != -1, (
                    f"Assistant turn {i} should be found"
                )

                # Verify that assistant responses have proper labels
                turn_labels = labels[start_idx:end_idx]
                assert all(label != IGNORE_TOKEN_ID for label in turn_labels), (
                    f"Assistant turn {i} should be unmasked"
                )
