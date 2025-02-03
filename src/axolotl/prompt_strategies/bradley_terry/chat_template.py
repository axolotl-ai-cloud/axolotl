"""
Bradley-Terry model with chat template prompt strategy.
"""

import logging
from typing import Any, Dict, Optional

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.utils.chat_templates import get_chat_template_from_config

# Configure the logger
LOG = logging.getLogger("axolotl.prompt_strategies.bradley_terry.chat_template")
LOG.setLevel(logging.INFO)


class BTChatTemplateStrategy(ChatTemplateStrategy):
    """
    Bradley-Terry reward model pairwise chat template prompt strategy.
    """

    @property
    def supports_batched(self) -> bool:
        return False

    def _tokenize_single_prompt(self, prompt):
        """

        :param prompt: the actual row of data from the underlying dataset
        :return:
        """

        max_length = self.prompter.max_length

        # pylint: disable=duplicate-code
        prompt["messages"] = []
        if prompt["system"]:
            prompt["messages"].append({"role": "system", "content": prompt["system"]})
        prompt["messages"].append({"role": "user", "content": prompt["input"]})
        prompt["messages"].append({"role": "assistant", "content": prompt["chosen"]})
        chosen_tokenized = super()._tokenize_single_prompt(prompt)

        if len(chosen_tokenized["input_ids"]) > max_length:
            LOG.warning(
                f"To-be-trimmed chosen sequence exceeds max sequence length: {len(chosen_tokenized['input_ids'])}",
            )

            chosen_tokenized["input_ids"] = chosen_tokenized["input_ids"][:max_length]
            chosen_tokenized["attention_mask"] = chosen_tokenized["attention_mask"][
                :max_length
            ]

        # pylint: disable=duplicate-code
        prompt["messages"] = []
        if prompt["system"]:
            prompt["messages"].append({"role": "system", "content": prompt["system"]})
        prompt["messages"].append({"role": "user", "content": prompt["input"]})
        prompt["messages"].append({"role": "assistant", "content": prompt["rejected"]})
        rejected_tokenized = super()._tokenize_single_prompt(prompt)

        if len(rejected_tokenized["input_ids"]) > max_length:
            LOG.warning(
                f"To-be-trimmed rejected sequence exceeds max sequence length: {len(rejected_tokenized['input_ids'])}",
            )

            rejected_tokenized["input_ids"] = rejected_tokenized["input_ids"][
                :max_length
            ]
            rejected_tokenized["attention_mask"] = rejected_tokenized["attention_mask"][
                :max_length
            ]

        return {
            "input_ids_chosen": chosen_tokenized["input_ids"],
            "attention_mask_chosen": chosen_tokenized["attention_mask"],
            "labels_chosen": 1.0,
            "input_ids_rejected": rejected_tokenized["input_ids"],
            "attention_mask_rejected": rejected_tokenized["attention_mask"],
            "labels_rejected": 0.0,
        }


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    ds_cfg = ds_cfg or {}
    chat_template_string = get_chat_template_from_config(
        cfg=cfg, ds_cfg=ds_cfg, tokenizer=tokenizer
    )

    prompter_params = {
        "tokenizer": tokenizer,
        "chat_template": chat_template_string,
        "message_property_mappings": ds_cfg.get(
            "message_property_mappings",
            {
                "role": "role",
                "content": "content",
            },
        ),
        "message_field_training": ds_cfg.get("message_field_training", None),
        "message_field_training_detail": ds_cfg.get(
            "message_field_training_detail", None
        ),
        "roles": ds_cfg.get("roles"),
        "drop_system_message": ds_cfg.get("drop_system_message", False),
        # we need to add one for detecting sequences with exceeding the `sequence_len` limit.
        "max_length": (
            cfg.sequence_len + 1 if not cfg.reward_model else cfg.sequence_len
        ),
    }

    strategy_params = {
        "train_on_inputs": cfg.train_on_inputs,
        "sequence_len": cfg.sequence_len,
        "roles_to_train": ds_cfg.get("roles_to_train", []),
        "train_on_eos": ds_cfg.get("train_on_eos", None),
    }

    strategy = BTChatTemplateStrategy(
        ChatTemplatePrompter(**prompter_params), tokenizer=tokenizer, **strategy_params
    )

    return strategy
