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

    def tokenize_prompt(self, prompt):
        """

        :param prompt: the actual row of data from the underlying dataset
        :return:
        """

        self.messages = "chosen_messages"
        # pylint: disable=duplicate-code
        prompt[self.messages] = []
        if prompt["system"]:
            prompt[self.messages].append(
                {"role": "system", "content": prompt["system"]}
            )
        prompt[self.messages].append({"role": "user", "content": prompt["input"]})
        prompt[self.messages].append({"role": "assistant", "content": prompt["chosen"]})
        chosen_tokenized = super().tokenize_prompt(prompt)

        self.messages = "rejected_messages"
        # pylint: disable=duplicate-code
        prompt[self.messages] = []
        if prompt["system"]:
            prompt[self.messages].append(
                {"role": "system", "content": prompt["system"]}
            )
        prompt[self.messages].append({"role": "user", "content": prompt["input"]})
        prompt[self.messages].append(
            {"role": "assistant", "content": prompt["rejected"]}
        )
        rejected_tokenized = super().tokenize_prompt(prompt)

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
        "message_field_role": ds_cfg.get("message_field_role", "role"),
        "message_field_content": ds_cfg.get("message_field_content", "content"),
        "message_field_training": ds_cfg.get("message_field_training", None),
        "message_field_training_detail": ds_cfg.get(
            "message_field_training_detail", None
        ),
        "roles": ds_cfg.get("roles"),
        "drop_system_message": ds_cfg.get("drop_system_message", False),
        # we need to add one for detecting sequences with exceeding the `sequence_len` limit.
        "max_length": cfg.sequence_len + 1
        if not cfg.reward_model
        else cfg.sequence_len,
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

    if "field_messages" in ds_cfg and hasattr(strategy, "messages"):
        strategy.messages = ds_cfg["field_messages"]

    return strategy
