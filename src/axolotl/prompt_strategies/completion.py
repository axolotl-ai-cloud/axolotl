"""
Basic completion text
"""
from typing import Any, Dict, Optional

from axolotl.prompt_tokenizers import CompletionPromptTokenizingStrategy
from axolotl.prompters import CompletionPrompter


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    strat = CompletionPromptTokenizingStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    if ds_cfg and "field" in ds_cfg:
        strat.field = ds_cfg["field"]

    return strat
