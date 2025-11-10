"""Data handling specific to SFT."""

import logging
from typing import Any, NoReturn, cast

from datasets import (
    Dataset,
    IterableDataset,
    Sequence,
    Value,
)
from transformers import PreTrainedTokenizer
from transformers.processing_utils import ProcessorMixin

from axolotl.datasets import TokenizedPromptDataset, wrap_dataset_for_tokenized_prompt
from axolotl.prompt_strategies import load
from axolotl.prompt_strategies.bradley_terry import load as bradley_terry_load
from axolotl.prompt_tokenizers import (
    AlpacaMultipleChoicePromptTokenizingStrategy,
    AlpacaPromptTokenizingStrategy,
    AlpacaReflectionPTStrategy,
    DatasetWrappingStrategy,
    GPTeacherPromptTokenizingStrategy,
    JeopardyPromptTokenizingStrategy,
    OpenAssistantPromptTokenizingStrategy,
    PromptTokenizingStrategy,
    SummarizeTLDRPromptTokenizingStrategy,
)
from axolotl.prompters import (
    AlpacaPrompter,
    GPTeacherPrompter,
    JeopardyPrompter,
    MultipleChoiceConcisePrompter,
    MultipleChoiceExplainPrompter,
    Prompter,
    ReflectAlpacaPrompter,
    SummarizeTLDRPrompter,
    UnsupportedPrompter,
)
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


def handle_unknown_dataset_strategy(dataset_config: DictDefault) -> NoReturn:
    """Raise error for unknown dataset strategy."""
    ds_type = dataset_config.type
    suffix = ""
    if ":load_" in ds_type:
        suffix = f"Did you mean {ds_type.replace(':load_', '.load_')}?"

    error_message = f"unhandled prompt tokenization strategy: {ds_type}. {suffix}"
    LOG.error(error_message)
    raise ValueError(error_message)


def get_dataset_wrapper(
    dataset_config: DictDefault,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset_base_type: str | None,
    dataset: Dataset | IterableDataset,
    dataset_prompt_style: str | None = None,
    processor: ProcessorMixin | None = None,
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Create an appropriate dataset wrapper and prompter based on dataset
    configuration.

    Args:
        dataset_config: Configuration for the dataset.
        tokenizer: Tokenizer to use for processing text.
        cfg: Global configuration object.
        dataset_base_type: The base type of the dataset.
        dataset: The actual dataset object.
        dataset_prompt_style: Optional prompt style specification.
        processor: Optional processor for multimodal datasets.

    Returns:
        tuple of (dataset_wrapper, dataset_prompter).
    """
    # Common parameters for dataset wrapping
    dataset_kwargs: dict[str, Any] = {
        "process_count": cfg.dataset_num_proc,
        "keep_in_memory": cfg.dataset_keep_in_memory is True,
    }

    LOG.info(
        f"Loading dataset: {dataset_config['path']} with base_type: "
        f"{dataset_base_type} and prompt_style: {dataset_prompt_style}"
    )

    # Dataset is already tokenized
    if _is_dataset_already_tokenized(dataset):
        return dataset, UnsupportedPrompter()

    # Custom dataset type definition
    if isinstance(dataset_config.type, DictDefault):
        return _handle_custom_dataset_type(
            dataset_config, tokenizer, cfg, dataset, dataset_kwargs
        )

    # Skip preparation if configured
    if cfg.skip_prepare_dataset:
        return dataset, None

    # Bradley-Terry dataset
    if dataset_config.type.startswith("bradley_terry"):
        return _handle_bradley_terry_dataset(
            dataset_config, tokenizer, cfg, dataset, dataset_kwargs
        )

    # Stepwise supervised dataset
    if dataset_config.type.startswith("stepwise_supervised"):
        return _handle_stepwise_supervised_dataset(
            dataset_config, tokenizer, cfg, dataset, dataset_kwargs
        )

    # Try to load prompt tokenizer / dataset wrapper strategy from registry
    dataset_strategy = load(
        dataset_config.type, tokenizer, cfg, dataset_config, processor=processor
    )
    if dataset_strategy:
        return _handle_loaded_strategy(dataset_strategy, dataset, dataset_kwargs)

    # Known dataset types with specific handling
    if dataset_base_type in DATASET_HANDLERS:
        handler = DATASET_HANDLERS[dataset_base_type]
        return handler(dataset_prompt_style, tokenizer, cfg, dataset, dataset_kwargs)

    # Unhandled dataset type
    handle_unknown_dataset_strategy(dataset_config)


def _is_dataset_already_tokenized(dataset: Dataset | IterableDataset) -> bool:
    """Check if the dataset is already tokenized."""
    return (
        isinstance(dataset, Dataset)
        and "input_ids" in dataset.features
        and "attention_mask" in dataset.features
        and "labels" in dataset.features
    )


def _handle_custom_dataset_type(
    dataset_config: DictDefault,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a custom dataset type defined in the configuration."""
    dataset_strategy = cast(
        PromptTokenizingStrategy,
        load("user_defined", tokenizer, cfg, dataset_config.type.to_dict()),
    )
    dataset_prompter = UnsupportedPrompter()
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_bradley_terry_dataset(
    dataset_config: DictDefault,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Handle a Bradley-Terry dataset."""
    bt_type = dataset_config.type.split(".", 1)[1]
    dataset_strategy = bradley_terry_load(bt_type, tokenizer, cfg, dataset_config)

    if not dataset_strategy:
        handle_unknown_dataset_strategy(dataset_config)

    dataset_prompter = UnsupportedPrompter()
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )

    return dataset_wrapper, dataset_prompter


def _handle_stepwise_supervised_dataset(
    dataset_config: DictDefault,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a stepwise supervised dataset."""
    dataset_prompter = UnsupportedPrompter()
    dataset_strategy = load(dataset_config.type, tokenizer, cfg, dataset_config)

    # We need to explicitly cast boolean labels to int
    # for compatibility with how trl's PRMTrainer works
    if isinstance(dataset, Dataset):
        dataset = dataset.cast_column("labels", Sequence(Value("int64")))

    dataset_wrapper = TokenizedPromptDataset(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_loaded_strategy(
    dataset_strategy: PromptTokenizingStrategy | DatasetWrappingStrategy,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Handle a dataset with a strategy loaded from the registry."""
    if isinstance(dataset_strategy, DatasetWrappingStrategy):
        return dataset_strategy.wrap_dataset(dataset, **dataset_kwargs), None

    dataset_prompter = UnsupportedPrompter()
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_alpaca_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle an Alpaca dataset."""
    dataset_prompter = AlpacaPrompter(dataset_prompt_style)
    dataset_strategy = AlpacaPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_explainchoice_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle an ExplainChoice dataset."""
    dataset_prompter = MultipleChoiceExplainPrompter(dataset_prompt_style)
    dataset_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_concisechoice_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a ConciseChoice dataset."""
    dataset_prompter = MultipleChoiceConcisePrompter(dataset_prompt_style)
    dataset_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_summarizetldr_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a SummarizeTLDR dataset."""
    dataset_prompter = SummarizeTLDRPrompter(dataset_prompt_style)
    dataset_strategy = SummarizeTLDRPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_jeopardy_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a Jeopardy dataset."""
    dataset_prompter = JeopardyPrompter(dataset_prompt_style)
    dataset_strategy = JeopardyPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_oasst_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle an OpenAssistant dataset."""
    dataset_prompter = AlpacaPrompter(dataset_prompt_style)
    dataset_strategy = OpenAssistantPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_gpteacher_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a GPTeacher dataset."""
    dataset_prompter = GPTeacherPrompter(dataset_prompt_style)
    dataset_strategy = GPTeacherPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_reflection_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a Reflection dataset."""
    dataset_prompter = ReflectAlpacaPrompter(dataset_prompt_style)
    dataset_strategy = AlpacaReflectionPTStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        dataset_strategy,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


DATASET_HANDLERS = {
    "alpaca": _handle_alpaca_dataset,
    "explainchoice": _handle_explainchoice_dataset,
    "concisechoice": _handle_concisechoice_dataset,
    "summarizetldr": _handle_summarizetldr_dataset,
    "jeopardy": _handle_jeopardy_dataset,
    "oasst": _handle_oasst_dataset,
    "gpteacher": _handle_gpteacher_dataset,
    "reflection": _handle_reflection_dataset,
}
