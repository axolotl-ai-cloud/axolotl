"""data handling specific to SFT"""

import logging
from typing import Any, cast

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
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


# pylint: disable=too-many-return-statements
def get_dataset_wrapper(
    config_dataset: DictDefault,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset_base_type: str | None,
    dataset: Dataset | IterableDataset,
    dataset_prompt_style: str | None = None,
    processor: ProcessorMixin | None = None,  # pylint: disable=unused-argument
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Create an appropriate dataset wrapper and prompter based on dataset
    configuration.

    Args:
        config_dataset: Configuration for the dataset.
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
        "process_count": cfg.dataset_processes,
        "keep_in_memory": cfg.dataset_keep_in_memory is True,
    }

    LOG.info(
        f"Loading dataset: {config_dataset['path']} with base_type: "
        f"{dataset_base_type} and prompt_style: {dataset_prompt_style}"
    )

    # Dataset is already tokenized
    if _is_dataset_already_tokenized(dataset):
        return dataset, UnsupportedPrompter()

    # Custom dataset type definition
    if isinstance(config_dataset.type, DictDefault):
        return _handle_custom_dataset_type(
            config_dataset, tokenizer, cfg, dataset, dataset_kwargs
        )

    # Skip preparation if configured
    if cfg.skip_prepare_dataset:
        return dataset, None

    # Bradley-Terry dataset
    if config_dataset.type.startswith("bradley_terry"):
        return _handle_bradley_terry_dataset(
            config_dataset, tokenizer, cfg, dataset, dataset_kwargs
        )

    # Stepwise supervised dataset
    if config_dataset.type.startswith("stepwise_supervised"):
        return _handle_stepwise_supervised_dataset(
            config_dataset, tokenizer, cfg, dataset, dataset_kwargs
        )

    # Try to load dataset strategy from registry
    prompt_tokenizer = load(
        config_dataset.type, tokenizer, cfg, config_dataset, processor=processor
    )
    if prompt_tokenizer:
        return _handle_loaded_strategy(prompt_tokenizer, dataset, dataset_kwargs)

    # Known dataset types with specific handling
    if dataset_base_type in DATASET_HANDLERS:
        handler = DATASET_HANDLERS[dataset_base_type]
        return handler(dataset_prompt_style, tokenizer, cfg, dataset, dataset_kwargs)

    # Unhandled dataset type
    return _handle_unhandled_dataset_type(config_dataset.type)


def _is_dataset_already_tokenized(
    dataset: Dataset | IterableDataset | DatasetDict | IterableDatasetDict,
) -> bool:
    """Check if the dataset is already tokenized."""
    return (
        isinstance(dataset, Dataset)
        and "input_ids" in dataset.features
        and "attention_mask" in dataset.features
        and "labels" in dataset.features
    )


def _handle_custom_dataset_type(
    config_dataset: DictDefault,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a custom dataset type defined in the configuration."""
    prompt_tokenizer = cast(
        PromptTokenizingStrategy,
        load("user_defined", tokenizer, cfg, config_dataset.type.to_dict()),
    )
    dataset_prompter = UnsupportedPrompter()
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_bradley_terry_dataset(
    config_dataset: DictDefault,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Handle a Bradley-Terry dataset."""
    bt_type = config_dataset.type.split(".", 1)[1]
    prompt_tokenizer = cast(
        PromptTokenizingStrategy | None,
        bradley_terry_load(bt_type, tokenizer, cfg, config_dataset),
    )

    if not prompt_tokenizer:
        return _handle_unhandled_dataset_type(config_dataset.type)

    dataset_prompter = UnsupportedPrompter()
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
        dataset,
        **dataset_kwargs,
    )

    return dataset_wrapper, dataset_prompter


def _handle_stepwise_supervised_dataset(
    config_dataset: DictDefault,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle a stepwise supervised dataset."""
    dataset_prompter = UnsupportedPrompter()
    prompt_tokenizer = cast(
        PromptTokenizingStrategy,
        load(config_dataset.type, tokenizer, cfg, config_dataset),
    )

    # We need to explicitly cast boolean labels to int
    # for compatibility with how trl's PRMTrainer works
    if isinstance(dataset, Dataset):
        dataset = dataset.cast_column("labels", Sequence(Value("int64")))

    dataset_wrapper = TokenizedPromptDataset(
        prompt_tokenizer,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_loaded_strategy(
    prompt_tokenizer: PromptTokenizingStrategy | DatasetWrappingStrategy,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Handle a dataset with a strategy loaded from the registry."""
    if isinstance(prompt_tokenizer, DatasetWrappingStrategy):
        return prompt_tokenizer.wrap_dataset(dataset, **dataset_kwargs), None

    dataset_prompter = UnsupportedPrompter()
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
        dataset,
        **dataset_kwargs,
    )
    return dataset_wrapper, dataset_prompter


def _handle_unhandled_dataset_type(
    dataset_type: str,
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Handle an unhandled dataset type by raising an error."""
    suffix = ""
    if ":load_" in dataset_type:
        suffix = f" Did you mean {dataset_type.replace(':load_', '.load_')}?"

    error_message = f"unhandled prompt tokenization strategy: {dataset_type}. {suffix}"
    LOG.error(error_message)
    raise ValueError(error_message)


def _handle_alpaca_dataset(
    dataset_prompt_style: str | None,
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    dataset: Dataset | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[Dataset | IterableDataset, Prompter]:
    """Handle an Alpaca dataset."""
    dataset_prompter = AlpacaPrompter(dataset_prompt_style)
    prompt_tokenizer = AlpacaPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
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
    prompt_tokenizer = AlpacaMultipleChoicePromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
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
    prompt_tokenizer = AlpacaMultipleChoicePromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
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
    prompt_tokenizer = SummarizeTLDRPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
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
    prompt_tokenizer = JeopardyPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
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
    prompt_tokenizer = OpenAssistantPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
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
    prompt_tokenizer = GPTeacherPromptTokenizingStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
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
    prompt_tokenizer = AlpacaReflectionPTStrategy(
        dataset_prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    dataset_wrapper = wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer,
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
