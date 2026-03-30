"""
Module containing work queue processing for datasets.
Completely bypasses datasets.map() for proper load balancing.
"""

import multiprocessing as mp
import queue
import sys
import time
from typing import List, Tuple

from datasets import Dataset
from tqdm import tqdm

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class WorkQueueTokenizedPromptDataset(Dataset):
    """Dataset that uses a work queue system, completely bypassing datasets.map().

    This implementation:
    1. Creates a shared work queue with individual examples
    2. Worker processes pull examples as they finish their current work
    3. No pre-allocation of work
    4. Processes results in order they complete
    """

    def __init__(
        self,
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: Dataset,
        process_count: int | None = None,
        keep_in_memory: bool | None = False,
        **kwargs,
    ):
        self.prompt_tokenizer = prompt_tokenizer
        self.process_count = process_count or mp.cpu_count()
        self.keep_in_memory = keep_in_memory

        # Process the dataset with work queue
        processed_data = self._process_with_work_queue(dataset)

        super().__init__(
            processed_data.data,
            **kwargs,
        )

    def _process_with_work_queue(self, dataset: Dataset) -> Dataset:
        """Process dataset using a work queue system."""
        total_examples = len(dataset)
        LOG.info(f"Processing {total_examples} examples with work queue system")
        LOG.info(f"Using {self.process_count} worker processes")

        # Convert dataset to list for easy indexing
        examples = list(dataset)

        # Create shared queues
        work_queue: mp.Queue = mp.Queue()
        result_queue: mp.Queue = mp.Queue()

        # Add all examples to work queue
        for idx, example in enumerate(examples):
            work_queue.put((idx, example))

        # Worker function
        def worker(worker_id):
            """Worker process that continuously pulls from work queue."""
            try:
                while True:
                    try:
                        # Get work with timeout
                        idx, example = work_queue.get(timeout=1)

                        # Tokenize the example
                        try:
                            tokenized = self.prompt_tokenizer.tokenize_prompt(example)
                            result_queue.put((idx, tokenized, None))
                        except Exception as e:
                            LOG.error(
                                f"Worker {worker_id}: Error tokenizing example {idx}: {e}"
                            )
                            result_queue.put((idx, None, str(e)))

                    except queue.Empty:
                        break

            except Exception as e:
                LOG.error(f"Worker {worker_id} crashed: {e}")

        # Start worker processes
        processes = []
        for i in range(self.process_count):
            p = mp.Process(target=worker, args=(i,))
            p.daemon = True
            p.start()
            processes.append(p)

        # Collect results with progress tracking
        results: list[dict[str, list] | None] = [None] * total_examples
        completed = 0
        errors = []

        # Progress bar
        pbar = tqdm(
            total=total_examples,
            desc="Tokenizing Prompts",
            unit="examples",
            file=sys.stdout,
            ncols=100,
        )

        start_time = time.time()
        last_update = start_time

        while completed < total_examples:
            try:
                # Get result with timeout
                idx, tokenized, error = result_queue.get(timeout=10)

                if error:
                    errors.append(f"Example {idx}: {error}")
                    # Create empty result for failed example
                    results[idx] = {"input_ids": [], "attention_mask": [], "labels": []}
                else:
                    results[idx] = tokenized

                completed += 1
                pbar.update(1)

                # Update rate every second
                current_time = time.time()
                if current_time - last_update > 1:
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({"examples/s": f"{rate:.1f}"})
                    last_update = current_time

            except queue.Empty:
                if all(not p.is_alive() for p in processes):
                    LOG.error("All workers died before completing all examples")
                    break
                LOG.warning("Timeout waiting for results, retrying...")

        pbar.close()

        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        # Report errors
        if errors:
            LOG.warning(f"Completed with {len(errors)} errors:")
            for error in errors[:5]:  # Show first 5 errors
                LOG.warning(f"  {error}")
            if len(errors) > 5:
                LOG.warning(f"  ... and {len(errors) - 5} more errors")

        # Combine results in order
        combined_results: dict[str, list] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for result in results:
            if result is not None:
                combined_results["input_ids"].append(result["input_ids"])
                combined_results["attention_mask"].append(result["attention_mask"])
                combined_results["labels"].append(result["labels"])
            else:
                # Empty result for failed examples
                combined_results["input_ids"].append([])
                combined_results["attention_mask"].append([])
                combined_results["labels"].append([])

        return Dataset.from_dict(combined_results)


def wrap_dataset_for_work_queue_tokenized_prompt(
    prompt_tokenizer: PromptTokenizingStrategy,
    dataset: Dataset,
    process_count: int | None = None,
    **kwargs,
) -> Dataset:
    """Wrap dataset with work queue processing."""
    return WorkQueueTokenizedPromptDataset(
        prompt_tokenizer=prompt_tokenizer,
        dataset=dataset,
        process_count=process_count,
        **kwargs,
    )


def wrap_multiple_datasets_for_work_queue_tokenized_prompt(
    datasets_with_strategies: List[Tuple[PromptTokenizingStrategy, Dataset]],
    process_count: int | None = None,
    **kwargs,
) -> List[Dataset]:
    """Process multiple datasets efficiently with a single work queue.

    This function:
    1. Combines all datasets into a single work queue
    2. Tracks which dataset each example came from
    3. Processes all examples in parallel
    4. Splits results back into separate datasets

    Args:
        datasets_with_strategies: List of (prompt_tokenizer, dataset) tuples
        process_count: Number of worker processes

    Returns:
        List of processed datasets in the same order as input
    """
    if not datasets_with_strategies:
        return []

    if len(datasets_with_strategies) == 1:
        # Single dataset - use standard function
        strategy, dataset = datasets_with_strategies[0]
        return [
            wrap_dataset_for_work_queue_tokenized_prompt(
                strategy, dataset, process_count, **kwargs
            )
        ]

    process_count = process_count or mp.cpu_count()

    # Calculate total examples and prepare combined work queue
    total_examples = sum(len(dataset) for _, dataset in datasets_with_strategies)
    LOG.info(
        f"Processing {total_examples} examples from {len(datasets_with_strategies)} datasets with work queue system"
    )
    LOG.info(f"Using {process_count} worker processes")

    # Create shared queues
    work_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    # Track dataset boundaries for splitting results later
    dataset_boundaries = []
    current_idx = 0

    # Add all examples to work queue with dataset info (but NOT the strategy)
    for dataset_idx, (_strategy, dataset) in enumerate(datasets_with_strategies):
        dataset_start = current_idx
        examples = list(dataset)

        for example_idx, example in enumerate(examples):
            # Store global index, dataset index, local index, and example only
            # Strategy will be looked up by dataset_idx in the worker
            work_queue.put((current_idx, dataset_idx, example_idx, example))
            current_idx += 1

        dataset_end = current_idx - 1
        dataset_boundaries.append((dataset_start, dataset_end, len(examples)))

    # Worker function that handles different strategies
    def worker(worker_id):
        """Worker process that continuously pulls from work queue."""
        try:
            while True:
                try:
                    # Get work with timeout
                    global_idx, dataset_idx, example_idx, example = work_queue.get(
                        timeout=1
                    )

                    # Get the strategy for this dataset
                    strategy = datasets_with_strategies[dataset_idx][0]

                    # Tokenize the example with its specific strategy
                    try:
                        tokenized = strategy.tokenize_prompt(example)
                        result_queue.put(
                            (global_idx, dataset_idx, example_idx, tokenized, None)
                        )
                    except Exception as e:
                        LOG.error(
                            f"Worker {worker_id}: Error tokenizing example {global_idx} from dataset {dataset_idx}: {e}"
                        )
                        result_queue.put(
                            (global_idx, dataset_idx, example_idx, None, str(e))
                        )

                except queue.Empty:
                    break

        except Exception as e:
            LOG.error(f"Worker {worker_id} crashed: {e}")

    # Start worker processes
    processes = []
    for i in range(process_count):
        p = mp.Process(target=worker, args=(i,))
        p.daemon = True
        p.start()
        processes.append(p)

    # Collect results with progress tracking
    results_by_dataset: list[list] = [
        [] for _ in datasets_with_strategies
    ]  # Results organized by dataset
    completed = 0
    errors = []

    # Progress bar
    pbar = tqdm(
        total=total_examples,
        desc="Tokenizing Prompts",
        unit="examples",
        file=sys.stdout,
        ncols=100,
    )

    start_time = time.time()
    last_update = start_time

    while completed < total_examples:
        try:
            # Get result with timeout
            global_idx, dataset_idx, example_idx, tokenized, error = result_queue.get(
                timeout=10
            )

            if error:
                errors.append(f"Dataset {dataset_idx}, Example {example_idx}: {error}")
                # Create empty result for failed example
                empty_result: dict[str, list] = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                }
                results_by_dataset[dataset_idx].append((example_idx, empty_result))
            else:
                results_by_dataset[dataset_idx].append((example_idx, tokenized))

            completed += 1
            pbar.update(1)

            # Update rate every second
            current_time = time.time()
            if current_time - last_update > 1:
                elapsed = current_time - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                pbar.set_postfix({"examples/s": f"{rate:.1f}"})
                last_update = current_time

        except queue.Empty:
            if all(not p.is_alive() for p in processes):
                LOG.error("All workers died before completing all examples")
                break
            LOG.warning("Timeout waiting for results, retrying...")

    pbar.close()

    # Wait for all processes to finish
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # Report errors
    if errors:
        LOG.warning(f"Completed with {len(errors)} errors:")
        for error in errors[:5]:
            LOG.warning(f"  {error}")
        if len(errors) > 5:
            LOG.warning(f"  ... and {len(errors) - 5} more errors")

    # Convert results back to datasets
    processed_datasets = []

    for dataset_idx, (_strategy, _original_dataset) in enumerate(
        datasets_with_strategies
    ):
        # Sort results by local index to maintain original order
        dataset_results = sorted(results_by_dataset[dataset_idx], key=lambda x: x[0])

        # Extract just the tokenized results
        tokenized_results = [result for _, result in dataset_results]

        # Combine results in order
        combined_results: dict[str, list] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for result in tokenized_results:
            if result is not None:
                combined_results["input_ids"].append(result["input_ids"])
                combined_results["attention_mask"].append(result["attention_mask"])
                combined_results["labels"].append(result["labels"])
            else:
                # Empty result for failed examples
                combined_results["input_ids"].append([])
                combined_results["attention_mask"].append([])
                combined_results["labels"].append([])

        # Create dataset
        processed_dataset = Dataset.from_dict(combined_results)
        processed_datasets.append(processed_dataset)

    return processed_datasets
