"""
Module containing true work queue processing for datasets.
Completely bypasses datasets.map() for proper load balancing.
"""

import multiprocessing as mp
import queue
import sys
import time
from typing import Any, Dict, List, Tuple

from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Apply monkey patches when module is imported
patch_dataset_methods()


class WorkQueueTokenizedPromptDataset(Dataset):
    """Dataset that uses a true work queue system, completely bypassing datasets.map().

    This implementation:
    1. Creates a shared work queue with individual examples
    2. Worker processes pull examples as they finish their current work
    3. No pre-allocation of work - true dynamic load balancing
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

        # Process the dataset with true work queue
        processed_data = self._process_with_work_queue(dataset)

        super().__init__(
            processed_data.data,
            **kwargs,
        )

    def _process_with_work_queue(self, dataset: Dataset) -> Dataset:
        """Process dataset using a true work queue system."""
        total_examples = len(dataset)
        LOG.info(f"Processing {total_examples} examples with work queue system")
        LOG.info(f"Using {self.process_count} worker processes")

        # Convert dataset to list for easy indexing
        examples = list(dataset)

        # Create shared queues
        work_queue = mp.Queue()
        result_queue = mp.Queue()

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
                            result_queue.put((idx, tokenized, None))  # None = no error
                        except Exception as e:
                            LOG.error(f"Worker {worker_id}: Error tokenizing example {idx}: {e}")
                            result_queue.put((idx, None, str(e)))  # Include error

                        # No task_done() needed for multiprocessing.Queue

                    except Exception:
                        # queue.Empty or any other exception - no more work
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
        results = [None] * total_examples
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
                LOG.error("Timeout waiting for results")
                break

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
        combined_results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for result in results:
            if result:
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
        return [wrap_dataset_for_work_queue_tokenized_prompt(strategy, dataset, process_count, **kwargs)]

    process_count = process_count or mp.cpu_count()

    # Calculate total examples and prepare combined work queue
    total_examples = sum(len(dataset) for _, dataset in datasets_with_strategies)
    LOG.info(f"Processing {total_examples} examples from {len(datasets_with_strategies)} datasets with work queue system")
    LOG.info(f"Using {process_count} worker processes")

    # Create shared queues
    work_queue = mp.Queue()
    result_queue = mp.Queue()

    # Track dataset boundaries for splitting results later
    dataset_boundaries = []
    current_idx = 0

    # Add all examples to work queue with dataset info (but NOT the strategy)
    for dataset_idx, (strategy, dataset) in enumerate(datasets_with_strategies):
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
                    global_idx, dataset_idx, example_idx, example = work_queue.get(timeout=1)

                    # Get the strategy for this dataset
                    strategy = datasets_with_strategies[dataset_idx][0]

                    # Tokenize the example with its specific strategy
                    try:
                        tokenized = strategy.tokenize_prompt(example)
                        result_queue.put((global_idx, dataset_idx, example_idx, tokenized, None))  # None = no error
                    except Exception as e:
                        LOG.error(f"Worker {worker_id}: Error tokenizing example {global_idx} from dataset {dataset_idx}: {e}")
                        result_queue.put((global_idx, dataset_idx, example_idx, None, str(e)))  # Include error

                except Exception:
                    # queue.Empty or any other exception - no more work
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
    results_by_dataset = [[] for _ in datasets_with_strategies]  # Results organized by dataset
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
            global_idx, dataset_idx, example_idx, tokenized, error = result_queue.get(timeout=10)

            if error:
                errors.append(f"Dataset {dataset_idx}, Example {example_idx}: {error}")
                # Create empty result for failed example
                empty_result = {"input_ids": [], "attention_mask": [], "labels": []}
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
            LOG.error("Timeout waiting for results")
            break

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

    # Convert results back to datasets
    processed_datasets = []

    for dataset_idx, (strategy, original_dataset) in enumerate(datasets_with_strategies):
        # Sort results by local index to maintain original order
        dataset_results = sorted(results_by_dataset[dataset_idx], key=lambda x: x[0])

        # Extract just the tokenized results
        tokenized_results = [result for _, result in dataset_results]

        # Combine results in order
        combined_results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for result in tokenized_results:
            if result:
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


def map_dataset_with_work_queue(
    dataset: Dataset,
    function: callable,
    process_count: int | None = None,
    batched: bool = False,
    batch_size: int = 1000,
    desc: str = "Processing",
    **kwargs,
) -> Dataset:
    """Apply a function to a dataset using a work queue for better load balancing.

    This is a drop-in replacement for dataset.map() that uses a work queue
    instead of the built-in multiprocessing to avoid load balancing issues.

    Args:
        dataset: The dataset to process
        function: The function to apply to each example/batch
        process_count: Number of worker processes
        batched: Whether to process batches (like dataset.map(batched=True))
        batch_size: Size of batches when batched=True
        desc: Description for progress bar
        **kwargs: Additional arguments passed to the function

    Returns:
        Processed dataset
    """
    process_count = process_count or mp.cpu_count()
    total_examples = len(dataset)

    LOG.info(f"Processing {total_examples} examples with work queue map")
    LOG.info(f"Using {process_count} worker processes")

    # Convert dataset to list for easy indexing
    examples = list(dataset)

    # Create shared queues
    work_queue = mp.Queue()
    result_queue = mp.Queue()

    # Prepare batches if batched processing
    if batched:
        batches = []
        for i in range(0, total_examples, batch_size):
            batch = examples[i:i + batch_size]
            batches.append((i, batch))

        # Add all batches to work queue
        for batch_idx, batch in batches:
            work_queue.put((batch_idx, batch))
    else:
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
                    idx, data = work_queue.get(timeout=1)

                    # Apply the function
                    try:
                        if batched:
                            # For batched processing, pass the batch directly
                            result = function(data, **kwargs)
                        else:
                            # For single examples, pass the example
                            result = function(data, **kwargs)
                        result_queue.put((idx, result, None))  # None = no error
                    except Exception as e:
                        LOG.error(f"Worker {worker_id}: Error processing {idx}: {e}")
                        result_queue.put((idx, None, str(e)))  # Include error

                except Exception:
                    # queue.Empty or any other exception - no more work
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
    if batched:
        results = [None] * len(batches)
        total_items = len(batches)
    else:
        results = [None] * total_examples
        total_items = total_examples

    completed = 0
    errors = []

    # Progress bar
    pbar = tqdm(
        total=total_items,
        desc=desc,
        unit="batches" if batched else "examples",
        file=sys.stdout,
        ncols=100,
    )

    start_time = time.time()
    last_update = start_time

    while completed < total_items:
        try:
            # Get result with timeout
            idx, result, error = result_queue.get(timeout=10)

            if error:
                errors.append(f"Item {idx}: {error}")
                # Create empty result for failed item
                results[idx] = None
            else:
                results[idx] = result

            completed += 1
            pbar.update(1)

            # Update rate every second
            current_time = time.time()
            if current_time - last_update > 1:
                elapsed = current_time - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                unit = "batches/s" if batched else "examples/s"
                pbar.set_postfix({unit: f"{rate:.1f}"})
                last_update = current_time

        except queue.Empty:
            LOG.error("Timeout waiting for results")
            break

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

    # Combine results
    if batched:
        # For batched results, we need to concatenate the batches
        combined_results = {}
        if results and results[0]:
            # Get the keys from the first non-None result
            keys = list(results[0].keys())
            for key in keys:
                combined_results[key] = []

            # Concatenate all batch results
            for batch_result in results:
                if batch_result:
                    for key in keys:
                        combined_results[key].extend(batch_result[key])
                else:
                    # Empty result for failed batch
                    for key in keys:
                        combined_results[key].extend([[]] * batch_size)
    else:
        # For non-batched results, just combine them
        combined_results = {}
        if results and results[0]:
            # Get the keys from the first non-None result
            keys = list(results[0].keys())
            for key in keys:
                combined_results[key] = []

            # Add all results
            for result in results:
                if result:
                    for key in keys:
                        combined_results[key].append(result[key])
                else:
                    # Empty result for failed example
                    for key in keys:
                        combined_results[key].append([])

    return Dataset.from_dict(combined_results)


def filter_dataset_with_work_queue(
    dataset: Dataset,
    function: callable,
    process_count: int | None = None,
    batched: bool = False,
    batch_size: int = 1000,
    desc: str = "Filtering",
    **kwargs,
) -> Dataset:
    """Filter a dataset using a work queue for better load balancing.

    This is a drop-in replacement for dataset.filter() that uses a work queue
    instead of the built-in multiprocessing to avoid load balancing issues.

    Args:
        dataset: The dataset to filter
        function: The function to apply to each example/batch (should return bool or list[bool])
        process_count: Number of worker processes
        batched: Whether to process batches (like dataset.filter(batched=True))
        batch_size: Size of batches when batched=True
        desc: Description for progress bar
        **kwargs: Additional arguments passed to the function

    Returns:
        Filtered dataset
    """
    process_count = process_count or mp.cpu_count()
    total_examples = len(dataset)

    LOG.info(f"Filtering {total_examples} examples with work queue")
    LOG.info(f"Using {process_count} worker processes")

    # Convert dataset to list for easy indexing
    examples = list(dataset)

    # Create shared queues
    work_queue = mp.Queue()
    result_queue = mp.Queue()

    # Prepare batches if batched processing
    if batched:
        batches = []
        for i in range(0, total_examples, batch_size):
            batch = examples[i:i + batch_size]
            batches.append((i, batch))

        # Add all batches to work queue
        for batch_idx, batch in batches:
            work_queue.put((batch_idx, batch))
    else:
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
                    idx, data = work_queue.get(timeout=1)

                    # Apply the function
                    try:
                        if batched:
                            # For batched processing, pass the batch directly
                            result = function(data, **kwargs)
                        else:
                            # For single examples, pass the example
                            result = function(data, **kwargs)
                        result_queue.put((idx, result, None))  # None = no error
                    except Exception as e:
                        LOG.error(f"Worker {worker_id}: Error processing {idx}: {e}")
                        result_queue.put((idx, None, str(e)))  # Include error

                except Exception:
                    # queue.Empty or any other exception - no more work
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
    if batched:
        results = [None] * len(batches)
        total_items = len(batches)
    else:
        results = [None] * total_examples
        total_items = total_examples

    completed = 0
    errors = []

    # Progress bar
    pbar = tqdm(
        total=total_items,
        desc=desc,
        unit="batches" if batched else "examples",
        file=sys.stdout,
        ncols=100,
    )

    start_time = time.time()
    last_update = start_time

    while completed < total_items:
        try:
            # Get result with timeout
            idx, result, error = result_queue.get(timeout=10)

            if error:
                errors.append(f"Item {idx}: {error}")
                # Create empty result for failed item (filter out)
                results[idx] = [] if batched else False
            else:
                results[idx] = result

            completed += 1
            pbar.update(1)

            # Update rate every second
            current_time = time.time()
            if current_time - last_update > 1:
                elapsed = current_time - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                unit = "batches/s" if batched else "examples/s"
                pbar.set_postfix({unit: f"{rate:.1f}"})
                last_update = current_time

        except queue.Empty:
            LOG.error("Timeout waiting for results")
            break

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

    # Apply filter
    if batched:
        # For batched results, we need to flatten the filter results
        keep_indices = []
        for batch_idx, batch_result in enumerate(results):
            if batch_result:
                # Calculate the actual indices in the original dataset
                start_idx = batch_idx * batch_size
                for i, keep in enumerate(batch_result):
                    if keep:
                        actual_idx = start_idx + i
                        if actual_idx < total_examples:
                            keep_indices.append(actual_idx)
    else:
        # For non-batched results, just collect the indices where result is True
        keep_indices = [i for i, result in enumerate(results) if result]

    # Select the kept examples
    if keep_indices:
        return dataset.select(keep_indices)
    else:
        # Return empty dataset if nothing matches
        return dataset.select([])


# Monkey patch dataset.map and dataset.filter to use work queue when num_proc is high
def patch_dataset_methods():
    """Monkey patch Dataset.map and Dataset.filter to use work queue when appropriate."""
    from datasets import Dataset

    original_map = Dataset.map
    original_filter = Dataset.filter

    def patched_map(self, *args, **kwargs):
        # Check if we should use work queue
        num_proc = kwargs.get('num_proc', 1)
        if num_proc and num_proc > 4:
            # Use work queue for high process counts
            function = args[0] if args else kwargs.get('function')
            if function:
                LOG.info(f"Using work queue for map with {num_proc} processes")
                return map_dataset_with_work_queue(
                    self, function,
                    process_count=num_proc,
                    **{k: v for k, v in kwargs.items() if k != 'num_proc'}
                )

        # Use original implementation
        return original_map(self, *args, **kwargs)

    def patched_filter(self, *args, **kwargs):
        # Check if we should use work queue
        num_proc = kwargs.get('num_proc', 1)
        if num_proc and num_proc > 4:
            # Use work queue for high process counts
            function = args[0] if args else kwargs.get('function')
            if function:
                LOG.info(f"Using work queue for filter with {num_proc} processes")
                return filter_dataset_with_work_queue(
                    self, function,
                    process_count=num_proc,
                    **{k: v for k, v in kwargs.items() if k != 'num_proc'}
                )

        # Use original implementation
        return original_filter(self, *args, **kwargs)

    # Apply monkey patches
    Dataset.map = patched_map
    Dataset.filter = patched_filter
