"""
Module containing true work queue processing for datasets.
Completely bypasses datasets.map() for proper load balancing.
"""

import multiprocessing as mp
import queue
import sys
import time
from typing import Any, Dict, List

from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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
