"""
This module defines the BatchInference class, and performs multi-GPU batch inferencing
"""

import json
import os
import time
from abc import ABC, abstractmethod
from glob import glob
from os.path import join
from typing import Dict, List, Literal, Optional, TypedDict

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import IterableDataset
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import gen_run_id

LOG = get_logger(__name__)

StatusType = Literal["SUCCESS", "FAILURE"]


class InferenceResponseRecord(TypedDict):
    """An individual response record, will be multiple when num_return_sequences > 1 in the generation config"""

    response: str
    tokens: int


class InferenceResponse(TypedDict):
    """Inferencing result object"""

    prompt: str
    prompt_tokens: int
    total_tokens: int
    total_response_tokens: int
    generate_time_sec: float
    responses: List[InferenceResponseRecord]


class InferenceRunResult(TypedDict):
    """Result object returned by BatchInference.run()"""

    status: StatusType
    run_id: str
    request_count: int
    response_count: int
    run_time_sec: float
    total_tokens_generated: int
    tokens_per_sec: float
    seed: Optional[int]
    response: List[InferenceResponse]


class AbstractPersistenceBackend(ABC):
    """Interface for a persistence backend"""

    @abstractmethod
    def write_tmp(
        self, run_id: str, device_id: str, device_responses: List[InferenceResponse]
    ) -> None:
        ...

    @abstractmethod
    def collect(self, run_id: str) -> List[InferenceResponse]:
        ...

    @abstractmethod
    def cleanup(self, run_id: str) -> None:
        ...


class FileSystemCollector(AbstractPersistenceBackend):
    """Collects inference results from a path on the local filesystem"""

    def __init__(self, output_dir: str, encoding="utf-8") -> None:
        super().__init__()

        self.output_dir = output_dir
        self.encoding = encoding

    def _get_tmp_files(self, run_id: str) -> List[str]:
        """Returns a list of temporary files forthe current run_id

        Parameters
        ----------
        run_id : str
            The run_id

        Returns
        -------
        List[str]
            List of matching files
        """
        return glob(
            join(
                str(self.output_dir),
                f"{run_id}_*.tmp.json",
            )
        )

    def write_tmp(
        self, run_id: str, device_id: str, device_responses: List[InferenceResponse]
    ) -> None:
        """Writes temporary file, must be compatible with collect

        Parameters
        ----------
        run_id : str
            Batch inference run identifier
        device_id : str
            Device identifier
        device_responses : List[InferenceResponse]
            Responses to write
        """
        tmp_file_path = join(
            str(self.output_dir),
            f"{run_id}_{device_id}.tmp.json",
        )
        LOG.info("Writing output to: %s", tmp_file_path)
        with open(
            tmp_file_path,
            "w",
            encoding="utf-8",
        ) as device_output_fp:
            json.dump(device_responses, device_output_fp)

    def collect(self, run_id: str) -> List[InferenceResponse]:
        """Collect results files from filesystem

        Parameters
        ----------
        run_id : str
            Batch inference run identifier

        Returns
        -------
        List[InferenceResponse]
            Collected result object
        """
        collected_results: List[InferenceResponse] = []
        for result_file in self._get_tmp_files(run_id=run_id):
            LOG.info("Collecting result file: %s", result_file)
            with open(result_file, "r", encoding=self.encoding) as input_fp:
                collected_results.extend(json.load(input_fp))

        return collected_results

    def cleanup(self, run_id: str) -> None:
        for tmp_file in self._get_tmp_files(run_id=run_id):
            LOG.info("Removing temporary file: %s", tmp_file)
            os.remove(tmp_file)


class AbstractPostProcessor(ABC):
    """Inferencing result post-processor, provides a hook for adding additional post processors in the future"""

    @abstractmethod
    def trigger(self, run_id: str, inference_run_result: InferenceRunResult) -> None:
        pass


class JsonFilePostProcessor(AbstractPostProcessor):
    """Saves inferencing results to the specified JSON file"""

    def __init__(
        self, output_dir: str, accelerator: Accelerator, pretty=True, encoding="utf-8"
    ) -> None:
        super().__init__()

        self.output_dir = output_dir
        self.accelerator = accelerator
        self.encoding = encoding
        self.pretty = pretty

    def trigger(self, run_id: str, inference_run_result: InferenceRunResult) -> None:
        # Derive output filename
        output_path = join(self.output_dir, f"{run_id}.json")

        LOG.info(
            "Writing files to %s",
            output_path,
        )
        with open(output_path, "w", encoding=self.encoding) as output_fp:
            json.dump(
                inference_run_result, output_fp, indent=2 if self.pretty else None
            )


class BatchInference:
    """Batch inferencing impementation"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: IterableDataset,
        accelerator: Accelerator,
        seed: Optional[int],
        output_dir: Optional[str],
        generation_config: Optional[Dict],
        strip_whitespace: Optional[bool],
        persistence_backend: AbstractPersistenceBackend,
        post_processors: Optional[List[AbstractPostProcessor]] = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.strip_whitespace = strip_whitespace
        self.persistence_backend = persistence_backend
        self.post_processors = post_processors
        self.seed = seed
        self.output_dir = output_dir
        self.generation_config = generation_config

    @staticmethod
    def validate_and_warn(cfg: DictDefault) -> DictDefault:
        """Validates that the configuration is appropriate for inferencing.

        Parameters
        ----------
        cfg : DictDefault
            Configuration dictionary

        Raises
        ------
        ValueError
            Raised when a required configuration is not provided and cannot be derived
        """

        # Problems
        if not cfg.datasets and len(cfg.datasets) < 1:
            raise ValueError(
                "At leaset one entry for 'datasets' must be provided for inference"
            )

        if not cfg.sequence_len:
            raise ValueError("'sequence_len' must be set for inference")

        if not cfg.output_dir:
            raise ValueError("'output_dir' must be set for inference")

        # Warnings
        if not cfg.generation_config:
            LOG.warning(
                "It is highly recommended to specify your own 'generation_config' for inference"
            )

        if not cfg.truncate_features or len(cfg.truncate_features) == 0:
            LOG.warning(
                "'truncate_features' was not set in the configuration. "
                "If you are inferencing on a dataset that doesn't contain response "
                "values then this warning can be ignored. If your dataset does contain "
                "a reponse feature it will be encoded in the prompt which likely isn't "
                "what you want. To resolve, you should add it to the 'truncate_feature' "
                "list so it can be automatically removed prior to inferencing."
            )

        if not cfg.split_name:
            LOG.warning("'split_name' is not set, defaulting to 'train'")
            cfg.split_name = "train"

        if cfg.strip_whitespace is None:
            LOG.warning("'strip_whitespace' was not set, defaulting to true")
            cfg.strip_whitespace = True

        # Forced defaults
        LOG.info("Setting train_on_inputs=False, required for inferencing")
        cfg.train_on_inputs = False

        return cfg

    def _inference(
        self,
        tokenized_prompt: List[int],
        generation_config: Optional[GenerationConfig],
        remove_prompt_in_result=True,
        skip_special_tokens=True,
    ) -> InferenceResponse:
        # Strip out any trailing EOS tokens from input, sometimes this causes a problem
        while tokenized_prompt and tokenized_prompt[-1] == self.tokenizer.eos_token_id:
            tokenized_prompt = tokenized_prompt[:-1]

        # Convert to [1,] dim tensor on the appropriate device
        input_ids = (
            torch.tensor(tokenized_prompt).unsqueeze(0).to(self.accelerator.device)
        )

        # Execute inference
        with torch.no_grad():
            # Execute generate
            start_time = time.perf_counter()
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
            )
            generate_time_sec = round(time.perf_counter() - start_time, 3)

        # Build response object
        total_response_tokens = 0
        decoded_responses: List[InferenceResponseRecord] = []
        for output in outputs:
            if remove_prompt_in_result:
                # Count tokens from trimmed output
                response_tokens = sum(
                    1
                    for token_id in output[input_ids.shape[1] :]
                    if token_id.item() not in self.tokenizer.all_special_ids
                )

                decoded_output = self.tokenizer.decode(
                    output[input_ids.shape[1] :],
                    skip_special_tokens=skip_special_tokens,
                )
            else:
                # else, count tokens from raw output
                response_tokens = sum(
                    1
                    for token_id in output[input_ids.shape[1] :]
                    if token_id.item() not in self.tokenizer.all_special_ids
                )

                decoded_output = self.tokenizer.decode(
                    output,
                    skip_special_tokens=skip_special_tokens,
                )

            if self.strip_whitespace:
                decoded_output = decoded_output.strip()

            decoded_responses.append(
                InferenceResponseRecord(response=decoded_output, tokens=response_tokens)
            )

            total_response_tokens = total_response_tokens + response_tokens

        # Construct and return the response object
        prompt_tokens = len(tokenized_prompt)
        total_tokens = prompt_tokens + total_response_tokens

        results: InferenceResponse = InferenceResponse(
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            total_response_tokens=total_response_tokens,
            generate_time_sec=generate_time_sec,
            prompt=self.tokenizer.decode(
                tokenized_prompt, skip_special_tokens=skip_special_tokens
            ),
            responses=decoded_responses,
        )

        return results

    def run(self) -> InferenceRunResult:
        """Run batch inference."""

        run_id = gen_run_id(self.accelerator)

        inference_run_result = InferenceRunResult(
            status="FAILURE",
            run_id=run_id,
            request_count=len(self.dataset),
            response_count=0,
            run_time_sec=0,
            total_tokens_generated=0,
            tokens_per_sec=0.0,
            seed=self.seed,
            response=[],
        )

        start_time = time.perf_counter()

        if self.seed is not None:
            transformers.enable_full_determinism(seed=self.seed, warn_only=True)

        # Parse user-provided generation configuration
        generation_config = GenerationConfig(
            **(self.generation_config if self.generation_config is not None else {}),
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=False,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )

        LOG.info("Using generation configuration: %s", generation_config)

        self.model.eval()
        self.model.to(self.accelerator.device)

        LOG.info("Running batch inference on %i samples", len(self.dataset))

        try:
            with self.accelerator.split_between_processes(
                [prompt["input_ids"] for prompt in self.dataset]
            ) as device_prompts:
                device_responses: List[InferenceResponse] = []

                for device_prompt in device_prompts:
                    prompt_response = self._inference(
                        tokenized_prompt=device_prompt,
                        generation_config=generation_config,
                    )

                    device_responses.append(prompt_response)

                # Write device output to the filesystem. This is preferable over a gather since
                # we have other non-Tensor metadata in the response object.
                device_type = str(self.accelerator.device.type)
                device_index = (
                    str(self.accelerator.device.index)
                    if self.accelerator.device.index
                    else ""
                )
                device_id = f"{device_type}{device_index}"

                self.persistence_backend.write_tmp(
                    run_id=run_id,
                    device_id=device_id,
                    device_responses=device_responses,
                )

            self.accelerator.wait_for_everyone()
            run_time_sec = round(time.perf_counter() - start_time, 3)

            # Once everyone is done, collect the results and create response statistics
            if self.accelerator.is_main_process:
                collected_results = self.persistence_backend.collect(run_id=run_id)

                # Calculate respnse metrics
                response_count = len(collected_results)

                total_tokens_generated = sum(
                    x["total_response_tokens"] for x in collected_results
                )

                tokens_per_sec = round(total_tokens_generated / run_time_sec, 3)

                # Update response object
                inference_run_result["status"] = "SUCCESS"
                inference_run_result["response_count"] = response_count
                inference_run_result["run_time_sec"] = run_time_sec
                inference_run_result["total_tokens_generated"] = total_tokens_generated
                inference_run_result["tokens_per_sec"] = tokens_per_sec
                inference_run_result["response"] = collected_results

                # Invoke post-processors
                if self.post_processors is not None:
                    # Invoke post-processor(s) on consolidated results
                    for post_processor in self.post_processors:
                        post_processor.trigger(
                            run_id=run_id, inference_run_result=inference_run_result
                        )
        # pylint: disable=bare-except
        except:  # noqa: E722
            LOG.exception("An exception was raised during batch inferencing")
            inference_run_result["status"] = "FAILURE"

        finally:
            # Cleanup temporary files
            if self.accelerator.is_main_process:
                self.persistence_backend.cleanup(run_id=run_id)

        return inference_run_result
