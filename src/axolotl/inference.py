"""
This module defines the BatchInference class, and performs multi-GPU batch inferencing
"""

import json
import os
import time
from abc import ABC, abstractmethod
from glob import glob
from os.path import join
from typing import Dict, List, Optional, TypedDict

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


class InferenceRunResult(TypedDict):
    """Result object returned by BatchInference.run"""

    run_time_sec: float


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


class AbstractPostProcessor(ABC):
    """Inferencing result post-processor, provides a hook for adding additional post processors in the future"""

    @abstractmethod
    def trigger(self, run_id: str, result_paths: List[str]) -> None:
        pass


class JsonFilePostProcessor(AbstractPostProcessor):
    """Saves inferencing results to the specified JSON file"""

    def __init__(self, output_dir: str, encoding="utf-8") -> None:
        super().__init__()

        self.output_dir = output_dir
        self.encoding = encoding

    def trigger(self, run_id: str, result_paths: List[str]) -> None:
        # Write to output file outside of the loop
        combined_results: List[Dict] = []
        for result_file in result_paths:
            LOG.info("Consolidating result file: %s", result_file)
            with open(result_file, "r", encoding=self.encoding) as input_fp:
                combined_results.extend(json.load(input_fp))

        # Derive output filename
        output_path = join(self.output_dir, f"{run_id}.json")

        LOG.info(
            "Writing %i consolidated result files to %s",
            len(combined_results),
            output_path,
        )
        with open(output_path, "w", encoding=self.encoding) as output_fp:
            json.dump(combined_results, output_fp)


class BatchInference:
    """Batch inferencing logic"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: IterableDataset,
        accelerator: Accelerator,
        seed: Optional[int],
        output_dir: Optional[str],
        generation_config: Optional[Dict],
        post_processors: Optional[List[AbstractPostProcessor]] = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.post_processors = post_processors
        self.seed = seed
        self.output_dir = output_dir
        self.generation_config = generation_config

    @staticmethod
    def validate_and_warn(cfg: DictDefault) -> None:
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

        # Forced defaults
        LOG.info("Setting train_on_inputs=False, required for inferencing")
        cfg.train_on_inputs = False

    def _inference(
        self,
        tokenized_prompt: List[int],
        generation_config: Optional[GenerationConfig],
        remove_prompt_in_result=True,
        skip_special_tokens=True,
    ) -> InferenceResponse:
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
                decoded_output = self.tokenizer.decode(
                    output[input_ids.shape[1] :],
                    skip_special_tokens=skip_special_tokens,
                )
            else:
                decoded_output = self.tokenizer.decode(
                    output,
                    skip_special_tokens=skip_special_tokens,
                )

            decoded_responses.append(
                InferenceResponseRecord(response=decoded_output, tokens=output.shape[0])
            )

            total_response_tokens = total_response_tokens + output.shape[0]

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

    def run(self) -> InferenceRunResult:
        """Run batch inference."""

        run_id = gen_run_id(self.accelerator)

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
                device_output_file = join(
                    str(self.output_dir),
                    f"{run_id}_{str(self.accelerator.device.type)}{str(self.accelerator.device.index)}.tmp.json",
                )
                LOG.info("Writing output to: %s", device_output_file)
                with open(
                    device_output_file,
                    "w",
                    encoding="utf-8",
                ) as device_output_fp:
                    json.dump(device_responses, device_output_fp)

            self.accelerator.wait_for_everyone()

            # Invoke post-processors
            if self.accelerator.is_main_process and self.post_processors is not None:
                # Invoke post-processor(s) on consolidated results
                for post_processor in self.post_processors:
                    post_processor.trigger(
                        run_id=run_id, result_paths=self._get_tmp_files(run_id=run_id)
                    )

            run_time_sec = round(time.perf_counter() - start_time, 3)

        finally:
            # Cleanup temporary files
            if self.accelerator.is_main_process:
                for tmp_file in self._get_tmp_files(run_id=run_id):
                    LOG.info("Removing temporary file: %s", tmp_file)
                    os.remove(tmp_file)

        return InferenceRunResult(run_time_sec=run_time_sec)
