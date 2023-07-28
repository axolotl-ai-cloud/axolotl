"""
This module defines the BatchInference class, and performs multi-GPU batch inferencing
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, TypedDict

import torch
import transformers
from accelerate import Accelerator
from datasets import IterableDataset
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


class InferenceRunResult(TypedDict):
    """Result object returned by BatchInference.run"""

    run_time_sec: float


class InferenceResponseRecord(TypedDict):
    """An individual response record, will be multiple when num_return_sequences > 1 in the generation config"""

    response: str
    tokens: int


class InferenceResponse(TypedDict):
    """Inferencing result object"""

    device: str
    prompt: str
    prompt_tokens: int
    total_tokens: int
    total_response_tokens: int
    generate_time_sec: float
    responses: List[InferenceResponseRecord]


class AbstractPostProcessor(ABC):
    """Inferencing result post-processor, provides a hook for adding additional post processors in the future"""

    def __init__(
        self,
        cfg: DictDefault,
    ) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def trigger(self, results: List[InferenceResponse]) -> None:
        pass


class JsonFilePostProcessor(AbstractPostProcessor):
    """Saves inferencing results to the specified JSON file"""

    def __init__(self, cfg: DictDefault, filename: str, encoding="utf-8") -> None:
        super().__init__(cfg=cfg)

        self.filename = filename
        self.encoding = encoding

    def trigger(self, results: List[InferenceResponse]) -> None:
        # Write to output file outside of the loop
        LOG.info("Writing %i results to %s", len(results), self.filename)
        with open(self.filename, "w", encoding=self.encoding) as output_fp:
            json.dump(results, output_fp)


class BatchInference:
    """Batch inferencing logic"""

    def __init__(
        self,
        cfg: DictDefault,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: IterableDataset,
        accelerator: Accelerator,
        post_processors: Optional[List[AbstractPostProcessor]] = None,
    ) -> None:
        self.cfg = cfg
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.post_processors = post_processors

    def validate_and_warn(self) -> None:
        """Validate configuration settings for batch inference"""

        if not self.cfg.generation_config:
            LOG.warning(
                "It is highly recommended to specify your own generation_config"
            )

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
            device=self.accelerator.device,
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
        """Run batch evaluation and return average loss and perplexity."""

        start_time = time.perf_counter()

        if self.cfg.seed is not None:
            transformers.enable_full_determinism(seed=self.cfg.seed)

        # Parse user-provided generation configuration
        generation_config = GenerationConfig(
            **(
                self.cfg.generation_config
                if self.cfg.generation_config is not None
                else {}
            ),
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

        if self.accelerator.is_local_main_process:
            LOG.info(
                "Running batch inference on %i samples",
                len(self.dataset),
            )

        with self.accelerator.split_between_processes(
            [prompt["input_ids"] for prompt in self.dataset]
        ) as device_prompts:
            device_responses: List[InferenceResponse] = []
            for device_prompt in device_prompts:
                prompt_response = self._inference(
                    tokenized_prompt=device_prompt, generation_config=generation_config
                )

                device_responses.append(prompt_response)

        gathered_responses = self.accelerator.gather(device_responses)

        # Invoke post-processors
        if self.accelerator.is_local_main_process and self.post_processors is not None:
            # Invoke post-processor(s) on consolidated results
            for post_processor in self.post_processors:
                post_processor.trigger(results=gathered_responses)

        run_time_sec = round(time.perf_counter() - start_time, 3)

        return InferenceRunResult(run_time_sec=run_time_sec)
