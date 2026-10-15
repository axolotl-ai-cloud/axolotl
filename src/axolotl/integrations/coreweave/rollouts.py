"""Synchronous GRPO generation using CoreWeave Dedicated Inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch.distributed as dist

from .checkpoints import CheckpointPublisher
from .client import CoreWeaveClient


def check_rank_errors(error):
    """Keep every trainer rank on the same side of the rollout boundary."""
    errors = [error]
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    for message in errors:
        if message is not None:
            raise RuntimeError(f"CoreWeave rollout failed: {message}")


MAX_CONCURRENT_SAMPLES = 8


class CoreWeaveRollouts:
    """Publish current weights once per optimizer step before sampling any rank."""

    def __init__(self, config):
        self.config = config
        self.client = None
        self.publisher = None
        self.step = None

    def __call__(self, prompts, trainer):
        error = None
        prompt_ids = None
        try:
            if not all(
                isinstance(prompt, list)
                and prompt
                and all(
                    isinstance(message, dict)
                    and isinstance(message.get("content"), str)
                    and message.get("role") in {"system", "user", "assistant"}
                    for message in prompt
                )
                for prompt in prompts
            ):
                raise ValueError(
                    "CoreWeave GRPO requires text-only conversational prompts"
                )
            prompt_ids, images, multimodal = trainer._tokenize_prompts(prompts)
            if images or multimodal:
                raise ValueError("CoreWeave GRPO does not support multimodal inputs")
            if self.client is None:
                self.client = CoreWeaveClient(self.config)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = str(exc)
        check_rank_errors(error)

        if self.step != trainer.state.global_step:
            # All ranks must participate in sharded state-dict gathering.
            weights = trainer.accelerator.get_state_dict(trainer.model_wrapped)
            error = None
            if trainer.accelerator.is_main_process:
                try:
                    if self.publisher is None:
                        self.publisher = CheckpointPublisher(
                            self.config,
                            self.client,
                            Path(trainer.args.output_dir) / "coreweave-checkpoints",
                        )
                    model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
                    self.publisher.publish(
                        weights, model.config.to_dict(), trainer.state.global_step
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    error = str(exc)
            del weights
            check_rank_errors(error)
            identity = [
                self.publisher.identity if trainer.accelerator.is_main_process else None
            ]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(identity, src=0)
            self.client.ready_identity = identity[0]
            self.step = trainer.state.global_step

        def sample(tokens):
            return self.client.sample(
                tokens,
                n=1,
                max_tokens=trainer.max_completion_length,
                temperature=trainer.temperature,
                top_p=trainer.top_p,
            )[0]

        completions, logprobs = [], []
        error = None
        try:
            workers = max(1, min(MAX_CONCURRENT_SAMPLES, len(prompt_ids)))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="coreweave-sample"
            ) as executor:
                for result in executor.map(sample, prompt_ids):
                    completions.append(result["tokens"])
                    logprobs.append(result["logprobs"])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = str(exc)
        check_rank_errors(error)
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completions,
            "logprobs": logprobs,
        }
