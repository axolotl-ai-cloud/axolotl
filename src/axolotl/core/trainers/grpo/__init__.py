"""GRPO Specific Strategy for training"""

import importlib
import inspect
import logging
from typing import Any

from trl.trainer.grpo_trainer import RewardFunc

from axolotl.core.trainers.grpo.args import AxolotlGRPOConfig
from axolotl.core.trainers.grpo.trainer import (
    AxolotlGRPOSequenceParallelTrainer,
    AxolotlGRPOTrainer,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.trl import TRLConfig

LOG = logging.getLogger(__name__)


class GRPOStrategy:
    """Strategy for GRPO training"""

    @classmethod
    def get_trainer_class(
        cls, sequence_parallel: bool
    ) -> type[AxolotlGRPOTrainer] | type[AxolotlGRPOSequenceParallelTrainer]:
        if sequence_parallel:
            return AxolotlGRPOSequenceParallelTrainer
        return AxolotlGRPOTrainer

    @classmethod
    def get_training_args_class(cls) -> type[AxolotlGRPOConfig]:
        return AxolotlGRPOConfig

    @classmethod
    def set_training_args_kwargs(cls, cfg: DictDefault) -> dict[str, Any]:
        grpo_args_kwargs: dict[str, Any] = {}

        if not hasattr(cfg, "trl") or not cfg.trl:
            return grpo_args_kwargs

        trl: TRLConfig = cfg.trl  # type: ignore

        if trl.use_vllm:
            grpo_args_kwargs["use_vllm"] = trl.use_vllm
            grpo_args_kwargs["vllm_server_host"] = trl.vllm_server_host or trl.vllm.host  # type: ignore[attr-defined]
            grpo_args_kwargs["vllm_server_port"] = trl.vllm_server_port or trl.vllm.port  # type: ignore[attr-defined]
            if trl.vllm_server_timeout:
                grpo_args_kwargs["vllm_server_timeout"] = trl.vllm_server_timeout
            if trl.vllm_guided_decoding_regex:
                grpo_args_kwargs["vllm_guided_decoding_regex"] = (
                    trl.vllm_guided_decoding_regex
                )

        if trl.num_generations:
            grpo_args_kwargs["num_generations"] = trl.num_generations

        if trl.sync_ref_model:
            grpo_args_kwargs["sync_ref_model"] = trl.sync_ref_model

            if trl.ref_model_mixup_alpha:
                grpo_args_kwargs["ref_model_mixup_alpha"] = trl.ref_model_mixup_alpha

            if trl.ref_model_sync_steps:
                grpo_args_kwargs["ref_model_sync_steps"] = trl.ref_model_sync_steps

        grpo_args_kwargs["max_completion_length"] = trl.max_completion_length
        grpo_args_kwargs["log_completions"] = trl.log_completions
        grpo_args_kwargs["num_completions_to_print"] = trl.num_completions_to_print

        if trl.reward_weights:
            grpo_args_kwargs["reward_weights"] = trl.reward_weights

        if trl.scale_rewards is not None:
            grpo_args_kwargs["scale_rewards"] = trl.scale_rewards

        if trl.loss_type is not None:
            grpo_args_kwargs["loss_type"] = trl.loss_type
        if trl.mask_truncated_completions is not None:
            grpo_args_kwargs["mask_truncated_completions"] = (
                trl.mask_truncated_completions
            )

        if trl.temperature is not None:
            grpo_args_kwargs["temperature"] = trl.temperature
        if trl.top_p is not None:
            grpo_args_kwargs["top_p"] = trl.top_p
        if trl.top_k is not None:
            grpo_args_kwargs["top_k"] = trl.top_k
        if trl.min_p is not None:
            grpo_args_kwargs["min_p"] = trl.min_p
        if trl.repetition_penalty is not None:
            grpo_args_kwargs["repetition_penalty"] = trl.repetition_penalty

        if trl.num_iterations is not None:
            grpo_args_kwargs["num_iterations"] = trl.num_iterations
        if trl.epsilon is not None:
            grpo_args_kwargs["epsilon"] = trl.epsilon
        if trl.epsilon_high is not None:
            grpo_args_kwargs["epsilon_high"] = trl.epsilon_high

        if trl.use_liger_loss is not None:
            grpo_args_kwargs["use_liger_loss"] = trl.use_liger_loss

        return grpo_args_kwargs

    @classmethod
    def set_trainer_args(cls, cfg: DictDefault) -> list[Any]:
        trainer_args = []
        if cfg.trl and cfg.trl.reward_funcs:
            reward_funcs = []
            for reward_func_fqn in cfg.trl.reward_funcs:
                reward_funcs.append(cls.get_reward_func(reward_func_fqn))
            trainer_args.append(reward_funcs)

        return trainer_args

    @classmethod
    def set_trainer_kwargs(cls, cfg: DictDefault) -> dict[str, Any]:
        trainer_kwargs = {}
        if cfg.trl and cfg.trl.reward_processing_classes:
            trainer_kwargs["reward_processing_classes"] = (
                cfg.trl.reward_processing_classes
            )
        return trainer_kwargs

    @classmethod
    def get_collator(cls, *args, **kwargs):  # pylint: disable=unused-argument
        # No data collation is needed in GRPO, handled by trl's trainer __init__
        return None

    @classmethod
    def get_blocklist_args_kwargs(cls) -> list[str]:
        return ["dataset_num_proc"]

    @classmethod
    def get_reward_func(cls, reward_func_fqn: str) -> RewardFunc:
        """
        Returns the reward function from the given fully qualified name, or the path to the reward function model.

        Args:
            reward_func_fqn (str): Fully qualified name of the reward function (e.g. r1_grpo.gsm8k_transform),
                or a HF hub path to the reward model.

        Returns:
            RewardFunc: A callable that accepts prompts and completions and returns rewards,
                or a path to a reward model.

        Raises:
            ValueError: If the reward function does not accept at least two arguments.
        """
        try:
            # use importlib to dynamically load the reward function from the module
            reward_func_module_name = reward_func_fqn.split(".")[-1]
            reward_func_module = importlib.import_module(
                ".".join(reward_func_fqn.split(".")[:-1])
            )
            reward_func = getattr(reward_func_module, reward_func_module_name)
            if not len(inspect.signature(reward_func).parameters) >= 2:
                raise ValueError(
                    "Reward function must accept at least two arguments: prompts: list and completions: list"
                )
            return reward_func
        except ModuleNotFoundError:
            # the user has passed a string (ideally indicating the path of a reward model)
            LOG.info(
                f"Reward function {reward_func_fqn} is a pre-trained model path - if this is unexpected, please check the reward function path."
            )
            return reward_func
