"""
GRPO Specific Strategy for training
"""
import importlib

from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer


class GRPOStrategy:
    """
    Strategy for GRPO training
    """

    @classmethod
    def get_trainer_class(cls):
        return AxolotlGRPOTrainer

    @classmethod
    def get_training_args_class(cls):
        from axolotl.core.trainers.grpo.args import AxolotlGRPOConfig

        return AxolotlGRPOConfig

    @classmethod
    def set_training_args_kwargs(cls, cfg):
        grpo_args_kwargs = {}
        if cfg.grpo_use_vllm:
            grpo_args_kwargs["use_vllm"] = cfg.grpo_use_vllm
            if cfg.grpo_vllm_device:
                grpo_args_kwargs["vllm_device"] = cfg.grpo_vllm_device
            else:
                grpo_args_kwargs["vllm_device"] = "auto"
            if cfg.grpo_vllm_gpu_memory_utilization:
                grpo_args_kwargs[
                    "vllm_gpu_memory_utilization"
                ] = cfg.grpo_vllm_gpu_memory_utilization
        return grpo_args_kwargs

    @classmethod
    def set_trainer_kwargs(cls, cfg):
        trainer_kwargs = {}
        if cfg.grpo_reward_funcs:
            reward_funcs = []
            for reward_func_module in cfg.grpo_reward_funcs:
                # use importlib to dynamically load the reward function from the module
                reward_func_module_name = reward_func_module.split(".")[-1]
                reward_func_module = importlib.import_module(reward_func_module)
                reward_func = getattr(reward_func_module, reward_func_module_name)
                reward_funcs.append(reward_func)
            trainer_kwargs["reward_funcs"] = reward_funcs

        return trainer_kwargs
