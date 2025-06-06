"""Module for TRL PPO trainer"""

import torch
from tqdm import tqdm
from trl import (
    CPOTrainer,
    KTOTrainer,
    ORPOTrainer,
    PPOTrainer,
    PRMTrainer,
    RewardTrainer,
)

from axolotl.core.trainers.mixins import RngLoaderMixin
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.core.trainers.mixins.scheduler import SchedulerMixin


class TRLPPOTrainer(PPOTrainer):
    """Wrapper for TRL PPO trainer to handle customizations"""

    tag_names = ["axolotl", "ppo"]

    def train(
        self,
        reward_pipe,
        resume_from_checkpoint=None,  # pylint: disable=unused-argument
    ):
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 32,
        }
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 16,
        }

        for _, batch in tqdm(enumerate(self.dataloader)):
            query_tensors = batch["input_ids"]

            # generate model response
            response_tensors, ref_response_tensors = self.generate(
                query_tensors,
                return_prompt=False,
                generate_ref_response=True,
                **generation_kwargs,
            )
            batch["response"] = self.tokenizer.batch_decode(response_tensors)
            batch["ref_response"] = self.tokenizer.batch_decode(ref_response_tensors)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_pipe(texts, **sent_kwargs)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
            ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
            ref_pipe_outputs = reward_pipe(ref_texts, **sent_kwargs)
            ref_rewards = [
                torch.tensor(output[1]["score"]) for output in ref_pipe_outputs
            ]
            batch["ref_rewards"] = ref_rewards

            # Run PPO step
            stats = self.step(query_tensors, response_tensors, rewards)
            self.log_stats(
                stats,
                batch,
                rewards,
                columns_to_log=["query", "response", "ref_response", "ref_rewards"],
            )


class AxolotlORPOTrainer(
    RngLoaderMixin, SchedulerMixin, OptimizerMixin, OptimizerInitMixin, ORPOTrainer
):
    """
    Extend the base ORPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "orpo"]


class AxolotlKTOTrainer(
    RngLoaderMixin, SchedulerMixin, OptimizerMixin, OptimizerInitMixin, KTOTrainer
):
    """
    Extend the base KTOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "kto"]


class AxolotlCPOTrainer(
    RngLoaderMixin, SchedulerMixin, OptimizerMixin, OptimizerInitMixin, CPOTrainer
):
    """
    Extend the base CPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "cpo"]


class AxolotlRewardTrainer(
    RngLoaderMixin, SchedulerMixin, OptimizerMixin, OptimizerInitMixin, RewardTrainer
):
    """
    Extend the base RewardTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "reward"]


class AxolotlPRMTrainer(
    RngLoaderMixin, SchedulerMixin, OptimizerMixin, OptimizerInitMixin, PRMTrainer
):
    """
    Extend the base trl.PRMTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "prm"]
