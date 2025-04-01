"""Module for TRL PPO trainer"""

from typing import Literal, Union

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


class AxolotlORPOTrainer(RngLoaderMixin, SchedulerMixin, ORPOTrainer):
    """
    Extend the base ORPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "orpo"]

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the ORPO loss and other metrics for the given batch of inputs for train or test."""

        # TODO remove once https://github.com/huggingface/trl/pull/3069 is included in a trl release

        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = (
            self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        )
        # full ORPO loss
        loss = policy_nll_loss - losses.mean()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(
            chosen_rewards
        ).mean()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(
            rejected_rewards
        ).mean()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(
            reward_accuracies
        ).mean()
        metrics[f"{prefix}rewards/margins"] = self.accelerator.gather_for_metrics(
            chosen_rewards - rejected_rewards
        ).mean()
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(policy_rejected_logps).detach().mean()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(policy_chosen_logps).detach().mean()
        )
        metrics[f"{prefix}logits/rejected"] = self.accelerator.gather_for_metrics(
            policy_rejected_logits.detach().mean()
        ).mean()
        metrics[f"{prefix}logits/chosen"] = self.accelerator.gather_for_metrics(
            policy_chosen_logits.detach().mean()
        ).mean()
        metrics[f"{prefix}nll_loss"] = (
            self.accelerator.gather_for_metrics(policy_nll_loss).detach().mean()
        )
        metrics[f"{prefix}log_odds_ratio"] = (
            self.accelerator.gather_for_metrics(log_odds_ratio).detach().mean()
        )
        metrics[f"{prefix}log_odds_chosen"] = (
            self.accelerator.gather_for_metrics(log_odds_chosen).detach().mean()
        )
        for k, v in metrics.items():
            metrics[k] = v.item()
        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * aux_loss

        return loss, metrics


class AxolotlKTOTrainer(RngLoaderMixin, SchedulerMixin, KTOTrainer):
    """
    Extend the base KTOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "kto"]


class AxolotlCPOTrainer(RngLoaderMixin, SchedulerMixin, CPOTrainer):
    """
    Extend the base CPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "cpo"]

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the CPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        losses, chosen_rewards, rejected_rewards = self.cpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )

        loss = losses.mean() + self.cpo_alpha * policy_nll_loss
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/accuracies"] = (
            self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        )
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards)
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(policy_rejected_logps)
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(policy_chosen_logps)
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(policy_rejected_logits.detach().mean())
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(policy_chosen_logits.detach().mean())
            .mean()
            .item()
        )
        metrics[f"{prefix}nll_loss"] = (
            self.accelerator.gather_for_metrics(policy_nll_loss).detach().mean().item()
        )

        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * aux_loss

        return loss, metrics


class AxolotlRewardTrainer(RngLoaderMixin, SchedulerMixin, RewardTrainer):
    """
    Extend the base RewardTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "reward"]


class AxolotlPRMTrainer(RngLoaderMixin, SchedulerMixin, PRMTrainer):
    """
    Extend the base trl.PRMTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "prm"]
