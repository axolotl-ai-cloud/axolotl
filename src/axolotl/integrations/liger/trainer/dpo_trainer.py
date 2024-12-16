"""
integration of liger dpo kernels with dpotrainer
"""
from typing import Dict, List, Literal, Union

import torch
from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
from liger_kernel.transformers.trainer.orpo_trainer import _FSDPForwardRedirection
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel

from axolotl.core.trainers.base import AxolotlDPOTrainer


class AxolotlLigerDPOTrainer(AxolotlDPOTrainer):
    """
    Extend the DPO Trainer to use LIGER kernels for DPO
    """

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ):
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together,
        and compute the DPO loss using Liger's fused kernel.

        This method replaces the original `concatenated_forward` implementation to use Liger.
        """

        # Prepare concatenated inputs
        concatenated_batch = self.concatenated_inputs(batch, self.padding_value)

        # Extract concatenated inputs
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        # For encoder-decoder models, you'd need to construct decoder_input_ids, etc.
        # This example assumes a causal decoder-only model.
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat(
            (prompt_attention_mask, completion_attention_mask), dim=1
        )

        # Align inputs by removing leading padding
        for i in range(attention_mask.size(0)):
            first_one_idx = torch.nonzero(attention_mask[i])[0].item()
            input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
            attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)

        # Remove trailing empty columns
        empty_cols = torch.sum(attention_mask, dim=0) == 0
        if empty_cols.any():
            first_empty_col = torch.nonzero(empty_cols)[0].item()
            input_ids = input_ids[:, :first_empty_col]
            attention_mask = attention_mask[:, :first_empty_col]

        if self.args.max_length is not None:
            input_ids = input_ids[:, : self.args.max_length]
            attention_mask = attention_mask[:, : self.args.max_length]

        # Labels are completion_input_ids shifted by one token right
        # For causal LM, labels are the completion part only
        labels = torch.cat(
            (torch.zeros_like(prompt_input_ids), completion_input_ids), dim=1
        )
        labels = labels[:, 1:]  # shift left by one
        attention_mask = attention_mask[:, 1:]
        labels = labels[:, : attention_mask.size(1)]

        # Mask out the prompt portion from loss
        labels[~attention_mask.bool()] = self.label_pad_token_id

        # Prepare reference model hidden states if ref_model exists
        use_ref_model = self.ref_model is not None and not self.reference_free

        # Run main model forward to get hidden states
        # If using FSDP, redirect forward calls
        if isinstance(model, FullyShardedDataParallel):
            outputs = _FSDPForwardRedirection()(
                model,
                model._fsdp_wrapped_module.model,  # pylint: disable=protected-access
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        else:
            # If model is a DataParallel, unwrap
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            outputs = model.model(
                input_ids, attention_mask=attention_mask, use_cache=False
            )

        last_hidden_state = outputs.last_hidden_state

        ref_last_hidden_state = None
        if use_ref_model:
            ref_model = self.ref_model
            if isinstance(ref_model, FullyShardedDataParallel):
                with torch.no_grad():
                    ref_outputs = _FSDPForwardRedirection()(
                        ref_model,
                        ref_model._fsdp_wrapped_module.model,  # pylint: disable=protected-accessåå
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            else:
                if isinstance(ref_model, torch.nn.DataParallel):
                    ref_model = ref_model.module
                with torch.no_grad():
                    ref_outputs = ref_model.model(
                        input_ids, attention_mask=attention_mask, use_cache=False
                    )
            ref_last_hidden_state = ref_outputs.last_hidden_state

        # Retrieve lm_head parameters
        lm_head = model.lm_head
        ref_lm_head = (
            self.ref_model.lm_head
            if (use_ref_model and self.ref_model is not None)
            else None
        )

        # Use Liger fused DPO loss
        dpo_loss_fn = LigerFusedLinearDPOLoss(
            ignore_index=self.label_pad_token_id,
            beta=self.beta,
            compute_nll_loss=False,
            compiled=True,
            use_ref_model=use_ref_model,
        )

        # call fused Liger DPO
        if use_ref_model:
            loss_acc, aux_outputs = dpo_loss_fn(
                lm_head.weight,  # lin_weight
                last_hidden_state,  # _input
                labels,  # target
                bias=lm_head.bias,
                ref_input=ref_last_hidden_state,
                ref_weight=ref_lm_head.weight,
                ref_bias=ref_lm_head.bias,
            )

            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits_mean,
                policy_rejected_logits_mean,
                policy_nll_loss,
            ) = aux_outputs[:5]

        else:
            # No reference model scenario: Liger kernel treats ref_logps as 0
            loss_acc, aux_outputs = dpo_loss_fn(
                lm_head.weight,
                last_hidden_state,
                labels,
                bias=lm_head.bias,
            )
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits_mean,
                policy_rejected_logits_mean,
                policy_nll_loss,
            ) = aux_outputs[:5]

        # Add aux loss if enabled
        if self.aux_loss_enabled and hasattr(outputs, "aux_loss"):
            loss_acc = loss_acc + self.aux_loss_coef * outputs.aux_loss

        # Add RPO loss if requested (RPO is a variant that adds NLL loss)
        if self.args.rpo_alpha is not None:
            # policy_nll_loss: average negative log-likelihood of chosen completions
            loss_acc = loss_acc + self.args.rpo_alpha * policy_nll_loss.mean()

        return (
            loss_acc,
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
            policy_nll_loss,
        )

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """
        Compute the DPO loss and other metrics for a given batch using the Liger fused kernel.
        """
        metrics = {}

        (
            loss,
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
            policy_nll_loss,
        ) = self.concatenated_forward(model, batch)

        # For metrics, we approximate chosen/rejected rewards as beta * (log π(y) - log π_ref(y)) if ref model used.
        # If no ref model is used, we can't compute reward_accuracies meaningfully. For simplicity, we assume ref_model presence.
        if self.ref_model is not None and not self.reference_free:
            # If you want full parity with original DPOTrainer metrics (like chosen_rewards, rejected_rewards),
            # you'd need to run reference forward or store reference log ps. The Liger kernel currently doesn't
            # return ref_chosen_logps/ref_rejected_logps explicitly. By design, Liger directly computes DPO.
            #
            # Here we approximate chosen_rewards and rejected_rewards from the difference in chosen/rejected logps.
            # Since Liger DPO does not output ref logps separately, you may need to modify the Liger kernel to
            # also output them if you need all the metrics. For now, we'll skip them or provide a placeholder.

            # Placeholder: chosen/rejected "rewards" can't be retrieved directly from Liger as-is.
            # If needed, integrate ref_chosen_logps/ref_rejected_logps into Liger kernel returns.
            chosen_rewards = policy_chosen_logps * self.beta  # approximation
            rejected_rewards = policy_rejected_logps * self.beta  # approximation
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            metrics[f"{train_eval}_rewards/chosen"] = chosen_rewards.mean().cpu().item()
            metrics[f"{train_eval}_rewards/rejected"] = (
                rejected_rewards.mean().cpu().item()
            )
            metrics[f"{train_eval}_rewards/accuracies"] = (
                reward_accuracies.mean().cpu().item()
            )
            metrics[f"{train_eval}_rewards/margins"] = (
                (chosen_rewards - rejected_rewards).mean().cpu().item()
            )

        metrics[f"{train_eval}_logps/chosen"] = policy_chosen_logps.mean().cpu().item()
        metrics[f"{train_eval}_logps/rejected"] = (
            policy_rejected_logps.mean().cpu().item()
        )
        metrics[f"{train_eval}_logits/chosen"] = (
            policy_chosen_logits_mean.detach().cpu().item()
        )
        metrics[f"{train_eval}_logits/rejected"] = (
            policy_rejected_logits_mean.detach().cpu().item()
        )

        if self.args.rpo_alpha is not None:
            metrics[f"{train_eval}_nll_loss"] = (
                policy_nll_loss.mean().detach().cpu().item()
            )

        return loss.mean(), metrics
