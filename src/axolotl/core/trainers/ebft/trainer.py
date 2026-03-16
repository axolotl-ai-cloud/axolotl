"""
EBFT Trainer — Energy-Based Fine-Tuning integrated via GRPOTrainer.

Extends AxolotlGRPOTrainer by plugging feature-matching rewards into
the standard GRPO reward function interface.

Paper: "Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models"
       (Jelassi et al., 2026) https://arxiv.org/abs/2603.12248
"""

import copy
from typing import Any

import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback

from axolotl.core.trainers.ebft.args import AxolotlEBFTConfig
from axolotl.core.trainers.ebft.rewards import (
    apply_embed_method,
    extract_hidden_states,
    get_alignment_rewards,
    get_diversity_rewards,
    whiten_embeddings_batched,
)
from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class AxolotlEBFTTrainer(AxolotlGRPOTrainer):
    """
    Energy-Based Fine-Tuning trainer.

    Extends GRPOTrainer by replacing external reward functions with
    feature-matching rewards from a frozen feature network. Reuses all
    of GRPO's infrastructure: vLLM generation, RLOO advantages, clipped
    policy gradient loss, distributed training, logging, etc.

    The key trick: we register a callable reward function that computes
    feature-matching rewards using the frozen network, ground-truth
    completions from the dataset, and the generated completions.
    """

    _tag_names = ["trl", "ebft", "axolotl"]

    def __init__(
        self,
        model: str | PreTrainedModel,
        args: AxolotlEBFTConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: Any | None = None,
    ):
        # Pass our feature-matching reward function to GRPOTrainer
        # It will be called with (prompts, completions, **kwargs) where
        # kwargs includes all extra dataset fields like "ground_truth"
        super().__init__(
            model=model,
            reward_funcs=[self._feature_matching_reward],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # --- Create frozen feature network ---
        LOG.info("Creating frozen feature network for EBFT...")
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.feature_network = copy.deepcopy(unwrapped)
        for param in self.feature_network.parameters():
            param.requires_grad = False
        self.feature_network.eval()

        # Compute layer indices from fractional depths
        num_layers = unwrapped.config.num_hidden_layers
        self.feature_layer_indices = [
            int(frac * num_layers) for frac in args.ebft_feature_layers
        ]
        LOG.info(
            f"EBFT feature extraction from layers {self.feature_layer_indices} "
            f"(of {num_layers} total), embed_method={args.ebft_embed_method}"
        )

    @torch.no_grad()
    def _feature_matching_reward(
        self,
        prompts: list,
        completions: list,
        ground_truth: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Compute feature-matching rewards for generated completions.

        This is called by GRPOTrainer's _generate_and_score_completions()
        as a standard reward function. The `ground_truth` field comes from
        the dataset via reward_kwargs.

        Args:
            prompts: List of prompt strings/messages
            completions: List of generated completion strings
            ground_truth: List of reference completion strings (from dataset)

        Returns:
            List of scalar rewards, one per completion
        """
        if ground_truth is None:
            LOG.warning("No ground_truth field in dataset — using zero rewards")
            return [0.0] * len(prompts)

        device = self.accelerator.device
        args = self.args
        num_gens = self.num_generations

        # --- Tokenize generated sequences: prompt + completion ---
        gen_texts = []
        for p, c in zip(prompts, completions):
            if isinstance(p, list):
                # Chat format — extract text
                prompt_text = self.processing_class.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = p
            if isinstance(c, list):
                comp_text = c[0].get("content", "") if c else ""
            else:
                comp_text = c
            gen_texts.append(prompt_text + comp_text)

        gen_encoded = self.processing_class(
            text=gen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.args, "max_length", None) or getattr(self.args, "max_seq_length", None) or 2048,
            add_special_tokens=False,
        )
        gen_ids = gen_encoded["input_ids"].to(device)
        gen_mask = gen_encoded["attention_mask"].to(device)

        # --- Tokenize ground-truth sequences: prompt + ground_truth ---
        gt_texts = []
        for i, (p, gt) in enumerate(zip(prompts, ground_truth)):
            if i % num_gens != 0:
                continue  # Only need one GT per prompt group
            if isinstance(p, list):
                prompt_text = self.processing_class.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = p
            gt_texts.append(prompt_text + gt)

        gt_encoded = self.processing_class(
            text=gt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.args, "max_length", None) or getattr(self.args, "max_seq_length", None) or 2048,
            add_special_tokens=False,
        )
        gt_ids = gt_encoded["input_ids"].to(device)
        gt_mask = gt_encoded["attention_mask"].to(device)

        # --- Extract features from frozen feature network ---
        gen_hidden = extract_hidden_states(
            self.feature_network, gen_ids, gen_mask, self.feature_layer_indices
        )
        gt_hidden = extract_hidden_states(
            self.feature_network, gt_ids, gt_mask, self.feature_layer_indices
        )

        # --- Pool to sequence-level embeddings ---
        gen_emb = apply_embed_method(gen_hidden, args.ebft_embed_method, gen_mask)
        gt_emb = apply_embed_method(gt_hidden, args.ebft_embed_method, gt_mask)

        # --- Optional whitening ---
        batch_size = gen_emb.shape[0]
        if args.ebft_use_whitening and batch_size > 1:
            num_prompts = batch_size // num_gens
            gen_reshaped = gen_emb.view(num_prompts, num_gens, -1)
            whitened_gen_list = []
            whitened_gt_list = []
            for i in range(num_prompts):
                w_gen, w_gt = whiten_embeddings_batched(
                    gen_reshaped[i], gt_emb[i : i + 1]
                )
                whitened_gen_list.append(w_gen)
                whitened_gt_list.append(w_gt)
            gen_emb = torch.cat(whitened_gen_list, dim=0)
            gt_emb = torch.cat(whitened_gt_list, dim=0)
        else:
            gen_emb = torch.nn.functional.normalize(gen_emb, p=2, dim=-1)
            gt_emb = torch.nn.functional.normalize(gt_emb, p=2, dim=-1)

        # Repeat gt_emb: each GT repeated num_generations times
        gt_emb_expanded = gt_emb.repeat_interleave(num_gens, dim=0)

        # --- Compute rewards ---
        alignment = get_alignment_rewards(gen_emb, gt_emb_expanded)
        diversity = get_diversity_rewards(gen_emb, num_gens)

        rewards = args.ebft_alignment_coef * alignment - args.ebft_diversity_coef * diversity

        return rewards.cpu().tolist()
