"""
EBFT Trainer — Energy-Based Fine-Tuning integrated via GRPOTrainer.

Extends AxolotlGRPOTrainer by plugging feature-matching rewards into
the standard GRPO reward function interface.

Paper: "Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models"
       (Jelassi et al., 2026) https://arxiv.org/abs/2603.12248
"""

import contextlib
import copy
from typing import TYPE_CHECKING, Any

import torch
from datasets import Dataset, IterableDataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback

from axolotl.core.trainers.ebft.args import AxolotlEBFTConfig
from axolotl.core.trainers.ebft.rewards import (
    apply_embed_method,
    extract_hidden_states,
    get_alignment_rewards,
    get_diversity_rewards,
    whiten_embeddings_batched,
)
from axolotl.core.trainers.grpo.trainer import (
    AxolotlAsyncGRPOTrainer,
    AxolotlGRPOTrainer,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from collections import defaultdict

    from accelerate import Accelerator
    from trl.generation.vllm_generation import VLLMGeneration

LOG = get_logger(__name__)


class EBFTMixin:
    """
    Mixin that adds EBFT feature-matching reward logic to any GRPO-based trainer.

    Provides:
    - Frozen feature network setup (shared weights for PEFT, deepcopy otherwise)
    - _feature_matching_reward() callable for GRPO reward function interface
    - _sequential_rollout() for multi-turn conversations
    """

    # Type stubs for attributes provided by the composed GRPOTrainer base class.
    # These are not defined here but accessed via cooperative multiple inheritance.
    if TYPE_CHECKING:
        accelerator: Accelerator
        model: PreTrainedModel
        args: AxolotlEBFTConfig
        processing_class: PreTrainedTokenizerBase
        num_generations: int
        vllm_generation: VLLMGeneration
        _metrics: defaultdict

    _tag_names = ["trl", "ebft", "axolotl"]

    def __init__(
        self,
        model: str | PreTrainedModel,
        args: AxolotlEBFTConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset
        | IterableDataset
        | dict[str, Dataset | IterableDataset]
        | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = (None, None),
        peft_config: Any | None = None,
    ):
        # Pass our feature-matching reward function to GRPOTrainer
        # It will be called with (prompts, completions, **kwargs) where
        # kwargs includes all extra dataset fields like "ground_truth"
        super().__init__(  # type: ignore[call-arg]
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
        assert args is not None

        # --- Feature network setup ---
        unwrapped = self.accelerator.unwrap_model(self.model)
        # Check for PEFT model — use hasattr for robustness across DDP/FSDP wrapping
        self._share_feature_weights = isinstance(unwrapped, PeftModel) or hasattr(
            unwrapped, "disable_adapter"
        )

        if self._share_feature_weights:
            # Share weights: use actor's base model with adapters disabled.
            # Saves a full model copy (~8 GB for 4B model).
            self.feature_network = None
            param_gb = sum(p.numel() for p in unwrapped.parameters()) * 2 / 1e9
            LOG.info(
                f"EBFT feature network shares actor weights (PEFT disable_adapter). "
                f"Saving ~{param_gb:.1f} GB"
            )
        else:
            LOG.info("Creating frozen feature network for EBFT (deepcopy)...")
            self.feature_network = copy.deepcopy(unwrapped)
            for param in self.feature_network.parameters():
                param.requires_grad = False
            self.feature_network.eval()

        # Compute layer indices from fractional depths
        # Handle VLM models where num_hidden_layers is on text_config
        config = unwrapped.config
        if hasattr(config, "text_config") and hasattr(
            config.text_config, "num_hidden_layers"
        ):
            config = config.text_config
        num_layers = config.num_hidden_layers
        self.feature_layer_indices = [
            int(frac * num_layers) for frac in args.ebft_feature_layers
        ]
        LOG.info(
            f"EBFT feature extraction from layers {self.feature_layer_indices} "
            f"(of {num_layers} total), embed_method={args.ebft_embed_method}"
        )
        if args.ebft_adaptive_max_tokens:
            LOG.info(
                f"EBFT adaptive max_tokens enabled "
                f"(gt_length_multiplier={args.ebft_gt_length_multiplier})"
            )

    _adaptive_max_lock = None  # initialized lazily

    def _generate_only(self, inputs, rank0_only=False):
        """Override to set per-batch max_tokens based on ground-truth length.

        Uses a lock to prevent race conditions in async mode where concurrent
        BG threads could interleave mutations of max_completion_length.
        """
        import threading

        args = self.args
        if (
            args.ebft_adaptive_max_tokens
            and hasattr(self, "vllm_generation")
            and inputs
        ):
            gt_texts = [
                x.get("ground_truth", "") for x in inputs if x.get("ground_truth")
            ]
            if gt_texts:
                gt_token_counts = [
                    len(self.processing_class.encode(gt, add_special_tokens=False))
                    for gt in gt_texts
                ]
                multiplier = args.ebft_gt_length_multiplier
                max_completion = self.vllm_generation.max_completion_length
                adaptive_max = max(
                    min(int(c * multiplier), max_completion) for c in gt_token_counts
                )
                adaptive_max = max(adaptive_max, 64)

                if self._adaptive_max_lock is None:
                    self._adaptive_max_lock = threading.Lock()
                with self._adaptive_max_lock:
                    original = self.vllm_generation.max_completion_length
                    self.vllm_generation.max_completion_length = adaptive_max
                    try:
                        return super()._generate_only(inputs, rank0_only)
                    finally:
                        self.vllm_generation.max_completion_length = original

        return super()._generate_only(inputs, rank0_only)

    @torch.no_grad()
    def _feature_matching_reward(
        self,
        prompts: list,
        completions: list,
        ground_truth: list[str] | None = None,
        remaining_turns: list | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Compute feature-matching rewards for generated completions.

        This is called by GRPOTrainer's _generate_and_score_completions()
        as a standard reward function. The `ground_truth` field comes from
        the dataset via reward_kwargs.

        For multi-turn conversations, `remaining_turns` contains the subsequent
        user/assistant turn pairs. When present, we do sequential rollouts:
        generate each assistant turn conditioned on history + previous generations,
        then compute feature-matching rewards on the full generated conversation.

        Args:
            prompts: List of prompt strings/messages
            completions: List of generated completion strings
            ground_truth: List of reference completion strings (from dataset)
            remaining_turns: List of remaining conversation turns after the
                first assistant turn (for multi-turn rollouts)

        Returns:
            List of scalar rewards, one per completion
        """
        if ground_truth is None:
            LOG.warning("No ground_truth field in dataset — using zero rewards")
            return [0.0] * len(prompts)

        device = self.accelerator.device
        args = self.args
        num_gens = self.num_generations

        # --- Multi-turn sequential rollout ---
        # If remaining_turns is provided, generate subsequent assistant turns
        # by calling vLLM for each turn, building up the full conversation.
        if remaining_turns is not None and hasattr(self, "vllm_generation"):
            completions = self._sequential_rollout(
                prompts, completions, remaining_turns, num_gens
            )

        # --- Tokenize generated sequences: prompt + completion ---
        gen_texts = []
        gen_prompt_texts = []
        for p, c in zip(prompts, completions, strict=True):
            if isinstance(p, list):
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
            gen_prompt_texts.append(prompt_text)

        gen_encoded = self.processing_class(
            text=gen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.args, "max_length", None)
            or getattr(self.args, "max_seq_length", None)
            or 2048,
            add_special_tokens=False,
        )
        gen_ids = gen_encoded["input_ids"].to(device)
        gen_mask = gen_encoded["attention_mask"].to(device)

        # Compute prompt lengths for completion_mean pooling
        gen_prompt_lengths = torch.tensor(
            [
                len(self.processing_class.encode(pt, add_special_tokens=False))
                for pt in gen_prompt_texts
            ],
            device=device,
        )

        # --- Tokenize ground-truth sequences: prompt + ground_truth ---
        # For multi-turn (remaining_turns present), render the full GT conversation
        # through the chat template to preserve role markers between turns.
        gt_texts = []
        gt_prompt_texts = []
        for i, (p, gt) in enumerate(zip(prompts, ground_truth, strict=True)):
            if i % num_gens != 0:
                continue  # Only need one GT per prompt group
            if isinstance(p, list):
                prompt_text = self.processing_class.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                # Multi-turn: build full GT conversation with remaining turns
                if remaining_turns is not None:
                    prompt_idx = i // num_gens
                    turns = (
                        remaining_turns[prompt_idx]
                        if prompt_idx < len(remaining_turns)
                        else []
                    )
                    if turns:
                        gt_conv = list(p) + [{"role": "assistant", "content": gt}]
                        gt_conv.extend(turns)
                        full_gt_text = self.processing_class.apply_chat_template(
                            gt_conv, tokenize=False, add_generation_prompt=False
                        )
                        gt_texts.append(full_gt_text)
                        gt_prompt_texts.append(prompt_text)
                        continue
            else:
                prompt_text = p
            gt_texts.append(prompt_text + gt)
            gt_prompt_texts.append(prompt_text)

        gt_encoded = self.processing_class(
            text=gt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.args, "max_length", None)
            or getattr(self.args, "max_seq_length", None)
            or 2048,
            add_special_tokens=False,
        )
        gt_ids = gt_encoded["input_ids"].to(device)
        gt_mask = gt_encoded["attention_mask"].to(device)

        gt_prompt_lengths = torch.tensor(
            [
                len(self.processing_class.encode(pt, add_special_tokens=False))
                for pt in gt_prompt_texts
            ],
            device=device,
        )

        # --- Extract features from frozen feature network ---
        # INVARIANT: disable_adapter() yields the unmodified base weights because
        # _sync_peft_weights_no_merge and _sync_lora_adapter never call
        # merge_adapter() — they compute merged weights as new tensors or save
        # the adapter to filesystem. Base weights are never modified in-place.
        if self._share_feature_weights:
            unwrapped = self.accelerator.unwrap_model(self.model)
            feature_ctx = unwrapped.disable_adapter()
        else:
            unwrapped = self.feature_network
            feature_ctx = contextlib.nullcontext()

        with feature_ctx:
            was_training = unwrapped.training
            unwrapped.eval()
            gen_hidden = extract_hidden_states(
                unwrapped, gen_ids, gen_mask, self.feature_layer_indices
            )
            gt_hidden = extract_hidden_states(
                unwrapped, gt_ids, gt_mask, self.feature_layer_indices
            )
            if was_training:
                unwrapped.train()

        # --- Pool to sequence-level embeddings ---
        gen_emb = apply_embed_method(
            gen_hidden,
            args.ebft_embed_method,
            gen_mask,
            prompt_lengths=gen_prompt_lengths,
        )
        gt_emb = apply_embed_method(
            gt_hidden,
            args.ebft_embed_method,
            gt_mask,
            prompt_lengths=gt_prompt_lengths,
        )

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

        # Scale by 2 per paper equation (7):
        #   r_j = 2*φ(ŷ_j)^T*φ(y) - 2/(n-1) * Σ_{j'≠j} φ(ŷ_j)^T*φ(ŷ_{j'})
        alignment = alignment * 2
        diversity = diversity * 2

        rewards = (
            args.ebft_alignment_coef * alignment - args.ebft_diversity_coef * diversity
        )

        # Compute CFM loss: ||E[φ(ŷ)] - φ(y)||^2 (paper eq 2)
        gen_reshaped = gen_emb.view(-1, num_gens, gen_emb.shape[-1])
        mean_gen = gen_reshaped.mean(dim=1)  # (num_prompts, D)
        cfm_loss = ((mean_gen - gt_emb) ** 2).sum(dim=-1).mean()

        # Log feature-matching metrics to console and wandb
        _align = alignment.mean().item()
        _divers = diversity.mean().item()
        _reward = rewards.mean().item()
        _cfm = cfm_loss.item()

        LOG.info(
            f"ebft reward | "
            f"align {_align:+.3f} ^ | "
            f"divers {_divers:+.3f} v | "
            f"cfm {_cfm:.3f} v | "
            f"reward {_reward:+.3f} ^"
        )

        # Log to wandb via trainer's _metrics (picked up by GRPO's logging)
        mode = "train" if self.model.training else "eval"
        if hasattr(self, "_metrics"):
            self._metrics[mode]["ebft/alignment"].append(_align)
            self._metrics[mode]["ebft/diversity"].append(_divers)
            self._metrics[mode]["ebft/cfm_loss"].append(_cfm)
            self._metrics[mode]["ebft/reward"].append(_reward)

        return rewards.cpu().tolist()

    @torch.no_grad()
    def _sequential_rollout(
        self,
        prompts: list,
        first_completions: list,
        remaining_turns: list,
        num_gens: int,
    ) -> list:
        """
        Extend single-turn completions into multi-turn conversations.

        For each prompt group, takes the first generated assistant turn and
        sequentially generates subsequent assistant turns by calling vLLM,
        building up a full multi-turn conversation.

        Args:
            prompts: List of prompt message lists (repeated num_gens times)
            first_completions: List of generated first-turn completions
            remaining_turns: List of remaining turn pairs after first assistant turn.
                Each element is a list of dicts: [{"role": "user", "content": "..."},
                {"role": "assistant", "content": "...GT..."}]
            num_gens: Number of generations per prompt

        Returns:
            Extended completions incorporating all generated turns
        """
        vllm_client = self.vllm_generation.vllm_client
        max_tokens = getattr(self.args, "max_completion_length", 256)
        temperature = getattr(self.args, "temperature", 0.7)
        gen_kwargs = getattr(self.args, "generation_kwargs", None) or {}

        extended_completions = []

        for idx in range(len(prompts)):
            prompt_msgs = prompts[idx] if isinstance(prompts[idx], list) else []
            first_comp = first_completions[idx]

            # Extract first completion text
            if isinstance(first_comp, list):
                first_text = first_comp[0].get("content", "") if first_comp else ""
            else:
                first_text = first_comp

            # Get remaining turns for this prompt (same for all num_gens copies)
            prompt_idx = idx // num_gens
            turns = (
                remaining_turns[prompt_idx] if prompt_idx < len(remaining_turns) else []
            )

            if not turns:
                extended_completions.append(first_text)
                continue

            # Build conversation with generated first turn
            conv = list(prompt_msgs) + [{"role": "assistant", "content": first_text}]

            # Generate subsequent turns
            for turn in turns:
                if turn["role"] == "user":
                    conv.append(turn)
                elif turn["role"] == "assistant":
                    try:
                        result = vllm_client.chat(
                            messages=[conv],
                            n=1,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            generation_kwargs=gen_kwargs,
                        )
                        gen_ids = result.get("completion_ids", [[]])[0]
                        gen_text = self.processing_class.decode(
                            gen_ids, skip_special_tokens=True
                        )
                    except Exception as e:
                        LOG.warning(f"Multi-turn rollout generation failed: {e}")
                        gen_text = ""

                    conv.append({"role": "assistant", "content": gen_text})

            # Render full conversation through chat template, then extract
            # everything after the original prompt as the "completion" text.
            # This preserves role markers and formatting between turns.
            full_rendered = self.processing_class.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            prompt_rendered = self.processing_class.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            completion_text = full_rendered[len(prompt_rendered) :]
            extended_completions.append(completion_text)

        return extended_completions


class AxolotlEBFTTrainer(EBFTMixin, AxolotlGRPOTrainer):
    """EBFT trainer using synchronous GRPO (standard vLLM generation)."""

    pass


class AxolotlAsyncEBFTTrainer(EBFTMixin, AxolotlAsyncGRPOTrainer):
    """EBFT trainer using async GRPO (prefetches next batch during training)."""

    pass
