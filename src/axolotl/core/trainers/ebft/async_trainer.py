"""Async EBFT on TRL's AsyncGRPOTrainer.

The rollout process (``EBFTRolloutLoop``) generates and leaves the advantage pending;
``EBFTScorer`` scores each optimizer step's groups on the trainer, collectively on
every rank: a no-grad forward of each completion's feature sequence and its group's
reference through the frozen feature network (the actor with adapters disabled, or a
frozen copy for full fine-tuning), pooled hidden states at ``ebft_feature_layers``,
then ``group_feature_rewards``. See ``axolotl.core.trainers.grpo.async_scoring``.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from accelerate.utils import is_peft_model
from datasets import Dataset
from transformers import PreTrainedModel

from axolotl.core.trainers.ebft.args import EBFTFieldsMixin
from axolotl.core.trainers.ebft.rewards import group_feature_rewards, pool_sequence
from axolotl.core.trainers.grpo.async_packing import sample_boundaries
from axolotl.core.trainers.grpo.async_scoring import (
    ScoredRow,
    TrainerScorer,
    pack_sequences,
    unpack_sequences,
)
from axolotl.core.trainers.grpo.async_trainer import (
    AxolotlAsyncGRPOConfig,
    AxolotlAsyncGRPOTrainer,
)

EBFT_ROLLOUT_LOOP = "axolotl.core.trainers.ebft.async_rollout.EBFTRolloutLoop"


@dataclass
class AxolotlAsyncEBFTConfig(EBFTFieldsMixin, AxolotlAsyncGRPOConfig):
    """``AxolotlAsyncGRPOConfig`` plus EBFT's fields."""

    generation_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Extra vLLM sampling parameters for EBFT rollouts, e.g. "
            "stop_token_ids."
        },
    )


class EBFTScorer(TrainerScorer):
    """EBFT's feature-matching reward, scored on the trainer."""

    name = "ebft"

    def __init__(
        self,
        layer_indices: list[int],
        embed_method: str,
        use_whitening: bool,
        alignment_coef: float,
        diversity_coef: float,
        feature_network: PreTrainedModel | None = None,
    ):
        self.layer_indices = layer_indices
        self.embed_method = embed_method
        self.use_whitening = use_whitening
        self.alignment_coef = alignment_coef
        self.diversity_coef = diversity_coef
        self.feature_network = feature_network

    def group_extras(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = samples[0]["score_payload"]
        return [
            {
                "input_ids": payload["reference_ids"],
                "pool_start": payload["reference_pool_start"],
            }
        ]

    def collate(self, rows: list[ScoredRow], index: int):
        per_row: dict[str, tuple[list[torch.Tensor], float]] = {}
        for row in rows:
            extras = [extra for extra in row.extras if extra["scorer"] == index]
            sequences = [s["score_payload"]["feature_ids"] for s in row.samples]
            sequences += [extra["input_ids"] for extra in extras]
            packed = pack_sequences(sequences, self.share_prefixes)
            packed["uids"] = torch.tensor(
                [s["score_uid"] for s in row.samples] + [e["uid"] for e in extras],
                dtype=torch.long,
            )
            packed["pool_start"] = torch.tensor(
                [s["score_payload"]["pool_start"] for s in row.samples]
                + [e["pool_start"] for e in extras],
                dtype=torch.long,
            )
            for key, value in packed.items():
                per_row.setdefault(key, ([], 0))[0].append(value)
        return per_row

    @contextlib.contextmanager
    def _feature_model(self, trainer):
        if self.feature_network is not None:
            yield self.feature_network
            return
        # The actor's base weights, which no weight sync modifies.
        model = trainer.accelerator.unwrap_model(trainer.model)
        was_training = model.training
        model.eval()
        try:
            with model.disable_adapter():
                yield model
        finally:
            model.train(was_training)

    @torch.no_grad()
    def features(self, trainer, inputs: dict[str, torch.Tensor]):
        unpacked = unpack_sequences(inputs)
        if not self.share_prefixes:
            unpacked["model_inputs"].update(sample_boundaries(unpacked["model_inputs"]))
        with self._feature_model(trainer) as model:
            outputs = model(
                **unpacked["model_inputs"],
                output_hidden_states=True,
                logits_to_keep=1,
                use_cache=False,
            )
        hidden = torch.cat(
            [outputs.hidden_states[i][0] for i in self.layer_indices], dim=-1
        )
        offsets = unpacked["offsets"].tolist()
        positions = unpacked["positions"]
        pool_starts = inputs["pool_start"][0].tolist()
        embeddings = torch.stack(
            [
                pool_sequence(
                    hidden[positions[start:end]].float(),
                    self.embed_method,
                    pool_starts[i],
                )
                for i, (start, end) in enumerate(
                    zip(offsets[:-1], offsets[1:], strict=True)
                )
            ]
        )
        return inputs["uids"][0, : len(offsets) - 1], embeddings

    def group_rewards(self, sample_features, extra_features):
        rewards, alignment, diversity, cfm_loss = group_feature_rewards(
            sample_features,
            extra_features[:1],
            self.use_whitening,
            self.alignment_coef,
            self.diversity_coef,
        )
        return rewards, {
            "ebft/alignment": alignment.mean().item(),
            "ebft/diversity": diversity.mean().item(),
            "ebft/cfm_loss": cfm_loss.item(),
            "ebft/reward": rewards.mean().item(),
        }


def feature_layer_indices(model: PreTrainedModel, fractions: list[float]) -> list[int]:
    config = model.config.get_text_config()
    return [int(fraction * config.num_hidden_layers) for fraction in fractions]


class AxolotlAsyncEBFTTrainer(AxolotlAsyncGRPOTrainer):
    """Structured-mode EBFT on async GRPO (see module doc)."""

    _tag_names = ["trl", "async-grpo", "ebft", "axolotl"]

    def __init__(
        self,
        model: PreTrainedModel,
        args: AxolotlAsyncEBFTConfig | None = None,
        train_dataset=None,
        processing_class=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        **kwargs,
    ):
        if args is None:
            raise ValueError("AxolotlAsyncEBFTTrainer needs an AxolotlAsyncEBFTConfig")
        if (
            isinstance(train_dataset, Dataset)
            and "ground_truth" not in train_dataset.column_names
        ):
            raise ValueError("EBFT needs a `ground_truth` column in the dataset")
        args.rollout_loop = args.rollout_loop or EBFT_ROLLOUT_LOOP
        args.rollout_loop_kwargs = {
            "adaptive_max_tokens": args.ebft_adaptive_max_tokens,
            "gt_length_multiplier": args.ebft_gt_length_multiplier,
            "generation_kwargs": args.generation_kwargs,
            **(args.rollout_loop_kwargs or {}),
        }
        scorer = EBFTScorer(
            layer_indices=feature_layer_indices(model, args.ebft_feature_layers),
            embed_method=args.ebft_embed_method,
            use_whitening=args.ebft_use_whitening,
            alignment_coef=args.ebft_alignment_coef,
            diversity_coef=args.ebft_diversity_coef,
        )
        super().__init__(
            model,
            reward_funcs=[],
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            trainer_scorers=[scorer],
            **kwargs,
        )
        if not is_peft_model(self.model):
            feature_network = copy.deepcopy(self.accelerator.unwrap_model(self.model))
            # TRL's fused-head forward closes over the original model's forward.
            feature_network.__dict__.pop("forward", None)
            feature_network.requires_grad_(False)
            scorer.feature_network = feature_network.eval()
