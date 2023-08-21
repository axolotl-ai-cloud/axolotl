"""Callbacks for Trainer class"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from optimum.bettertransformer import BetterTransformer
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, IntervalStrategy

from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.distributed import is_main_process, zero_first, zero_only

if TYPE_CHECKING:
    from axolotl.utils.trainer import AxolotlTrainingArguments

LOG = logging.getLogger("axolotl.callbacks")
IGNORE_INDEX = -100


class SavePeftModelCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
    """Callback to save the PEFT adapter"""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(
            peft_model_path, save_safetensors=args.save_safetensors
        )

        return control


class SaveBetterTransformerModelCallback(
    TrainerCallback
):  # pylint: disable=too-few-public-methods
    """Callback to save the BetterTransformer wrapped model"""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            control.should_save = True

        if control.should_save:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
            )

            model = BetterTransformer.reverse(kwargs["model"])
            model.save_pretrained(checkpoint_folder)
            # FIXME - need to cleanup old checkpoints

            # since we're saving here, we don't need the trainer loop to attempt to save too b/c
            # the trainer will raise an exception since it can't save a BetterTransformer wrapped model
            control.should_save = False
        return control


class GPUStatsCallback(
    TrainerCallback
):  # pylint: disable=too-few-public-methods disable=unused-argument
    """Callback to track GPU utilization"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.logged = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.logged and state.global_step > 1:
            log_gpu_memory_usage(LOG, "while training", self.cfg.device)
            self.logged = True
        return control


def mmlu_eval_callback_factory(trainer, tokenizer):
    accuracy = evaluate.load("accuracy")
    abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
        tokenizer("B", add_special_tokens=False).input_ids[0],
        tokenizer("C", add_special_tokens=False).input_ids[0],
        tokenizer("D", add_special_tokens=False).input_ids[0],
    ]
    mmlu_split = "eval"
    if trainer.args.mmlu_dataset == "sampled":
        mmlu_dataset = load_dataset(
            "pharaouk/dharma-1",
            data_files={
                "eval": "pharaouk/dharma-1",
            },
        )
        # mmlu_dataset = mmlu_dataset.remove_columns("subject")
    if trainer.args.mmlu_dataset == "mmlu-zs":
        mmlu_dataset = load_dataset(
            "openaccess-ai-collective/mmlu-evals",
            data_files={
                "eval": "zero_shot_mmlu_val.json",
                "test": "zero_shot_mmlu_test.json",
            },
        )
        # mmlu_dataset = mmlu_dataset.remove_columns("subject")
    # MMLU Five-shot (Eval/Test only)
    elif trainer.args.mmlu_dataset in ["mmlu", "mmlu-fs"]:
        mmlu_dataset = load_dataset(
            "openaccess-ai-collective/mmlu-evals",
            data_files={
                "eval": "five_shot_mmlu_val.json",
                "test": "five_shot_mmlu_test.json",
            },
        )
        # mmlu_dataset = mmlu_dataset.remove_columns('subject')
    else:
        raise ValueError("unhandled value for mmlu_dataset training args")
    mmlu_dataset = mmlu_dataset[trainer.args.mmlu_split]
    if trainer.args.max_mmlu_samples is not None:
        mmlu_dataset = mmlu_dataset.select(range(trainer.args.max_mmlu_samples))

    def tokenize_evals(example):
        source = f"{tokenizer.bos_token}{example['input']}"
        target = f"{example['output']}{tokenizer.eos_token}"

        tokenized_source = tokenizer(
            source,
            max_length=2048,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_target = tokenizer(
            target,
            max_length=2048,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = tokenized_source["input_ids"] + tokenized_target["input_ids"]
        labels = [-100] * len(tokenized_source["input_ids"]) + tokenized_target[
            "input_ids"
        ]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "subject": example["subject"],
        }

    with zero_first(is_main_process()):
        mmlu_dataset = mmlu_dataset.map(tokenize_evals)

    class BenchEvalCallback(TrainerCallback):
        """
        TrainerCallback that runs the MMLU evals
        """

        def on_evaluate(
            self,
            args: AxolotlTrainingArguments,
            state: TrainerState,  # pylint: disable=unused-argument
            control: TrainerControl,  # pylint: disable=unused-argument
            metrics: Dict[str, float],  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
        ):
            with zero_only(is_main_process()):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.max_length
                trainer.data_collator.max_length = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(
                        trainer.model,
                        batch,
                        prediction_loss_only=False,
                    )
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch["labels"][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id - 1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {"bench_loss": loss_mmlu / len(data_loader)}
                subject = mmlu_dataset["subject"]
                subjects: dict = {s: {"refs": [], "preds": []} for s in set(subject)}
                for s, p, r in zip(  # pylint: disable=invalid-name
                    subject, preds, refs
                ):
                    subjects[s]["preds"].append(p)
                    subjects[s]["refs"].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]["refs"],
                        predictions=subjects[subject]["preds"],
                    )["accuracy"]
                    if not pd.isna(subject_score):
                        results[
                            f"bench_{mmlu_split}_accuracy_{subject}"
                        ] = subject_score
                        subject_scores.append(subject_score)
                results[f"bench_{mmlu_split}_accuracy"] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.max_length = source_max_len

    return BenchEvalCallback
