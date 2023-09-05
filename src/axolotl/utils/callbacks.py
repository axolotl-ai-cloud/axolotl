"""Callbacks for Trainer class"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset
from optimum.bettertransformer import BetterTransformer
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, IntervalStrategy

import wandb
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.distributed import (
    barrier,
    gather_scalar_from_all_ranks,
    get_world_size,
    is_distributed,
    is_main_process,
    zero_first,
)

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


def bench_eval_callback_factory(trainer, tokenizer):
    accuracy = evaluate.load("accuracy")
    abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
        tokenizer("B", add_special_tokens=False).input_ids[0],
        tokenizer("C", add_special_tokens=False).input_ids[0],
        tokenizer("D", add_special_tokens=False).input_ids[0],
        tokenizer("E", add_special_tokens=False).input_ids[0],
        tokenizer("F", add_special_tokens=False).input_ids[0],
        tokenizer("G", add_special_tokens=False).input_ids[0],
    ]
    bench_split = "eval"

    def transform_bench_subject(example):
        # Split on ':' and trim whitespace
        parts = example["subject"].split(":")
        first_part = (
            parts[0].strip().lower().replace("-", "_")
        )  # Lowercase the first part
        second_part = (
            parts[1].strip().replace("-", "_") if len(parts) > 1 else "all"
        )  # Replace hyphens with underscores

        # Return the transformed values
        return {"name": first_part, "subject": second_part}

    if trainer.args.bench_dataset == "mmlu-zs":
        bench_dataset = load_dataset(
            "openaccess-ai-collective/mmlu-evals",
            data_files={
                "eval": "zero_shot_mmlu_val.json",
                "test": "zero_shot_mmlu_test.json",
            },
        )
        # bench_dataset = bench_dataset.remove_columns("subject")
    # MMLU Five-shot (Eval/Test only)
    elif trainer.args.bench_dataset in ["mmlu", "mmlu-fs"]:
        bench_dataset = load_dataset(
            "openaccess-ai-collective/mmlu-evals",
            data_files={
                "eval": "five_shot_mmlu_val.json",
                "test": "five_shot_mmlu_test.json",
            },
        )
        # bench_dataset = bench_dataset.remove_columns('subject')
    elif "/" in trainer.args.bench_dataset:
        bench_ds = trainer.args.bench_dataset
        bench_ds_name = "/".join(bench_ds.split("/", 2)[:2])
        bench_ds_data_file = "/".join(bench_ds.split("/", 2)[2:])
        bench_dataset = load_dataset(
            bench_ds_name,
            data_files={
                "eval": bench_ds_data_file,
            },
        )
        bench_dataset["eval"] = bench_dataset["eval"].map(transform_bench_subject)
    else:
        raise ValueError(
            f"unhandled value `{trainer.args.bench_dataset}` for bench_dataset training args"
        )
    bench_dataset = bench_dataset[trainer.args.bench_split]
    if trainer.args.max_bench_samples is not None:
        bench_dataset = bench_dataset.select(range(trainer.args.max_bench_samples))

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
        labels = [IGNORE_INDEX] * len(tokenized_source["input_ids"]) + tokenized_target[
            "input_ids"
        ]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "subject": example["subject"],
        }

    with zero_first(is_main_process()):
        bench_dataset = bench_dataset.map(tokenize_evals)
        bench_dataset = bench_dataset.filter(lambda x: x["labels"][-2] in abcd_idx)

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
            data_loader = trainer.get_bench_dataloader(
                bench_dataset.remove_columns(["input", "subject", "output", "name"])
            )
            trainer.model.eval()
            preds, refs = [], []
            loss_bench = 0
            for batch in tqdm(data_loader, total=len(data_loader)):
                (loss, logits, labels) = trainer.prediction_step(
                    trainer.model,
                    batch,
                    prediction_loss_only=False,
                )
                # There are two tokens, the output, and eos token.
                for i, logit in enumerate(logits):
                    label_non_zero_id = (batch["labels"][i] != IGNORE_INDEX).nonzero()[
                        0
                    ][0]
                    logit_abcd = logit[label_non_zero_id - 1][abcd_idx]
                    preds.append(torch.argmax(logit_abcd).item())
                labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
                refs += [
                    abcd_idx.index(label) if label in abcd_idx else -1
                    for label in labels.tolist()
                ]
                loss_bench += loss.item()
            # Extract results by subject.
            bench_name = bench_dataset["name"]
            bench_names: dict = {s: {"refs": [], "preds": []} for s in set(bench_name)}
            for s, p, r in zip(bench_name, preds, refs):  # pylint: disable=invalid-name
                bench_names[s]["preds"].append(p)
                bench_names[s]["refs"].append(r)
            barrier()
            local_bench_names = bench_names
            gathered_bench_names: List[Dict] = [{} for _ in range(get_world_size())]
            # Gather results from all GPUs to GPU 0

            loss_bench_ranks = gather_scalar_from_all_ranks(
                lambda: loss_bench, get_world_size()
            )
            len_data_loader_ranks = gather_scalar_from_all_ranks(
                lambda: len(data_loader), get_world_size()
            )

            if is_distributed() and not is_main_process():
                dist.gather_object(local_bench_names, dst=0)
            else:
                if is_distributed():
                    dist.gather_object(local_bench_names, gathered_bench_names, dst=0)
                else:
                    gathered_bench_names = [local_bench_names]
                bench_loss = sum(loss_bench_ranks) / sum(len_data_loader_ranks)
                results = {f"{bench_split}_bench_loss": bench_loss}

                # Combine results from all GPUs
                combined_bench_names: Dict[str, Dict[str, List]] = {}
                for bench_name in gathered_bench_names:
                    for name, data in bench_name.items():
                        if name not in combined_bench_names:
                            combined_bench_names[name] = {"refs": [], "preds": []}
                        combined_bench_names[name]["refs"].extend(data["refs"])
                        combined_bench_names[name]["preds"].extend(data["preds"])

                bench_scores = []
                bench_refs = []
                bench_preds = []
                for (
                    bench_name
                ) in combined_bench_names:  # pylint: disable=consider-using-dict-items
                    bench_score = accuracy.compute(
                        references=combined_bench_names[bench_name]["refs"],
                        predictions=combined_bench_names[bench_name]["preds"],
                    )["accuracy"]
                    bench_refs.extend(combined_bench_names[bench_name]["refs"])
                    bench_preds.extend(combined_bench_names[bench_name]["preds"])
                    if not pd.isna(bench_score):
                        results[
                            f"{bench_split}_bench_accuracy_{bench_name}"
                        ] = bench_score
                        bench_scores.append(bench_score)
                    else:
                        results[f"{bench_split}_bench_accuracy_{bench_name}"] = 0.0
                        bench_scores.append(0.0)
                results[f"{bench_split}_bench_average_accuracy"] = np.mean(bench_scores)
                results[f"{bench_split}_bench_total_accuracy"] = accuracy.compute(
                    references=bench_refs, predictions=bench_preds
                )["accuracy"]
                trainer.log(results)

    return BenchEvalCallback


def log_prediction_callback_factory(trainer: Trainer, tokenizer):
    class LogPredictionCallback(TrainerCallback):
        """Callback to log prediction values during each evaluation"""

        def __init__(self, cfg):
            self.cfg = cfg
            self.logged = False

        def on_evaluate(
            self,
            args: AxolotlTrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            # model,
            # tokenizer,
            eval_dataloader,
            **kwargs,
        ):
            LOG.info("logging predictions")

            # Initialize an empty wandb.Table
            table = wandb.Table(columns=["id", "Prompt", "Correct Completion", "Predicted Completion"])

            # Iterate over the evaluation data
            # for batch in eval_dataloader:
            #     inputs, labels = batch
            #     predictions = model(inputs)

            #     # Convert the predictions and labels to a readable format
            #     predictions = tokenizer.decode(predictions)
            #     labels = tokenizer.decode(labels)

            #     # Add the data to the wandb.Table
            #     table.add_data(predictions, labels)

            # Generate fake data for the table
            # for _ in range(10):
            #     fake_prediction = "Fake Prediction " + str(_)
            #     fake_ground_truth = "Fake Ground Truth " + str(_)
            #     table.add_data(fake_prediction, fake_ground_truth)

            # print(dir(eval_dataloader))

            # eval_loop = trainer.prediction_loop if trainer.args.use_legacy_prediction_loop else trainer.evaluation_loop
            # output = eval_loop(
            #     eval_dataloader,
            #     description="Evaluation",
            #     # No point gathering the predictions if there are no metrics, otherwise we defer to
            #     # self.args.prediction_loss_only
            #     # prediction_loss_only=True if trainer.compute_metrics is None else None,
            #     prediction_loss_only=False,
            #     # ignore_keys=ignore_keys,
            #     # metric_key_prefix=metric_key_prefix,
            # )

            # print(type(output))
            # print(dir(output))
            # print(output.predictions)
            # print(output.label_ids)
            # print(output.metrics)

            # # Extract the predictions and labels from the output
            # predictions = output.predictions
            # labels = output.label_ids
            # # Convert the predictions and labels to a readable format
            # predictions = [tokenizer.decode(p) for p in predictions]
            # labels = [tokenizer.decode(l) for l in labels]

            # # Add the data to the wandb.Table
            # for prediction, label in zip(predictions, labels):
            #     table.add_data(prediction, label)

            trainer.model.eval()

            def logits_to_tokens(logits) -> str:

                probabilities = torch.softmax(logits, dim=-1)

                # Get the predicted token ids (the ones with the highest probability)
                predicted_token_ids = torch.argmax(probabilities, dim=-1)

                # Decode the predicted token ids to get the plaintext
                # predicted_tokens = tokenizer.batch_decode(predicted_token_ids)
                # predicted_tokens = tokenizer.decode(predicted_token_ids)
                # return predicted_tokens

                return predicted_token_ids


            # preds, refs = [], []
            # loss_bench = 0
            # predictions = []
            id = 0
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):

                # batch.data['labels'].shape
                # torch.Size([2, 320])
                # values at front with -100 are supposed to be prompt tokens
                # values after are completion tokens

                # batch.data['input_ids'].shape
                # torch.Size([2, 320])
                
                # # Extract prompt and completion tokens from input_ids based on labels
                # prompt_token_ids = batch.data['input_ids'][batch.data['labels'] == IGNORE_INDEX]
                # completion_token_ids = batch.data['input_ids'][batch.data['labels'] != IGNORE_INDEX]

                # # prompt_texts = tokenizer.batch_decode(batch.data['input_ids'])
                # prompt_texts = tokenizer.batch_decode(prompt_token_ids)
                # completion_texts = tokenizer.batch_decode(completion_token_ids)

                (loss, logits, labels) = trainer.prediction_step(
                    trainer.model,
                    batch,
                    prediction_loss_only=False,
                )

                # prompt_completion_pairs = zip(prompt_texts, logits)

                # print("logits", logits)
                # print("labels", labels)

                # pred_tokens = []
                # for i, logit in enumerate(logits):
                for i, (logit, labels_i) in enumerate(zip(logits, labels)):
                    # for i, (prompt_text, logit) in enumerate(prompt_completion_pairs):
                    # print(dir(logit))
                    # print(logit)
                    # print(logit.shape)
                    # # Convert the logits to probabilities using softmax
                    # probabilities = torch.softmax(logit, dim=-1)

                    # # Get the predicted token id (the one with the highest probability)
                    # predicted_token_id = torch.argmax(probabilities).item()

                    # # Decode the predicted token id to get the plaintext
                    # predicted_token = tokenizer.decode([predicted_token_id])

                    # # Append the predicted token to the preds list
                    # pred_tokens.append(predicted_token)

                    # # Convert the logits to probabilities using softmax
                    # probabilities = torch.softmax(logit, dim=-1)

                    # # Get the predicted token ids (the ones with the highest probability)
                    # predicted_token_ids = torch.argmax(probabilities, dim=-1)

                    # # Decode the predicted token ids to get the plaintext
                    # predicted_tokens = tokenizer.batch_decode(predicted_token_ids)

                    # 
                    # label_non_zero_indices = (batch["labels"][i] != IGNORE_INDEX).nonzero().transpose(0, 1)[0] # FIXME: clean up?

                    prompt_token_indices = (batch["labels"][i] == IGNORE_INDEX).nonzero().transpose(0, 1)[0] # FIXME: clean up?
                    completion_token_indices = (batch["labels"][i] != IGNORE_INDEX).nonzero().transpose(0, 1)[0] # FIXME: clean up?

                    # Extract prompt and completion tokens from input_ids based on labels
                    # prompt_token_ids = batch['input_ids'][batch['labels'] == IGNORE_INDEX]
                    # completion_token_ids = batch['input_ids'][batch['labels'] != IGNORE_INDEX]

                    # prompt_token_ids = batch['input_ids'][batch['labels'] == IGNORE_INDEX]
                    # prompt_token_ids = batch['input_ids'][label_non_zero_indices]
                    # prompt_token_ids = batch['input_ids'][i][label_non_zero_indices]
                    # prompt_token_ids = batch['input_ids'][i]

                    prompt_token_ids = batch['input_ids'][i][prompt_token_indices]
                    completion_token_ids = batch['input_ids'][i][completion_token_indices]

                    # prompt_texts = tokenizer.batch_decode(batch.data['input_ids'])
                    # prompt_texts = tokenizer.batch_decode(prompt_token_ids)
                    prompt_text = tokenizer.decode(prompt_token_ids)
                    completion_text = tokenizer.decode(completion_token_ids)

                    completion_logit = logit[completion_token_indices]
                    # predicted_tokens = logits_to_tokens(logit)
                    predicted_tokens = logits_to_tokens(completion_logit)

                    # Append the predicted tokens to the preds list
                    # pred_tokens.extend(predicted_tokens)
                    # pred_string = " ".join(predicted_tokens) # FIXME: missing spaces
                    prediction_text = tokenizer.decode(predicted_tokens)

                    # print("=" * 80)
                    # print("Prompt:")
                    # print(prompt_text)
                    # print("=" * 80)
                    # print("Expected Completion:")
                    # print(completion_text)
                    # print("=" * 80)
                    # print("Predicted Completion:")
                    # print(prediction_text)
                    # print("=" * 80)

                    table.add_data(id, prompt_text, completion_text, prediction_text)
                    id += 1

                # add prediction
                # convert pred_tokens to a single string
                # pred_string = " ".join(pred_tokens)
                # predictions.append(pred_string)

                # table.add_data(prompt_text, pred_string, "Ground Truth")

            #     # Convert the predictions and labels to a readable format
            #     # predictions = [tokenizer.decode(p) for p in logits]
            #     # labels = [tokenizer.decode(l) for l in labels]

            #     # Add the data to the wandb.Table
            #     for prediction, label in zip(predictions, labels):
            #         table.add_data(prediction, label)

            # using trainer.model generate prediction tokens for each input in eval_dataloader
            # predictions = []
            # for batch in eval_dataloader:
            #     inputs, _ = batch
            #     print(inputs)
            #     with torch.no_grad():
            #         outputs = trainer.model(inputs)
            #     print(outputs)
            #     next_pred = [tokenizer.decode(p) for p in outputs.logits.argmax(dim=-1).tolist()]
            #     print(next_pred)
            #     predictions.extend(next_pred)

            # add the predictions to the table
            # for prediction in predictions:
            #     table.add_data(prediction, "Ground Truth")

            # print table size
            # print("Table size:", len(table.data))

            # print first entry in table
            # print("First entry in table:", table.data[0])

            # Log the wandb.Table
            wandb.run.log({"Predictions vs Ground Truth": table})

            return control

    return LogPredictionCallback
