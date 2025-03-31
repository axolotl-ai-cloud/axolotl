"""Callbacks for Trainer class"""

from __future__ import annotations

import gc
import logging
import os
import traceback
from shutil import copyfile
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Dict, List

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from optimum.bettertransformer import BetterTransformer
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, IntervalStrategy
from trl.models import unwrap_model_for_generation

from axolotl.utils import is_comet_available, is_mlflow_available
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.callbacks.perplexity import Perplexity
from axolotl.utils.distributed import (
    barrier,
    broadcast_dict,
    gather_scalar_from_all_ranks,
    get_world_size,
    is_distributed,
    is_main_process,
    zero_first,
)
from axolotl.utils.schemas.config import AxolotlInputConfig

if TYPE_CHECKING:
    from axolotl.core.trainer_builder import AxolotlTrainingArguments


IGNORE_INDEX = -100
LOG = logging.getLogger("axolotl.callbacks")


class EvalFirstStepCallback(
    TrainerCallback
):  # pylint: disable=too-few-public-methods disable=unused-argument
    """
    Callback to trigger evals on the first step
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.eval_strategy == IntervalStrategy.STEPS and state.global_step == 1:
            control.should_evaluate = True
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


class LossWatchDogCallback(TrainerCallback):
    """Callback to track loss and stop training if loss is too high"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.logged = False
        self.violations = 0
        self.threshold = cfg.loss_watchdog_threshold
        self.patience = cfg.loss_watchdog_patience or 3

    def on_step_end(
        self,
        _args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **_kwargs,
    ):
        if len(state.log_history) > 0 and "loss" in state.log_history[-1]:
            if state.log_history[-1]["loss"] > self.threshold:
                self.violations += 1
                if self.violations >= self.patience:
                    LOG.warning(
                        "Loss is too high, stopping training (loss_watchdog_threshold)"
                    )
                    control.should_training_stop = True
            else:
                self.violations = 0
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

            results = {}
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
                        results[f"{bench_split}_bench_accuracy_{bench_name}"] = (
                            bench_score
                        )
                        bench_scores.append(bench_score)
                    else:
                        results[f"{bench_split}_bench_accuracy_{bench_name}"] = 0.0
                        bench_scores.append(0.0)
                results[f"{bench_split}_bench_average_accuracy"] = np.mean(bench_scores)
                results[f"{bench_split}_bench_total_accuracy"] = accuracy.compute(
                    references=bench_refs, predictions=bench_preds
                )["accuracy"]
                trainer.log(results)

            results = broadcast_dict(results)
            for key, val in results.items():
                metrics[key] = val

    return BenchEvalCallback


def causal_lm_bench_eval_callback_factory(trainer: Trainer, tokenizer):
    class CausalLMBenchEvalCallback(TrainerCallback):
        """Callback to log prediction values during each evaluation"""

        def __init__(self, cfg):
            self.cfg = cfg
            self.logged = False
            self.metrics = self.__maybe_load_metrics()

        def __maybe_load_metrics(self):
            metrics = {}
            for metric in self.cfg.eval_causal_lm_metrics:
                if metric == "perplexity":
                    max_seq_len = self.cfg.eval_max_new_tokens
                    metrics[metric] = Perplexity(
                        tokenizer=tokenizer,
                        max_seq_len=max_seq_len,
                    )
                else:
                    try:
                        metrics[metric] = evaluate.load(metric)
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        LOG.warning(f"{metric}: {exc.args}")
            return metrics

        def on_evaluate(
            self,
            args: AxolotlTrainingArguments,  # pylint: disable=unused-argument
            state: TrainerState,
            control: TrainerControl,
            train_dataloader,  # pylint: disable=unused-argument
            eval_dataloader,
            **kwargs,  # pylint: disable=unused-argument
        ):
            trainer.model_wrapped.eval()

            device = torch.device(
                self.cfg.device
            )  # Use this instead of trainer.model_wrapped.device as it may return cpu if fsdp offloaded

            # pylint: disable=duplicate-code
            generation_config = GenerationConfig(
                max_new_tokens=self.cfg.eval_max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )

            def find_ranges(lst):
                ranges = []
                start = 0
                for i in range(1, len(lst)):
                    if lst[i] == 0:
                        ranges.append((start, i - 1))
                        start = i
                end = len(lst) - 1
                ranges.append((start, end))
                return ranges

            def compute(metric: evaluate.Metric, **kwargs):
                # safely compute a metric and return the score if the format is correct
                metric_score = None
                try:
                    # Only pass the kwargs that are in the metric's feature list
                    metric_kwargs = {
                        k: kwargs[k]
                        for k in metric._feature_names()  # pylint: disable=protected-access
                        if k in kwargs
                    }

                    if isinstance(metric, Perplexity):
                        metric_kwargs["model"] = trainer.model_wrapped

                    metric_score = metric.compute(**metric_kwargs)
                    return (
                        metric_score["score"]
                        if "score" in metric_score
                        else metric_score["mean_score"]
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    traceback.print_exc()
                    LOG.debug(
                        f"Failed to compute metric {metric.name} with kwargs {kwargs.keys()}"
                    )
                return metric_score

            def evaluate_preds(sources, predictions, references):
                scores = {}

                for metric_name, metric in self.metrics.items():
                    score = compute(
                        metric,
                        references=references,
                        predictions=predictions,
                        sources=sources,
                    )
                    if score is None:
                        score = compute(
                            metric,
                            references=[[r] for r in references],
                            predictions=predictions,
                        )
                    scores["eval_" + metric_name] = score
                return scores

            def predict_with_generate():
                eval_src, eval_pred, eval_ref = [], [], []

                with unwrap_model_for_generation(
                    trainer.model_wrapped, trainer.accelerator
                ) as unwrapped_model:
                    for batch in tqdm(eval_dataloader, disable=not is_main_process()):
                        batch_labels = batch["labels"].to(device)
                        batch_input_ids = batch["input_ids"].to(device)

                        if "position_ids" in batch:
                            batch_pos_ids = batch["position_ids"].tolist()
                        else:
                            batch_pos_ids = [None] * len(batch["input_ids"])

                        prompt_token_ids_list = []
                        completion_token_ids_list = []

                        for input_ids_all, labels_all, pos_ids in zip(
                            batch_input_ids,
                            batch_labels,
                            batch_pos_ids,
                        ):
                            if pos_ids is None:
                                pos_ranges = [(0, len(input_ids_all) - 1)]
                            else:
                                pos_ranges = find_ranges(pos_ids)

                            for pos_range in pos_ranges:
                                start, end = pos_range
                                if start == end:
                                    continue

                                input_ids = input_ids_all[start : end + 1]
                                labels = labels_all[start : end + 1]

                                tokens_without_loss = labels == IGNORE_INDEX
                                tokens_with_loss = labels != IGNORE_INDEX
                                tokens_exclude_padding = (
                                    input_ids != tokenizer.pad_token_id
                                )
                                prompt_token_includes = (
                                    tokens_without_loss & tokens_exclude_padding
                                )

                                prompt_token_ids = input_ids[prompt_token_includes]
                                prompt_token_ids_list.append(prompt_token_ids)

                                completion_token_ids = input_ids[tokens_with_loss]
                                completion_token_ids_list.append(completion_token_ids)

                        prompt_texts = tokenizer.batch_decode(
                            prompt_token_ids_list, skip_special_tokens=True
                        )
                        completion_texts = tokenizer.batch_decode(
                            completion_token_ids_list, skip_special_tokens=True
                        )

                        with torch.no_grad():
                            prompt_encoding = tokenizer(
                                prompt_texts, padding=True, return_tensors="pt"
                            ).to(device)

                            predictions = unwrapped_model.generate(
                                **prompt_encoding, generation_config=generation_config
                            )

                            del prompt_encoding

                        prediction_all_tokens = predictions["sequences"].cpu().tolist()
                        prediction_without_prompt_tokens_list = []
                        for prompt_token_ids, prediction_tokens in zip(
                            prompt_token_ids_list, prediction_all_tokens
                        ):
                            prediction_without_prompt_tokens = prediction_tokens[
                                len(prompt_token_ids) :
                            ]
                            prediction_without_prompt_tokens_list.append(
                                prediction_without_prompt_tokens
                            )

                        predicted_texts = tokenizer.batch_decode(
                            prediction_without_prompt_tokens_list,
                            skip_special_tokens=True,
                        )

                        eval_src.extend(prompt_texts)
                        eval_pred.extend(predicted_texts)
                        eval_ref.extend(completion_texts)

                return eval_src, eval_pred, eval_ref

            eval_preds = predict_with_generate()
            trainer.log(evaluate_preds(*eval_preds))

            return control

    return CausalLMBenchEvalCallback


def log_prediction_callback_factory(trainer: Trainer, tokenizer, logger: str):
    class LogPredictionCallback(TrainerCallback):
        """Callback to log prediction values during each evaluation"""

        def __init__(self, cfg):
            self.cfg = cfg
            self.logged = False

        def on_evaluate(
            self,
            args: AxolotlTrainingArguments,  # pylint: disable=unused-argument
            state: TrainerState,
            control: TrainerControl,
            train_dataloader,  # pylint: disable=unused-argument
            eval_dataloader,
            **kwargs,  # pylint: disable=unused-argument
        ):
            eval_table_size = self.cfg.eval_table_size

            if eval_table_size <= 0:
                return control

            trainer.model.eval()
            device = torch.device(self.cfg.device)

            # pylint: disable=duplicate-code
            generation_config = GenerationConfig(
                max_new_tokens=self.cfg.eval_max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )

            def logits_to_tokens(logits) -> torch.Tensor:
                probabilities = torch.softmax(logits, dim=-1)
                # Get the predicted token ids (the ones with the highest probability)
                predicted_token_ids = torch.argmax(probabilities, dim=-1)
                return predicted_token_ids

            def find_ranges(lst):
                ranges = []
                start = 0
                for i in range(1, len(lst)):
                    if lst[i] == 0:
                        ranges.append((start, i - 1))
                        start = i
                end = len(lst) - 1
                ranges.append((start, end))
                return ranges

            def log_table_from_dataloader(name: str, table_dataloader):
                table_data: Dict[str, List[Any]] = {
                    "id": [],
                    "Prompt": [],
                    "Correct Completion": [],
                    "Predicted Completion (model.generate)": [],
                    "Predicted Completion (trainer.prediction_step)": [],
                }
                row_index = 0

                for batch in tqdm(table_dataloader):
                    if row_index > eval_table_size:
                        break

                    batch_labels = batch["labels"].to(device)
                    batch_input_ids = batch["input_ids"].to(device)

                    if "position_ids" in batch:
                        batch_pos_ids = batch["position_ids"].tolist()
                    else:
                        batch_pos_ids = [None] * len(batch["input_ids"])

                    (_, batch_logits, _) = trainer.prediction_step(
                        trainer.model,
                        batch,
                        prediction_loss_only=False,
                    )

                    prompt_token_ids_list = []
                    pred_step_token_ids_list = []
                    completion_token_ids_list = []

                    for input_ids_all, labels_all, pos_ids, logits in zip(
                        batch_input_ids,
                        batch_labels,
                        batch_pos_ids,
                        batch_logits,
                    ):
                        if pos_ids is None:
                            pos_ranges = [(0, len(input_ids_all) - 1)]
                        else:
                            pos_ranges = find_ranges(pos_ids)

                        for pos_range in pos_ranges:
                            start, end = pos_range
                            if start == end:
                                continue

                            input_ids = input_ids_all[start : end + 1]
                            labels = labels_all[start : end + 1]

                            tokens_without_loss = labels == IGNORE_INDEX
                            tokens_with_loss = labels != IGNORE_INDEX
                            tokens_exclude_padding = input_ids != tokenizer.pad_token_id
                            prompt_token_includes = (
                                tokens_without_loss & tokens_exclude_padding
                            )

                            prompt_token_ids = input_ids[prompt_token_includes]
                            prompt_token_ids_list.append(prompt_token_ids)

                            completion_token_ids = input_ids[tokens_with_loss]
                            completion_token_ids_list.append(completion_token_ids)

                            pred_step_token_ids = logits_to_tokens(
                                logits[start : end + 1]
                            )[tokens_with_loss]
                            pred_step_token_ids_list.append(pred_step_token_ids)

                    prompt_texts = tokenizer.batch_decode(
                        prompt_token_ids_list, skip_special_tokens=True
                    )
                    completion_texts = tokenizer.batch_decode(
                        completion_token_ids_list, skip_special_tokens=True
                    )
                    pred_step_texts = tokenizer.batch_decode(
                        pred_step_token_ids_list, skip_special_tokens=True
                    )

                    with torch.no_grad():
                        prompt_encoding = tokenizer(
                            prompt_texts, padding=True, return_tensors="pt"
                        ).to(self.cfg.device)
                        predictions = trainer.model.generate(
                            **prompt_encoding, generation_config=generation_config
                        )

                    prediction_all_tokens = predictions["sequences"].cpu().tolist()
                    prediction_without_prompt_tokens_list = []
                    for prompt_token_ids, prediction_tokens in zip(
                        prompt_token_ids_list, prediction_all_tokens
                    ):
                        prediction_without_prompt_tokens = prediction_tokens[
                            len(prompt_token_ids) :
                        ]
                        prediction_without_prompt_tokens_list.append(
                            prediction_without_prompt_tokens
                        )

                    predicted_texts = tokenizer.batch_decode(
                        prediction_without_prompt_tokens_list, skip_special_tokens=True
                    )

                    for (
                        prompt_text,
                        completion_text,
                        prediction_text,
                        pred_step_text,
                    ) in zip(
                        prompt_texts, completion_texts, predicted_texts, pred_step_texts
                    ):
                        table_data["id"].append(row_index)
                        table_data["Prompt"].append(prompt_text)
                        table_data["Correct Completion"].append(completion_text)
                        table_data["Predicted Completion (model.generate)"].append(
                            prediction_text
                        )
                        table_data[
                            "Predicted Completion (trainer.prediction_step)"
                        ].append(pred_step_text)
                        row_index += 1
                if logger == "wandb":
                    wandb.run.log({f"{name} - Predictions vs Ground Truth": pd.DataFrame(table_data)})  # type: ignore[attr-defined]
                elif logger == "mlflow" and is_mlflow_available():
                    import mlflow

                    tracking_uri = AxolotlInputConfig(
                        **self.cfg.to_dict()
                    ).mlflow_tracking_uri
                    mlflow.log_table(
                        data=table_data,
                        artifact_file="PredictionsVsGroundTruth.json",
                        tracking_uri=tracking_uri,
                    )
                elif logger == "comet_ml" and is_comet_available():
                    import comet_ml

                    experiment = comet_ml.get_running_experiment()
                    if experiment:
                        experiment.log_table(
                            f"{name} - Predictions vs Ground Truth.csv",
                            pd.DataFrame(table_data),
                        )

            if is_main_process():
                log_table_from_dataloader("Eval", eval_dataloader)

            return control

    return LogPredictionCallback


class SaveAxolotlConfigtoWandBCallback(TrainerCallback):
    """Callback to save axolotl config to wandb"""

    def __init__(self, axolotl_config_path):
        self.axolotl_config_path = axolotl_config_path

    def on_train_begin(
        self,
        args: AxolotlTrainingArguments,  # pylint: disable=unused-argument
        state: TrainerState,  # pylint: disable=unused-argument
        control: TrainerControl,
        **kwargs,  # pylint: disable=unused-argument
    ):
        if is_main_process():
            try:
                # sync config to top level in run, cannot delete file right away because wandb schedules it to be synced even w/policy = 'now', so let OS delete it later.
                with NamedTemporaryFile(
                    mode="w", delete=False, suffix=".yml", prefix="axolotl_config_"
                ) as temp_file:
                    copyfile(self.axolotl_config_path, temp_file.name)
                    artifact = wandb.Artifact(
                        f"config-{wandb.run.id}", type="axolotl-config"
                    )
                    artifact.add_file(temp_file.name)
                    wandb.log_artifact(artifact)
                    wandb.save(temp_file.name)
                LOG.info(
                    "The Axolotl config has been saved to the WandB run under files."
                )
            except (FileNotFoundError, ConnectionError) as err:
                LOG.warning(f"Error while saving Axolotl config to WandB: {err}")
        return control


class GCCallback(TrainerCallback):
    """Callback to garbage collect torch cache"""

    def __init__(self, gc_steps=None):
        self.gc_steps = gc_steps

    def on_step_end(
        self, args, state, control, **kwargs  # pylint: disable=unused-argument
    ):
        if self.gc_steps > 0 and state.global_step % self.gc_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def on_epoch_end(
        self, args, state, control, **kwargs  # pylint: disable=unused-argument
    ):
        torch.cuda.empty_cache()
        gc.collect()
