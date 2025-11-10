"""Callbacks for Trainer class"""

from __future__ import annotations

import gc
import json
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
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import (
    SaveStrategy,
)
from trl.models import unwrap_model_for_generation

from axolotl.utils import is_comet_available, is_mlflow_available
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
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.config import AxolotlInputConfig

if TYPE_CHECKING:
    from axolotl.core.training_args import AxolotlTrainingArguments


IGNORE_INDEX = -100
LOG = get_logger(__name__)


class LossWatchDogCallback(TrainerCallback):
    """Callback to track loss and stop training if loss is too high"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.violations = 0
        self.threshold = cfg.loss_watchdog_threshold
        self.patience = cfg.loss_watchdog_patience or 3

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **_kwargs,
    ) -> TrainerControl:
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


class SaveModelOnFirstStepCallback(TrainerCallback):
    """Callback to save the model on the first step of training if enabled"""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **_kwargs,
    ) -> TrainerControl:
        if state.global_step == 1:
            control.should_save = True
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
            state: TrainerState,
            control: TrainerControl,
            metrics: Dict[str, float],
            **kwargs,
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
            for s, p, r in zip(bench_name, preds, refs, strict=False):
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
                for bench_name in combined_bench_names:
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
                    except Exception as exc:
                        LOG.warning(f"{metric}: {exc.args}")
            return metrics

        def on_evaluate(
            self,
            args: AxolotlTrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            train_dataloader,
            eval_dataloader,
            **kwargs,
        ):
            trainer.model_wrapped.eval()

            device = torch.device(
                self.cfg.device
            )  # Use this instead of trainer.model_wrapped.device as it may return cpu if fsdp offloaded

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
                        k: kwargs[k] for k in metric._feature_names() if k in kwargs
                    }

                    if isinstance(metric, Perplexity):
                        metric_kwargs["model"] = trainer.model_wrapped

                    metric_score = metric.compute(**metric_kwargs)
                    return (
                        metric_score["score"]
                        if "score" in metric_score
                        else metric_score["mean_score"]
                    )
                except Exception:
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
                            strict=False,
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
                            prompt_token_ids_list, prediction_all_tokens, strict=False
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
            args: AxolotlTrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            train_dataloader,
            eval_dataloader,
            **kwargs,
        ):
            eval_table_size = self.cfg.eval_table_size

            if eval_table_size <= 0:
                return control

            trainer.model.eval()
            device = torch.device(self.cfg.device)

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
                        strict=False,
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
                        prompt_token_ids_list, prediction_all_tokens, strict=False
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
                        prompt_texts,
                        completion_texts,
                        predicted_texts,
                        pred_step_texts,
                        strict=False,
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
                    # type: ignore[attr-defined]
                    wandb.run.log(
                        {
                            f"{name} - Predictions vs Ground Truth": pd.DataFrame(
                                table_data
                            )
                        }
                    )
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
        args: AxolotlTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
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

            try:
                with open(self.axolotl_config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

                chat_tpl = cfg.get("chat_template_jinja")
                if chat_tpl:
                    with NamedTemporaryFile(
                        mode="w", delete=True, suffix=".jinja", prefix="chat_template_"
                    ) as temp_ct_file:
                        if (
                            isinstance(chat_tpl, str)
                            and os.path.exists(chat_tpl)
                            and os.path.isfile(chat_tpl)
                        ):
                            copyfile(chat_tpl, temp_ct_file.name)
                        else:
                            temp_ct_file.write(str(chat_tpl))
                            temp_ct_file.flush()

                        artifact = wandb.Artifact(
                            f"chat-template-{wandb.run.id}", type="jinja-template"
                        )
                        artifact.add_file(temp_ct_file.name)
                        wandb.log_artifact(artifact)
                        wandb.save(temp_ct_file.name)
                        LOG.info(
                            "The chat_template_jinja has been saved to the WandB run under files."
                        )
            except (FileNotFoundError, ConnectionError, yaml.YAMLError) as err:
                LOG.warning(f"Error while saving chat_template_jinja to WandB: {err}")

            if args.deepspeed:
                try:
                    # sync config to top level in run, cannot delete file right away because wandb schedules it to be synced even w/policy = 'now', so let OS delete it later.
                    with NamedTemporaryFile(
                        mode="w",
                        delete=False,
                        suffix=".json",
                        prefix="deepspeed_config_",
                    ) as temp_file:
                        skip_upload = False
                        if isinstance(args.deepspeed, dict):
                            json.dump(args.deepspeed, temp_file, indent=4)
                        elif isinstance(args.deepspeed, str) and os.path.exists(
                            args.deepspeed
                        ):
                            copyfile(args.deepspeed, temp_file.name)
                        else:
                            skip_upload = True
                        if not skip_upload:
                            artifact = wandb.Artifact(
                                f"deepspeed-config-{wandb.run.id}",
                                type="deepspeed-config",
                            )
                            artifact.add_file(temp_file.name)
                            wandb.log_artifact(artifact)
                            wandb.save(temp_file.name)
                            LOG.info(
                                "The DeepSpeed config has been saved to the WandB run under files."
                            )
                except (FileNotFoundError, ConnectionError) as err:
                    LOG.warning(f"Error while saving DeepSpeed config to WandB: {err}")

        return control


class GCCallback(TrainerCallback):
    """Callback to garbage collect torch cache"""

    def __init__(self, gc_steps: int | None = -1):
        self.gc_steps: int = gc_steps or -1
        self.next_gc_on_begin_step: int = -1

    def _gc(self):
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_begin(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        self._gc()

    def on_step_begin(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if self.next_gc_on_begin_step == state.global_step or state.global_step == 0:
            self._gc()

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if control.should_evaluate:
            # automatically GC before evals so the eval memory spike from the CEL doesn't OOM the trainer
            self._gc()
            # also GC on the start of the next step after the eval
            self.next_gc_on_begin_step = state.global_step + 1
        elif self.gc_steps > 0 and state.global_step % self.gc_steps == 0:
            self._gc()
        elif (
            args.save_strategy == SaveStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            # gc on save steps in case anything is loaded to CPU RAM like offloaded tensors
            self._gc()
        elif state.global_step >= state.max_steps:
            if args.save_strategy == SaveStrategy.STEPS:
                # gc on save steps in case anything is loaded to CPU RAM like offloaded tensors
                self._gc()

    def on_epoch_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        self._gc()


def colab_inference_post_train_callback(trainer: Trainer):
    class ColabCallback(TrainerCallback):
        """Callback to prep model for inference on Google Colab"""

        def __init__(self, cfg):
            self.gpu_name = torch.cuda.get_device_name(0)
            self.cfg = cfg

        def on_train_end(self, args, state, control, **kwargs):
            """
            handle T4 gpu, we need to convert attention to eager for inference
            """
            if "Tesla T4" in self.gpu_name and self.cfg.xformers_attention:
                trainer.model.config._attn_implementation = "eager"
            trainer.model.gradient_checkpointing_disable()
            trainer.model.config.use_cache = True
            trainer.model.eval()

    return ColabCallback
