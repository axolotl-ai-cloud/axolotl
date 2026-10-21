"""Exercise grouped evaluation with an actual CPU Trainer."""

import pytest
import torch
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from axolotl.core.trainers.mixins.named_eval import named_eval_trainer_class


class LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels=None):
        return {
            "loss": input_ids.float().mean() * self.weight,
            "logits": input_ids.float(),
        }


class CaptureEvaluation(TrainerCallback):
    def __init__(self):
        self.metrics = []
        self.dataset_sizes = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics.append(metrics.copy())
        self.dataset_sizes.append(len(kwargs["eval_dataloader"].dataset))


def make_trainer(tmp_path, persistent=False):
    aggregate = Dataset.from_dict(
        {"input_ids": [[1], [3], [8], [8]], "labels": [[0]] * 4}
    )
    datasets = {
        "": aggregate,
        "math": aggregate.select([0, 1]),
        "code": aggregate.select([2]),
    }
    callback = CaptureEvaluation()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        disable_tqdm=True,
        per_device_eval_batch_size=2,
        dataloader_pin_memory=False,
        dataloader_num_workers=1 if persistent else 0,
        dataloader_persistent_workers=persistent,
        metric_for_best_model="eval/math/loss",
    )
    cls = named_eval_trainer_class(Trainer, datasets)
    trainer = cls(
        model=LossModel(), args=args, eval_dataset=datasets, callbacks=[callback]
    )
    return trainer, callback


def test_losses_and_single_callback(tmp_path):
    trainer, callback = make_trainer(tmp_path)
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] == 5
    assert metrics["eval/math/loss"] == 2
    assert metrics["eval/code/loss"] == 8
    assert callback.metrics == [metrics]
    assert callback.dataset_sizes == [4]
    assert len(trainer.get_eval_dataloader().dataset) == 4


def test_explicit_mapping_and_prefix(tmp_path):
    trainer, callback = make_trainer(tmp_path)
    metrics = trainer.evaluate(trainer.eval_dataset, metric_key_prefix="validation")
    assert metrics["validation_loss"] == 5
    assert metrics["validation/math/loss"] == 2
    assert len(callback.metrics) == 1
    metrics = trainer.evaluate(
        trainer.eval_dataset["math"], metric_key_prefix="separate"
    )
    assert metrics["separate_loss"] == 2
    assert "separate_math_loss" not in metrics


def test_early_stopping_sees_selected_group_once(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    stopping = EarlyStoppingCallback(early_stopping_patience=2)
    trainer.add_callback(stopping)
    trainer.state.best_metric = 1
    trainer.evaluate()
    assert stopping.early_stopping_patience_counter == 1
    assert not trainer.control.should_training_stop
    trainer.evaluate()
    assert stopping.early_stopping_patience_counter == 2
    assert trainer.control.should_training_stop


def test_failure_restores_callback_handler(tmp_path):
    trainer, callback = make_trainer(tmp_path)
    handler = trainer.callback_handler
    trainer.eval_dataset["math"] = trainer.eval_dataset["math"].select([])
    with pytest.raises(ValueError, match="math is empty"):
        trainer.evaluate()
    assert trainer.callback_handler is handler
    assert trainer._eval_metric_key_prefix is None
    assert callback.metrics == []


def test_no_names_keeps_trainer_class():
    assert named_eval_trainer_class(Trainer, Dataset.from_dict({"x": [1]})) is Trainer


def test_best_checkpoint_uses_slash_metric(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    trainer.args.greater_is_better = False
    metrics = trainer.evaluate()
    assert trainer._determine_best_metric(metrics, trial=None)
    assert trainer.state.best_metric == 2
    assert not trainer._determine_best_metric(metrics, trial=None)
    assert "eval_eval/math/loss" not in list(metrics)


def test_reporting_integration_metric_paths(tmp_path):
    from transformers.integrations.integration_utils import rewrite_logs

    trainer, _ = make_trainer(tmp_path)
    trainer.evaluate()
    logged = {}
    for entry in trainer.state.log_history:
        logged.update(rewrite_logs(entry))
    assert logged["eval/math/loss"] == 2
    assert logged["eval/code/loss"] == 8
    assert logged["eval/loss"] == 5


def test_axolotl_perplexity_for_each_group(tmp_path):
    import math

    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.training_args import AxolotlTrainingArguments

    trainer, _ = make_trainer(tmp_path)
    args = AxolotlTrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        disable_tqdm=True,
        per_device_eval_batch_size=2,
        dataloader_pin_memory=False,
    )
    cls = named_eval_trainer_class(AxolotlTrainer, trainer.eval_dataset)
    axo_trainer = cls(
        model=LossModel(),
        args=args,
        eval_dataset=trainer.eval_dataset,
        eval_data_collator=trainer.data_collator,
    )
    metrics = axo_trainer.evaluate()
    assert metrics["eval/math/ppl"] == pytest.approx(math.exp(2), abs=1e-5)
    assert metrics["eval/code/ppl"] == pytest.approx(math.exp(8), abs=1e-5)
    assert metrics["eval_ppl"] == pytest.approx(math.exp(5), abs=1e-5)


def test_persistent_workers_do_not_reuse_another_groups_loader(tmp_path):
    trainer, _ = make_trainer(tmp_path, persistent=True)
    first = trainer.evaluate()
    second = trainer.evaluate()
    for metrics in (first, second):
        assert metrics["eval_loss"] == 5
        assert metrics["eval/math/loss"] == 2
        assert metrics["eval/code/loss"] == 8


def test_trainer_specific_metrics_are_namespaced(tmp_path):
    from transformers.integrations.integration_utils import rewrite_logs

    class ExtraMetricsTrainer(Trainer):
        def log(self, logs, *args, **kwargs):
            logs["eval_rewards/accuracy"] = 0.75
            return super().log(logs, *args, **kwargs)

    trainer, _ = make_trainer(tmp_path)
    cls = named_eval_trainer_class(ExtraMetricsTrainer, trainer.eval_dataset)
    trainer = cls(
        model=LossModel(), args=trainer.args, eval_dataset=trainer.eval_dataset
    )
    metrics = trainer.evaluate()
    assert metrics["eval/math/rewards/accuracy"] == 0.75
    assert metrics["eval_rewards/accuracy"] == 0.75
    logged = {}
    for entry in trainer.state.log_history:
        logged.update(rewrite_logs(entry))
    assert logged["eval/math/rewards/accuracy"] == 0.75
    assert logged["eval/code/rewards/accuracy"] == 0.75
