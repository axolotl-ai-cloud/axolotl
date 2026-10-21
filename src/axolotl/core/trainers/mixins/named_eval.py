"""Evaluation over named groups with a single combined callback event."""


class _EvaluationMetrics(dict):
    def __missing__(self, key):
        # Transformers prepends eval_ to metric_for_best_model, even for eval/name/loss.
        if isinstance(key, str) and key.startswith("eval_eval/"):
            return self[key[len("eval_") :]]
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class _EvaluationCallbackHandler:
    def __init__(self, handler):
        self.handler = handler
        self.metric_prefix = None
        self.group_name = None

    def __getattr__(self, name):
        return getattr(self.handler, name)

    @property
    def eval_dataloader(self):
        return self.handler.eval_dataloader

    @eval_dataloader.setter
    def eval_dataloader(self, dataloader):
        self.handler.eval_dataloader = dataloader

    def on_evaluate(self, args, state, control, metrics):
        return control

    def _prefix_logs(self, logs):
        if not self.group_name:
            return
        # Reporting integrations rewrite eval_ to eval/.
        for key in list(logs):
            for prefix in (self.metric_prefix + "_", "eval_"):
                if key.startswith(prefix) and not key.startswith(
                    self.metric_prefix + "/"
                ):
                    logs[self.metric_prefix + "/" + key[len(prefix) :]] = logs.pop(key)
                    break

    def on_log(self, args, state, control, logs):
        self._prefix_logs(logs)
        if state.log_history:
            self._prefix_logs(state.log_history[-1])
        return self.handler.on_log(args, state, control, logs)


class NamedEvaluationMixin:
    """Keep aggregate metric names and notify callbacks with all group metrics."""

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and isinstance(self.eval_dataset, dict):
            eval_dataset = ""
        return super().get_eval_dataloader(eval_dataset)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        datasets = self.eval_dataset if eval_dataset is None else eval_dataset
        if not isinstance(datasets, dict) or "" not in datasets:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        metrics = _EvaluationMetrics()
        handler = self.callback_handler
        aggregate_dataloader = handler.eval_dataloader
        self.callback_handler = _EvaluationCallbackHandler(handler)
        try:
            for name, dataset in datasets.items():
                if len(dataset) == 0:
                    raise ValueError(
                        f"Evaluation dataset {name or 'aggregate'} is empty"
                    )
                prefix = f"{metric_key_prefix}_{name}" if name else metric_key_prefix
                self._eval_metric_key_prefix = prefix
                self.callback_handler.metric_prefix = prefix
                self.callback_handler.group_name = name
                group_metrics = super().evaluate(
                    eval_dataset=name if datasets is self.eval_dataset else dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=prefix,
                )
                if not name:
                    aggregate_dataloader = handler.eval_dataloader
                for key, value in group_metrics.items():
                    if name and (
                        key.startswith(prefix + "_") or key.startswith(prefix + "/")
                    ):
                        key = f"{metric_key_prefix}/{name}/{key[len(prefix) + 1 :]}"
                    metrics[key] = value
        finally:
            self.callback_handler = handler
            handler.eval_dataloader = aggregate_dataloader
            self._eval_metric_key_prefix = None
        self.control = handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


def named_eval_trainer_class(trainer_cls, eval_dataset):
    """Add grouped evaluation only when the loader supplied named groups."""
    if not isinstance(eval_dataset, dict) or "" not in eval_dataset:
        return trainer_cls
    return type(
        f"NamedEvaluation{trainer_cls.__name__}",
        (NamedEvaluationMixin, trainer_cls),
        {"__module__": trainer_cls.__module__},
    )
