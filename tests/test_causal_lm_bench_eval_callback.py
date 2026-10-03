"""unit tests for the causal LM generation eval callback"""

import torch
from accelerate import Accelerator
from pytest import fixture
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from axolotl.utils.callbacks import causal_lm_bench_eval_callback_factory

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

LONG_PROMPT = (
    "Question: What is the capital city of France, and why did it become "
    "the capital?\nAnswer:"
)
SHORT_PROMPT = "Hi."
COMPLETION = " Paris."


@fixture()
def tokenizer():
    tokenizer_ = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer_.add_special_tokens({"pad_token": "<|endoftext|>"})
    # generation pads on the left, which is what makes the prompt prefix in the
    # generated rows wider than the individual prompts
    tokenizer_.padding_side = "left"
    return tokenizer_


@fixture()
def model():
    model_ = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="float32")
    model_.eval()
    return model_


class StubTrainer:
    """Minimal stand-in for the pieces of Trainer the callback touches."""

    def __init__(self, model):
        self.model_wrapped = model
        self.accelerator = Accelerator()
        self.logged: dict = {}

    def log(self, logs):
        self.logged.update(logs)


class CapturingMetric:
    """Stands in for an ``evaluate`` metric so the predictions are observable."""

    name = "capture"

    def __init__(self):
        self.predictions: list[str] | None = None
        self.references: list[str] | None = None

    def _feature_names(self):
        return ["predictions", "references", "sources"]

    def compute(self, predictions=None, references=None, sources=None):
        self.predictions = predictions
        self.references = references
        return {"score": 0.0}


class Cfg(dict):
    """Attribute access over a plain dict, as the callback reads cfg.<name>."""

    __getattr__ = dict.get


def _eval_batch(tokenizer, prompts, completion):
    """One right-padded eval batch, prompt tokens masked out of the labels."""
    rows = [
        (
            tokenizer(prompt, add_special_tokens=False)["input_ids"],
            tokenizer(completion, add_special_tokens=False)["input_ids"],
        )
        for prompt in prompts
    ]
    width = max(
        len(prompt_ids) + len(completion_ids) for prompt_ids, completion_ids in rows
    )

    input_ids = []
    labels = []
    for prompt_ids, completion_ids in rows:
        padding = width - len(prompt_ids) - len(completion_ids)
        input_ids.append(
            prompt_ids + completion_ids + [tokenizer.pad_token_id] * padding
        )
        labels.append([-100] * len(prompt_ids) + completion_ids + [-100] * padding)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
    }


def _run_callback(model, tokenizer, prompts):
    metric = CapturingMetric()
    callback_cls = causal_lm_bench_eval_callback_factory(StubTrainer(model), tokenizer)
    callback = callback_cls(
        Cfg(device="cpu", eval_max_new_tokens=8, eval_causal_lm_metrics=[])
    )
    callback.metrics = {"capture": metric}
    callback.on_evaluate(
        None, None, "control", None, [_eval_batch(tokenizer, prompts, COMPLETION)]
    )
    return metric


def test_prompt_is_not_prepended_to_the_prediction(model, tokenizer):
    """A short prompt in a padded batch must not leak into its own prediction.

    ``generate`` returns rows that begin with the padded prompt, so dropping only
    as many tokens as the unpadded prompt has leaves the prompt itself at the
    front of the prediction that is then scored.
    """
    metric = _run_callback(model, tokenizer, [LONG_PROMPT, SHORT_PROMPT])

    assert metric.predictions is not None
    assert len(metric.predictions) == 2
    for prompt, prediction in zip(
        [LONG_PROMPT, SHORT_PROMPT], metric.predictions, strict=True
    ):
        assert prompt not in prediction


def test_single_prompt_batch_is_unaffected(model, tokenizer):
    """With one prompt there is no padding, so nothing about this path changes."""
    metric = _run_callback(model, tokenizer, [SHORT_PROMPT])

    assert metric.predictions is not None
    assert len(metric.predictions) == 1
    assert SHORT_PROMPT not in metric.predictions[0]
