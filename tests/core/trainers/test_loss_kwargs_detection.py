"""Tests for detection of whether a model's loss consumes ``num_items_in_batch``."""

# pylint: disable=missing-class-docstring,too-few-public-methods

import inspect

import pytest
from transformers.loss.loss_utils import (
    ForCausalLMLoss,
    ForSequenceClassificationLoss,
    ForTokenClassification,
)

from axolotl.core.trainers import base as base_module
from axolotl.core.trainers.base import model_loss_accepts_num_items_in_batch


class TestRealTransformersLossSignatures:
    """Pin the upstream signatures the detection depends on.

    If transformers changes which losses take ``num_items_in_batch`` these fail
    loudly, which is the intent: the detection would then be making the wrong
    call for those model types and needs revisiting.
    """

    def test_causal_lm_loss_takes_num_items_in_batch(self):
        params = inspect.signature(ForCausalLMLoss).parameters
        assert "num_items_in_batch" in params

    @pytest.mark.parametrize(
        "loss_fn", [ForSequenceClassificationLoss, ForTokenClassification]
    )
    def test_non_causal_losses_do_not_take_num_items_in_batch(self, loss_fn):
        params = inspect.signature(loss_fn).parameters
        assert "num_items_in_batch" not in params


class TestDetection:
    def test_causal_lm_loss_is_accepted(self):
        class Model:
            loss_function = staticmethod(ForCausalLMLoss)

        assert model_loss_accepts_num_items_in_batch(Model()) is True

    @pytest.mark.parametrize(
        "loss_fn", [ForSequenceClassificationLoss, ForTokenClassification]
    )
    def test_losses_without_the_kwarg_are_forced_off(self, loss_fn):
        """The case the maintainer asked about.

        A sequence- or token-classification model's loss genuinely does not take
        ``num_items_in_batch``, so the flag must be forced False for it.
        """

        class Model:
            loss_function = staticmethod(loss_fn)

        assert model_loss_accepts_num_items_in_batch(Model()) is False

    def test_peft_wrapper_is_unwrapped(self):
        class Inner:
            loss_function = staticmethod(ForSequenceClassificationLoss)

        class PeftLike:
            # A PEFT wrapper's own loss_function would resolve to the causal
            # default; only the wrapped model's loss is authoritative.
            loss_function = staticmethod(ForCausalLMLoss)

            def get_base_model(self):
                return Inner()

        assert model_loss_accepts_num_items_in_batch(PeftLike()) is False

    def test_does_not_descend_through_base_model_property(self):
        """Regression test for the bug this replaced.

        ``base_model`` is a property on every ``PreTrainedModel``, so walking
        ``.base_model`` / ``.model`` generically descended from the causal-LM
        head into the bare backbone, which has no loss at all, and the flag was
        forced False for every model.
        """

        class Backbone:
            """Stands in for e.g. LlamaModel: computes no loss."""

        class CausalLM:
            loss_function = staticmethod(ForCausalLMLoss)

            @property
            def base_model(self):
                return Backbone()

            @property
            def model(self):
                return Backbone()

        assert model_loss_accepts_num_items_in_batch(CausalLM()) is True

    def test_model_without_loss_function_left_alone(self):
        class Model:
            pass

        assert model_loss_accepts_num_items_in_batch(Model()) is True

    @pytest.mark.parametrize("exc", [ValueError, TypeError])
    def test_uninspectable_loss_left_alone(self, monkeypatch, exc):
        """The fallback for a callable whose signature cannot be retrieved.

        Which builtins expose a signature varies by CPython version, so rather
        than pick one and hope, force the failure the fallback exists for.
        """

        def raises(*_args, **_kwargs):
            raise exc("no signature found")

        monkeypatch.setattr(base_module.inspect, "signature", raises)

        class Model:
            loss_function = staticmethod(ForCausalLMLoss)

        assert model_loss_accepts_num_items_in_batch(Model()) is True
