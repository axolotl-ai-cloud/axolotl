"""
This module contains unit tests for the `freeze_layers_except` function.

The `freeze_layers_except` function is used to freeze layers in a model, except for the specified layers.
The unit tests in this module verify the behavior of the `freeze_layers_except` function in different scenarios.
"""

import unittest

import torch
from torch import nn

from axolotl.utils.freeze import freeze_layers_except

ZERO = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ONE_TO_TEN = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


class TestFreezeLayersExcept(unittest.TestCase):
    """
    A test case class for the `freeze_layers_except` function.
    """

    def setUp(self):
        self.model = _TestModel()

    def test_freeze_layers_with_dots_in_name(self):
        freeze_layers_except(self.model, ["features.layer"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

    def test_freeze_layers_without_dots_in_name(self):
        freeze_layers_except(self.model, ["classifier"])
        self.assertFalse(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertTrue(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

    def test_freeze_layers_regex_patterns(self):
        # The second pattern cannot match because only characters 'a' to 'c' are allowed after the word 'class', whereas it should be matching the character 'i'.
        freeze_layers_except(self.model, [r"^features.[a-z]+.weight$", r"class[a-c]+"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

    def test_all_layers_frozen(self):
        freeze_layers_except(self.model, [])
        self.assertFalse(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be frozen.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

    def test_all_layers_unfrozen(self):
        freeze_layers_except(self.model, ["features.layer", "classifier"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertTrue(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be trainable.",
        )

    def test_freeze_layers_with_range_pattern_start_end(self):
        freeze_layers_except(self.model, ["features.layer[1:5]"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

        self._assert_gradient_output(
            [
                ZERO,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ZERO,
                ZERO,
                ZERO,
                ZERO,
                ZERO,
            ]
        )

    def test_freeze_layers_with_range_pattern_single_index(self):
        freeze_layers_except(self.model, ["features.layer[5]"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

        self._assert_gradient_output(
            [ZERO, ZERO, ZERO, ZERO, ZERO, ONE_TO_TEN, ZERO, ZERO, ZERO, ZERO]
        )

    def test_freeze_layers_with_range_pattern_start_omitted(self):
        freeze_layers_except(self.model, ["features.layer[:5]"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

        self._assert_gradient_output(
            [
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ZERO,
                ZERO,
                ZERO,
                ZERO,
                ZERO,
            ]
        )

    def test_freeze_layers_with_range_pattern_end_omitted(self):
        freeze_layers_except(self.model, ["features.layer[4:]"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

        self._assert_gradient_output(
            [
                ZERO,
                ZERO,
                ZERO,
                ZERO,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
            ]
        )

    def test_freeze_layers_with_range_pattern_merge_included(self):
        freeze_layers_except(self.model, ["features.layer[4:]", "features.layer[5:6]"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

        self._assert_gradient_output(
            [
                ZERO,
                ZERO,
                ZERO,
                ZERO,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
            ]
        )

    def test_freeze_layers_with_range_pattern_merge_intersect(self):
        freeze_layers_except(self.model, ["features.layer[4:7]", "features.layer[6:8]"])
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

        self._assert_gradient_output(
            [
                ZERO,
                ZERO,
                ZERO,
                ZERO,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ONE_TO_TEN,
                ZERO,
                ZERO,
            ]
        )

    def test_freeze_layers_with_range_pattern_merge_separate(self):
        freeze_layers_except(
            self.model,
            ["features.layer[1:2]", "features.layer[3:4]", "features.layer[5:6]"],
        )
        self.assertTrue(
            self.model.features.layer.weight.requires_grad,
            "model.features.layer should be trainable.",
        )
        self.assertFalse(
            self.model.classifier.weight.requires_grad,
            "model.classifier should be frozen.",
        )

        self._assert_gradient_output(
            [
                ZERO,
                ONE_TO_TEN,
                ZERO,
                ONE_TO_TEN,
                ZERO,
                ONE_TO_TEN,
                ZERO,
                ZERO,
                ZERO,
                ZERO,
            ]
        )

    def _assert_gradient_output(self, expected):
        input_tensor = torch.tensor([ONE_TO_TEN], dtype=torch.float32)

        self.model.features.layer.weight.grad = None  # Reset gradients
        output = self.model.features.layer(input_tensor)
        loss = output.sum()
        loss.backward()

        expected_grads = torch.tensor(expected)
        torch.testing.assert_close(
            self.model.features.layer.weight.grad, expected_grads
        )


class _SubLayerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)


class _TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = _SubLayerModule()
        self.classifier = nn.Linear(10, 2)


if __name__ == "__main__":
    unittest.main()
