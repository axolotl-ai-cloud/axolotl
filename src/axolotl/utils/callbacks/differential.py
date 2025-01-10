"""
Monitor and log differential attention components during training.

This module provides a callback for tracking the behavior of differential attention
mechanisms, including lambda parameters and attention statistics.
"""

from typing import Any

import torch
import wandb
from torch import nn
from transformers import TrainerCallback

from axolotl.utils.distributed import is_main_process


class DifferentialAttentionMonitorCallback(TrainerCallback):
    """
    Callback to monitor differential attention components and lambda parameters.

    This callback tracks attention statistics across all layers and provides detailed
    monitoring for a specified number of layers evenly spaced through the model.
    """

    def __init__(
        self,
        log_every: int = 250,
        num_monitor_layers: int = 3,
        warmup_steps: int | None = None,
    ):
        """
        Initialize the differential attention monitor.

        Args:
            log_every: Number of steps between logging events.
            num_monitor_layers: Number of individual layers to monitor in detail.
            warmup_steps: Optional parameter for negative attention component warmup.
        """
        self.log_every = log_every
        self.num_monitor_layers = num_monitor_layers
        self.warmup_steps = warmup_steps
        self.monitor_layers: list[int] | None = None  # Will be set in on_train_begin

    # pylint: disable=unused-argument
    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: torch.nn.Module,
        **kwargs,
    ) -> None:
        """
        Set up layer monitoring at the start of training.

        Args:
            args: Training arguments.
            state: Training state.
            control: Training control object.
            model: The model being trained.
            **kwargs: Additional arguments passed by the trainer.
        """
        if is_main_process():
            num_layers = len(model.model.layers)
            self.num_monitor_layers = min(self.num_monitor_layers, num_layers)

            stride = (
                (num_layers - 1) / (self.num_monitor_layers - 1)
                if self.num_monitor_layers > 1
                else 0
            )
            self.monitor_layers = [
                round(i * stride) for i in range(self.num_monitor_layers)
            ]
            print(f"Monitoring layers {self.monitor_layers} in detail")

    # pylint: disable=unused-argument
    def on_step_end(
        self, args: Any, state: Any, control: Any, model: torch.nn.Module, **kwargs
    ) -> None:
        """
        Log attention metrics at the end of each step.

        Collects and logs:
            - Lambda parameter norms and values.
            - Attention statistics (mean and std).
            - Both per-layer and aggregate metrics.

        Args:
            args: Training arguments.
            state: Training state.
            control: Training control object.
            model: The model being trained.
            **kwargs: Additional arguments passed by the trainer.
        """
        if not is_main_process() or state.global_step % self.log_every != 0:
            return

        assert self.monitor_layers is not None

        # Aggregate stats across all layers
        all_q1_norms = []
        all_q2_norms = []
        all_k1_norms = []
        all_k2_norms = []
        all_lambda1 = []
        all_lambda2 = []
        all_lambda_full = []

        metrics = {}
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn

            # Collect stats for aggregation
            all_q1_norms.append(attn.lambda_q1.norm().item())
            all_q2_norms.append(attn.lambda_q2.norm().item())
            all_k1_norms.append(attn.lambda_k1.norm().item())
            all_k2_norms.append(attn.lambda_k2.norm().item())

            lambda1 = torch.exp(torch.sum(attn.lambda_q1 * attn.lambda_k1)).item()
            lambda2 = torch.exp(torch.sum(attn.lambda_q2 * attn.lambda_k2)).item()
            all_lambda1.append(lambda1)
            all_lambda2.append(lambda2)
            all_lambda_full.append(attn.lambda_full)

            # Log detailed metrics for monitored layers
            if layer_idx in self.monitor_layers:
                metrics.update(
                    {
                        f"layer_{layer_idx}/lambda_q1_norm": attn.lambda_q1.norm().item(),
                        f"layer_{layer_idx}/lambda_k1_norm": attn.lambda_k1.norm().item(),
                        f"layer_{layer_idx}/lambda_q2_norm": attn.lambda_q2.norm().item(),
                        f"layer_{layer_idx}/lambda_k2_norm": attn.lambda_k2.norm().item(),
                        f"layer_{layer_idx}/lambda1": lambda1,
                        f"layer_{layer_idx}/lambda2": lambda2,
                        f"layer_{layer_idx}/lambda_init": attn.lambda_init.item(),
                        f"layer_{layer_idx}/lambda_full": lambda1
                        - lambda2
                        + attn.lambda_init.item(),
                        f"layer_{layer_idx}/attn1_mean": attn.attn1.mean().item(),
                        f"layer_{layer_idx}/attn2_mean": attn.attn2.mean().item(),
                        f"layer_{layer_idx}/attn1_std": attn.attn1.std().item(),
                        f"layer_{layer_idx}/attn2_std": attn.attn2.std().item(),
                    }
                )

        # Add aggregate metrics
        metrics.update(
            {
                "aggregate/lambda_q1_norm_mean": torch.tensor(all_q1_norms)
                .mean()
                .item(),
                "aggregate/lambda_q1_norm_std": torch.tensor(all_q1_norms).std().item(),
                "aggregate/lambda_q2_norm_mean": torch.tensor(all_q2_norms)
                .mean()
                .item(),
                "aggregate/lambda_q2_norm_std": torch.tensor(all_q2_norms).std().item(),
                "aggregate/lambda_k1_norm_mean": torch.tensor(all_k1_norms)
                .mean()
                .item(),
                "aggregate/lambda_k1_norm_std": torch.tensor(all_k1_norms).std().item(),
                "aggregate/lambda_k2_norm_mean": torch.tensor(all_k2_norms)
                .mean()
                .item(),
                "aggregate/lambda_k2_norm_std": torch.tensor(all_k2_norms).std().item(),
                "aggregate/lambda1_mean": torch.tensor(all_lambda1).mean().item(),
                "aggregate/lambda1_std": torch.tensor(all_lambda1).std().item(),
                "aggregate/lambda2_mean": torch.tensor(all_lambda2).mean().item(),
                "aggregate/lambda2_std": torch.tensor(all_lambda2).std().item(),
                "aggregate/lambda_full_mean": torch.tensor(all_lambda_full)
                .mean()
                .item(),
                "aggregate/lambda_full_std": torch.tensor(all_lambda_full).std().item(),
            }
        )

        if self.warmup_steps:
            metrics["aggregate/diff_attn_mix"] = attn.diff_attn_mix

        wandb.log(metrics, step=state.global_step)


class DifferentialAttentionMixingCallback(TrainerCallback):
    """
    Callback to gradually increase the weight of negative attention components during
    training.
    """

    def __init__(self, warmup_steps: int):
        """
        Args:
            warmup_steps: Number of steps to linearly increase negative attention
                weight from 0 to 1. If `None`, negative attention has full weight from
                start.
        """
        self.warmup_steps = warmup_steps
        self.diff_attention_layers: list[nn.Module] | None = None

    # pylint: disable=unused-argument
    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: torch.nn.Module,
        **kwargs,
    ) -> None:
        """Cache the differential attention layers at the start of training."""
        if model is not None:
            # Get the actual model if it's wrapped
            if hasattr(model, "module"):
                model = model.module

            # Cache all differential attention layers
            self.diff_attention_layers = [
                module for module in model.modules() if hasattr(module, "diff_attn_mix")
            ]

    def on_step_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: torch.nn.Module = None,
        **kwargs,
    ) -> None:
        if self.diff_attention_layers and self.warmup_steps:
            # Calculate mixing parameter (0 to 1)
            mix = min(1.0, state.global_step / self.warmup_steps)

            # Update cached layers
            for layer in self.diff_attention_layers:
                layer.diff_attn_mix = mix
