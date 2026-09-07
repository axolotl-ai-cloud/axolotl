"""Example: Custom Trainer with SwanLab Profiling

This example demonstrates how to add SwanLab profiling to your custom trainer.

Features:
- @swanlab_profile decorator for automatic profiling
- swanlab_profiling_context for fine-grained profiling
- ProfilingConfig for advanced filtering and throttling

Usage:
    1. Create your custom trainer extending AxolotlTrainer
    2. Add @swanlab_profile decorators to methods you want to profile
    3. Use swanlab_profiling_context for fine-grained profiling within methods
    4. Enable SwanLab in your config (use_swanlab: true)

See also:
    - examples/swanlab/lora-swanlab-profiling.yml for config
    - src/axolotl/integrations/swanlab/profiling.py for implementation
"""

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.swanlab.profiling import (
    ProfilingConfig,
    swanlab_profile,
    swanlab_profiling_context,
    swanlab_profiling_context_advanced,
)


class CustomTrainerWithProfiling(AxolotlTrainer):
    """Custom trainer with SwanLab profiling enabled.

    This trainer demonstrates three profiling patterns:
    1. Decorator-based profiling (@swanlab_profile)
    2. Context manager profiling (swanlab_profiling_context)
    3. Advanced profiling with filtering (ProfilingConfig)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create custom profiling config for high-frequency operations
        self.fast_op_config = ProfilingConfig(
            enabled=True,
            min_duration_ms=0.5,  # Only log if duration > 0.5ms
            log_interval=50,  # Log every 50th call
        )

    # ========================================================================
    # Pattern 1: Decorator-based Profiling
    # ========================================================================
    # Best for: Methods you always want to profile
    # Overhead: ~2-5 microseconds per call (negligible)

    @swanlab_profile
    def training_step(self, model, inputs):
        """Main training step - always profile.

        Profiling metric: profiling/Time taken: CustomTrainerWithProfiling.training_step
        """
        return super().training_step(model, inputs)

    @swanlab_profile
    def compute_loss(self, model, inputs, return_outputs=False):
        """Loss computation - always profile.

        Profiling metric: profiling/Time taken: CustomTrainerWithProfiling.compute_loss
        """
        return super().compute_loss(model, inputs, return_outputs)

    @swanlab_profile
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Prediction step - always profile.

        Profiling metric: profiling/Time taken: CustomTrainerWithProfiling.prediction_step
        """
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    # ========================================================================
    # Pattern 2: Fine-grained Context Manager Profiling
    # ========================================================================
    # Best for: Profiling specific code blocks within a method
    # Use case: When you want to profile forward vs backward separately

    def complex_training_step(self, model, inputs):
        """Training step with fine-grained profiling.

        Profiling metrics:
        - profiling/Time taken: CustomTrainerWithProfiling.forward_pass
        - profiling/Time taken: CustomTrainerWithProfiling.backward_pass
        - profiling/Time taken: CustomTrainerWithProfiling.optimizer_step
        """
        # Profile just the forward pass
        with swanlab_profiling_context(self, "forward_pass"):
            outputs = model(**inputs)
            loss = outputs.loss

        # Profile just the backward pass
        with swanlab_profiling_context(self, "backward_pass"):
            loss.backward()

        # Profile optimizer step
        with swanlab_profiling_context(self, "optimizer_step"):
            self.optimizer.step()
            self.optimizer.zero_grad()

        return outputs

    # ========================================================================
    # Pattern 3: Advanced Profiling with Filtering
    # ========================================================================
    # Best for: High-frequency operations where you want to throttle logging
    # Use case: Methods called 100+ times per step

    def _prepare_inputs(self, inputs):
        """Prepare inputs - throttled profiling.

        This method is called frequently (once per batch), so we throttle
        profiling to reduce overhead:
        - Only log if duration > 0.5ms (skip very fast operations)
        - Only log every 50th call (reduce logging frequency)

        Profiling metric: profiling/Time taken: CustomTrainerWithProfiling.prepare_inputs
        """
        with swanlab_profiling_context_advanced(
            self, "prepare_inputs", config=self.fast_op_config
        ):
            return super()._prepare_inputs(inputs)

    def _prepare_input_for_model(self, input_ids):
        """Another high-frequency operation - throttled profiling.

        Profiling metric: profiling/Time taken: CustomTrainerWithProfiling.prepare_input_for_model
        """
        with swanlab_profiling_context_advanced(
            self, "prepare_input_for_model", config=self.fast_op_config
        ):
            # Your custom input preparation logic
            return input_ids

    # ========================================================================
    # Pattern 4: Exception-safe Profiling
    # ========================================================================
    # Profiling is exception-safe: duration is logged even if method raises

    @swanlab_profile
    def potentially_failing_method(self):
        """This method may raise an exception.

        SwanLab profiling will still log the duration before re-raising.
        Profiling metric: profiling/Time taken: CustomTrainerWithProfiling.potentially_failing_method
        """
        # Do some work
        result = self._do_risky_computation()

        # If this raises, profiling duration is still logged
        if result < 0:
            raise ValueError("Invalid result")

        return result

    def _do_risky_computation(self):
        """Placeholder for risky computation."""
        return 42


# ============================================================================
# Advanced Example: Custom ProfilingConfig Per Method
# ============================================================================


class AdvancedProfilingTrainer(AxolotlTrainer):
    """Trainer with method-specific profiling configurations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Different profiling configs for different method types
        self.critical_path_config = ProfilingConfig(
            enabled=True,
            min_duration_ms=0.0,  # Log everything on critical path
            log_interval=1,  # Log every call
        )

        self.fast_path_config = ProfilingConfig(
            enabled=True,
            min_duration_ms=1.0,  # Only log if > 1ms
            log_interval=100,  # Log every 100th call
        )

        self.debug_config = ProfilingConfig(
            enabled=True,
            min_duration_ms=0.0,  # Log everything
            log_interval=1,  # Log every call
        )

    def training_step(self, model, inputs):
        """Critical path - log everything."""
        with swanlab_profiling_context_advanced(
            self, "training_step", config=self.critical_path_config
        ):
            return super().training_step(model, inputs)

    def _prepare_inputs(self, inputs):
        """Fast path - throttle logging."""
        with swanlab_profiling_context_advanced(
            self, "prepare_inputs", config=self.fast_path_config
        ):
            return super()._prepare_inputs(inputs)

    def _debug_method(self, data):
        """Debug-only method - verbose logging."""
        with swanlab_profiling_context_advanced(
            self, "debug_method", config=self.debug_config
        ):
            # Your debug logic
            pass


# ============================================================================
# How to Use This Custom Trainer
# ============================================================================

"""
To use this custom trainer:

1. Save this file to your project (e.g., my_custom_trainer.py)

2. Create a config file that uses your custom trainer:

    # config.yml
    base_model: NousResearch/Llama-3.2-1B

    # ... other config ...

    plugins:
      - axolotl.integrations.swanlab.SwanLabPlugin

    use_swanlab: true
    swanlab_project: my-profiling-experiment

    # Optional: Specify custom trainer
    # (Or modify axolotl to use your custom trainer class)

3. Run training:

    export SWANLAB_API_KEY=your-api-key
    accelerate launch -m axolotl.cli.train config.yml

4. View profiling metrics in SwanLab dashboard:
   - profiling/Time taken: CustomTrainerWithProfiling.training_step
   - profiling/Time taken: CustomTrainerWithProfiling.forward_pass
   - profiling/Time taken: CustomTrainerWithProfiling.backward_pass
   - etc.

5. Compare profiling metrics across runs:
   - Run baseline without optimizations
   - Run with flash_attention enabled
   - Run with gradient_checkpointing enabled
   - Compare profiling metrics to see performance impact
"""

# ============================================================================
# Tips for Effective Profiling
# ============================================================================

"""
1. Profile the critical path first:
   - training_step, compute_loss, prediction_step
   - These methods are called most frequently and have biggest impact

2. Use throttling for high-frequency operations:
   - Methods called 100+ times per step
   - Use log_interval=50 or log_interval=100
   - Reduces profiling overhead and dashboard clutter

3. Filter noise with min_duration_ms:
   - Set min_duration_ms=1.0 to skip very fast operations
   - Focus on operations that actually take time

4. Compare across runs:
   - Run same config multiple times to check consistency
   - Compare different optimization strategies
   - Track profiling trends over time

5. Monitor distributed training:
   - Check for per-rank timing differences
   - Look for stragglers (slower ranks)
   - Identify synchronization bottlenecks

6. Disable profiling in production:
   - from axolotl.integrations.swanlab.profiling import DEFAULT_PROFILING_CONFIG
   - DEFAULT_PROFILING_CONFIG.enabled = False

7. Exception handling:
   - Profiling is exception-safe
   - Duration logged even if method raises
   - Useful for debugging methods that fail intermittently
"""
