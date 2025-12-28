# SwanLab Integration for Axolotl

SwanLab is an open-source, lightweight AI experiment tracking and visualization tool that provides a platform for tracking, recording, comparing, and collaborating on experiments.

This integration enables seamless experiment tracking and visualization of Axolotl training runs using SwanLab.

## Features

- 📊 **Automatic Metrics Logging**: Training loss, learning rate, and other metrics are automatically logged
- 🎯 **Hyperparameter Tracking**: Model configuration and training parameters are tracked
- 📈 **Real-time Visualization**: Monitor training progress in real-time through SwanLab dashboard
- ☁️ **Cloud & Local Support**: Works in both cloud-synced and offline modes
- 🔄 **Experiment Comparison**: Compare multiple training runs easily
- 🤝 **Team Collaboration**: Share experiments with team members
- 🎭 **RLHF Completion Logging**: Automatically log model outputs during DPO/KTO/ORPO/GRPO training for qualitative analysis
- ⚡ **Performance Profiling**: Built-in profiling decorators to measure and optimize training performance
- 🔔 **Lark Notifications**: Send real-time training updates to team chat (Feishu/Lark integration)

## Installation

```bash
pip install swanlab
```

## Quick Start

### 1. Register for SwanLab (Optional for cloud mode)

If you want to use cloud sync features, register at [https://swanlab.cn](https://swanlab.cn) to get your API key.

### 2. Configure Axolotl Config File

Add SwanLab configuration to your Axolotl YAML config:

```yaml
# Enable SwanLab plugin
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

# SwanLab configuration
use_swanlab: true
swanlab_project: my-llm-project
swanlab_experiment_name: qwen-finetune-v1
swanlab_mode: cloud  # Options: cloud, local, offline, disabled
swanlab_workspace: my-team  # Optional: organization name
swanlab_api_key: YOUR_API_KEY  # Optional: can also use env var SWANLAB_API_KEY
```

### 3. Run Training

```bash
# Set API key via environment variable (recommended)
export SWANLAB_API_KEY=your-api-key-here

# Or login once
swanlab login

# Run training as usual
accelerate launch -m axolotl.cli.train your-config.yaml
```

## Configuration Options

### Basic Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_swanlab` | bool | `false` | Enable SwanLab tracking |
| `swanlab_project` | str | `None` | Project name (required) |
| `swanlab_experiment_name` | str | `None` | Experiment name |
| `swanlab_description` | str | `None` | Experiment description |
| `swanlab_mode` | str | `cloud` | Sync mode: `cloud`, `local`, `offline`, `disabled` |

### Advanced Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `swanlab_workspace` | str | `None` | Workspace/organization name |
| `swanlab_api_key` | str | `None` | API key (prefer env var) |
| `swanlab_web_host` | str | `None` | Private deployment web host |
| `swanlab_api_host` | str | `None` | Private deployment API host |
| `swanlab_log_model` | bool | `false` | Log model checkpoints (coming soon) |
| `swanlab_lark_webhook_url` | str | `None` | Lark (Feishu) webhook URL for team notifications |
| `swanlab_lark_secret` | str | `None` | Lark webhook HMAC secret for authentication |
| `swanlab_log_completions` | bool | `true` | Enable RLHF completion table logging (DPO/KTO/ORPO/GRPO) |
| `swanlab_completion_log_interval` | int | `100` | Steps between completion logging |
| `swanlab_completion_max_buffer` | int | `128` | Max completions to buffer (memory bound) |

## Configuration Examples

### Example 1: Basic Cloud Sync

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: llama-finetune
swanlab_experiment_name: llama-3-8b-instruct-v1
swanlab_mode: cloud
```

### Example 2: Offline/Local Mode

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: local-experiments
swanlab_experiment_name: test-run-1
swanlab_mode: local  # or 'offline'
```

### Example 3: Team Workspace

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: research-project
swanlab_experiment_name: experiment-42
swanlab_workspace: my-research-team
swanlab_mode: cloud
```

### Example 4: Private Deployment

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: internal-project
swanlab_experiment_name: secure-training
swanlab_mode: cloud
swanlab_web_host: https://swanlab.yourcompany.com
swanlab_api_host: https://api.swanlab.yourcompany.com
```

## Team Notifications with Lark (Feishu)

SwanLab supports sending real-time training notifications to your team chat via Lark (Feishu), ByteDance's enterprise collaboration platform. This is especially useful for:
- **Production training monitoring**: Get alerts when training starts, completes, or encounters errors
- **Team collaboration**: Keep your ML team informed about long-running experiments
- **Multi-timezone teams**: Team members can check training progress without being online

### Prerequisites

1. **Lark Bot Setup**: Create a custom bot in your Lark group chat
2. **Webhook URL**: Get the webhook URL from your Lark bot settings
3. **HMAC Secret** (recommended): Enable signature verification in your Lark bot for security

For detailed Lark bot setup instructions, see [Lark Custom Bot Documentation](https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN).

### Example 5: Basic Lark Notifications

Send training notifications to a Lark group chat:

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: production-training
swanlab_experiment_name: llama-3-finetune-v2
swanlab_mode: cloud

# Lark notification (basic, no HMAC verification)
swanlab_lark_webhook_url: https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxxxx
```

**Note**: This configuration will work, but you'll see a security warning recommending HMAC secret configuration.

### Example 6: Lark Notifications with HMAC Security (Recommended)

For production use, enable HMAC signature verification:

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: production-training
swanlab_experiment_name: llama-3-finetune-v2
swanlab_mode: cloud

# Lark notification with HMAC authentication
swanlab_lark_webhook_url: https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxxxx
swanlab_lark_secret: your-webhook-secret-key
```

**Why HMAC secret matters**:
- Prevents unauthorized parties from sending fake notifications to your Lark group
- Ensures notifications genuinely come from your training jobs
- Required for production deployments with sensitive training data

### Example 7: Team Workspace + Lark Notifications

Combine team workspace collaboration with Lark notifications:

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: research-project
swanlab_experiment_name: multimodal-experiment-42
swanlab_workspace: ml-research-team
swanlab_mode: cloud

# Notify team via Lark when training starts/completes
swanlab_lark_webhook_url: https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxxxx
swanlab_lark_secret: your-webhook-secret-key
```

### What Notifications Are Sent?

SwanLab's Lark integration sends notifications for key training events:
- **Training Start**: When your experiment begins
- **Training Complete**: When training finishes successfully
- **Training Errors**: If training crashes or encounters critical errors
- **Metric Milestones**: Configurable alerts for metric thresholds (if configured in SwanLab)

Each notification includes:
- Experiment name and project
- Training status
- Key metrics (loss, learning rate)
- Direct link to SwanLab dashboard

### Lark Configuration Validation

The plugin validates your Lark configuration at startup:

#### ✅ Valid Configurations

```yaml
# Option 1: No Lark (default)
use_swanlab: true
swanlab_project: my-project
# No swanlab_lark_webhook_url → Lark disabled, no warnings

# Option 2: Lark with HMAC secret (recommended)
use_swanlab: true
swanlab_project: my-project
swanlab_lark_webhook_url: https://open.feishu.cn/open-apis/bot/v2/hook/xxx
swanlab_lark_secret: your-secret
# ✅ Logs: "Registered Lark notification callback with HMAC authentication"

# Option 3: Lark without secret (works but not recommended)
use_swanlab: true
swanlab_project: my-project
swanlab_lark_webhook_url: https://open.feishu.cn/open-apis/bot/v2/hook/xxx
# ⚠️ Logs: "Registered Lark notification callback (no HMAC secret)"
# ⚠️ Warning: "Lark webhook has no secret configured. For production use, set 'swanlab_lark_secret'..."
```

### Security Best Practices

1. **Always use HMAC secret in production**:
   ```yaml
   swanlab_lark_webhook_url: https://open.feishu.cn/...
   swanlab_lark_secret: your-secret-key  # ✅ Add this!
   ```

2. **Store secrets in environment variables** (even better):
   ```yaml
   # In your training script/environment
   export SWANLAB_LARK_WEBHOOK_URL="https://open.feishu.cn/..."
   export SWANLAB_LARK_SECRET="your-secret-key"
   ```

   Then in config:
   ```yaml
   # SwanLab plugin will auto-detect environment variables
   use_swanlab: true
   swanlab_project: my-project
   # Lark URL and secret read from env vars
   ```

3. **Rotate webhook secrets periodically**: Update your Lark bot's secret every 90 days

4. **Use separate webhooks for dev/prod**: Don't mix development and production notifications

### Distributed Training

Lark notifications are automatically deduplicated in distributed training:
- Only **rank 0** sends notifications
- Other GPU ranks skip Lark registration
- Prevents duplicate messages in multi-GPU training

```bash
# Running on 4 GPUs
torchrun --nproc_per_node=4 -m axolotl.cli.train config.yml

# Expected logs:
# [Rank 0] Registered Lark notification callback with HMAC authentication
# [Rank 1-3] (no Lark registration messages)
```

## RLHF Completion Table Logging

For RLHF (Reinforcement Learning from Human Feedback) training methods like DPO, KTO, ORPO, and GRPO, SwanLab can log model completions (prompts, chosen/rejected responses, rewards) to a visual table for qualitative analysis. This helps you:

- **Inspect model behavior**: See actual model outputs during training
- **Debug preference learning**: Compare chosen vs rejected responses
- **Track reward patterns**: Monitor how rewards evolve over training
- **Share examples with team**: Visual tables in SwanLab dashboard

### Features

- ✅ **Automatic detection**: Works with DPO, KTO, ORPO, GRPO trainers
- ✅ **Memory-safe buffering**: Bounded buffer prevents memory leaks in long training runs
- ✅ **Periodic logging**: Configurable logging interval to reduce overhead
- ✅ **Rich visualization**: SwanLab tables show prompts, responses, and metrics side-by-side

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `swanlab_log_completions` | bool | `true` | Enable completion logging for RLHF trainers |
| `swanlab_completion_log_interval` | int | `100` | Log completions to SwanLab every N training steps |
| `swanlab_completion_max_buffer` | int | `128` | Maximum completions to buffer (memory bound) |

### Example: DPO Training with Completion Logging

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: dpo-training
swanlab_experiment_name: llama-3-dpo-v1
swanlab_mode: cloud

# RLHF completion logging (enabled by default)
swanlab_log_completions: true
swanlab_completion_log_interval: 100  # Log every 100 steps
swanlab_completion_max_buffer: 128    # Keep last 128 completions

# DPO-specific config
rl: dpo
datasets:
  - path: /path/to/preference_dataset
    type: chatml.intel
```

### Example: Disable Completion Logging

If you're doing a quick test run or don't need completion tables:

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: dpo-training

# Disable completion logging
swanlab_log_completions: false
```

### Supported RLHF Trainers

The completion logging callback automatically activates for these trainer types:

- **DPO (Direct Preference Optimization)**: Logs prompts, chosen, rejected, reward_diff
- **KTO (Kahneman-Tversky Optimization)**: Logs prompts, completions, labels, rewards
- **ORPO (Odds Ratio Preference Optimization)**: Logs prompts, chosen, rejected, log_odds_ratio
- **GRPO (Group Relative Policy Optimization)**: Logs prompts, completions, rewards, advantages
- **CPO (Constrained Policy Optimization)**: Logs prompts, chosen, rejected

For non-RLHF trainers (standard supervised fine-tuning), the completion callback is automatically skipped.

### How It Works

1. **Auto-detection**: Plugin detects trainer type at initialization
2. **Buffering**: Completions are buffered in memory (up to `swanlab_completion_max_buffer`)
3. **Periodic logging**: Every `swanlab_completion_log_interval` steps, buffer is logged to SwanLab
4. **Memory safety**: Old completions are automatically dropped when buffer is full (uses `collections.deque`)
5. **Final flush**: Remaining completions are logged when training completes

### Viewing Completion Tables

After training starts, you can view completion tables in your SwanLab dashboard:

1. Navigate to your experiment in SwanLab
2. Look for the "rlhf_completions" table in the metrics panel
3. The table shows:
   - **step**: Training step when completion was generated
   - **prompt**: Input prompt
   - **chosen**: Preferred response (DPO/ORPO)
   - **rejected**: Non-preferred response (DPO/ORPO)
   - **completion**: Model output (KTO/GRPO)
   - **reward_diff/reward**: Reward metrics
   - Trainer-specific metrics (e.g., log_odds_ratio for ORPO)

### Memory Management

The completion buffer is **memory-bounded** to prevent memory leaks:

```python
# Internal implementation uses deque with maxlen
from collections import deque

buffer = deque(maxlen=128)  # Old completions automatically dropped
```

**Memory usage estimate**:
- Average completion: ~500 characters (prompt + responses)
- Buffer size 128: ~64 KB (negligible)
- Buffer size 1024: ~512 KB (still small)

**Recommendation**: Default buffer size (128) works well for most cases. Increase to 512-1024 only if you need to review more historical completions.

### Performance Impact

Completion logging has minimal overhead:

- **Buffering**: O(1) append operation, negligible CPU/memory
- **Logging**: Only happens every N steps (default: 100)
- **Network**: SwanLab batches table uploads efficiently

**Expected overhead**: < 0.5% per training step

### Troubleshooting

#### Completions not appearing in SwanLab

**Cause**: Trainer may not be logging completion data in the expected format.

**Diagnostic steps**:
1. Check trainer type detection in logs:
   ```
   INFO: SwanLab RLHF completion logging enabled for DPOTrainer (type: dpo)
   ```
2. Verify your trainer is an RLHF trainer (DPO/KTO/ORPO/GRPO)
3. Check if trainer logs completion data (this depends on TRL version)

**Note**: The current implementation expects trainers to log completion data in the `logs` dict during `on_log()` callback. Some TRL trainers may not expose this data by default. You may need to patch the trainer to expose completions.

#### Buffer fills up too quickly

**Cause**: High logging frequency with small buffer size.

**Solution**: Increase buffer size or logging interval:
```yaml
swanlab_completion_log_interval: 200  # Log less frequently
swanlab_completion_max_buffer: 512    # Larger buffer
```

#### Memory usage growing over time

**Cause**: Buffer should be bounded, so this indicates a bug.

**Solution**:
1. Verify `swanlab_completion_max_buffer` is set
2. Check SwanLab version is up to date
3. Report issue with memory profiling data

## Performance Profiling

SwanLab integration includes profiling utilities to measure and log execution time of trainer methods. This helps you:

- **Identify bottlenecks**: Find slow operations in your training loop
- **Optimize performance**: Track improvements after optimization changes
- **Monitor distributed training**: See per-rank timing differences
- **Debug hangs**: Detect methods that take unexpectedly long

### Features

- ✅ **Zero-config profiling**: Automatic timing of key trainer methods
- ✅ **Decorator-based**: Easy to add profiling to custom methods with `@swanlab_profile`
- ✅ **Context manager**: Fine-grained profiling with `swanlab_profiling_context()`
- ✅ **Advanced filtering**: `ProfilingConfig` for throttling and minimum duration thresholds
- ✅ **Exception-safe**: Logs duration even if function raises an exception

### Basic Usage: Decorator

Add profiling to any trainer method with the `@swanlab_profile` decorator:

```python
from axolotl.integrations.swanlab.profiling import swanlab_profile

class MyCustomTrainer(AxolotlTrainer):
    @swanlab_profile
    def training_step(self, model, inputs):
        # Your training step logic
        return super().training_step(model, inputs)

    @swanlab_profile
    def prediction_step(self, model, inputs, prediction_loss_only):
        # Your prediction logic
        return super().prediction_step(model, inputs, prediction_loss_only)
```

The decorator automatically:
1. Measures execution time with high-precision timer
2. Logs to SwanLab as `profiling/Time taken: ClassName.method_name`
3. Only logs if SwanLab is enabled (`use_swanlab: true`)
4. Gracefully handles exceptions (logs duration, then re-raises)

### Advanced Usage: Context Manager

For fine-grained profiling within a method:

```python
from axolotl.integrations.swanlab.profiling import swanlab_profiling_context

class MyTrainer(AxolotlTrainer):
    def complex_training_step(self, model, inputs):
        # Profile just the forward pass
        with swanlab_profiling_context(self, "forward_pass"):
            outputs = model(**inputs)

        # Profile just the backward pass
        with swanlab_profiling_context(self, "backward_pass"):
            loss = outputs.loss
            loss.backward()

        return outputs
```

### Advanced Usage: ProfilingConfig

Filter and throttle profiling logs with `ProfilingConfig`:

```python
from axolotl.integrations.swanlab.profiling import (
    swanlab_profiling_context_advanced,
    ProfilingConfig,
)

# Create custom profiling config
profiling_config = ProfilingConfig(
    enabled=True,
    min_duration_ms=1.0,    # Only log if duration > 1ms
    log_interval=10,        # Log every 10th call
)

class MyTrainer(AxolotlTrainer):
    def frequently_called_method(self, data):
        with swanlab_profiling_context_advanced(
            self,
            "frequent_op",
            config=profiling_config
        ):
            # This only logs every 10th call, and only if it takes > 1ms
            result = expensive_computation(data)
        return result
```

**ProfilingConfig Parameters**:
- `enabled`: Enable/disable profiling globally (default: `True`)
- `min_duration_ms`: Minimum duration to log in milliseconds (default: `0.1`)
- `log_interval`: Log every Nth function call (default: `1` = log all)

**Use cases**:
- **High-frequency methods**: Use `log_interval=100` to reduce logging overhead
- **Filter noise**: Use `min_duration_ms=1.0` to skip very fast operations
- **Debugging**: Use `log_interval=1, min_duration_ms=0.0` to log everything

### Viewing Profiling Metrics

In your SwanLab dashboard, profiling metrics appear under the "profiling" namespace:

```
profiling/Time taken: AxolotlTrainer.training_step
profiling/Time taken: AxolotlTrainer.prediction_step
profiling/Time taken: MyTrainer.forward_pass
profiling/Time taken: MyTrainer.backward_pass
```

You can:
- **Track over time**: See if methods get faster/slower during training
- **Compare runs**: Compare profiling metrics across experiments
- **Identify regressions**: Detect if a code change slowed down training

### Configuration in Axolotl Config

Profiling is automatically enabled when SwanLab is enabled. No additional config needed:

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: my-project

# Profiling is automatically enabled
# Add @swanlab_profile decorators to your custom trainer methods
```

To disable profiling while keeping SwanLab enabled:

```python
# In your custom trainer code
from axolotl.integrations.swanlab.profiling import DEFAULT_PROFILING_CONFIG

# Disable profiling globally
DEFAULT_PROFILING_CONFIG.enabled = False
```

### Performance Impact

- **Decorator overhead**: ~2-5 microseconds per call (negligible)
- **Context manager overhead**: ~1-3 microseconds (negligible)
- **Logging overhead**: Only when SwanLab is enabled and method duration exceeds threshold
- **Network overhead**: SwanLab batches metrics efficiently

**Expected overhead**: < 0.1% per training step (effectively zero)

### Best Practices

1. **Profile bottlenecks first**: Start by profiling suspected slow operations
2. **Use min_duration_ms**: Filter out fast operations (< 1ms) to reduce noise
3. **Throttle high-frequency calls**: Use `log_interval` for methods called > 100 times/step
4. **Profile across runs**: Compare profiling metrics before/after optimization
5. **Monitor distributed training**: Check for rank-specific slowdowns

### Example: Complete Profiling Setup

```python
from axolotl.integrations.swanlab.profiling import (
    swanlab_profile,
    swanlab_profiling_context,
    ProfilingConfig,
)

class OptimizedTrainer(AxolotlTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Custom profiling config for high-frequency operations
        self.fast_op_config = ProfilingConfig(
            enabled=True,
            min_duration_ms=0.5,
            log_interval=50,
        )

    @swanlab_profile
    def training_step(self, model, inputs):
        """Main training step - always profile."""
        return super().training_step(model, inputs)

    @swanlab_profile
    def compute_loss(self, model, inputs, return_outputs=False):
        """Loss computation - always profile."""
        return super().compute_loss(model, inputs, return_outputs)

    def _prepare_inputs(self, inputs):
        """High-frequency operation - throttled profiling."""
        with swanlab_profiling_context_advanced(
            self,
            "prepare_inputs",
            config=self.fast_op_config,
        ):
            return super()._prepare_inputs(inputs)
```

### Troubleshooting

#### Profiling metrics not appearing in SwanLab

**Cause**: SwanLab is not enabled or not initialized.

**Solution**:
```yaml
# Ensure SwanLab is enabled
use_swanlab: true
swanlab_project: my-project
```

Check logs for:
```
INFO: SwanLab initialized for project: my-project
```

#### Too many profiling metrics cluttering dashboard

**Cause**: Profiling every function call for high-frequency operations.

**Solution**: Use `ProfilingConfig` with throttling:
```python
config = ProfilingConfig(
    min_duration_ms=1.0,    # Skip fast ops
    log_interval=100,       # Log every 100th call
)
```

#### Profiling overhead impacting training speed

**Cause**: Profiling itself should have negligible overhead (< 0.1%). If you see > 1% slowdown, this indicates a bug.

**Solution**:
1. Disable profiling temporarily to confirm:
   ```python
   DEFAULT_PROFILING_CONFIG.enabled = False
   ```
2. Report issue with profiling data and trainer details

#### Profiling shows inconsistent timing

**Cause**: Normal variation due to GPU warmup, data loading, or system load.

**Solution**:
- Ignore first few steps (warmup period)
- Look at average/median timing over many steps
- Use `log_interval` to reduce noise from individual outliers

## Complete Config Example

Here's a complete example integrating SwanLab with your RVQ-Alpha training:

```yaml
base_model: /path/to/your/model
model_type: Qwen2ForCausalLM

# SwanLab Integration
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin

use_swanlab: true
swanlab_project: RVQ-Alpha-Training
swanlab_experiment_name: Qwen2.5-7B-MetaQA-Perturb-P020
swanlab_description: "Training on MetaQA and Perturbation datasets with NEW-RVQ encoding"
swanlab_mode: cloud
swanlab_workspace: single-cell-genomics

# Training configuration
sequence_len: 32768
micro_batch_size: 1
gradient_accumulation_steps: 1
num_epochs: 2
learning_rate: 2e-5
optimizer: adamw_torch_fused

# Datasets
datasets:
  - path: /path/to/dataset
    type: chat_template

# Output
output_dir: ./outputs
```

## Modes Explained

### `cloud` Mode (Default)
- Syncs experiments to SwanLab cloud in real-time
- Requires API key and internet connection
- Best for: Team collaboration, remote monitoring

### `local` Mode
- Saves experiments locally only
- No cloud sync
- Best for: Local development, air-gapped environments

### `offline` Mode
- Saves metadata locally
- Can sync to cloud later using `swanlab sync`
- Best for: Unstable internet, sync later

### `disabled` Mode
- Turns off SwanLab completely
- No logging or tracking
- Best for: Debugging, testing

## Configuration Validation & Conflict Detection

SwanLab integration includes comprehensive validation and conflict detection to help you catch configuration errors early and avoid performance issues.

### Required Fields Validation

The plugin validates your configuration at startup and provides clear error messages with solutions:

#### Missing Project Name

```yaml
# ❌ INVALID: use_swanlab enabled but no project
use_swanlab: true
# Error: SwanLab enabled but 'swanlab_project' is not set.
```

**Solution**:
```yaml
# ✅ VALID: Provide project name
use_swanlab: true
swanlab_project: my-project
```

#### Invalid Mode

```yaml
# ❌ INVALID: Unknown mode
use_swanlab: true
swanlab_project: my-project
swanlab_mode: invalid-mode
# Error: Invalid swanlab_mode: 'invalid-mode'. Valid options: cloud, local, offline, disabled
```

**Solution**:
```yaml
# ✅ VALID: Use one of the valid modes
use_swanlab: true
swanlab_project: my-project
swanlab_mode: cloud  # or: local, offline, disabled
```

#### Empty Project Name

```yaml
# ❌ INVALID: Empty string project name
use_swanlab: true
swanlab_project: ""
# Error: swanlab_project cannot be an empty string.
```

**Solution**:
```yaml
# ✅ VALID: Provide non-empty project name
use_swanlab: true
swanlab_project: my-project
```

### Cloud Mode API Key Warning

When using `cloud` mode without an API key, you'll receive a warning with multiple solutions:

```yaml
use_swanlab: true
swanlab_project: my-project
swanlab_mode: cloud
# No API key set
# Warning: SwanLab cloud mode enabled but no API key found.
```

**Solutions**:
1. Set environment variable: `export SWANLAB_API_KEY=your-api-key`
2. Add to config (less secure): `swanlab_api_key: your-api-key`
3. Run `swanlab login` before training
4. Use `swanlab_mode: local` for offline tracking

### Multi-Logger Performance Warnings

Using multiple logging tools simultaneously (SwanLab + WandB + MLflow + Comet) can impact training performance:

#### Two Loggers - Warning

```yaml
use_swanlab: true
swanlab_project: my-project

use_wandb: true
wandb_project: my-project

# Warning: Multiple logging tools enabled: SwanLab, WandB
# Expected overhead: ~3.0% per training step.
```

**Impact**:
- Performance overhead: ~1-2% per logger (cumulative)
- Increased memory usage
- Longer training time per step
- Potential config/callback conflicts

**Recommendations**:
- Choose ONE primary logging tool for production training
- Use multiple loggers only for:
  - Migration period (transitioning between tools)
  - Short comparison runs
  - Debugging specific tool issues
- Monitor system resources (CPU, memory) during training

#### Three+ Loggers - Error-Level Warning

```yaml
use_swanlab: true
swanlab_project: my-project

use_wandb: true
wandb_project: my-project

use_mlflow: true
mlflow_tracking_uri: http://localhost:5000

# ERROR: 3 logging tools enabled simultaneously!
# Expected overhead: ~4.5% per training step.
# STRONGLY RECOMMEND: Disable all but ONE logging tool
```

**Why This Matters**:
- With 3 loggers: ~4-5% overhead per step → significant slowdown over long training
- Example: 10,000 steps at 2s/step → ~400-500 seconds extra (6-8 minutes)
- Memory overhead scales with number of loggers
- Rare edge cases with callback ordering conflicts

### Auto-Enable Logic

For convenience, SwanLab will auto-enable if you specify a project without setting `use_swanlab`:

```yaml
# This configuration:
swanlab_project: my-project

# Automatically becomes:
use_swanlab: true
swanlab_project: my-project
```

### Distributed Training Detection

In distributed training scenarios (multi-GPU), the plugin automatically detects and reports:

```yaml
use_swanlab: true
swanlab_project: my-project
swanlab_mode: cloud

# When running with torchrun --nproc_per_node=4:
# Info: Distributed training detected (world_size=4)
# Info: SwanLab mode: cloud
# Info: Only rank 0 will initialize SwanLab
# Info: Other ranks will skip SwanLab to avoid conflicts
```

**Why Only Rank 0**:
- Avoids duplicate experiment runs
- Reduces network/cloud API overhead on worker ranks
- Prevents race conditions in metric logging

## Authentication

### Method 1: Environment Variable (Recommended)
```bash
export SWANLAB_API_KEY=your-api-key-here
```

### Method 2: Login Command
```bash
swanlab login
# Enter your API key when prompted
```

### Method 3: Config File
```yaml
swanlab_api_key: your-api-key-here
```

## What Gets Logged?

### Automatically Logged Metrics
- Training loss
- Learning rate
- Gradient norm
- Training steps
- Epoch progress

### Automatically Logged Config
- Model configuration (base_model, model_type)
- Training hyperparameters (learning_rate, batch_size, etc.)
- Optimizer settings
- Parallelization settings (FSDP, DeepSpeed, Context Parallel)
- Axolotl configuration file
- DeepSpeed configuration (if used)

## Viewing Your Experiments

### Cloud Mode
Visit [https://swanlab.cn](https://swanlab.cn) and navigate to your project to view:
- Real-time training metrics
- Hyperparameter comparison
- System resource usage
- Configuration files

### Local Mode
```bash
# Start local dashboard
swanlab watch ./swanlog

# Open browser to http://localhost:5092
```

## Integration with Existing Tools

SwanLab can work alongside other tracking tools:

```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

# Use both SwanLab and Wandb
use_swanlab: true
swanlab_project: my-project

use_wandb: true
wandb_project: my-project
```

## Troubleshooting

### Configuration Errors

#### Error: "SwanLab enabled but 'swanlab_project' is not set"

**Cause**: You enabled SwanLab (`use_swanlab: true`) but forgot to specify a project name.

**Solution**:
```yaml
use_swanlab: true
swanlab_project: my-project  # Add this line
```

#### Error: "Invalid swanlab_mode: 'xxx'"

**Cause**: You provided an invalid mode value.

**Solution**: Use one of the valid modes:
```yaml
swanlab_mode: cloud     # or: local, offline, disabled
```

#### Error: "swanlab_project cannot be an empty string"

**Cause**: You set `swanlab_project: ""` (empty string).

**Solution**: Either provide a valid name or remove the field:
```yaml
# Option 1: Provide valid name
swanlab_project: my-project

# Option 2: Remove the field entirely
# swanlab_project: ""  <- Remove this line
```

### Import Errors

#### Error: "SwanLab is not installed"

**Cause**: SwanLab package is not installed in your environment.

**Solution**:
```bash
pip install swanlab
# or
pip install swanlab>=0.3.0
```

### Performance Issues

#### Warning: "Multiple logging tools enabled"

**Cause**: You have multiple experiment tracking tools enabled (e.g., SwanLab + WandB + MLflow).

**Impact**: ~1-2% performance overhead per logger, cumulative.

**Solution**: For production training, disable all but one logger:
```yaml
# Option 1: Keep only SwanLab
use_swanlab: true
swanlab_project: my-project
use_wandb: false      # Disable others
use_mlflow: false

# Option 2: Keep only WandB
use_swanlab: false
use_wandb: true
wandb_project: my-project
```

**Exception**: Multiple loggers are acceptable for:
- Short comparison runs (< 100 steps)
- Migration testing between logging tools
- Debugging logger-specific issues

### Distributed Training Issues

#### SwanLab creates duplicate runs in multi-GPU training

**Cause**: All ranks are initializing SwanLab instead of just rank 0.

**Expected Behavior**: The plugin automatically ensures only rank 0 initializes SwanLab. You should see:
```
Info: Distributed training detected (world_size=4)
Info: Only rank 0 will initialize SwanLab
Info: Other ranks will skip SwanLab to avoid conflicts
```

**If you see duplicates**:
1. Check your plugin is loaded correctly
2. Verify you're using the latest SwanLab integration code
3. Check logs for initialization messages on all ranks

### SwanLab not logging metrics

**Solution**: Ensure SwanLab is initialized before training starts. The plugin automatically handles this in `pre_model_load`.

### API Key errors

**Solution**:
```bash
# Verify API key
echo $SWANLAB_API_KEY

# Re-login
swanlab login
```

### Cloud sync issues

**Solution**: Use `offline` mode and sync later:
```yaml
swanlab_mode: offline
```

Then sync when ready:
```bash
swanlab sync ./swanlog
```

### Plugin not loaded

**Solution**: Verify plugin path in config:
```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin  # Correct path
```

### Lark Notification Issues

#### Error: "Failed to import SwanLab Lark plugin"

**Cause**: Your SwanLab version doesn't include the Lark plugin (requires SwanLab >= 0.3.0).

**Solution**:
```bash
# Upgrade SwanLab to latest version
pip install --upgrade swanlab

# Or install specific version
pip install 'swanlab>=0.3.0'
```

#### Warning: "Lark webhook has no secret configured"

**Cause**: You provided `swanlab_lark_webhook_url` but no `swanlab_lark_secret`.

**Impact**: Lark notifications will work, but without HMAC authentication (security risk).

**Solution**: Add HMAC secret for production use:
```yaml
swanlab_lark_webhook_url: https://open.feishu.cn/open-apis/bot/v2/hook/xxx
swanlab_lark_secret: your-webhook-secret  # Add this line
```

**When it's OK to skip secret**:
- Local development and testing
- Internal networks with restricted access
- Non-sensitive training experiments

**When secret is required**:
- Production training jobs
- Training with proprietary data
- Multi-team shared Lark groups

#### Error: "Failed to register Lark callback"

**Cause**: Invalid webhook URL or network connectivity issues.

**Diagnostic steps**:
```bash
# 1. Test webhook URL manually
curl -X POST "YOUR_WEBHOOK_URL" \
  -H 'Content-Type: application/json' \
  -d '{"msg_type":"text","content":{"text":"Test from Axolotl"}}'

# 2. Check SwanLab version
pip show swanlab

# 3. Verify webhook URL format
# Should start with: https://open.feishu.cn/open-apis/bot/v2/hook/
```

**Solution**:
1. Verify webhook URL is correct (copy from Lark bot settings)
2. Check network connectivity to Lark API
3. Ensure webhook is not expired (Lark webhooks can expire)
4. Regenerate webhook URL in Lark bot settings if needed

#### Lark notifications not received

**Cause**: Multiple possible causes.

**Diagnostic checklist**:

1. **Check training logs** for Lark registration confirmation:
   ```
   # Expected log message (rank 0 only):
   INFO: Registered Lark notification callback with HMAC authentication
   ```

2. **Verify webhook in Lark**: Test webhook manually (see above)

3. **Check distributed training**: Only rank 0 sends notifications
   ```bash
   # If running multi-GPU, check rank 0 logs specifically
   grep "Registered Lark" logs/rank_0.log
   ```

4. **Verify SwanLab is initialized**: Lark callback needs SwanLab to be running
   ```yaml
   use_swanlab: true  # Must be enabled
   swanlab_project: my-project  # Must be set
   ```

5. **Check Lark bot permissions**: Ensure bot is added to the target group chat

#### Duplicate Lark notifications in multi-GPU training

**Expected Behavior**: Should NOT happen - only rank 0 sends notifications.

**If you see duplicates**:
1. Check that all GPUs are using the same config file
2. Verify plugin is loaded correctly on all ranks
3. Check logs for unexpected Lark initialization on non-zero ranks
4. Ensure `RANK` or `LOCAL_RANK` environment variables are set correctly

**Solution**: This is a bug if it occurs. Report with:
- Full training command
- Logs from all ranks
- Config file

## Comparison: SwanLab vs WandB

| Feature | SwanLab | WandB |
|---------|---------|-------|
| Open Source | ✅ Yes | ❌ No |
| Self-Hosting | ✅ Easy | ⚠️ Complex |
| Free Tier | ✅ Generous | ⚠️ Limited |
| Chinese Support | ✅ Native | ⚠️ Limited |
| Offline Mode | ✅ Full support | ✅ Supported |
| Integration | 🆕 New | ✅ Mature |

## Advanced Usage

### Custom Logging

You can add custom metrics in your callbacks:

```python
import swanlab

# In your custom callback
swanlab.log({
    "custom_metric": value,
    "epoch": epoch_num
})
```

### Experiment Comparison

```bash
# Compare multiple experiments
swanlab compare run1 run2 run3
```

## Support

- **Documentation**: [https://docs.swanlab.cn](https://docs.swanlab.cn)
- **GitHub**: [https://github.com/SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab)
- **Issues**: Report bugs at [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues)

## License

This integration follows the Axolotl Community License Agreement.

## Acknowledgements

This integration is built on top of:
- [SwanLab](https://github.com/SwanHubX/SwanLab) - Experiment tracking tool
- [Transformers](https://github.com/huggingface/transformers) - SwanLabCallback
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - Training framework






