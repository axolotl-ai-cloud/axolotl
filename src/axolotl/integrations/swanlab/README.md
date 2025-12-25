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






