# SwanLab Integration Examples

This directory contains example configurations demonstrating SwanLab integration with Axolotl.

## Examples Overview

### 1. DPO with Completion Logging
**File**: `dpo-swanlab-completions.yml`

Demonstrates DPO (Direct Preference Optimization) training with RLHF completion table logging.

**Features**:
- Basic SwanLab experiment tracking
- Completion table logging (prompts, chosen/rejected responses, rewards)
- Memory-bounded buffer for long training runs
- Cloud sync configuration

**Best for**: RLHF practitioners who want to analyze model outputs qualitatively

**Quick start**:
```bash
export SWANLAB_API_KEY=your-api-key
accelerate launch -m axolotl.cli.train examples/swanlab/dpo-swanlab-completions.yml
```

---

### 2. LoRA with Performance Profiling
**File**: `lora-swanlab-profiling.yml`

Demonstrates standard LoRA fine-tuning with performance profiling enabled.

**Features**:
- SwanLab experiment tracking
- Automatic profiling of trainer methods
- Profiling metrics visualization
- Performance optimization guidance

**Best for**: Engineers optimizing training performance and comparing different configurations

**Quick start**:
```bash
export SWANLAB_API_KEY=your-api-key
accelerate launch -m axolotl.cli.train examples/swanlab/lora-swanlab-profiling.yml
```

---

### 3. Full-Featured DPO Production Setup
**File**: `dpo-swanlab-full-featured.yml`

Comprehensive production-ready configuration with ALL SwanLab features enabled.

**Features**:
- Experiment tracking with team workspace
- RLHF completion logging
- Performance profiling
- Lark (Feishu) team notifications
- Private deployment support
- Production checklist and troubleshooting

**Best for**: Production RLHF training with team collaboration

**Quick start**:
```bash
export SWANLAB_API_KEY=your-api-key
export SWANLAB_LARK_WEBHOOK_URL=https://open.feishu.cn/...
export SWANLAB_LARK_SECRET=your-webhook-secret
accelerate launch -m axolotl.cli.train examples/swanlab/dpo-swanlab-full-featured.yml
```

---

### 4. Custom Trainer Profiling (Python)
**File**: `custom_trainer_profiling.py`

Python code examples showing how to add SwanLab profiling to custom trainers.

**Features**:
- `@swanlab_profile` decorator examples
- Context manager profiling for fine-grained timing
- `ProfilingConfig` for advanced filtering and throttling
- Multiple profiling patterns and best practices

**Best for**: Advanced users creating custom trainers

**Usage**:
```python
from custom_trainer_profiling import CustomTrainerWithProfiling
# See file for detailed examples and patterns
```

---

## Feature Matrix

| Example | Tracking | Completion Logging | Profiling | Lark Notifications | Team Workspace |
|---------|----------|-------------------|-----------|-------------------|----------------|
| dpo-swanlab-completions.yml | ✅ | ✅ | ✅ (auto) | ➖ (commented) | ➖ (commented) |
| lora-swanlab-profiling.yml | ✅ | ➖ (disabled) | ✅ (auto) | ➖ (commented) | ➖ (commented) |
| dpo-swanlab-full-featured.yml | ✅ | ✅ | ✅ (auto) | ✅ | ✅ |
| custom_trainer_profiling.py | N/A | N/A | ✅ (manual) | N/A | N/A |

---

## Configuration Quick Reference

### Basic SwanLab Setup
```yaml
plugins:
  - axolotl.integrations.swanlab.SwanLabPlugin

use_swanlab: true
swanlab_project: my-project
swanlab_experiment_name: my-experiment
swanlab_mode: cloud  # cloud, local, offline, disabled
```

### RLHF Completion Logging
```yaml
swanlab_log_completions: true
swanlab_completion_log_interval: 100  # Log every 100 steps
swanlab_completion_max_buffer: 128    # Memory-bounded buffer
```

### Lark Team Notifications
```yaml
swanlab_lark_webhook_url: https://open.feishu.cn/...
swanlab_lark_secret: your-webhook-secret  # Required for production
```

### Team Workspace
```yaml
swanlab_workspace: my-research-team
```

### Private Deployment
```yaml
swanlab_web_host: https://swanlab.yourcompany.com
swanlab_api_host: https://api.swanlab.yourcompany.com
```

---

## Authentication

### Recommended: Environment Variable
```bash
export SWANLAB_API_KEY=your-api-key
export SWANLAB_LARK_WEBHOOK_URL=https://open.feishu.cn/...
export SWANLAB_LARK_SECRET=your-webhook-secret
```

### Alternative: Config File (less secure)
```yaml
swanlab_api_key: your-api-key
swanlab_lark_webhook_url: https://open.feishu.cn/...
swanlab_lark_secret: your-webhook-secret
```

---

## Common Use Cases

### Use Case 1: Migrate from WandB to SwanLab
Start with `lora-swanlab-profiling.yml`, add your model/dataset config, disable WandB:
```yaml
use_swanlab: true
use_wandb: false
```

### Use Case 2: Analyze DPO Model Outputs
Use `dpo-swanlab-completions.yml`, adjust completion logging interval based on your training length:
```yaml
swanlab_completion_log_interval: 50   # More frequent for short training
swanlab_completion_log_interval: 200  # Less frequent for long training
```

### Use Case 3: Optimize Training Performance
Use `lora-swanlab-profiling.yml`, run multiple experiments with different optimizations:
- Baseline: `flash_attention: false, gradient_checkpointing: false`
- Flash Attention: `flash_attention: true`
- Gradient Checkpointing: `gradient_checkpointing: true`
- Both: `flash_attention: true, gradient_checkpointing: true`

Compare profiling metrics in SwanLab dashboard.

### Use Case 4: Production RLHF with Team Collaboration
Use `dpo-swanlab-full-featured.yml`, set up team workspace and Lark notifications:
```yaml
swanlab_workspace: ml-team
swanlab_lark_webhook_url: ...
swanlab_lark_secret: ...
```

---

## Viewing Your Experiments

### Cloud Mode
Visit [https://swanlab.cn](https://swanlab.cn) and navigate to your project.

**Dashboard sections**:
- **Metrics**: Training loss, learning rate, profiling metrics
- **Tables**: RLHF completions (for DPO/KTO/ORPO/GRPO)
- **Config**: Hyperparameters and configuration
- **System**: Resource usage (GPU, memory, CPU)
- **Files**: Logged artifacts

### Local Mode
```bash
swanlab watch ./swanlog
# Open browser to http://localhost:5092
```

---

## Troubleshooting

### SwanLab not initializing
```bash
# Check API key
echo $SWANLAB_API_KEY

# Verify SwanLab is installed
pip show swanlab

# Check config
grep -A 5 "use_swanlab" your-config.yml
```

### Completions not appearing
- Verify you're using an RLHF trainer (DPO/KTO/ORPO/GRPO)
- Check `swanlab_log_completions: true`
- Wait for `swanlab_completion_log_interval` steps
- Look for "Registered SwanLab RLHF completion logging" in logs

### Lark notifications not working
- Test webhook manually: `curl -X POST "$SWANLAB_LARK_WEBHOOK_URL" ...`
- Verify `SWANLAB_LARK_SECRET` is set correctly
- Check bot is added to Lark group chat
- Look for "Registered Lark notification callback" in logs

### Profiling metrics not appearing
- Verify `use_swanlab: true`
- Check SwanLab is initialized (look for init log message)
- Profiling metrics are under "profiling/" namespace
- Profiling auto-enabled when SwanLab is enabled

---

## Performance Notes

### Overhead Comparison

| Feature | Overhead per Step | Memory Usage |
|---------|------------------|--------------|
| Basic tracking | < 0.1% | ~10 MB |
| Completion logging | < 0.5% | ~64 KB (buffer=128) |
| Profiling | < 0.1% | ~1 KB |
| **Total** | **< 0.7%** | **~10 MB** |

### Best Practices
1. Use ONE logging tool in production (disable WandB/MLflow when using SwanLab)
2. Adjust completion log interval based on training length (100-200 steps)
3. Keep completion buffer size reasonable (128-512)
4. Profile critical path methods first (training_step, compute_loss)
5. Use ProfilingConfig to throttle high-frequency operations

---

## Further Reading

- **Full Documentation**: [src/axolotl/integrations/swanlab/README.md](../../src/axolotl/integrations/swanlab/README.md)
- **SwanLab Docs**: [https://docs.swanlab.cn](https://docs.swanlab.cn)
- **Axolotl Docs**: [https://axolotl-ai-cloud.github.io/axolotl/](https://axolotl-ai-cloud.github.io/axolotl/)
- **DPO Paper**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

---

## Contributing

Found an issue or have an improvement? Please submit a PR or open an issue:
- [Axolotl Issues](https://github.com/axolotl-ai-cloud/axolotl/issues)
- [SwanLab Issues](https://github.com/SwanHubX/SwanLab/issues)
