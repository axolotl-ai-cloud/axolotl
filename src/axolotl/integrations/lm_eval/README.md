# LM Eval Harness

Run evaluation on model using the popular lm-evaluation-harness library.

See https://github.com/EleutherAI/lm-evaluation-harness

## Usage

There are two ways to use the LM Eval integration:

### 1. Post-Training Evaluation

When training with the plugin enabled, evaluation runs automatically after training completes:

```yaml
plugins:
  - axolotl.integrations.lm_eval.LMEvalPlugin

lm_eval_tasks:
  - gsm8k
  - hellaswag
  - arc_easy

lm_eval_batch_size: # Batch size for evaluation

# Directory to save evaluation results.
# The final model is loaded from this directory
# unless specified otherwise (see below)
output_dir:
```

Run training as usual:
```bash
axolotl train config.yml
```

### 2. Standalone CLI Evaluation

Evaluate any model directly without training:

```yaml
lm_eval_model: meta-llama/Llama-2-7b-hf

plugins:
  - axolotl.integrations.lm_eval.LMEvalPlugin

lm_eval_tasks:
  - gsm8k
  - hellaswag
  - arc_easy

lm_eval_batch_size: 8
output_dir: ./outputs
```

Run evaluation:
```bash
axolotl lm-eval config.yml
```

## Model Selection Priority

The model to evaluate is selected in the following priority order:

1. **`lm_eval_model`** - Explicit model path or HuggingFace repo (highest priority)
2. **`hub_model_id`** - Trained model pushed to HuggingFace Hub
3. **`output_dir`** - Local checkpoint directory containing trained model weights

## Citation

```bib
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```
