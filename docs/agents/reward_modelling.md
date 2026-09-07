# Reward Modelling — Agent Reference

Train models to score responses for use as reward signals in RL. For full docs, see [reward_modelling.qmd](../reward_modelling.qmd).

## Types

### Outcome Reward Models (ORM)

Train a classifier to predict preference over entire interactions. Uses `AutoModelForSequenceClassification`.

```yaml
base_model: google/gemma-2-2b
model_type: AutoModelForSequenceClassification
num_labels: 1
reward_model: true
chat_template: gemma
datasets:
  - path: argilla/distilabel-intel-orca-dpo-pairs
    type: bradley_terry.chat_template
```

Dataset format: `{"system": "...", "input": "...", "chosen": "...", "rejected": "..."}`

### Process Reward Models (PRM)

Train a token classifier to score each reasoning step. Uses `AutoModelForTokenClassification`.

```yaml
base_model: Qwen/Qwen2.5-3B
model_type: AutoModelForTokenClassification
num_labels: 2
process_reward_model: true
datasets:
  - path: trl-lib/math_shepherd
    type: stepwise_supervised
```

Dataset format: see [stepwise_supervised.qmd](../dataset-formats/stepwise_supervised.qmd).

## File Map

```
src/axolotl/
  core/builders/causal.py                    # Handles reward_model flag in trainer builder
  prompt_strategies/bradley_terry/           # Bradley-Terry prompt strategies
  prompt_strategies/stepwise_supervised.py   # PRM dataset strategy
  utils/schemas/config.py                    # reward_model, process_reward_model config fields
```
