<h1>LLM Post Training- Full fine-tune, LoRA, QLoRa etc. Llama/Mistral/Gemma and more</h1>

# Configuration Options

This document outlines all available configuration options for training models. The configuration can be provided as a JSON request.

## Usage

You can use these configuration Options:

1. As a JSON request body:

```json
{
  "input": {
    "user_id": "user",
    "model_id": "model-name",
    "run_id": "run-id",
    "credentials": {
      "wandb_api_key": "", # add your Weights & biases key. TODO:  you will be able to set this in Enviornment variables.
      "hf_token": "", # add your HF_token. TODO:  you will be able to set this in Enviornment variables.
    },
    "args": {
      "base_model": "NousResearch/Llama-3.2-1B",
      // ... other options
    }
  }
}
```

## Configuration Options

### Model Configuration

| Option              | Description                                                                                   | Default              |
| ------------------- | --------------------------------------------------------------------------------------------- | -------------------- |
| `base_model`        | Path to the base model (local or HuggingFace)                                                 | Required             |
| `base_model_config` | Configuration path for the base model                                                         | Same as base_model   |
| `revision_of_model` | Specific model revision from HuggingFace hub                                                  | Latest               |
| `tokenizer_config`  | Custom tokenizer configuration path                                                           | Optional             |
| `model_type`        | Type of model to load                                                                         | AutoModelForCausalLM |
| `tokenizer_type`    | Type of tokenizer to use                                                                      | AutoTokenizer        |
| `hub_model_id`      | Repository ID where the model will be pushed on Hugging Face Hub (format: username/repo-name) | Optional             |

## Model Family Identification

| Option                     | Default | Description                    |
| -------------------------- | ------- | ------------------------------ |
| `is_falcon_derived_model`  | `false` | Whether model is Falcon-based  |
| `is_llama_derived_model`   | `false` | Whether model is LLaMA-based   |
| `is_qwen_derived_model`    | `false` | Whether model is Qwen-based    |
| `is_mistral_derived_model` | `false` | Whether model is Mistral-based |

## Model Configuration Overrides

| Option                                          | Default    | Description                        |
| ----------------------------------------------- | ---------- | ---------------------------------- |
| `overrides_of_model_config.rope_scaling.type`   | `"linear"` | RoPE scaling type (linear/dynamic) |
| `overrides_of_model_config.rope_scaling.factor` | `1.0`      | RoPE scaling factor                |

### Model Loading Options

| Option         | Description                   | Default |
| -------------- | ----------------------------- | ------- |
| `load_in_8bit` | Load model in 8-bit precision | false   |
| `load_in_4bit` | Load model in 4-bit precision | false   |
| `bf16`         | Use bfloat16 precision        | false   |
| `fp16`         | Use float16 precision         | false   |
| `tf32`         | Use tensor float 32 precision | false   |

## Memory and Device Settings

| Option             | Default   | Description             |
| ------------------ | --------- | ----------------------- |
| `gpu_memory_limit` | `"20GiB"` | GPU memory limit        |
| `lora_on_cpu`      | `false`   | Load LoRA on CPU        |
| `device_map`       | `"auto"`  | Device mapping strategy |
| `max_memory`       | `null`    | Max memory per device   |

## Training Hyperparameters

| Option                        | Default   | Description                 |
| ----------------------------- | --------- | --------------------------- |
| `gradient_accumulation_steps` | `1`       | Gradient accumulation steps |
| `micro_batch_size`            | `2`       | Batch size per GPU          |
| `eval_batch_size`             | `null`    | Evaluation batch size       |
| `num_epochs`                  | `4`       | Number of training epochs   |
| `warmup_steps`                | `100`     | Warmup steps                |
| `warmup_ratio`                | `0.05`    | Warmup ratio                |
| `learning_rate`               | `0.00003` | Learning rate               |
| `lr_quadratic_warmup`         | `false`   | Quadratic warmup            |
| `logging_steps`               | `null`    | Logging frequency           |
| `eval_steps`                  | `null`    | Evaluation frequency        |
| `evals_per_epoch`             | `null`    | Evaluations per epoch       |
| `save_strategy`               | `"epoch"` | Checkpoint saving strategy  |
| `save_steps`                  | `null`    | Saving frequency            |
| `saves_per_epoch`             | `null`    | Saves per epoch             |
| `save_total_limit`            | `null`    | Maximum checkpoints to keep |
| `max_steps`                   | `null`    | Maximum training steps      |

### Dataset Configuration

```yaml
datasets:
  - path: vicgalle/alpaca-gpt4 # HuggingFace dataset or TODO: You will be able to add the local path.
    type: alpaca # Format type (alpaca, gpteacher, oasst, etc.)
    ds_type: json # Dataset type
    data_files: path/to/data # Source data files
    train_on_split: train # Dataset split to use
```

## Chat Template Settings

| Option                   | Default                          | Description            |
| ------------------------ | -------------------------------- | ---------------------- |
| `chat_template`          | `"tokenizer_default"`            | Chat template type     |
| `chat_template_jinja`    | `null`                           | Custom Jinja template  |
| `default_system_message` | `"You are a helpful assistant."` | Default system message |

## Dataset Processing

| Option                            | Default                    | Description                         |
| --------------------------------- | -------------------------- | ----------------------------------- |
| `dataset_prepared_path`           | `"data/last_run_prepared"` | Path for prepared dataset           |
| `push_dataset_to_hub`             | `""`                       | Push dataset to HF hub              |
| `dataset_processes`               | `4`                        | Number of preprocessing processes   |
| `dataset_keep_in_memory`          | `false`                    | Keep dataset in memory              |
| `shuffle_merged_datasets`         | `true`                     | Shuffle merged datasets             |
| `shuffle_before_merging_datasets` | `false`                    | Shuffle each dataset before merging |
| `dataset_exact_deduplication`     | `true`                     | Deduplicate datasets                |

## LoRA Configuration

| Option                     | Default                | Description                    |
| -------------------------- | ---------------------- | ------------------------------ |
| `adapter`                  | `"lora"`               | Adapter type (lora/qlora)      |
| `lora_model_dir`           | `""`                   | Directory with pretrained LoRA |
| `lora_r`                   | `8`                    | LoRA attention dimension       |
| `lora_alpha`               | `16`                   | LoRA alpha parameter           |
| `lora_dropout`             | `0.05`                 | LoRA dropout                   |
| `lora_target_modules`      | `["q_proj", "v_proj"]` | Modules to apply LoRA          |
| `lora_target_linear`       | `false`                | Target all linear modules      |
| `peft_layers_to_transform` | `[]`                   | Layers to transform            |
| `lora_modules_to_save`     | `[]`                   | Modules to save                |
| `lora_fan_in_fan_out`      | `false`                | Fan in/out structure           |

## Optimization Settings

| Option                    | Default | Description                |
| ------------------------- | ------- | -------------------------- |
| `train_on_inputs`         | `false` | Train on input prompts     |
| `group_by_length`         | `false` | Group by sequence length   |
| `gradient_checkpointing`  | `false` | Use gradient checkpointing |
| `early_stopping_patience` | `3`     | Early stopping patience    |

## Learning Rate Scheduling

| Option                     | Default    | Description          |
| -------------------------- | ---------- | -------------------- |
| `lr_scheduler`             | `"cosine"` | Scheduler type       |
| `lr_scheduler_kwargs`      | `{}`       | Scheduler parameters |
| `cosine_min_lr_ratio`      | `null`     | Minimum LR ratio     |
| `cosine_constant_lr_ratio` | `null`     | Constant LR ratio    |
| `lr_div_factor`            | `null`     | LR division factor   |

## Optimizer Settings

| Option                 | Default      | Description         |
| ---------------------- | ------------ | ------------------- |
| `optimizer`            | `"adamw_hf"` | Optimizer choice    |
| `optim_args`           | `{}`         | Optimizer arguments |
| `optim_target_modules` | `[]`         | Target modules      |
| `weight_decay`         | `null`       | Weight decay        |
| `adam_beta1`           | `null`       | Adam beta1          |
| `adam_beta2`           | `null`       | Adam beta2          |
| `adam_epsilon`         | `null`       | Adam epsilon        |
| `max_grad_norm`        | `null`       | Gradient clipping   |

## Attention Implementations

| Option                     | Default | Description                   |
| -------------------------- | ------- | ----------------------------- |
| `flash_optimum`            | `false` | Use better transformers       |
| `xformers_attention`       | `false` | Use xformers                  |
| `flash_attention`          | `false` | Use flash attention           |
| `flash_attn_cross_entropy` | `false` | Flash attention cross entropy |
| `flash_attn_rms_norm`      | `false` | Flash attention RMS norm      |
| `flash_attn_fuse_mlp`      | `false` | Fuse MLP operations           |
| `sdp_attention`            | `false` | Use scaled dot product        |
| `s2_attention`             | `false` | Use shifted sparse attention  |

## Tokenizer Modifications

| Option           | Default | Description                  |
| ---------------- | ------- | ---------------------------- |
| `special_tokens` | -       | Special tokens to add/modify |
| `tokens`         | `[]`    | Additional tokens            |

## Distributed Training

| Option                  | Default | Description           |
| ----------------------- | ------- | --------------------- |
| `fsdp`                  | `null`  | FSDP configuration    |
| `fsdp_config`           | `null`  | FSDP config options   |
| `deepspeed`             | `null`  | Deepspeed config path |
| `ddp_timeout`           | `null`  | DDP timeout           |
| `ddp_bucket_cap_mb`     | `null`  | DDP bucket capacity   |
| `ddp_broadcast_buffers` | `null`  | DDP broadcast buffers |

<details>
<summary><h3>Example Configuration Request:</h3></summary>

Here's a complete example for fine-tuning a LLaMA model using LoRA:

```json
{
  "input": {
    "user_id": "user",
    "model_id": "llama-test",
    "run_id": "test-run",
    "credentials": {
      "wandb_api_key": "",
      "hf_token": ""
    },
    "args": {
      "base_model": "NousResearch/Llama-3.2-1B",
      "load_in_8bit": false,
      "load_in_4bit": false,
      "strict": false,
      "datasets": [
        {
          "path": "teknium/GPT4-LLM-Cleaned",
          "type": "alpaca"
        }
      ],
      "dataset_prepared_path": "last_run_prepared",
      "val_set_size": 0.1,
      "output_dir": "./outputs/lora-out",
      "adapter": "lora",
      "sequence_len": 2048,
      "sample_packing": true,
      "eval_sample_packing": true,
      "pad_to_sequence_len": true,
      "lora_r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.05,
      "lora_target_modules": [
        "gate_proj",
        "down_proj",
        "up_proj",
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
      ],
      "gradient_accumulation_steps": 2,
      "micro_batch_size": 2,
      "num_epochs": 1,
      "optimizer": "adamw_8bit",
      "lr_scheduler": "cosine",
      "learning_rate": 0.0002,
      "train_on_inputs": false,
      "group_by_length": false,
      "bf16": "auto",
      "tf32": false,
      "gradient_checkpointing": true,
      "logging_steps": 1,
      "flash_attention": true,
      "loss_watchdog_threshold": 5,
      "loss_watchdog_patience": 3,
      "warmup_steps": 10,
      "evals_per_epoch": 4,
      "saves_per_epoch": 1,
      "weight_decay": 0,
      "hub_model_id": "runpod/llama-fr-lora",
      "wandb_name": "test-run-1",
      "wandb_project": "test-run-1",
      "wandb_entity": "axo-test",
      "special_tokens": {
        "pad_token": "<|end_of_text|>"
      }
    }
  }
}
```

</details>

### Advanced Features

#### Wandb Integration

- `wandb_project`: Project name for Weights & Biases
- `wandb_entity`: Team name in W&B
- `wandb_watch`: Monitor model with W&B
- `wandb_name`: Name of the W&B run
- `wandb_run_id`: ID for the W&B run

#### Performance Optimization

- `sample_packing`: Enable efficient sequence packing
- `eval_sample_packing`: Use sequence packing during evaluation
- `torch_compile`: Enable PyTorch 2.0 compilation
- `flash_attention`: Use Flash Attention implementation
- `xformers_attention`: Use xFormers attention implementation

### Available Optimizers

The following optimizers are supported:

- `adamw_hf`: HuggingFace's AdamW implementation
- `adamw_torch`: PyTorch's AdamW
- `adamw_torch_fused`: Fused AdamW implementation
- `adamw_torch_xla`: XLA-optimized AdamW
- `adamw_apex_fused`: NVIDIA Apex fused AdamW
- `adafactor`: Adafactor optimizer
- `adamw_anyprecision`: Anyprecision AdamW
- `adamw_bnb_8bit`: 8-bit AdamW from bitsandbytes
- `lion_8bit`: 8-bit Lion optimizer
- `lion_32bit`: 32-bit Lion optimizer
- `sgd`: Stochastic Gradient Descent
- `adagrad`: Adagrad optimizer

## Notes

- Set `load_in_8bit: true` or `load_in_4bit: true` for memory-efficient training
- Enable `flash_attention: true` for faster training on modern GPUs
- Use `gradient_checkpointing: true` to reduce memory usage
- Adjust `micro_batch_size` and `gradient_accumulation_steps` based on your GPU memory

For more detailed information, please refer to the [documentation](https://axolotl-ai-cloud.github.io/axolotl/docs/config-reference.html).

### Errors:

- if you face any issues with the Flash Attention-2, Delete yoor worker and Re-start.
