# Knowledge Distillation

## Usage

```yaml
plugins:
  - "axolotl.integrations.kd.KDPlugin"

kd_trainer: True
kd_ce_alpha: 0.1
kd_alpha: 0.9
kd_temperature: 1.0

torch_compile: True  # torch>=2.5.1, recommended to reduce vram

datasets:
  - path: ...
    type: "axolotl.integrations.kd.chat_template"
    field_messages: "messages_combined"
    logprobs_field: "llm_text_generation_vllm_logprobs"  # for kd only, field of logprobs
```

An example dataset can be found at [`axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample`](https://huggingface.co/datasets/axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample)
