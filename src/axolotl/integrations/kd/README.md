# Knowledge Distillation

## Usage

```yaml
plugins:
  - "axolotl.integrations.kd.KDPlugin"

kd_trainer: True
kd_ce_alpha: 0.1
kd_alpha: 0.9
kd_temperature: 1.0

torch_compile: True  # recommended to reduce vram

datasets:
  - path: ...
    type: "axolotl.integrations.kd.chat_template"
    field_messages: "messages_combined"
    logprobs_field: "llm_text_generation_vllm_logprobs"  # for kd only, field of logprobs
```

An example dataset can be found at [`axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample`](https://huggingface.co/datasets/axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample)

## Implementation notes

The live KD loss is the fused Liger kernel
(`kernels/liger.py::LigerFusedLinearKLTopKLogprobLoss`). A readable, dependency-free
reference of the same top-k forward-KL is kept in
[`topk_logprob/forward_kl.py`](topk_logprob/forward_kl.py) for correctness
comparisons and as a non-Liger fallback; it is not wired into the trainer by default.

For **on-policy** distillation (student rollouts + in-process teacher), see the
[GKD plugin](../gkd/README.md).
