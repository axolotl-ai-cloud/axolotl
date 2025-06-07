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

## Online KD (sglang)

```bash
export UV_TORCH_BACKEND=cu124
uv venv sglang --python 3.11
source sglang/bin/activate
uv pip install --upgrade pip
uv pip install setuptools
uv pip install torch~=2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install sgl-kernel --force-reinstall --no-deps
uv pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
```

## Online KD (vllm)

```bash
VLLM_USE_V1=0 vllm serve open-r1/OlympicCoder-32B --max-model-len 16400 --port 8888 --max-logprobs 128 --return-tokens-as-token-ids --tensor-parallel-size 8 --max-num-seqs
256 --gpu_memory_utilization 0.2 --enable-chunked-prefill
```

```bash
vllm serve open-r1/OlympicCoder-32B --max-model-len 16400 --port 8888 --max-logprobs 128 --return-tokens-as-token-ids --tensor-parallel-size 8 --no-enable-prefix-caching --gpu-memory-utilization 0.3 --max-num-batched-tokens 131072 --host 0.0.0.0
```


```bash
python -m sglang.launch_server --model-path open-r1/OlympicCoder-32B --tensor-parallel-size 8 --port 8080 --host 0.0.0.0 --max-running-requests 256 --context-length 16400 --mem-fraction-static 0.2 --schedule-conservativeness 0.3 --chunked-prefill-size 131072 --schedule-policy fcfs --skip-tokenizer-init
```
