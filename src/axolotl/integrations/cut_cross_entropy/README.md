# Cut Cross Entropy

Cut Cross Entropy (CCE) reduces VRAM usage through optimization on the cross-entropy operation during loss calculation.

See https://github.com/apple/ml-cross-entropy

## Requirements

- PyTorch 2.4.0 or higher

## Installation

Run the following command to install `cut_cross_entropy[transformers]` if you don't have it already.

- If you are in dev environment
```bash
python scripts/cutcrossentropy_install.py | sh
```

- If you are installing from pip
```bash
pip3 uninstall -y cut-cross-entropy && pip3 install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@v0.1.0-rc0"
```

## Usage

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
```

### Options

```yaml
cut_cross_entropy: true                      # default when the plugin is loaded
cut_cross_entropy_accum_c_fp32: true         # fp32 classifier (lm_head) gradient accumulation
cut_cross_entropy_c_grad_chunk_size: auto    # or a positive multiple of 128, e.g. 32768
```

`cut_cross_entropy_accum_c_fp32` improves numerical stability for large vocabularies at the cost of a full fp32 copy of the `lm_head` gradient during the backward pass.

`cut_cross_entropy_c_grad_chunk_size` bounds that fp32 buffer to the given number of vocabulary rows, launching the backward kernel once per chunk. `auto` resolves once at model load from `micro_batch_size`, `sequence_len` (divided by `context_parallel_size`) and the model's vocab and hidden size, picking a size that keeps the GPU busy while capping the scratch buffer at 1 GiB; the chosen value is logged. The gradient is identical to the unchunked path; only peak memory and throughput change. Requires `cut_cross_entropy_accum_c_fp32: true` and Triton >= 3.2.

LoRA and DoRA adapters on `lm_head` (for example `lora_target_modules: [..., lm_head]`) are folded into the loss, so they train under CCE like any other target module.

## Supported Models

- afmoe
- apertus
- arcee
- cohere
- cohere2
- cohere2_moe
- cohere2_vision
- cohere_compass
- cohere_compass_text
- deepseek_v2
- deepseek_v3
- deepseek_v4
- exaone4
- exaone4_5
- exaone_moe
- gemma
- gemma2
- gemma3
- gemma3_text
- gemma3n
- gemma3n_text
- gemma4
- gemma4_text
- gemma4_unified
- gemma4_unified_text
- glm
- glm4
- glm4_moe
- glm4_moe_lite
- glm46v
- glm4v
- glm4v_moe
- glm_image
- glm_moe_dsa
- gpt_oss
- granite
- granitemoe
- granitemoehybrid
- granitemoeshared
- hunyuan_v1_dense
- hunyuan_v1_moe
- internvl
- kimi_linear
- lfm2
- lfm2_moe
- lfm2_vl
- llama
- llama4
- llama4_text
- llava
- minimax
- minimax_m2
- ministral
- ministral3
- mistral
- mistral3
- mistral4
- mixtral
- mllama
- muse_glimmer
- nemotron_h
- olmo
- olmo2
- olmo3
- olmoe
- phi
- phi3
- phi4_multimodal
- qwen2
- qwen2_5_vl
- qwen2_moe
- qwen2_vl
- qwen3
- qwen3_5
- qwen3_5_text
- qwen3_5_moe
- qwen3_5_moe_text
- qwen3_moe
- qwen3_next
- qwen3_vl
- qwen3_vl_moe
- qwen4_exp
- qwen4_exp_text
- seed_oss
- smollm3
- step3p5
- step3p7
- voxtral

## Citation

```bib
@article{wijmans2024cut,
  author       = {Erik Wijmans and
                  Brody Huval and
                  Alexander Hertzberg and
                  Vladlen Koltun and
                  Philipp Kr\"ahenb\"uhl},
  title        = {Cut Your Losses in Large-Vocabulary Language Models},
  journal      = {arXiv},
  year         = {2024},
  url          = {https://arxiv.org/abs/2411.09009},
}
```
