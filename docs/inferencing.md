# Axolotl Inferencing Guide

Batch inferencing takes a list of json-formatted datasets, applies the configured prompt template, executes the inferencing process, and saves the output to ``output_dir``. Inferencing operations distribute prompts across GPUs via [accelerate](https://huggingface.co/docs/accelerate/index).

To launch a multi-GPU batch inferencing job:

```shell
accelerate launch --config_file=/work/accelerate/development_gpu_all.yaml -m \
    axolotl --config=/workspace/work/atheos/empty.yaml --log_level=DEBUG inference batch \
        --base_model=/models/llama-7b-hf \
        --base_model_config=/models/llama-7b-hf \
        --model_type=LlamaForCausalLM \
        --tokenizer_type=LlamaTokenizer \
        --adapter=lora \
        --lora_model_dir=/workspace/work/atheos/output1 \
        --dataset=name=Sample,path=/workspace/data/test/sample-inference-6.json,type=gpteacher \
        --dataset_prepared_path=/tmp/gen \
        --no_train_on_inputs \
        --sequence_len=2048 \
        --split_name=train \
        --micro_batch_size=4 \
        --seed=10 \
        --generation_config=use_cache=true,num_return_sequences=1,do_sample=true,num_beams=1,temperature=0.9,top_p=0.95,top_k=50,typical_p=0.95,max_new_tokens=2048,min_new_tokens=20,repetition_penalty=1.1,prepend_bos=false,renormalize_logits=false \
        --output_dir=/work/tmp
```

The output produced will look like this:

```json

```




To get help via the CLI:

```shell
axolotl inference batch --help
```

For a complete list of generation configuration options please see the corresponding [HuggingFace documentation](https://huggingface.co/docs/transformers/main/main_classes/text_generation#transformers.GenerationConfig).

## Example Accelerate Configuration

This section provides a few example GPU configurations.
### Single GPU

The example below is for a single GPU configuration on 1 machine:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### Multi GPU

The example below is for a 3 GPU configuration on 1 machine:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: "no"
gpu_ids: all
num_processes: 3
machine_rank: 0
main_training_function: main
mixed_precision: "no"
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
