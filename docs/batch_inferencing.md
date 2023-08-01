# Axolotl Batch Inferencing Guide

Batch inferencing takes a list of JSON-formatted datasets, applies the configured prompt template, executes the inferencing process, and saves the output to ``output_dir``. Inferencing operations distribute prompts across GPUs via Accelerate. Currently, batch inferencing supports multiple GPUs on a single host.

To run the example below, either install axolotl as a module in your environment or add the axolotl source path to ``PYTHONPATH``.

Next, use ``accelerate`` to launch an inferencing job with the Llama-2-chat model:

```shell
accelerate launch --config_file=accelerate.yaml -m \
    axolotl -c config.yaml inference batch \
        --base_model=/models/Llama-2-7b-chat-hf \
        --base_model_config=/models/Llama-2-7b-chat-hf \
        --dataset=name=Sample,path=/workspace/data/test/sample-response-10.json,type=gpteacher \
        --dataset_prepared_path=/tmp \
        --sequence_len=4096 \
        --truncate_features=response \
        --generation_config=num_return_sequences=3,repetition_penalty=1.18,max_new_tokens=500,temperature=0.5,top_p=0.5,top_k=20,do_sample=true,use_cache=false,prepend_bos=true \
        --output_dir=/work/tmp
```

The output produced will look like this:

```json
{
  "status": "SUCCESS",
  "run_id": "20230801T130712Z",
  "request_count": 10,
  "response_count": 10,
  "run_time_sec": 12.925,
  "total_tokens_generated": 200,
  "tokens_per_sec": 15.474,
  "seed": 10,
  "response": [
    {
      "prompt_tokens": 101,
      "total_tokens": 121,
      "total_response_tokens": 20,
      "generate_time_sec": 1.416,
      "prompt": "...",
      "responses": [
        {
          "response": "...",
          "tokens": 20
        }
      ]
    },
    {
      "prompt_tokens": 65,
      "total_tokens": 85,
      "total_response_tokens": 20,
      "generate_time_sec": 0.912,
      "prompt": "...",
      "responses": [
        {
          "response": "...",
          "tokens": 20
        }
      ]
    }
  ]
}
```

All parameter defaults should be set in ``config.yaml``. This file can also be set in the environment via ``AXOLOTL_CONFIG``. The table below identifies what

| Param | Description |
| ----- | ----------- |
| ``base_model`` | Path to a directory containing the model |
| ``base_model_config`` | Path to the model configuration |
| ``dataset`` | Datasets to use for inferencing, multiple datasets are supported |
| ``dataset_prepared_path`` | Path to save cached data files |
| ``sequence_len`` | Set to the maximum sequence length, ex 4096 for llama2  |
| ``truncate_features`` | When the input json file contains responses, this contains a list of fields that will be set to empty strings. This is desirable when inferencing on a dataset that already contains a response. |
| ``generation_config`` | The transformers [generation config](https://huggingface.co/docs/transformers/main/main_classes/text_generation#transformers.GenerationConfig) parameters  |
| ``output_dir`` | Directory on the filesystem where output results will be saved |

To get help via the CLI run the command below:

```shell
axolotl inference batch --help
```

## Features

* Automatically generates prompts with supported axolotl prompt templates
* Multi-GPU on a single node
* Supports multiple generaitons per prompt
* Trims trailing EOS tokens from prompt, during testing this positively affected results under some configurations
* Optionally strips whitespace out of responses
* New CLI, options can be overridden via parameters or environment variables
* Detailed inferencing statistics
* The last line of output is a parsable JSON string, useful for external orchestration frameworks (such as [Airflow Xcoms](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html))

## Limitations

* Currently, batch inferencing will only collect results on a single node. This would just require an implementation of ``AbstractPersistenceBackend`` with storage accessible to all nodes (such as a database or S3)
* Likely, prompt distribution logic could be optimized more
* Left padding of prompts is triggered when ``num_return_sequences`` > 1 and is best suited for causal language models 


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
