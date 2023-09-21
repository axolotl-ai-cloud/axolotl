# Multi Node

You will need to create a configuration for accelerate, either by using `accelerate config` and follow the instructions or you can use one of the preset below:

~/.cache/huggingface/accelerate/default_config.yaml
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
machine_rank: 0 # Set to 0 for the main machine, increment by one for other machines
main_process_ip: 10.0.0.4 # Set to main machine's IP
main_process_port: 5000
main_training_function: main
mixed_precision: bf16
num_machines: 2 # Change to the number of machines
num_processes: 4 # That's the total number of GPUs, (for example: if you have 2 machines with 4 GPU, put 8)
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Configure your model to use FSDP with for example:
```yaml
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_offload_params: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

## Machine configuration

On each machine you need a copy of Axolotl, we suggest using the same commit to ensure compatibility.

You will also need to have the same configuration file for your model on each machine.

On the main machine only, make sure the port you set as `main_process_port` is open in TCP and reachable by other machines.

All you have to do now is launch using accelerate as you would usually do on each machine and voila, the processes will start once you have launched accelerate on every machine.
