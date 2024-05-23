# todo list

- [] Validation of parameters for combinations that won't work



## things that are known not to work

- FSDP offload and gradient_checkpointing - https://github.com/pytorch/pytorch/issues/82203
- adamw_bnb_8bit doesn't play well with FSDP offload
