# Streaming Dataset Examples

This directory contains example configurations for using Axolotl's streaming dataset
functionality, which enables memory-efficient training with large datasets.

## Examples

Run the following examples with e.g. `axolotl train examples/streaming/sft.yaml`; no
`axolotl preprocess` required!

### Pretraining (`pretrain.yaml`)

Demonstrates streaming configuration for pretraining tasks using the fineweb-edu dataset
with SmolLM2-135M.

- Uses `pretraining_dataset` configuration for automatic streaming
- Multipack attention control to prevent cross-attention between packed sequences
- Buffer size configuration for memory management

### SFT (`sft.yaml`)

Shows how to use streaming for supervised fine-tuning with the Alpaca dataset.

- Explicit `streaming: true` flag for SFT datasets
- Memory-efficient training on instruction datasets
- Evaluation datasets are currently not streamed

## Key Configuration Options

### `streaming`
- Enables streaming mode for standard datasets
- Automatically enabled for `pretraining_dataset`

### `streaming_multipack_buffer_size`
- Controls buffer size for sample packing (default: 10,000)
- Larger values improve packing efficiency but use more memory
- Adjust based on available memory

### `shuffle_merged_datasets`
- Enables shuffling of streaming datasets
- Requires additional memory for shuffle buffer

### `sample_packing`
- Packs multiple samples into single sequences
- Minimize per-step padding tokens

## Performance Tips

- Download small / frequently-used datasets locally for better performance
- Larger buffer sizes improve packing efficiency
