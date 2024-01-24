# Overview

This is a simple example of how to finetune TinyLlama1.1B using either lora or qlora:

LoRa:

```
accelerate launch -m axolotl.cli.train examples/tiny-llama/lora.yml
```

qLoRa:

```
accelerate launch -m axolotl.cli.train examples/tiny-llama/qlora.yml
```

Both take about 10 minutes to complete on a 4090.
