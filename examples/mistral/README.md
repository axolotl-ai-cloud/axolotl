**Mistral 7B** is a language model with a total of 7.3 billion parameters, showcasing a notable performance across a variety of benchmarks.

Fine Tune:
```shell
accelerate launch -m axolotl.cli.train examples/mistral/config.yml

```

If you run into CUDA OOM, use deepspeed with config zero2.json:
```shell
accelerate launch -m axolotl.cli.train examples/mistral/config.yml --deepspeed deepspeed/zero2.json
```
