# Overview

This is an example of a Yi-34B-Chat configuration. It demonstrates that it is possible to finetune a 34B model on a GPU with 24GB of VRAM.

Tested on an RTX 4090 with `python -m axolotl.cli.train examples/mistral/qlora.yml`, a single epoch of finetuning on the alpaca dataset using qlora runs in 47 mins, using 97% of available memory.
