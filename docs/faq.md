# Axolotl FAQ's


> The trainer stopped and hasn't progressed in several minutes.

Usually an issue with the GPU's communicating with each other. See the [NCCL doc](../docs/nccl.md)

> Exitcode -9

This usually happens when you run out of system RAM.

> Exitcode -7 while using deepspeed

Try upgrading deepspeed w: `pip install -U deepspeed`

> AttributeError: 'DummyOptim' object has no attribute 'step'

You may be using deepspeed with single gpu. Please don't set `deepspeed:` in yaml or cli.
