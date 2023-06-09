# Python 12B

- Single-GPU A100 only (?)

```shell
python scripts/finetune.py examples/pythia-12b/config.yml
```

⚠️ Multiple-GPU A100 - Doesn't seem to work with multi-gpu without causing OOM! ⚠️
