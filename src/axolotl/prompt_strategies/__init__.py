import importlib
from functools import cache

@cache
def load(strategy, tokenizer, cfg):
    try:
        m = importlib.import_module(f".{strategy}", axolotl.prompt_strategies)
        fn = getattr(m, "load")
        return fn(tokenizer, cfg)
    except:
        pass
