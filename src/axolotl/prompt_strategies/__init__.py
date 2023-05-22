import importlib

def load(strategy, tokenizer, cfg):
    try:
        load_fn = "load"
        if strategy.split(".")[-1].startswith("load_"):
            load_fn = strategy.split(".")[-1]
            strategy = ".".join(strategy.split(".")[:-1])
        m = importlib.import_module(f".{strategy}", "axolotl.prompt_strategies")
        fn = getattr(m, load_fn)
        return fn(tokenizer, cfg)
    except:
        pass
