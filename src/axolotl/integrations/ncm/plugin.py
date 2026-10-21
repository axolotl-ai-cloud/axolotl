"""Next-Concept Modeling plugin."""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault

from . import loading
from .configuration import NCMTokenizerConfig

_CONFIG_PATH = "axolotl.integrations.ncm.configuration"


class NCMPlugin(BasePlugin):
    """Two-stage NCM training: ``ncm.stage: tokenizer`` then ``ncm.stage: generator``."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.ncm.NCMArgs"

    def register(self, cfg: dict):
        ncm = cfg.get("ncm") or {}
        stage = ncm.get("stage")
        if stage == "tokenizer":
            if not cfg.get("base_model"):
                cfg["base_model"] = cfg.get("tokenizer_config")
            if not cfg["base_model"]:
                return
            if (
                loading.checkpoint_model_type(cfg["base_model"])
                == NCMTokenizerConfig.model_type
            ):
                return
            config_cls = f"{_CONFIG_PATH}.NCMScratchTokenizerConfig"
            cfg["reinit_weights"] = True
        elif stage == "generator":
            if cfg.get("reinit_weights"):
                raise ValueError(
                    "ncm.stage=generator loads its tokenizer from base_model; "
                    "reinit_weights would replace it with a random one."
                )
            config_cls = f"{_CONFIG_PATH}.NCMGeneratorConfig"
        else:
            return
        if cfg.get("cls_model_config") not in (None, config_cls):
            raise ValueError(f"ncm.stage={stage} loads its config through {config_cls}")
        cfg["cls_model_config"] = config_cls

    def get_trainer_cls(self, cfg: DictDefault):
        if cfg.ncm:
            from .trainer import NCMTrainer

            return NCMTrainer
        return None

    def add_callbacks_post_trainer(self, cfg: DictDefault, trainer):
        if not cfg.ncm or cfg.ncm.stage != "generator" or not cfg.ncm.sample_every:
            return []
        from .trainer import NCMSampleCallback

        return [
            NCMSampleCallback(
                trainer.processing_class,
                every=cfg.ncm.sample_every,
                num_samples=cfg.ncm.num_samples,
                temperature=cfg.ncm.sample_temperature,
            )
        ]
