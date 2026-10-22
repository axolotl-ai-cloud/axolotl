"""Next-Concept Modeling plugin."""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault

from . import loading
from .configuration import NCMGeneratorConfig, NCMTokenizerConfig

_CONFIG_PATH = "axolotl.integrations.ncm.configuration"


class NCMPlugin(BasePlugin):
    """Two-stage NCM training: ``ncm.stage: tokenizer`` then ``ncm.stage: generator``."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.ncm.NCMArgs"

    def register(self, cfg: dict):
        ncm = cfg.get("ncm") or {}
        stage = ncm.get("stage") if isinstance(ncm, dict) else None
        if stage not in ("tokenizer", "generator"):
            return
        for key in ("sample_packing", "fsdp", "fsdp_config", "deepspeed"):
            if cfg.get(key):
                raise ValueError(f"ncm does not support {key} yet")

        base, text_tok = cfg.get("base_model"), cfg.get("tokenizer_config")
        source = base or text_tok
        if not source:
            return
        model_type = loading.checkpoint_model_type(
            source, revision=cfg.get("revision_of_model")
        )
        if stage == "tokenizer":
            if model_type == NCMTokenizerConfig.model_type:
                return
            if base and base != text_tok:
                raise ValueError(
                    f"ncm.stage=tokenizer: base_model {base!r} is not an NCM tokenizer "
                    f"checkpoint (model_type={model_type}). Leave base_model unset to "
                    "train one from scratch over tokenizer_config."
                )
            cfg["base_model"] = source
            cfg["reinit_weights"] = True
            config_cls = f"{_CONFIG_PATH}.NCMScratchTokenizerConfig"
        else:
            if model_type not in (
                NCMTokenizerConfig.model_type,
                NCMGeneratorConfig.model_type,
            ):
                raise ValueError(
                    f"ncm.stage=generator: base_model {source!r} must be an NCM tokenizer "
                    f"or generator checkpoint (got model_type={model_type})."
                )
            if cfg.get("reinit_weights"):
                raise ValueError(
                    "ncm.stage=generator loads its tokenizer from base_model; "
                    "reinit_weights would replace it with a random one."
                )
            config_cls = f"{_CONFIG_PATH}.NCMGeneratorConfig"
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
