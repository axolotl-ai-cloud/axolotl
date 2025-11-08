"""SwanLab Plugin for Axolotl"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.utils.dict import DictDefault

LOG = get_logger(__name__)


class SwanLabPlugin(BasePlugin):
    """
    SwanLab integration plugin for Axolotl.
    
    Provides experiment tracking, visualization, and logging capabilities
    using SwanLab (https://swanlab.cn).
    
    Usage in config.yaml:
        plugins:
          - axolotl.integrations.swanlab.SwanLabPlugin
        
        use_swanlab: true
        swanlab_project: my-project
        swanlab_experiment_name: my-experiment
        swanlab_mode: cloud  # or 'local', 'offline', 'disabled'
    """

    def __init__(self):
        super().__init__()
        self.swanlab_initialized = False
        LOG.info("SwanLab plugin initialized")

    def get_input_args(self) -> str:
        """Returns the configuration model for SwanLab integration."""
        return "axolotl.integrations.swanlab.SwanLabConfig"

    def register(self, cfg: dict):
        """Register SwanLab plugin with configuration."""
        LOG.info("Registering SwanLab plugin")

        # Enable SwanLab if project is specified
        if cfg.get("swanlab_project") and not cfg.get("use_swanlab"):
            cfg["use_swanlab"] = True
            LOG.info("Automatically enabled use_swanlab because swanlab_project is set")

    def pre_model_load(self, cfg: DictDefault):
        """Initialize SwanLab before model loading."""
        if not cfg.use_swanlab:
            return

        try:
            import swanlab
        except ImportError:
            LOG.error(
                "SwanLab is not installed. Install it with: pip install swanlab"
            )
            return

        # Only initialize SwanLab on the main process (rank 0)
        # to avoid creating multiple runs in distributed training
        from axolotl.utils.distributed import is_main_process

        if not is_main_process():
            LOG.debug(
                "Skipping SwanLab initialization on non-main process"
            )
            return

        # Setup SwanLab environment variables
        self._setup_swanlab_env(cfg)

        # Initialize SwanLab run
        try:
            init_kwargs = self._get_swanlab_init_kwargs(cfg)
            swanlab.init(**init_kwargs)
            self.swanlab_initialized = True
            LOG.info(f"SwanLab initialized with project: {cfg.swanlab_project}")

            # Log configuration (with error handling)
            try:
                config_dict = self._prepare_config_for_logging(cfg)
                swanlab.config.update(config_dict)
                LOG.debug("Successfully logged config to SwanLab")
            except Exception as config_err:  # pylint: disable=broad-except
                LOG.warning(f"Failed to log config to SwanLab: {config_err}. Continuing anyway.")

        except Exception as err:  # pylint: disable=broad-except
            LOG.error(f"Failed to initialize SwanLab: {err}")
            self.swanlab_initialized = False

    def add_callbacks_pre_trainer(self, cfg: DictDefault, model):
        """Add SwanLab callbacks before trainer creation."""
        callbacks = []

        if not cfg.use_swanlab:
            return callbacks

        if not self.swanlab_initialized:
            LOG.warning(
                "SwanLab not initialized, skipping callback registration"
            )
            return callbacks

        try:
            from axolotl.utils.callbacks.swanlab import (
                CustomSwanLabCallback,
                SaveAxolotlConfigtoSwanLabCallback,
            )

            # Add our custom lightweight SwanLabCallback
            # (avoids omegaconf/antlr4 version conflicts)
            swanlab_callback = CustomSwanLabCallback()
            callbacks.append(swanlab_callback)
            LOG.info("Added CustomSwanLabCallback for metrics logging")

            # Add Axolotl config logging callback
            if cfg.axolotl_config_path:
                config_callback = SaveAxolotlConfigtoSwanLabCallback(
                    cfg.axolotl_config_path
                )
                callbacks.append(config_callback)
                LOG.info("Added SaveAxolotlConfigtoSwanLabCallback")

        except ImportError as err:
            LOG.error(f"Failed to import SwanLab callbacks: {err}")

        return callbacks

    def post_trainer_create(self, cfg: DictDefault, trainer):
        """Post-trainer creation hook."""
        if cfg.use_swanlab and self.swanlab_initialized:
            try:
                import swanlab

                # Log additional trainer information (with safe conversion)
                trainer_config = {
                    "total_steps": int(trainer.state.max_steps) if trainer.state.max_steps else None,
                    "num_train_epochs": float(trainer.args.num_train_epochs) if trainer.args.num_train_epochs else None,
                    "train_batch_size": int(trainer.args.train_batch_size) if hasattr(trainer.args, 'train_batch_size') else None,
                    "gradient_accumulation_steps": int(trainer.args.gradient_accumulation_steps) if trainer.args.gradient_accumulation_steps else None,
                }
                # Remove None values
                trainer_config = {k: v for k, v in trainer_config.items() if v is not None}
                
                if trainer_config:
                    swanlab.config.update(trainer_config)
                    LOG.info("Logged trainer configuration to SwanLab")
            except Exception as err:  # pylint: disable=broad-except
                LOG.debug(f"Failed to log trainer config to SwanLab: {err}")

    def _setup_swanlab_env(self, cfg: DictDefault):
        """Setup SwanLab environment variables from config."""
        import os

        env_mapping = {
            "swanlab_api_key": "SWANLAB_API_KEY",
            "swanlab_mode": "SWANLAB_MODE",
            "swanlab_web_host": "SWANLAB_WEB_HOST",
            "swanlab_api_host": "SWANLAB_API_HOST",
        }

        for cfg_key, env_key in env_mapping.items():
            value = getattr(cfg, cfg_key, None)
            if value and isinstance(value, str) and len(value) > 0:
                os.environ[env_key] = value
                LOG.debug(f"Set environment variable {env_key}")

    def _get_swanlab_init_kwargs(self, cfg: DictDefault) -> dict:
        """Prepare kwargs for swanlab.init()."""
        init_kwargs = {}

        # Project name (required)
        if cfg.swanlab_project:
            init_kwargs["project"] = cfg.swanlab_project

        # Experiment name
        if cfg.swanlab_experiment_name:
            init_kwargs["experiment_name"] = cfg.swanlab_experiment_name

        # Description
        if cfg.swanlab_description:
            init_kwargs["description"] = cfg.swanlab_description

        # Workspace (organization)
        if cfg.swanlab_workspace:
            init_kwargs["workspace"] = cfg.swanlab_workspace

        # Mode: cloud, local, offline, disabled
        if cfg.swanlab_mode:
            init_kwargs["mode"] = cfg.swanlab_mode

        # Log model checkpoints (coming soon in SwanLab)
        if cfg.swanlab_log_model:
            init_kwargs["log_model"] = cfg.swanlab_log_model

        return init_kwargs

    def _prepare_config_for_logging(self, cfg: DictDefault) -> dict:
        """Prepare configuration dict for logging to SwanLab."""
        def safe_convert(value):
            """Convert value to JSON-serializable type."""
            if value is None:
                return None
            if isinstance(value, (int, float, bool)):
                return value
            if isinstance(value, str):
                return value
            # Convert everything else to string
            return str(value)
        
        try:
            # Extract important training parameters with safe conversion
            config_dict = {
                "base_model": safe_convert(getattr(cfg, "base_model", "")),
                "model_type": safe_convert(getattr(cfg, "model_type", "")),
                "sequence_len": safe_convert(getattr(cfg, "sequence_len", None)),
                "micro_batch_size": safe_convert(getattr(cfg, "micro_batch_size", None)),
                "gradient_accumulation_steps": safe_convert(getattr(cfg, "gradient_accumulation_steps", None)),
                "num_epochs": safe_convert(getattr(cfg, "num_epochs", None)),
                "max_steps": safe_convert(getattr(cfg, "max_steps", None)),
                "learning_rate": safe_convert(getattr(cfg, "learning_rate", None)),
                "lr_scheduler": safe_convert(getattr(cfg, "lr_scheduler", "")),
                "optimizer": safe_convert(getattr(cfg, "optimizer", "")),
                "warmup_ratio": safe_convert(getattr(cfg, "warmup_ratio", None)),
                "weight_decay": safe_convert(getattr(cfg, "weight_decay", None)),
                "seed": safe_convert(getattr(cfg, "seed", None)),
                "bf16": safe_convert(getattr(cfg, "bf16", None)),
                "tf32": safe_convert(getattr(cfg, "tf32", None)),
                "flash_attention": safe_convert(getattr(cfg, "flash_attention", None)),
                "sample_packing": safe_convert(getattr(cfg, "sample_packing", None)),
            }

            # Add FSDP/parallel config - only boolean flags
            if hasattr(cfg, "fsdp_config") and cfg.fsdp_config:
                config_dict["fsdp_enabled"] = True
                config_dict["fsdp_version"] = safe_convert(getattr(cfg, "fsdp_version", None))

            if hasattr(cfg, "deepspeed") and cfg.deepspeed:
                config_dict["deepspeed_enabled"] = True

            # Add context parallel info
            if hasattr(cfg, "context_parallel_size"):
                config_dict["context_parallel_size"] = safe_convert(getattr(cfg, "context_parallel_size", None))
            if hasattr(cfg, "tensor_parallel_size"):
                config_dict["tensor_parallel_size"] = safe_convert(getattr(cfg, "tensor_parallel_size", None))
            if hasattr(cfg, "dp_shard_size"):
                config_dict["dp_shard_size"] = safe_convert(getattr(cfg, "dp_shard_size", None))

            # Remove None values and empty strings
            config_dict = {
                k: v for k, v in config_dict.items() 
                if v is not None and v != "" and v != "None"
            }

            return config_dict
        except Exception as err:  # pylint: disable=broad-except
            LOG.warning(f"Failed to prepare config for logging: {err}")
            # Return minimal config
            return {
                "base_model": str(getattr(cfg, "base_model", "unknown")),
                "learning_rate": float(getattr(cfg, "learning_rate", 0.0)) if getattr(cfg, "learning_rate", None) else None,
            }

