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
        """Register SwanLab plugin with configuration and conflict detection."""
        LOG.info("Registering SwanLab plugin")

        # === Conflict Detection: Required Fields ===

        # Check if SwanLab is enabled
        if cfg.get("use_swanlab"):
            # 1. Validate project name is set
            if not cfg.get("swanlab_project"):
                raise ValueError(
                    "SwanLab enabled but 'swanlab_project' is not set.\n\n"
                    "Solutions:\n"
                    "  1. Add 'swanlab_project: your-project-name' to your config\n"
                    "  2. Set 'use_swanlab: false' to disable SwanLab\n\n"
                    "See: src/axolotl/integrations/swanlab/README.md for examples"
                )

            # 2. Validate swanlab_mode value
            valid_modes = ["cloud", "local", "offline", "disabled"]
            mode = cfg.get("swanlab_mode")
            if mode and mode not in valid_modes:
                raise ValueError(
                    f"Invalid swanlab_mode: '{mode}'.\n\n"
                    f"Valid options: {', '.join(valid_modes)}\n\n"
                    f"Example:\n"
                    f"  swanlab_mode: cloud  # Sync to SwanLab cloud\n"
                    f"  swanlab_mode: local  # Local only, no cloud sync\n"
                )

            # 3. Check API key for cloud mode
            import os
            mode = cfg.get("swanlab_mode", "cloud")  # Default is cloud
            if mode == "cloud":
                api_key = cfg.get("swanlab_api_key") or os.environ.get("SWANLAB_API_KEY")
                if not api_key:
                    LOG.warning(
                        "SwanLab cloud mode enabled but no API key found.\n"
                        "SwanLab may fail to initialize during training.\n\n"
                        "Solutions:\n"
                        "  1. Set SWANLAB_API_KEY environment variable:\n"
                        "     export SWANLAB_API_KEY=your-api-key\n"
                        "  2. Add 'swanlab_api_key: your-api-key' to config (less secure)\n"
                        "  3. Run 'swanlab login' before training\n"
                        "  4. Use 'swanlab_mode: local' for offline tracking\n"
                    )

        # === Conflict Detection: Multi-Logger Performance Warning ===

        # Detect all active logging tools
        active_loggers = []
        if cfg.get("use_wandb"):
            active_loggers.append("WandB")
        if cfg.get("use_mlflow"):
            active_loggers.append("MLflow")
        if cfg.get("comet_api_key") or cfg.get("comet_project_name"):
            active_loggers.append("Comet")
        if cfg.get("use_swanlab"):
            active_loggers.append("SwanLab")

        if len(active_loggers) > 1:
            LOG.warning(
                f"\n{'='*70}\n"
                f"Multiple logging tools enabled: {', '.join(active_loggers)}\n"
                f"{'='*70}\n"
                f"This may cause:\n"
                f"  - Performance overhead (~1-2% per logger, cumulative)\n"
                f"  - Increased memory usage\n"
                f"  - Longer training time per step\n"
                f"  - Potential config/callback conflicts\n\n"
                f"Recommendations:\n"
                f"  - Choose ONE primary logging tool for production training\n"
                f"  - Use multiple loggers only for:\n"
                f"    * Migration period (transitioning between tools)\n"
                f"    * Short comparison runs\n"
                f"    * Debugging specific tool issues\n"
                f"  - Monitor system resources (CPU, memory) during training\n"
                f"{'='*70}\n"
            )

            if len(active_loggers) >= 3:
                LOG.error(
                    f"\n{'!'*70}\n"
                    f"WARNING: {len(active_loggers)} logging tools enabled simultaneously!\n"
                    f"{'!'*70}\n"
                    f"This is likely unintentional and WILL significantly impact performance.\n"
                    f"Expected overhead: ~{len(active_loggers) * 1.5:.1f}% per training step.\n\n"
                    f"STRONGLY RECOMMEND:\n"
                    f"  - Disable all but ONE logging tool\n"
                    f"  - Use config inheritance to manage multiple configs\n"
                    f"{'!'*70}\n"
                )

        # === Auto-Enable Logic ===

        # Enable SwanLab if project is specified
        if cfg.get("swanlab_project") and not cfg.get("use_swanlab"):
            cfg["use_swanlab"] = True
            LOG.info("Automatically enabled use_swanlab because swanlab_project is set")

    def pre_model_load(self, cfg: DictDefault):
        """Initialize SwanLab before model loading with runtime checks."""
        if not cfg.use_swanlab:
            return

        # === Runtime Check: Import Availability ===
        try:
            import swanlab
        except ImportError as err:
            raise ImportError(
                "SwanLab is not installed.\n\n"
                "Install with:\n"
                "  pip install swanlab\n\n"
                "Or add to requirements:\n"
                "  swanlab>=0.3.0\n\n"
                f"Original error: {err}"
            ) from err

        # Log SwanLab version
        try:
            swanlab_version = swanlab.__version__
            LOG.info(f"SwanLab version: {swanlab_version}")
        except AttributeError:
            LOG.warning("Could not determine SwanLab version")

        # === Runtime Check: Distributed Training Setup ===
        from axolotl.utils.distributed import is_main_process, get_world_size

        world_size = get_world_size()
        if world_size > 1:
            mode = getattr(cfg, "swanlab_mode", "cloud")
            LOG.info(
                f"\n{'='*70}\n"
                f"Distributed training detected (world_size={world_size})\n"
                f"SwanLab mode: {mode}\n"
                f"{'='*70}\n"
                f"Behavior:\n"
                f"  - Only rank 0 will initialize SwanLab\n"
                f"  - Other ranks will skip SwanLab to avoid conflicts\n"
            )

            if mode == "cloud":
                LOG.info(
                    f"  - Only rank 0 will upload to SwanLab cloud\n"
                    f"  - Other ranks run without SwanLab overhead\n"
                    f"{'='*70}\n"
                )

        # Only initialize SwanLab on the main process (rank 0)
        # to avoid creating multiple runs in distributed training
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

