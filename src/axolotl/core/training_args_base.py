"""
Base Axolotl Training Mixins shared across various trainer configs
"""

from dataclasses import dataclass, field
from typing import Optional

from PIL.Image import Resampling


@dataclass
class AxolotlTrainingMixins:
    """
    Mixin class for the Axolotl training args.
    """

    model_type: Optional[str] = field(
        default=None, metadata={"help": "HF model configuration model_type."}
    )
    lr_quadratic_warmup: bool = field(
        default=False,
        metadata={"help": "Use quadratic warmup for cosine scheduling."},
    )
    pretraining: bool = field(
        default=False,
        metadata={
            "help": "Indicates to trainer whether we are doing continued pretraining."
        },
    )
    sample_packing: bool = field(
        default=False,
        metadata={"help": "Use sample packing for efficient training."},
    )
    sample_packing_sequentially: bool = field(
        default=False,
        metadata={
            "help": "Use next-fit sample packing that preserves the order of samples coming from the sampler. Use in combination with curriculum_sampling for fully sequential packing."
        },
    )
    sample_packing_mp_start_method: str | None = field(
        default=None,
        metadata={"help": "The multiprocessing start method to use."},
    )
    sample_packing_drop_attention_mask: bool = field(
        default=False,
        metadata={"help": "Drop attention mask from inputs when using packing."},
    )
    multipack_real_batches: bool = field(
        default=False,
        metadata={"help": "Use real batches for efficient training."},
    )
    include_tkps: bool = field(
        default=True,
        metadata={
            "help": "Whether to include tokens per second in the training metrics."
        },
    )
    eval_sample_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Use sample packing for efficient evals."},
    )
    sample_packing_efficiency: float = field(
        default=1.0,
        metadata={"help": "Sample packing efficiency for calculating batch length."},
    )
    sample_packing_bin_size: int = field(
        default=200,
        metadata={
            "help": "The max number of samples that packed sample can contain after packing. Increase for better packing."
        },
    )
    sample_packing_group_size: int = field(
        default=100000,
        metadata={
            "help": "The number of samples to group together for packing. Increase for better packing."
        },
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length the model can handle"},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for data processing"},
    )
    relora_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how often to reset for ReLoRA"},
    )
    relora_prune_ratio: Optional[float] = field(
        default=0.9,
        metadata={"help": "prune ratio for magnitude pruning of the optimizer"},
    )
    jagged_restart_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how often to reset for jagged restarts"},
    )
    jagged_restart_warmup_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "how many warmup steps to take after reset for jagged restarts"
        },
    )
    jagged_restart_anneal_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "how many anneal steps to take before reset for jagged restarts"
        },
    )
    bench_split: Optional[str] = field(
        default="eval", metadata={"help": "The benchmark split to run on"}
    )
    bench_dataset: Optional[str] = field(
        default="pharaouk/dharma-1/dharma_1_mini.json",
        metadata={
            "help": "Benchmark dataset to use: options are `mmlu-zs`, `mmlu-fs`, or the full path to the dataset file"
        },
    )
    do_bench_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the Benchmark evaluation."}
    )
    do_causal_lm_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the Causal LM evaluation."}
    )
    max_bench_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, only evaluates on `max_bench_samples` of the benchmark dataset."
        },
    )
    bench_source_max_len: int = field(
        default=2048, metadata={"help": "Maximum source sequence length for bench."}
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={"help": "prefetch_factor argument to the dataloader"},
    )
    cosine_min_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum learning rate is min_lr_ratio * learning_rate"},
    )
    cosine_constant_lr_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "Starting constant learning rate step is cosine_constant_lr_ratio * max_steps"
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=1e-6,
        metadata={"help": "loraplus learning rate for lora embedding layers."},
    )
    embedding_lr_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Scale the learning rate for the embedding layers."},
    )
    lr_groups: Optional[list[dict]] = field(
        default=None,
        metadata={"help": "Specify learning rate groups for with different LRs."},
    )
    embedding_lr: Optional[float] = field(
        default=None,
        metadata={"help": "absolute learning rate for the embedding layers."},
    )
    qlora: bool = field(
        default=False,
        metadata={"help": "whether this is a qlora training"},
    )
    orpo_alpha: Optional[float] = field(
        default=None,
    )
    lisa_n_layers: Optional[int] = field(
        default=None,
        metadata={"help": "the number of activate layers in LISA"},
    )
    lisa_step_interval: Optional[int] = field(
        default=None,
        metadata={"help": "how often to switch layers in LISA"},
    )
    lisa_layers_attribute: Optional[str] = field(
        default=None,
        metadata={"help": "path under the model to access the layers"},
    )
    curriculum_sampling: Optional[bool] = field(
        default=None,
        metadata={"help": "whether to use sequential sampling for curriculum learning"},
    )
    alternate_lr_scheduler_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "workaround to pass an alternate lr scheduler to the HF trainer"
        },
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "Chat template converting chat messages to text"},
    )

    # kd_ce_alpha: Optional[float] = field(
    #     default=None,
    #     metadata={
    #         "help": "The alpha scaling parameter for SFT cross entropy loss when using KD"
    #     },
    # )
    #
    # kd_alpha: Optional[float] = field(
    #     default=1.0,
    #     metadata={"help": "The alpha scaling parameter for KD loss"},
    # )
    #
    # kd_temperature: Optional[float] = field(
    #     default=1.0,
    #     metadata={
    #         "help": "the temperature parameter for KL divergence loss when using KD"
    #     },
    # )

    adam_beta3: Optional[float] = field(
        default=None,
        metadata={
            "help": "The beta3 hyperparameter used in some optimizers such as CAME"
        },
    )
    adam_epsilon2: Optional[float] = field(
        default=None,
        metadata={
            "help": "The epsilon2 hyperparameter used in some optimizers such as CAME"
        },
    )

    activation_offloading: bool | None = field(
        default=None,
        metadata={"help": "Use activation offloading with CUDA streams for training."},
    )

    # multi-modal section

    image_size: int | tuple[int, int] | None = field(
        default=None,
        metadata={"help": "The size of the image to resize to"},
    )

    image_resize_algorithm: Resampling | None = field(
        default=None,
        metadata={"help": "The algorithm to use for image resizing"},
    )

    # end of multi-modal section

    dion_learning_rate: float | None = field(
        default=None,
        metadata={"help": "The learning rate for Dion"},
    )
    dion_momentum: float | None = field(
        default=None,
        metadata={"help": "The momentum for Dion"},
    )
    dion_rank_fraction: float | None = field(
        default=None,
    )
    dion_rank_multiple_of: int | None = field(
        default=None,
    )
