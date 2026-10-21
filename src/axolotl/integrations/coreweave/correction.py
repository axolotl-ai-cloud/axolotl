"""Enable pinned TRL's sampler correction without starting its vLLM backend."""

from functools import wraps


def install_sampler_correction(trainer):
    """Scope TRL's vLLM correction gate to scoring and loss, excluding generation."""
    if trainer.use_vllm:
        raise ValueError("CoreWeave requires TRL's vLLM backend to be disabled")
    if getattr(trainer, "_coreweave_correction_installed", False):
        return
    if trainer.loss_type == "vespo" and trainer.vllm_importance_sampling_correction:
        raise ValueError("VESPO requires vllm_importance_sampling_correction: false")
    if trainer.use_liger_kernel:
        raise ValueError(
            "CoreWeave sampler correction currently requires use_liger_kernel: false"
        )

    def scoped(method, enabled):
        @wraps(method)
        def call(*args, **kwargs):
            previous = trainer.use_vllm
            # TRL 1.14.1 gates both correction and backend lifecycle on this flag.
            trainer.use_vllm = enabled
            try:
                return method(*args, **kwargs)
            finally:
                trainer.use_vllm = previous

        return call

    trainer._generate = scoped(trainer._generate, False)
    trainer._generate_and_score_completions = scoped(
        trainer._generate_and_score_completions, True
    )
    trainer._compute_loss = scoped(trainer._compute_loss, True)
    trainer._coreweave_correction_installed = True
