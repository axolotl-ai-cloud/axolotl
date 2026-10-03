"""Compatibility exports for diffusion generation callbacks."""

from axolotl.integrations.diffusion.lm.callbacks import (
    DiffusionGenerationCallback as _CoreDiffusionGenerationCallback,
)
from axolotl.integrations.diffusion.lm.generation import generate_samples


class DiffusionGenerationCallback(_CoreDiffusionGenerationCallback):
    """Legacy callback retaining its module-level sample-generation seam."""

    def _generate_samples(self, **kwargs):
        from axolotl.model_support.native_generation import (
            generate_native_samples,
            uses_native_generation,
        )

        if uses_native_generation(kwargs["model"]):
            return generate_native_samples(**kwargs)
        return generate_samples(**kwargs)


__all__ = ["DiffusionGenerationCallback", "generate_samples"]
