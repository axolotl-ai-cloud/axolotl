"""
Transformers trainer callbacks to schedule the KD temperature during training
"""

import math

from transformers.trainer_callback import TrainerCallback


class KDTemperatureSchedulerCallback(TrainerCallback):
    """
    KD temperature scheduler callback for the trainer.
    """

    def __init__(self, temperature_start, temperature_min, trainer):
        self.temperature_start = temperature_start
        self.temperature_min = temperature_min
        self.temperature = temperature_start

        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        # cosine decay temperature over the max steps

        progress = state.global_step / state.max_steps
        # Cosine decay factor: 0.5 * (1 + cos(pi * progress))
        # This factor goes from 1 (at progress=0) to 0 (at progress=1)
        decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.temperature = self.temperature_start - (
            (self.temperature_start - self.temperature_min) * (1.0 - decay_factor)
        )

        if hasattr(self.trainer.data_collator, "kd_temperature"):
            self.trainer.data_collator.kd_temperature = self.temperature
