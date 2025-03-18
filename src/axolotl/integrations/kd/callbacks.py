from transformers import TrainerCallback


class KDAlphaSchedulerCallback(TrainerCallback):
    """Callback to for scheduling KD alpha during training."""

    def on_epoch_begin(
        self, args, state, control, **kwargs  # pylint: disable=unused-argument
    ):
        if int(state.epoch) == 0:
            state.kd_alpha = args.kd_alpha
            state.kd_ce_alpha = args.kd_ce_alpha
        elif int(state.epoch) == state.num_train_epochs - 1:
            if args.kd_alpha_end is not None:
                control.kd_alpha = args.kd_alpha_end
            if args.kd_ce_alpha_end is not None:
                control.kd_ce_alpha = args.kd_ce_alpha_end
        else:
            epoch_steps = state.num_train_epochs - 1
            scale = int(state.epoch) / epoch_steps
            if args.kd_alpha_end is not None:
                control.kd_alpha = (
                    args.kd_alpha + (args.kd_alpha_end - args.kd_alpha) * scale
                )
            if args.kd_ce_alpha_end is not None:
                control.kd_ce_alpha = (
                    args.kd_ce_alpha + (args.kd_ce_alpha_end - args.kd_ce_alpha) * scale
                )
