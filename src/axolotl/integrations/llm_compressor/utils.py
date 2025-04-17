from transformers import Trainer

def save_compressed_model(
        model, output_dir, trainer: Trainer, safe_serialization: bool, save_compressed:bool
):
    from llmcompressor.transformers.sparsification.compressed_tensors_utils import modify_save_pretrained
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        modify_save_pretrained(model)
        model.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization,
            save_compressed=save_compressed,
            skip_sparsity_compression_stats=not save_compressed,
        )