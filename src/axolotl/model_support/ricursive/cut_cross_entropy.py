"""Cut Cross Entropy patch for the in-tree Ricursive model, reusing the fork's Llama forward."""

from types import MethodType

from transformers import PreTrainedModel


def patch_ricursive(maybe_model, patch_options, remote_model_id=None):
    """CCE ``PATCH_FNS`` entry; remote-code loads resolve to the in-tree class patched here."""
    del remote_model_id
    from cut_cross_entropy.transformers import llama as llama_patch

    from .modeling_ricursive import RicursiveForCausalLM

    llama_patch._PATCH_OPTS = patch_options

    if isinstance(maybe_model, PreTrainedModel):
        if not isinstance(maybe_model, RicursiveForCausalLM):
            raise ValueError(
                f"Expected RicursiveForCausalLM, got {type(maybe_model).__name__}"
            )
        maybe_model.forward = MethodType(llama_patch.cce_forward, maybe_model)
        return maybe_model

    RicursiveForCausalLM.forward = llama_patch.cce_forward
    return None
