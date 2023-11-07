"""
patch to add noisy embeddings per https://arxiv.org/abs/2310.05914
"""

import torch
import transformers.models.mistral.modeling_mistral
from transformers.utils import logging

logger = logging.get_logger(__name__)


def replace_mistral_embeddings_with_uniform_distribution(noise_alpha=5):
    # pylint: disable=duplicate-code
    def noised_embed(orig_embed, noise_alpha, model):
        def new_func(input_ids):
            # during training, we add noise to the embedding
            # during generation, we don't add noise to the embedding
            if model.training:
                embed_init = orig_embed(input_ids)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha / torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(
                    -mag_norm, mag_norm
                )
            return orig_embed(input_ids)

        return new_func

    def post_init(orig_post_init):
        def new_func(self):
            orig_post_init(self)
            self.embed_tokens.forward = noised_embed(
                self.embed_tokens.forward, noise_alpha, self
            )

        return new_func

    transformers.models.mistral.modeling_mistral.MistralModel.post_init = post_init(
        transformers.models.mistral.modeling_mistral.MistralModel.post_init
    )
