import torch
from transformers.utils import logging
import transformers.models.llama.modeling_llama

logger = logging.get_logger(__name__)

def replace_llama_embeddings_with_uniform_distribution(noise_alpha=5):
    def noised_embed(orig_embed, noise_alpha, model):
        def new_func(x):
            # during training, we add noise to the embedding
            # during generation, we don't add noise to the embedding
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha/torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
            else:
                return orig_embed(x)
        return new_func
    
    def post_init(orig_post_init):
        def new_func(self):
            orig_post_init(self)
            self.embed_tokens.forward = noised_embed(self.embed_tokens.forward, noise_alpha, self)
        return new_func

    transformers.models.llama.modeling_llama.LlamaModel.post_init = post_init(transformers.models.llama.modeling_llama.LlamaModel.post_init)