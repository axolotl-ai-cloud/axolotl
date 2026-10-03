"""CPU proof for reserved151 initialization before PEFT modules-to-save cloning."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PretrainedConfig

from axolotl.integrations.decision.template import (
    RESERVED151_TOKEN_IDS,
)


class TinyUntiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.embed_tokens = torch.nn.Embedding(512, 8)
        self.diffusion_head = torch.nn.Linear(8, 512, bias=False)
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.config = PretrainedConfig()
        with torch.no_grad():
            self.encoder.embed_tokens.weight[200:351].fill_(0.001)
            self.diffusion_head.weight[200:351].fill_(0.002)

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def get_output_embeddings(self):
        return self.diffusion_head

    def forward(self, input_ids):
        hidden = self.q_proj(self.encoder.embed_tokens(input_ids))
        return type("Output", (), {"logits": self.diffusion_head(hidden)})()


def test_reserved151_sparse_input_and_saved_head_survive_gradient_and_reload(tmp_path):
    model = TinyUntiedModel()
    input_before = model.encoder.embed_tokens.weight.detach().clone()
    output_before = model.diffusion_head.weight[200].detach().clone()

    peft = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj"],
            trainable_token_indices=list(RESERVED151_TOKEN_IDS),
            modules_to_save=["diffusion_head"],
            task_type=None,
        ),
    )
    trainable = dict(peft.named_parameters())
    delta = next(
        value for name, value in trainable.items() if "trainable_tokens_delta" in name
    )
    head = peft.get_output_embeddings()
    optimizer = torch.optim.SGD(peft.parameters(), lr=0.1)
    loss = peft(torch.tensor([[200]])).logits[..., 200].sum()
    loss.backward()
    optimizer.step()
    assert delta.grad is not None
    assert head.weight.grad is not None
    assert not torch.equal(delta[0], torch.zeros_like(delta[0]))
    assert torch.equal(model.encoder.embed_tokens.weight[199], input_before[199])
    assert not torch.equal(head.weight[200], output_before)

    adapter = tmp_path / "adapter"
    peft.save_pretrained(adapter)
    restored = TinyUntiedModel()
    reloaded = PeftModel.from_pretrained(restored, adapter)
    restored_delta = next(
        value
        for name, value in reloaded.named_parameters()
        if "trainable_tokens_delta" in name
    )
    assert torch.equal(restored_delta, delta)
    assert torch.equal(reloaded.get_output_embeddings().weight[200], head.weight[200])
    assert RESERVED151_TOKEN_IDS[-1] == 350
