from axolotl.monkeypatch.moe.moe import SparseMoeBlock
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

for name, module in model.named_modules():
    if isinstance(module, MixtralSparseMoeBlock):
        smoe = SparseMoeBlock(
            experts=module.experts,
            gate=module.gate,
            hidden_dim=module.hidden_dim,
            ffn_dim=module.ffn_dim,
            num_experts=module.num_experts,
            top_k=module.top_k,
        )
        setattr(model, name, smoe)

tokenizer = AutoTokenizer.from_pretrained(model_path)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = "[INST] {prompt} [/INST]"

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

tokens = tokenizer(
    prompt_template.format(prompt=prompt), 
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=512
)