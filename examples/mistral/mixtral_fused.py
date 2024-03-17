import torch
from tqdm import tqdm
from axolotl.monkeypatch.moe.moe import SparseMoeBlock
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

def compute_memory_used_pct(device):
    memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)
    memory_pct = (
        memory_used
        / (torch.cuda.get_device_properties(device).total_memory / (1024**3))
        * 100
    )
    return memory_pct

model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
modules = {k:v for k,v in model.named_modules() if isinstance(v, MixtralSparseMoeBlock)}

with tqdm(modules.items(), desc="scatter moe") as pbar:
    for name, module in pbar:
        smoe = SparseMoeBlock(
            experts=module.experts,
            gate=module.gate,
            hidden_dim=module.hidden_dim,
            ffn_dim=module.ffn_dim,
            num_experts=module.num_experts,
            top_k=module.top_k,
        )
        setattr(model, name, smoe)
        for device_index in range(torch.cuda.device_count()):
            device_memory_pct = compute_memory_used_pct(device_index)
            print(device_index, device_memory_pct)

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