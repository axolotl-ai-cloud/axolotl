# pylint: skip-file

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Hymba-1.5B-Base", trust_remote_code=True
)

repr(model.config)
