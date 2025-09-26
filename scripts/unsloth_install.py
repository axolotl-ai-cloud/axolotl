"""Print the uv commands required to install Unsloth."""

UNSLOTH_BASE = "uv pip install --system unsloth-zoo==2025.9.12"
UNSLOTH_HF = 'uv pip install --system --no-deps "unsloth[huggingface]==2025.9.9"'

print(f"{UNSLOTH_BASE} && {UNSLOTH_HF}")
