"""Helpers for diffusion-mode inference in CLI and Gradio."""

from __future__ import annotations

from axolotl.integrations.diffusion.generation import generate
from axolotl.integrations.diffusion.utils import resolve_mask_token_id
from axolotl.utils.dict import DictDefault


def parse_commands(text: str):
    """
    Parse leading diffusion commands.

    Supported at start of input (can be chained):
      :complete N  -> completion mode with N tokens (default 64)
      :mask R      -> random masking with ratio R in [0, 1]
    """
    tokens = text.strip().split()
    i = 0
    mode = "random"
    completion_tokens = 0
    target_mask_ratio = None
    consumed = 0
    while i < len(tokens) and tokens[i].startswith(":"):
        cmd = tokens[i]
        i += 1
        consumed = i
        if cmd == ":complete":
            mode = "completion"
            if i < len(tokens):
                try:
                    completion_tokens = int(tokens[i])
                    i += 1
                    consumed = i
                except Exception:
                    completion_tokens = 64
            else:
                completion_tokens = 64
        elif cmd == ":mask":
            mode = "random"
            if i < len(tokens):
                try:
                    target_mask_ratio = float(tokens[i])
                    i += 1
                    consumed = i
                except Exception:
                    target_mask_ratio = None
        else:
            i -= 1
            consumed = i
            break

    cleaned = " ".join(tokens[consumed:])
    return mode, completion_tokens, target_mask_ratio, cleaned


def run_diffusion(
    *,
    model,
    tokenizer,
    cfg: DictDefault,
    prompt: str,
    chat_template_str: str | None,
    mode: str = "random",
    target_mask_ratio: float | None = None,
    completion_tokens: int = 0,
):
    """Run a single diffusion generation and return a structured result dict."""
    if chat_template_str:
        batch = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_special_tokens=True,
            add_generation_prompt=True,
            chat_template=chat_template_str,
            tokenize=True,
            return_dict=True,
        )
    else:
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    mask_token_id = resolve_mask_token_id(tokenizer, cfg, allow_add=False)

    seq = batch["input_ids"].to(cfg.device)
    gen_mode = "random" if mode == "mask" else "completion"
    comp_tokens = int(completion_tokens) if gen_mode == "completion" else 0

    result = generate(
        model,
        tokenizer,
        original_sequence=seq[:1],
        num_diffusion_steps=cfg.diffusion_num_diffusion_steps,
        temperature=cfg.diffusion_generation_temperature,
        mask_token_id=int(mask_token_id),
        mode=gen_mode,  # type: ignore[arg-type]
        completion_tokens=comp_tokens,
        target_mask_ratio=target_mask_ratio,
    )

    masked_text = result.get("masked") if isinstance(result, dict) else None
    mask_ratio = result.get("mask_ratio") if isinstance(result, dict) else None
    generated_ids = result.get("generated_ids") if isinstance(result, dict) else None
    masked_positions = (
        set(result.get("masked_positions") or []) if isinstance(result, dict) else set()
    )
    orig_ids = seq[0].detach().cpu().tolist()

    return {
        "masked_text": masked_text,
        "mask_ratio": mask_ratio,
        "generated_ids": generated_ids,
        "masked_positions": masked_positions,
        "orig_ids": orig_ids,
    }
