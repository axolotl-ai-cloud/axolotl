"""Helpers for diffusion-mode inference in CLI and Gradio."""

from __future__ import annotations

from colorama import Fore, Style

from axolotl.integrations.diffusion import generate, resolve_mask_token_id
from axolotl.utils.dict import DictDefault


def diffusion_inference(
    model,
    tokenizer,
    cfg,
    prompt: str,
    chat_template_str: str | None = None,
):
    """Diffusion inference helper method."""
    mode = "random"
    completion_tokens = 0
    target_mask_ratio = None
    mode, completion_tokens, target_mask_ratio, cleaned = _parse_commands(prompt)

    if cleaned:
        prompt = cleaned

    info = _run_diffusion(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        prompt=prompt,
        chat_template_str=chat_template_str,
        mode=mode,
        target_mask_ratio=target_mask_ratio,
        completion_tokens=completion_tokens,
    )
    masked_text = info["masked_text"]
    mask_ratio = info["mask_ratio"]
    generated_ids = info["generated_ids"]
    masked_positions = info["masked_positions"]
    orig_ids = info["orig_ids"]

    # Display with masked preview and colored diff
    if masked_text is not None and mask_ratio is not None:
        print(f"Masked ({mask_ratio:.1%}):\n{masked_text}\n")
    if generated_ids is not None:
        # Compute per-token style
        styles: list[str] = []
        for i, tid in enumerate(generated_ids):
            if i in masked_positions:
                if i < len(orig_ids) and tid == orig_ids[i]:
                    styles.append("green")  # correct fill
                elif i < len(orig_ids):
                    styles.append("red")  # incorrect fill
                else:
                    styles.append("normal")  # appended
            else:
                same = i < len(orig_ids) and tid == orig_ids[i]
                styles.append("dim" if same else "normal")

        # Group contiguous spans by style
        styled_spans: list[tuple[str, int, int]] = []
        if generated_ids:
            current_style = styles[0]
            start = 0
            for i in range(1, len(generated_ids)):
                s = styles[i]
                if s != current_style:
                    styled_spans.append((current_style, start, i))
                    current_style, start = s, i
            styled_spans.append((current_style, start, len(generated_ids)))

        out_parts = []
        for style_name, a, b in styled_spans:
            chunk_text = tokenizer.decode(generated_ids[a:b], skip_special_tokens=False)
            if style_name == "green":
                out_parts.append(Fore.GREEN + chunk_text + Style.RESET_ALL)
            elif style_name == "red":
                out_parts.append(Fore.RED + chunk_text + Style.RESET_ALL)
            else:
                if style_name == "dim":
                    out_parts.append(Style.DIM + chunk_text + Style.RESET_ALL)
                else:
                    out_parts.append(chunk_text)
        print("Generated:\n" + "".join(out_parts))
    else:
        print("Generated:\n(no output)")


def _parse_commands(text: str):
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


def _run_diffusion(
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
    gen_mode = "completion" if mode == "completion" else "random"
    comp_tokens = int(completion_tokens) if gen_mode == "completion" else 0

    result = generate(
        model,
        tokenizer,
        original_sequence=seq[:1],
        num_diffusion_steps=cfg.diffusion.num_diffusion_steps,
        temperature=cfg.diffusion.generation_temperature,
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
