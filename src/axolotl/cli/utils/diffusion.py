"""Helpers for diffusion-mode inference in CLI and Gradio."""

from __future__ import annotations

from axolotl.integrations.diffusion.generation import generate as diffusion_generate
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


def infer_mask_token_id(tokenizer, cfg: DictDefault) -> int:
    """Infer mask token id from tokenizer/config, preferring a saved diffusion token."""
    mts = cfg.get("mask_token_str")
    if mts:
        try:
            tid = tokenizer.convert_tokens_to_ids(mts)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            specials = (getattr(tokenizer, "all_special_tokens", []) or []) + (
                getattr(tokenizer, "additional_special_tokens", []) or []
            )
            if (
                isinstance(tid, int)
                and tid >= 0
                and (unk_id is None or tid != unk_id)
                and mts in specials
            ):
                return tid
        except Exception:  # pragma: no cover
            pass
    mid = cfg.get("mask_token_id")
    if isinstance(mid, int):
        try:
            if 0 <= mid < len(tokenizer):
                return mid
        except Exception:  # pragma: no cover
            pass
    specials = (getattr(tokenizer, "all_special_tokens", []) or []) + (
        getattr(tokenizer, "additional_special_tokens", []) or []
    )

    def _ok(s: str) -> bool:
        low = s.lower()
        if any(
            bad in low
            for bad in [
                "im_start",
                "im_end",
                "bos",
                "eos",
                "pad",
                "system",
                "assistant",
                "user",
            ]
        ):
            return False
        return ("diffusion" in low) or ("mask" in low)

    for s in specials:
        if _ok(s):
            token_id = tokenizer.convert_tokens_to_ids(s)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if (
                isinstance(token_id, int)
                and token_id >= 0
                and (unk_id is None or token_id != unk_id)
            ):
                return token_id

    return getattr(tokenizer, "unk_token_id", 0) or 0


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

    steps = cfg.get("generation_steps") or cfg.get("num_diffusion_steps") or 128
    temperature = cfg.get("generation_temperature", 0.0)
    mask_token_id = infer_mask_token_id(tokenizer, cfg)

    seq = batch["input_ids"].to(cfg.device)
    gen_mode = "random" if mode == "mask" else "completion"
    comp_tokens = int(completion_tokens) if gen_mode == "completion" else 0

    result = diffusion_generate(
        model,
        tokenizer,
        original_sequence=seq[:1],
        num_diffusion_steps=int(steps),
        temperature=float(temperature),
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


def render_html(
    generated_ids: list[int] | None,
    orig_ids: list[int],
    masked_positions: set[int],
    tokenizer,
) -> str:
    """Render HTML with colored spans for diffusion correctness visualization."""
    if not generated_ids:
        return "<pre>Generated: (no output)</pre>"

    def _style_for(i: int, tid: int) -> str:
        if i in masked_positions:
            if i < len(orig_ids) and tid == orig_ids[i]:
                return "green"
            if i < len(orig_ids):
                return "red"
            return "normal"
        same = i < len(orig_ids) and tid == orig_ids[i]
        return "dim" if same else "normal"

    spans: list[tuple[str, int, int]] = []
    if generated_ids:
        cur = _style_for(0, generated_ids[0])
        start = 0
        for i in range(1, len(generated_ids)):
            s = _style_for(i, generated_ids[i])
            if s != cur:
                spans.append((cur, start, i))
                cur, start = s, i
        spans.append((cur, start, len(generated_ids)))

    html_parts = []
    for style_name, a, b in spans:
        txt = tokenizer.decode(generated_ids[a:b], skip_special_tokens=False)
        if style_name == "green":
            html_parts.append(f'<span style="color:#2e7d32">{txt}</span>')
        elif style_name == "red":
            html_parts.append(f'<span style="color:#c62828">{txt}</span>')
        elif style_name == "dim":
            html_parts.append(f'<span style="opacity:0.6">{txt}</span>')
        else:
            html_parts.append(txt)
    legend = (
        '<div style="font-size:0.9em;margin-bottom:4px">'
        '<span style="color:#2e7d32">correct</span>, '
        '<span style="color:#c62828">incorrect</span>, '
        '<span style="opacity:0.6">unchanged</span>'
        "</div>"
    )
    return (
        legend
        + '<pre style="white-space:pre-wrap">Generated:\n'
        + "".join(html_parts)
        + "</pre>"
    )
