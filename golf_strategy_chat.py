#!/usr/bin/env python3
"""
Interactive Golf Strategy Chat
Chat with your trained LoRA model to get golf strategy recommendations and hole descriptions.

Basic usage (interactive):
    python golf_strategy_chat.py --adapter_dir outputs/bethpage-lora/checkpoint-1_hour-gpt2 \
        --data-file data/bethpage_black/train_multitask_case_fixed.jsonl

One-shot example:
    python golf_strategy_chat.py --adapter_dir outputs/bethpage-lora/checkpoint-1_hour-gpt2 \
        --data-file data/bethpage_black/train_multitask_case_fixed.jsonl \
        --prompt "Hole 5, I drive 290 yards, what strategy?"
"""

import os
import json
import re
import argparse
import tempfile
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from peft import PeftModel, PeftConfig
from safetensors.torch import load_file, save_file

GUIDANCE = (
    "Write one cohesive paragraph (120–180 words). Focus on hazards, wind, elevation, landing zones, and approach angles. "
    "Use specific details from the prompt. Avoid generic phrases and meta language (e.g., 'Success depends…', 'combines strategic positioning'). "
    "Do not include headings or labels (no 'Hints:', 'Guidance:', bullets, or sections). Do not use words like 'diagram' or 'Zone'. "
    "Only include distances when they are meaningful and express them as '<number> yards'. Do not list multiple unrelated yardages. "
    "Do not restate the hole number incorrectly; if you mention it, ensure it matches the prompt."
)


def prepare_adapter_folder(adapter_dir: str) -> str:
    """Create a temp folder with inference_mode=True and correctly prefixed LoRA weights."""
    tmpdir = tempfile.mkdtemp(prefix="fixed_lora_")
    os.makedirs(tmpdir, exist_ok=True)

    # Copy config and enforce inference_mode
    cfg_src = os.path.join(adapter_dir, "adapter_config.json")
    cfg_dst = os.path.join(tmpdir, "adapter_config.json")
    shutil.copy(cfg_src, cfg_dst)
    peft_cfg = PeftConfig.from_pretrained(tmpdir)
    peft_cfg.inference_mode = True
    peft_cfg.save_pretrained(tmpdir)

    # Load and ensure weights are prefixed as expected by PeftModel
    st_src = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = load_file(st_src, device="cpu")
    prefix = "base_model.model."
    needs_prefix = not next(iter(sd)).startswith(prefix)
    fixed = {}
    if needs_prefix:
        for k, v in sd.items():
            fixed[prefix + k] = v
    else:
        fixed = sd
    save_file(fixed, os.path.join(tmpdir, "adapter_model.safetensors"))

    return tmpdir


def fix_mojibake(s: str) -> str:
    repl = {
        "â€™": "’",
        "â€“": "–",
        "â€”": "—",
        "â€œ": "“",
        "â€": "”",
        "â€˜": "‘",
        "â€¦": "…",
        "Ã©": "é",
        "Ã": "A",
        "Â": "",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def clean_bleed(text: str) -> str:
    parts = re.split(r"\[Hole\s*\d+\]", text, maxsplit=1)
    return parts[0].strip()


HOLE_RE = re.compile(r"\bHole\s*(\d+)\b", flags=re.IGNORECASE)


def extract_expected_hole(prompt: str) -> int | None:
    m = HOLE_RE.search(prompt)
    return int(m.group(1)) if m else None


def sanitize_hole_header(text: str, expected_hole: int | None) -> str:
    if expected_hole is None:
        return text
    m = re.match(r"^\s*Hole\s*(\d+)\b", text, flags=re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if num != expected_hole:
            return re.sub(r"^\s*Hole\s*\d+\b\s*[:\-]?\s*", "This hole ", text, count=1, flags=re.IGNORECASE)
    return text


def de_template(text: str) -> str:
    patterns = [
        re.compile(r"combines strategic positioning with technical execution", re.IGNORECASE),
        re.compile(r"\b[Ss]uccess\b[^.!?]{0,180}\bdepends\b[^.!?]*[.!?]"),
    re.compile(r"\b[Ss]uccess\s*On\b[^.!?]*", re.IGNORECASE),
    re.compile(r"\b[Ss]uccessOn\b[^.!?]*", re.IGNORECASE),
    re.compile(r"\bdiagram\b[^.!?]*", re.IGNORECASE),
    re.compile(r"\bZone\s*\d+\b[^.!?]*", re.IGNORECASE),
    re.compile(r"\bdecision\s+sharing\b[^.!?]*", re.IGNORECASE),
    re.compile(r"\btechnical\s+execut(ion|ing)\b[^.!?]*", re.IGNORECASE),
    ]
    for pat in patterns:
        text = pat.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_boundary_trim(text: str, min_words: int = 90, target_words: int = 160, max_words: int = 220) -> str:
    """
    Trim to whole sentences near target word count.
    - Keep at least min_words.
    - Prefer not to exceed max_words.
    - If already shorter than min_words, return as-is.
    """
    # Normalize whitespace
    t = re.sub(r"\s+", " ", text).strip()
    words = t.split()
    if len(words) <= min_words:
        return t

    # Simple sentence split by punctuation followed by space or end.
    # Keeps delimiters.
    parts = re.split(r"(?<=[.!?])\s+", t)
    acc = []
    wcount = 0
    for sent in parts:
        acc.append(sent)
        wcount += len(sent.split())
        if wcount >= target_words:
            break
    trimmed = " ".join(acc).strip()

    # If we overshot max_words, back off one sentence if possible
    if len(trimmed.split()) > max_words and len(acc) > 1:
        trimmed = " ".join(acc[:-1]).strip()

    # Ensure ends with sentence punctuation
    if not re.search(r"[.!?]$", trimmed):
        # Try to find last punctuation in original text
        m = list(re.finditer(r"[.!?]", trimmed))
        if m:
            trimmed = trimmed[: m[-1].end()].strip()
    return trimmed


def normalize_user_input(text: str) -> str:
    """Lightweight normalization to handle common typos and variants."""
    t = text
    # Lowercase for predictable replacements, but preserve original casing for numbers
    lower = t.lower()
    replacements = {
        "stratergy": "strategy",
        "stratery": "strategy",
        "strategry": "strategy",
        "maximun": "maximum",
        "maxium": "maximum",
        "max": "maximum",
        "yrds": "yards",
        "yds": "yards",
        "yd": "yards",
        # hole phrasing typos
        "what hole": "hole",
    }
    for wrong, right in replacements.items():
        lower = re.sub(rf"\b{re.escape(wrong)}\b", right, lower)
    return lower


# Heuristics for deterministic description synthesis
KEY_HAZARD_TERMS = [
    "bunker", "bunkers", "sand", "rough", "fairway", "narrow", "dogleg",
    "wind", "elevation", "uphill", "downhill", "carry", "landing", "landing zone",
    "water", "creek", "pond", "out of bounds", "OB", "green", "tier", "slope",
    "angle", "approach", "runoff", "false front", "hazard", "mound"
]

GENERIC_NOISE = [
    r"strategic demands?",
    r"decision[- ]?making",
    r"combines? strategic positioning",
    r"technical execut(ion|ing)",
    r"Success\s*depends",
]

# Theme guidance: map strategic_theme tokens to natural language directives and synonyms
THEME_DIRECTIVES: dict[str, list[str]] = {
    "angle_priority": [
        "Favor the side of the fairway that opens the approach angle.",
        "Avoid positions that leave a blocked or acute angle into the green.",
    ],
    "positioning": [
        "Prioritize placement over raw distance from the tee.",
        "Choose a club that holds the chosen side of the fairway.",
    ],
    "risk_reward": [
        "Outline both the safer layup line and the aggressive carry and their consequences.",
        "Only take on the hazard if the landing zone is wide enough to hold the shot shape.",
    ],
    "elevation_adjustment": [
        "Account for uphill/downhill in club selection and rollout.",
        "Keep approaches under the hole when slopes are severe.",
    ],
    "wind_read": [
        "Note prevailing or crosswinds and how they affect start lines and curvature.",
        "Use trajectory control to hold the fairway or green into the wind.",
    ],
    "precision_tee": [
        "Emphasize accuracy off the tee; less than driver is acceptable to find short grass.",
        "Pick a target and commit to a playable miss on one side.",
    ],
    "layup_choice": [
        "Describe preferred layup yardage and shelf to leave a full number into the green.",
        "Avoid cross bunkers or narrowing in the second-shot zone.",
    ],
    "carry": [
        "State the benefit and risk of carrying the primary hazard versus laying back.",
    ],
}

THEME_SYNONYMS: dict[str, list[str]] = {
    "angle_priority": ["angle", "approach angle", "opening angle", "better angle", "line into the green"],
    "positioning": ["position", "placement", "line", "side of the fairway"],
    "risk_reward": ["risk", "reward", "aggressive", "safer", "lay up", "layup", "carry"],
    "elevation_adjustment": ["uphill", "downhill", "elevation", "slope"],
    "wind_read": ["wind", "crosswind", "breeze", "gust"],
    "precision_tee": ["precision", "accurate", "narrow", "find the fairway", "less than driver"],
    "layup_choice": ["lay up", "layup", "second shot", "shelf", "lay-up"],
    "carry": ["carry", "cover", "over the bunker", "over the hazard"],
}

def keep_only_official_yardage(text: str, official: int | None) -> str:
    if not isinstance(official, int):
        return text
    # Remove all yardage mentions that aren't the official number
    def repl(m: re.Match) -> str:
        num = int(m.group(1))
        return f"{official} yards" if num == official else ""
    text = re.sub(r"(\d{2,4})\s*(?:yards?|yds?)\+?", repl, text)
    # Compress spaces after removals
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_generic_phrases(text: str) -> str:
    out = text
    for pat in GENERIC_NOISE:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = de_template(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def rank_insights(items: list[str]) -> list[str]:
    def score(s: str) -> int:
        s_low = s.lower()
        hits = sum(1 for k in KEY_HAZARD_TERMS if k in s_low)
        length = min(len(s_low) // 40, 5)  # mild preference for longer, capped
        return hits * 5 + length
    return sorted(items, key=score, reverse=True)

def compose_description(
    hole_num: int,
    par: int | None,
    yardage: int | None,
    insights: list[str],
    hazards: list[str] | set[str] | None = None,
    preferred_miss: str | None = None,
    theme: str | None = None,
) -> str:
    # Pick top 4-6 insights
    top = rank_insights([fix_mojibake(s).strip() for s in insights if s and len(s) > 8])[:6]
    lead = f"Hole {hole_num} at Bethpage Black"
    if isinstance(par, int) and isinstance(yardage, int):
        header = f"{lead} is a Par {par} playing {yardage} yards."
    elif isinstance(yardage, int):
        header = f"{lead} plays {yardage} yards."
    elif isinstance(par, int):
        header = f"{lead} is a Par {par}."
    else:
        header = f"{lead} demands precise placement."

    body = []
    for s in top:
        s = sanitize_hole_header(s, hole_num)
        s = strip_generic_phrases(s)
        # Avoid ending with colons from bullets
        s = s.rstrip(": ")
        body.append(s if s.endswith(('.', '!', '?')) else s + ".")

    # Append concise structured bits when present
    extras = []
    if hazards:
        hz = sorted(set(hazards))[:5]
        if hz:
            extras.append("Key hazards: " + ", ".join(hz) + ".")
    if isinstance(theme, str) and theme.strip():
        extras.append(f"Strategic theme: {theme.strip()}.")
    if isinstance(preferred_miss, str) and preferred_miss.strip():
        extras.append(f"Preferred miss: {preferred_miss.strip()}.")

    text = header + " " + " ".join(body + (extras if extras else []))
    text = keep_only_official_yardage(text, yardage)
    text = sentence_boundary_trim(text, min_words=60, target_words=130, max_words=180)
    return text


def compose_description_structured(
    hole_num: int,
    par: int | None,
    yardage: int | None,
    attrs: dict | None,
    theme: str | None,
    preferred_miss: str | None,
    hazards: list[str] | set[str] | None,
) -> str:
    """Compose a grounded paragraph from structured attributes with par-aware narrative (no copying)."""
    attrs = attrs or {}
    lead = f"Hole {hole_num} at Bethpage Black"
    if isinstance(par, int) and isinstance(yardage, int):
        header = f"{lead} is a Par {par} playing {yardage} yards."
    elif isinstance(yardage, int):
        header = f"{lead} plays {yardage} yards."
    elif isinstance(par, int):
        header = f"{lead} is a Par {par}."
    else:
        header = f"{lead} demands precise placement."

    bits: list[str] = []
    gradient = attrs.get("gradient")
    shape = attrs.get("hole_shape")
    elev_note = attrs.get("elevation_note")
    risk_note = attrs.get("risk_note")
    angle_note = attrs.get("angle_note")
    pri_hz = attrs.get("primary_hazard_type")
    sec_hz = attrs.get("secondary_hazard_type")
    key_hz = attrs.get("key_hazards") or []

    # Sanitize short text attributes
    def _clean_txt(x: str | None) -> str | None:
        if not isinstance(x, str):
            return None
        return re.sub(r"\s+", " ", x).strip()
    gradient = _clean_txt(gradient)
    shape = _clean_txt(shape)
    elev_note = _clean_txt(elev_note)

    # Normalize hazards
    hz_all = list(sorted(set([*(key_hz or []), *(h for h in [pri_hz, sec_hz] if h)])))
    if hazards:
        hz_all.extend(list(hazards))
    hz_all = [h for h in {str(h).lower().strip() for h in hz_all} if h and h not in {"none", "n/a", "hazard"}]

    if par == 3:
        # Par-3 narrative: concise, label-free prose
        sentences: list[str] = []
        # Opener with shape/gradient
        opener_bits: list[str] = []
        if shape:
            opener_bits.append(f"It’s a {shape.replace('_', ' ')}")
        if gradient:
            verb = "plays" if not opener_bits else "that plays"
            opener_bits.append(f"{verb} {gradient.replace('_', ' ')} from the tee")
        if opener_bits:
            sentences.append(" ".join(opener_bits) + ".")
        # Hazards phrase
        hz_phrase = None
        if hz_all:
            has_water = any("water" in h for h in hz_all)
            has_bunker = any("bunker" in h for h in hz_all)
            if has_water and has_bunker:
                hz_phrase = "The shot must carry water to a green guarded by bunkers."
            elif has_water:
                hz_phrase = "The shot must carry water to reach the putting surface."
            elif has_bunker:
                hz_phrase = "Bunkers tightly guard the green and collect misses."
            else:
                hz_phrase = "The green is well defended, rewarding precise distance control."
        if hz_phrase:
            sentences.append(hz_phrase)
        # Elevation and theme-aware guidance
        if elev_note or gradient:
            if theme and any(t.strip() in (theme or "").split("+") for t in ["wind_read", "elevation_adjustment"]):
                sentences.append("Account for the elevation when selecting the club and use spin to hold the green.")
            else:
                sentences.append("Club selection matters with the elevation change; favor a flight that lands softly.")
        # Preferred miss, angles, and risk notes folded into prose
        tail_bits: list[str] = []
        if preferred_miss:
            tail_bits.append(f"The safer miss is {preferred_miss}.")
        if angle_note:
            tail_bits.append(f"Better angles come from {angle_note}.")
        if risk_note:
            tail_bits.append(f"Be mindful of {risk_note}.")
        if tail_bits:
            sentences.append(" ".join(tail_bits))
        text = header + " " + " ".join(sentences)
    else:
        # Par-4/5 narrative
        if shape or gradient:
            if shape and gradient:
                bits.append(f"It’s a {shape.replace('_', ' ')} that plays {gradient.replace('_', ' ')} from the tee.")
            elif shape:
                bits.append(f"It’s a {shape.replace('_', ' ')} with emphasis on line and distance control.")
            else:
                bits.append(f"It plays {gradient.replace('_', ' ')} from the tee.")
        if hz_all:
            has_water = any("water" in h for h in hz_all)
            has_bunker = any("bunker" in h for h in hz_all)
            if has_water and has_bunker:
                bits.append("Key trouble includes water and deep bunkers.")
            elif has_water:
                bits.append("Water is in play off the tee or on approach.")
            elif has_bunker:
                bits.append("Deep bunkers frame the landing and green complexes.")
        if elev_note:
            bits.append(f"Elevation: {elev_note}.")
        if angle_note:
            bits.append(f"Angles: {angle_note}.")
        if risk_note:
            bits.append(f"Risk: {risk_note}.")
        if theme:
            tokens = [t.strip() for t in str(theme).split("+") if t.strip()]
            selected = []
            for tok in tokens:
                d = THEME_DIRECTIVES.get(tok)
                if d:
                    selected.append(d[0])
            if selected:
                bits.append(" ".join(selected[:2]))
        if preferred_miss:
            bits.append(f"Preferred miss: {preferred_miss}.")
        text = header + " " + " ".join(bits)

    text = keep_only_official_yardage(text, yardage)
    text = sentence_boundary_trim(text, min_words=50, target_words=110, max_words=160)
    return text


class GolfStrategyChat:
    def __init__(
        self,
        adapter_path: str,
        data_file: str,
        device: str = "auto",
        use_guidance: bool = True,
        base_model_name: str = "gpt2",
        base_only: bool = False,
        explain: bool = True,
        desc_mode: str = "model",
        sample_desc: bool = False,
        desc_fallback: bool = True,
        desc_grounding_min_signals: int = 2,
    ) -> None:
        self.adapter_path = adapter_path
        self.data_file = data_file
        self.hole_data = {}
        self.current_hole = None
        self.device = self._resolve_device(device)
        self.use_guidance = use_guidance
        self.base_model_name = base_model_name
        self.base_only = base_only
        self.explain = explain
        # description generation mode: 'model' (default) or 'deterministic'
        self.desc_mode = desc_mode or "model"
        # description decoding mode: greedy by default unless sampling requested
        self.sample_desc = bool(sample_desc)
        self.desc_fallback = bool(desc_fallback)
        self.desc_grounding_min_signals = max(0, int(desc_grounding_min_signals))
        # metrics
        self.last_desc_meta: dict | None = None
        self.desc_stats = {
            "model_attempts": 0,
            "model_accepted": 0,
            "fallback_used": 0,
            "signals_matched_sum": 0,
            "signals_possible_sum": 0,
        }
        self.load_model()
        self.load_hole_data()

    def _resolve_device(self, choice: str) -> torch.device:
        if choice == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if choice == "cpu":
            return torch.device("cpu")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the trained LoRA model"""
        print("Loading model and tokenizer...")

        if not self.base_only and not os.path.isdir(self.adapter_path):
            raise FileNotFoundError(
                f"Adapter folder not found: {self.adapter_path}. "
                "Pass --adapter_dir to point at your trained checkpoint, or use --base-only to compare base model output."
            )
        # Aggregators
        pars: dict[int, Counter] = {}
        yardages: dict[int, Counter] = {}
        themes: dict[int, Counter] = {}
        misses: dict[int, Counter] = {}
        hazards_counts: dict[int, Counter] = {}

        # Use utf-8-sig to tolerate potential BOM on Windows
        with open(self.data_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                hole = example.get("hole")
                if not hole:
                    continue

                # Ensure structure
                if hole not in self.hole_data:
                    self.hole_data[hole] = {
                        # We'll finalize par/yardage using majority vote
                        "par": None,
                        "yardage": None,
                        "descriptions": set(),
                        "insights": set(),
                        "strategies": [],
                        "hazards": set(),
                        "preferred_miss": None,
                        "strategic_theme": None,
                        "attributes": {},
                    }
                    pars[hole] = Counter()
                    yardages[hole] = Counter()
                    themes[hole] = Counter()
                    misses[hole] = Counter()
                    hazards_counts[hole] = Counter()

                # Track par/yardage candidates
                if isinstance(example.get("par"), int):
                    pars[hole][example["par"]] += 1
                if isinstance(example.get("yardage"), int):
                    yardages[hole][example["yardage"]] += 1

                # Optional enriched fields
                hz = example.get("hazards")
                if isinstance(hz, list):
                    for h in hz:
                        if isinstance(h, str) and h.strip():
                            hazards_counts[hole][h.strip().lower()] += 1
                elif isinstance(hz, str) and hz.strip():
                    for h in re.split(r",|/|;|\n", hz):
                        h = h.strip()
                        if h:
                            hazards_counts[hole][h.lower()] += 1

                pm = example.get("preferred_miss")
                if isinstance(pm, str) and pm.strip():
                    misses[hole][pm.strip().lower()] += 1

                th = example.get("strategic_theme")
                if isinstance(th, str) and th.strip():
                    themes[hole][th.strip()] += 1

                task_type = example.get("task_type")
                if task_type == "strategy_selection":
                    strategies = example.get("tee_shot_strategies") or []
                    # Fallbacks from other strategy variants
                    available_cutoffs = example.get("available_cutoffs") or []
                    if strategies:
                        existing = {s.get("cutoff_distance") for s in self.hole_data[hole]["strategies"]}
                        for s in strategies:
                            cutoff = s.get("cutoff_distance")
                            if isinstance(cutoff, int) and cutoff not in existing:
                                self.hole_data[hole]["strategies"].append(s)
                                existing.add(cutoff)
                    elif available_cutoffs:
                        existing = {s.get("cutoff_distance") for s in self.hole_data[hole]["strategies"]}
                        for cutoff in available_cutoffs:
                            if isinstance(cutoff, int) and cutoff not in existing:
                                self.hole_data[hole]["strategies"].append({
                                    "cutoff_distance": cutoff,
                                    "remaining_distance": None,
                                    "advantage": None,
                                })
                                existing.add(cutoff)
                elif task_type == "description_synthesis":
                    # Use the completion text as a candidate description
                    comp = (example.get("completion") or "").strip()
                    if comp and len(comp) > 20:
                        self.hole_data[hole]["descriptions"].add(fix_mojibake(comp))
                    # Also parse insights from the prompt bullets for cleaner synthesis
                    prompt_text = example.get("prompt") or ""
                    # Extract simple insights from numbered bullets if present
                    for bl in re.findall(r"\n\s*\d+\.\s*(.+?)\n", prompt_text + "\n", flags=re.DOTALL):
                        bl = fix_mojibake(bl.strip())
                        if len(bl) > 8:
                            self.hole_data[hole]["insights"].add(bl)
                        # Do NOT boost generic hazard keywords from insights; they pollute hazards

                # Parse Facts-style lines from prompts to harvest hazards/theme/miss and attributes
                ptxt = (example.get("prompt") or "")
                if ptxt:
                    # KeyHazards: a; b; c
                    m = re.search(r"KeyHazards:\s*(.+)", ptxt)
                    if m:
                        raw = m.group(1)
                        for h in re.split(r";|,|/", raw):
                            h = h.strip()
                            if h:
                                hazards_counts[hole][h.lower()] += 1
                        self.hole_data[hole].setdefault("attributes", {})["key_hazards"] = [
                            s.strip() for s in re.split(r";|,|/", raw) if s.strip()
                        ]
                    # Primary/SecondaryHazard
                    for lab in ("PrimaryHazard", "SecondaryHazard"):
                        m2 = re.search(rf"{lab}:\s*([A-Za-z][\w\- ]+)", ptxt)
                        if m2:
                            hz = m2.group(1).strip().lower()
                            hazards_counts[hole][hz] += 1
                            key = "primary_hazard_type" if lab == "PrimaryHazard" else "secondary_hazard_type"
                            self.hole_data[hole].setdefault("attributes", {})[key] = hz
                    # PreferredMiss
                    m3 = re.search(r"PreferredMiss:\s*([^\n]+)", ptxt)
                    if m3:
                        pmv = m3.group(1).strip().lower()
                        if pmv:
                            misses[hole][pmv] += 1
                            self.hole_data[hole]["preferred_miss"] = pmv
                    # Theme
                    m4 = re.search(r"Theme:\s*([^\n]+)", ptxt)
                    if m4:
                        thv = m4.group(1).strip()
                        if thv:
                            themes[hole][thv] += 1
                            self.hole_data[hole]["strategic_theme"] = thv
                    # Additional attributes sometimes present in prompts
                    m5 = re.search(r"Gradient:\s*([^\n]+)", ptxt)
                    if m5:
                        self.hole_data[hole].setdefault("attributes", {})["gradient"] = m5.group(1).strip()
                    m6 = re.search(r"Shape:\s*([^\n]+)", ptxt)
                    if m6:
                        self.hole_data[hole].setdefault("attributes", {})["hole_shape"] = m6.group(1).strip()
                    m7 = re.search(r"ElevationNote:\s*([^\n]+)", ptxt)
                    if m7:
                        self.hole_data[hole].setdefault("attributes", {})["elevation_note"] = m7.group(1).strip()
                    m8 = re.search(r"RiskNote:\s*([^\n]+)", ptxt)
                    if m8:
                        self.hole_data[hole].setdefault("attributes", {})["risk_note"] = m8.group(1).strip()
                    m9 = re.search(r"AngleNote:\s*([^\n]+)", ptxt)
                    if m9:
                        self.hole_data[hole].setdefault("attributes", {})["angle_note"] = m9.group(1).strip()

                # Also capture any base description on original records
                base_desc = (example.get("description") or "").strip()
                if base_desc and len(base_desc) > 20:
                    clean = fix_mojibake(base_desc)
                    self.hole_data[hole]["descriptions"].add(clean)
                    low = clean.lower()
                    for kw in KEY_HAZARD_TERMS:
                        if kw in low:
                            hazards_counts[hole][kw] += 1

        # Convert description sets to lists for easier handling
        for h, hole_data in self.hole_data.items():
            hole_data["descriptions"] = list(hole_data["descriptions"])
            hole_data["insights"] = list(hole_data["insights"])
            # Finalize par/yardage as most common values seen for this hole
            if pars.get(h):
                hole_data["par"] = pars[h].most_common(1)[0][0]
            if yardages.get(h):
                hole_data["yardage"] = yardages[h].most_common(1)[0][0]

            # Finalize hazards/theme/preferred_miss
            if hazards_counts.get(h):
                top_hz = [k for k, _ in hazards_counts[h].most_common(10)]
                # Clean hazards: drop 'none' and generic tokens
                generic_drop = {"none", "n/a", "na", "no hazard", "hazard"}
                # Also drop generic keywords we might have accumulated earlier
                generic_drop.update(w.lower() for w in KEY_HAZARD_TERMS)
                cleaned = []
                for k in top_hz:
                    kk = (k or "").strip().lower()
                    if not kk or kk in generic_drop:
                        continue
                    # Too short to be meaningful
                    if len(kk) < 3:
                        continue
                    cleaned.append(kk)
                hole_data["hazards"] = set(cleaned[:6])
            if themes.get(h) and themes[h]:
                hole_data["strategic_theme"] = themes[h].most_common(1)[0][0]
            if misses.get(h) and misses[h]:
                hole_data["preferred_miss"] = misses[h].most_common(1)[0][0]

            # Fill in missing remaining_distance for strategies using hole yardage
            yd = hole_data.get("yardage")
            par = hole_data.get("par")
            if isinstance(yd, int) and hole_data.get("strategies"):
                for s in hole_data["strategies"]:
                    cutoff = s.get("cutoff_distance")
                    if s.get("remaining_distance") in (None, "", "null") and isinstance(cutoff, int):
                        if par == 3:
                            s["remaining_distance"] = 0
                        else:
                            rem = max(yd - cutoff, 0)
                            s["remaining_distance"] = rem

        print(f"Loaded data for {len(self.hole_data)} holes")
        # Load tokenizer and model (+adapter) for generation
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        # Ensure pad token is set for GPT-2 style models
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.base_only:
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        else:
            # Prepare adapter to ensure proper keys and inference_mode
            try:
                tmp_adapter = prepare_adapter_folder(self.adapter_path)
            except Exception:
                # Fallback: try to use the adapter folder directly
                tmp_adapter = self.adapter_path
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            self.model = PeftModel.from_pretrained(base_model, tmp_adapter)
            # Keep ref to temp path for the session (no cleanup necessary during run)
            self._adapter_tmp = tmp_adapter
        # Move to device and eval mode
        self.model.to(self.device)
        self.model.eval()

    # Back-compat: previously hole data was loaded in a separate method; we now do it in load_model
    def load_hole_data(self) -> None:
        return
    
    def extract_hole_number(self, text):
        """Extract hole number from user input"""
        patterns = [
            r"hole\s*(\d+)",
            r"#(\d+)",
            r"(\d+)(?:st|nd|rd|th)?\s*hole",
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                hole_num = int(match.group(1))
                if 1 <= hole_num <= 18:
                    return hole_num
        return None
    
    def extract_drive_distance(self, text):
        """Extract drive distance from user input"""
        patterns = [
            r"drive\s*(?:it\s*)?(\d{2,3})\s*(?:yards?|yds?|yrds?)",
            r"hit\s*(?:it\s*)?(\d{2,3})\s*(?:yards?|yds?|yrds?)",
            r"(\d{2,3})\s*(?:yard|yd|yds|yrds)\s*drive",
            r"(\d{2,3})\s*(?:yards?|yds?|yrds?)",
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                distance = int(match.group(1))
                if 150 <= distance <= 400:  # Reasonable drive distances
                    return distance
        return None
    
    def format_hole_info(self, hole_num):
        """Format hole information for display using model-generated description"""
        if hole_num not in self.hole_data:
            return f"No data for hole {hole_num}."
        
        hole = self.hole_data[hole_num]
        info = f"Hole {hole_num} - Bethpage Black\n"
        info += f"Par {hole['par']}, {hole['yardage']} yards\n"
        
        # Get model-generated description
        description = self.get_hole_description(hole_num)
        if description:
            info += f"Description: {description}\n"
            # Append concise grounding/fallback score if available
            meta = getattr(self, "last_desc_meta", None)
            if isinstance(meta, dict):
                mode = meta.get("mode")
                if mode == "model":
                    matched = meta.get("signals_matched")
                    possible = meta.get("signals_possible")
                    fallback_used = meta.get("fallback_used")
                    threshold = meta.get("grounding_threshold")
                    disabled = meta.get("fallback_disabled", False)
                    reason = meta.get("fallback_reason")
                    score_line = f"Grounding: {matched}/{possible} signals; Fallback: {'Yes' if fallback_used else 'No'}"
                    if disabled:
                        score_line += " (fallback disabled)"
                    else:
                        score_line += f" (threshold {threshold})"
                    if fallback_used and reason:
                        score_line += f"; Reason: {reason}"
                else:
                    score_line = "Mode: deterministic (no model used)"
                info += score_line + "\n"
        
        # Suppress strategy listing on par-3 holes; tee shot is to the green
        if hole['strategies'] and int(hole.get('par') or 0) != 3:
            info += f"\nAvailable Strategies:\n"
            for i, strategy in enumerate(hole['strategies'], 1):
                cutoff = strategy.get('cutoff_distance')
                remaining = strategy.get('remaining_distance')
                if remaining in (None, "", "null"):
                    remaining = 'N/A'
                advantage = strategy.get('advantage', '')
                
                info += f"  {i}. {cutoff} yards - Leaves {remaining} yards to pin"
                if advantage:
                    info += f" ({advantage})"
                info += "\n"
        
        return info
    
    def get_hole_description(self, hole_num):
        """Generate hole description using either:
        - 'model': LoRA model conditioned on structured facts (with optional fallback)
        - 'deterministic': a structured, template-based composer that uses only attributes/hazards/theme
          and never copies description text from the dataset.
        """
        if hole_num not in self.hole_data:
            return "No data for that hole."

        hole = self.hole_data[hole_num]
        mode = getattr(self, "desc_mode", "model")
        if mode == "model":
            return self._generate_model_description(hole_num, hole)
        else:
            # Strictly structured deterministic composer: never copy dataset description text.
            desc = compose_description_structured(
                hole_num,
                hole.get("par"),
                hole.get("yardage"),
                hole.get("attributes"),
                hole.get("strategic_theme"),
                hole.get("preferred_miss"),
                hole.get("hazards"),
            )
            # Record meta for display
            self.last_desc_meta = {
                "mode": "deterministic",
            }
            return desc

    def _generate_model_description(self, hole_num: int, hole: dict) -> str:
        """Generate a hole description using the model+adapter, conditioning on structured facts.
        Post-process to keep it specific and within length and yardage constraints."""
        par = hole.get("par")
        yard = hole.get("yardage")
        # Locally sanitize hazards for prompting
        raw_hz = list(hole.get("hazards") or [])
        bad = {"none", "n/a", "na", "no hazard", "hazard"}
        hz = []
        for h in raw_hz:
            hh = (h or "").strip().lower()
            if not hh or hh in bad:
                continue
            if len(hh) < 3:
                continue
            hz.append(hh)
        hz = sorted(set(hz))[:6]
        theme = hole.get("strategic_theme")
        pmiss = hole.get("preferred_miss")
        # Use at most 1 high-signal insight to reduce noise
        insights = rank_insights([fix_mojibake(s).strip() for s in hole.get("insights", [])])[:1]

        facts = []
        if isinstance(par, int) and isinstance(yard, int):
            facts.append(f"Par {par}, {yard} yards")
        elif isinstance(par, int):
            facts.append(f"Par {par}")
        elif isinstance(yard, int):
            facts.append(f"{yard} yards")
        # include structured attributes for stronger grounding
        attrs = hole.get("attributes") or {}
        gradient = attrs.get("gradient")
        shape = attrs.get("hole_shape")
        elev_note = attrs.get("elevation_note")
        risk_note = attrs.get("risk_note")
        angle_note = attrs.get("angle_note")
        if hz:
            facts.append("Hazards: " + ", ".join(hz))
        if theme:
            # Keep the original stable behavior: include theme label as a Fact line
            facts.append(f"Strategic theme: {theme}")
        if pmiss:
            facts.append(f"Preferred miss: {pmiss}")
        if gradient:
            facts.append(f"Gradient: {gradient}")
        if shape:
            facts.append(f"Shape: {shape}")
        if elev_note:
            facts.append(f"ElevationNote: {elev_note}")
        if risk_note:
            facts.append(f"RiskNote: {risk_note}")
        if angle_note:
            facts.append(f"AngleNote: {angle_note}")

        prompt_parts = [
            "TASK: description_synthesis",
            f"Hole {hole_num} at Bethpage Black.",
        ]
        if facts:
            prompt_parts.append("Facts: " + " | ".join(facts))
        if insights:
            prompt_parts.append(f"Insight: {insights[0]}")
        if self.use_guidance:
            prompt_parts.append("Guidance: " + GUIDANCE)
        prompt_parts.append("Synthesized description:")
        prompt = "\n".join(prompt_parts).strip()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        do_sample = bool(self.sample_desc)
        gen_kwargs = dict(
            max_new_tokens=200,
            do_sample=do_sample,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=0.7, top_p=0.85)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Keep only the part after the prompt header
        gen = text.split("Synthesized description:", 1)
        gen_text = gen[1] if len(gen) == 2 else text
        gen_text = gen_text.strip()

        # Post-process for specificity and constraints
        gen_text = fix_mojibake(gen_text)
        gen_text = sanitize_hole_header(gen_text, hole_num)
        gen_text = strip_generic_phrases(gen_text)
        gen_text = keep_only_official_yardage(gen_text, yard)
        gen_text = sentence_boundary_trim(gen_text, min_words=60, target_words=130, max_words=180)

        # Optional grounding check with deterministic fallback
        def _signals(txt: str) -> tuple[int, int]:
            low = txt.lower()
            hz = [h.lower() for h in (hole.get("hazards") or [])]
            theme = (hole.get("strategic_theme") or "").lower()
            pmiss = (hole.get("preferred_miss") or "").lower()
            signals = 0
            possible = 0
            if hz:
                possible += 1
                # Require at least one concrete hazard mention
                signals += 1 if any(h in low for h in hz) else 0
            if theme:
                possible += 1
                # Only count exact token match to avoid overfitting to generic synonyms
                token_hits = 0
                for tok in [t.strip() for t in theme.split("+") if t.strip()]:
                    if tok in low:
                        token_hits = 1
                        break
                signals += token_hits
            if pmiss:
                possible += 1
                signals += 1 if pmiss in low else 0
            return signals, possible

        def _semantic_ok(txt: str) -> bool:
            # Simple checks to catch obvious nonsense
            low = txt.lower()
            # Par phrasing sanity
            if re.search(r"par\s*[-]?[\d]+\.?[\d]*", low):
                # Reject things like par-4.5 or odd decimals
                if re.search(r"par\s*[-]?\d+\.\d+", low):
                    return False
            # Hole 18 specifics: do not mention par-3s below the green, etc.
            if re.search(r"par-?3s?\s+below\s+the\s+green", low):
                return False
            # Avoid bizarre phrases
            if "center of mass" in low:
                return False
            return True

        signals_matched, signals_possible = _signals(gen_text)
        threshold = getattr(self, "desc_grounding_min_signals", 2)
        self.desc_stats["model_attempts"] += 1
        self.desc_stats["signals_matched_sum"] += signals_matched
        self.desc_stats["signals_possible_sum"] += signals_possible

        fallback_disabled = not bool(self.desc_fallback)
        not_enough_signals = signals_matched < threshold
        semantic_bad = not _semantic_ok(gen_text)
        needs_fallback = (not_enough_signals or semantic_bad) and (not fallback_disabled)
        if needs_fallback:
            self.desc_stats["fallback_used"] += 1
            desc = compose_description_structured(
                hole_num,
                hole.get("par"),
                hole.get("yardage"),
                hole.get("attributes"),
                hole.get("strategic_theme"),
                hole.get("preferred_miss"),
                hole.get("hazards"),
            )
            self.last_desc_meta = {
                "mode": "model",
                "signals_matched": signals_matched,
                "signals_possible": signals_possible,
                "grounding_threshold": threshold,
                "fallback_used": True,
                "fallback_disabled": False,
                "fallback_reason": "semantic" if semantic_bad else "grounding",
            }
            return desc

        # Model accepted
        self.desc_stats["model_accepted"] += 1
        self.last_desc_meta = {
            "mode": "model",
            "signals_matched": signals_matched,
            "signals_possible": signals_possible,
            "grounding_threshold": threshold,
            "fallback_used": False,
            "fallback_disabled": fallback_disabled,
        }
        return gen_text if gen_text else f"Hole {hole_num} at Bethpage Black."

    def get_strategy_recommendation(self, hole_num, drive_distance):
        """Get strategy recommendation using the trained model"""
        if hole_num not in self.hole_data:
            return "No data for that hole."
        # Create prompt aligned to training format with explicit cutoff list
        hole = self.hole_data[hole_num]
        available_cutoffs = [
            s["cutoff_distance"] for s in hole["strategies"] if isinstance(s.get("cutoff_distance"), int)
        ]
        available_cutoffs = sorted(set(available_cutoffs))
        cutoff_list_text = ", ".join(f"{c} yards" for c in available_cutoffs) if available_cutoffs else ""
        prompt = (
            "TASK: strategy_selection\n"
            f"Hole {hole_num} at Bethpage Black. Par {hole['par']}, {hole['yardage']} yards.\n\n"
            "Tee Strategy Selection Task:\n"
            f"Player average drive: {drive_distance} yards. "
            + (f"Available tee strategy cutoffs: {cutoff_list_text}.\n" if cutoff_list_text else "")
            + "Return only: 'Strategy: <number> yards' for the recommended tee shot cutoff distance."
        )

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=250).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        full_response = self.tokenizer.decode(outputs[0].detach().cpu(), skip_special_tokens=True)
        response = full_response[len(prompt) :].strip()

        # Extract strategy number
        strategy_match = re.search(r"Strategy:\s*(\d{2,4})\s*(?:yards?)?", response, re.IGNORECASE)
        if not strategy_match:
            # Fallback: any 2–4 digit number
            strategy_match = re.search(r"(\d{2,4})", response)

        # Guarded selection coerced to valid cutoffs (available_cutoffs already computed above)
        eligible = [c for c in available_cutoffs if c <= drive_distance]
        if eligible:
            coerced = max(eligible)
        else:
            coerced = min(available_cutoffs) if available_cutoffs else None

        if strategy_match:
            model_pick = int(strategy_match.group(1))
            final_choice = coerced if coerced is not None else model_pick
            note = " (coerced from model suggestion)" if coerced is not None and model_pick != coerced else ""
            rec_text = f"\nRecommended Strategy: {final_choice} yards{note}\n"
        else:
            final_choice = coerced
            rec_text = (
                f"\nRecommended Strategy: {final_choice} yards (coerced)\n" if coerced is not None else "\nRecommended Strategy: unavailable\n"
            )

        rec_text += f"Available strategies: {available_cutoffs} yards\n"
        rec_text += (
            f"Eligible for {drive_distance}-yard drive: {eligible if eligible else ['None - using shortest option']}\n"
        )
        rec_text += "Logic: Choose highest available <= drive distance"
        # Short rationale using structured fields
        if getattr(self, "explain", True):
            why = []
            if hole.get("strategic_theme"):
                why.append(f"Theme: {hole['strategic_theme']}")
            if hole.get("preferred_miss"):
                why.append(f"Preferred miss: {hole['preferred_miss']}")
            if hole.get("hazards"):
                hz = ", ".join(sorted(set(hole["hazards"]))[:4])
                if hz:
                    why.append(f"Hazards: {hz}")
            if why:
                rec_text += "\nWhy: " + "; ".join(why)
        return rec_text

    def process_input(self, user_input):
        """Process user input and generate appropriate response"""
        original = user_input.strip()
        user_input = normalize_user_input(original)
        
        if not user_input:
            return "Ask me about golf strategies. Try: 'What are the strategies for hole 12?'"
        
        # Check for hole information request
        hole_num = self.extract_hole_number(user_input)
        drive_distance = self.extract_drive_distance(user_input)
        
        # Handle different types of queries
        if "strategies" in user_input.lower() or "available" in user_input.lower():
            if hole_num:
                self.current_hole = hole_num
                return self.format_hole_info(hole_num)
            else:
                return "Which hole are you asking about? Try: 'What are the strategies for hole 5?'"
        
        elif "strategy" in user_input.lower() or "should i use" in user_input.lower() or "recommend" in user_input.lower():
            if hole_num and drive_distance:
                self.current_hole = hole_num
                return self.get_strategy_recommendation(hole_num, drive_distance)
            elif drive_distance and self.current_hole:
                return self.get_strategy_recommendation(self.current_hole, drive_distance)
            elif hole_num and not drive_distance:
                self.current_hole = hole_num
                return f"{self.format_hole_info(hole_num)}\nHow far can you drive? Try: 'I can drive 250 yards, which strategy should I use?'"
            else:
                return "I need the hole number and your drive distance. Try: 'Hole 5, I drive 280 yards, what strategy?'"
        
        elif drive_distance and self.current_hole:
            return self.get_strategy_recommendation(self.current_hole, drive_distance)
        
        elif hole_num:
            self.current_hole = hole_num
            return self.format_hole_info(hole_num)
        
        else:
            return (
                "Golf Strategy Chat Help\n\n"
                "Try:\n"
                "- What are the strategies for hole 12?\n"
                "- I can drive 250 yards, which strategy should I use?\n"
                "- Hole 5, I drive 280 yards, what strategy?\n"
                "- Show me hole 3\n"
                "- 280 yards (if we're already discussing a hole)\n\n"
                + "Currently discussing: "
                + (f"Hole {self.current_hole}" if self.current_hole else "None")
            )

def main():
    parser = argparse.ArgumentParser(description="Interactive chat for golf strategies and descriptions")
    parser.add_argument("--adapter_dir", required=False, default="outputs/bethpage-lora/checkpoint-1_hour-gpt2")
    parser.add_argument("--data-file", required=False, default="data/bethpage_black/train_multitask_enriched_v3.jsonl")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", help="Optional one-shot prompt; if provided, runs once and exits", default=None)
    parser.add_argument("--no_desc_guidance", action="store_true", help="Disable extra guidance for description synthesis prompts")
    parser.add_argument("--base-model", default="gpt2", help="Base model name used during training (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--desc-sample", action="store_true", help="Enable sampling for description generation (default: greedy)")
    parser.add_argument("--base-only", action="store_true", help="Use base model without loading the LoRA adapter (for comparison)")
    parser.add_argument("--no-explain", action="store_true", help="Omit rationale lines for strategies (theme/miss/hazards)")
    parser.add_argument("--desc-mode", choices=["model", "deterministic"], default="model", help="Choose description engine: model (LoRA) or deterministic composer")
    parser.add_argument("--no-desc-fallback", action="store_true", help="Disable automatic fallback to deterministic description if model output seems ungrounded")
    parser.add_argument("--desc-grounding-min-signals", type=int, default=3, help="Minimum number of grounding signals (hazard/theme/miss) the model text must contain to avoid fallback")
    args = parser.parse_args()

    print("Golf Strategy Chat - Bethpage Black")
    print("=" * 50)
    print("Loading your trained model...")

    try:
        chat = GolfStrategyChat(
            adapter_path=args.adapter_dir,
            data_file=args.data_file,
            device=args.device,
            use_guidance=(not args.no_desc_guidance),
            base_model_name=args.base_model,
            base_only=args.base_only,
            explain=(not args.no_explain),
            desc_mode=args.desc_mode,
            sample_desc=args.desc_sample,
            desc_fallback=(not args.no_desc_fallback),
            desc_grounding_min_signals=args.desc_grounding_min_signals,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure --adapter_dir points to your trained adapter and --data-file to the dataset.")
        return

    if args.prompt:
        # One-shot mode
        out = chat.process_input(args.prompt)
        print(out)
        return

    print("\nReady to chat! Type 'quit' to exit.")
    print("Try: 'What are the strategies for hole 12?'")
    print("-" * 50)

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye.")
                break
            response = chat.process_input(user_input)
            print(f"\n{response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
