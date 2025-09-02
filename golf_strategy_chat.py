#!/usr/bin/env python3
"""
Interactive Golf Strategy Chat
Chat with your trained LoRA model to get golf strategy recommendations and hole descriptions.

Basic usage (interactive):
    python golf_strategy_chat.py --adapter_dir outputs/bethpage-lora/checkpoint-8_hour \
        --data-file data/bethpage_black/train_multitask_case_fixed.jsonl

One-shot example:
    python golf_strategy_chat.py --adapter_dir outputs/bethpage-lora/checkpoint-8_hour \
        --data-file data/bethpage_black/train_multitask_case_fixed.jsonl \
        --prompt "Hole 5, I drive 290 yards, what strategy?"
"""

import os
import json
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

GUIDANCE = (
    "Write one cohesive paragraph (120–180 words). Focus on hazards, wind, elevation, landing zones, and approach angles. "
    "Use specific details from the prompt. Avoid generic phrases and meta language (e.g., 'Success depends…', 'combines strategic positioning'). "
    "Do not include headings or labels (no 'Hints:', 'Guidance:', bullets, or sections). Do not use words like 'diagram' or 'Zone'. "
    "Only include distances when they are meaningful and express them as '<number> yards'. Do not list multiple unrelated yardages. "
    "Do not restate the hole number incorrectly; if you mention it, ensure it matches the prompt."
)


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


class GolfStrategyChat:
    def __init__(self, adapter_path: str, data_file: str, device: str = "auto", use_guidance: bool = True, base_model_name: str = "gpt2", base_only: bool = False):
        self.adapter_path = adapter_path
        self.data_file = data_file
        self.hole_data = {}
        self.current_hole = None
        self.device = self._resolve_device(device)
        self.use_guidance = use_guidance
        self.base_model_name = base_model_name
        self.base_only = base_only
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

        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter unless base-only
        if self.base_only:
            self.model = base_model.to(self.device)
            self.model.eval()
        else:
            try:
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
                self.model.eval()
                self.model.to(self.device)
            except Exception as e:
                # Commonly due to hidden-size mismatch when swapping base models
                print(f"Warning: failed to load adapter on base '{self.base_model_name}': {e}")
                print("Falling back to base-only mode. For the trained adapter, use --base-model gpt2 (the training base).")
                self.model = base_model.to(self.device)
                self.model.eval()

        print("Model loaded.")
        
    def load_hole_data(self):
        """Load hole information from the multi-task dataset (strategies + descriptions)."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Data file not found: {self.data_file}. Pass --data-file to point at the dataset."
            )

        from collections import Counter
        pars = {}
        yardages = {}

        with open(self.data_file, "r", encoding="utf-8") as f:
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
                    }
                    pars[hole] = Counter()
                    yardages[hole] = Counter()

                # Track par/yardage candidates
                if isinstance(example.get("par"), int):
                    pars[hole][example["par"]] += 1
                if isinstance(example.get("yardage"), int):
                    yardages[hole][example["yardage"]] += 1

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
                    # Extract lines that look like numbered bullets
                    for bl in re.findall(r"\n\s*\d+\.\s*(.+?)\n", prompt_text + "\n", flags=re.DOTALL):
                        bl = fix_mojibake(bl.strip())
                        if len(bl) > 8:
                            self.hole_data[hole]["insights"].add(bl)

                # Also capture any base description on original records
                base_desc = (example.get("description") or "").strip()
                if base_desc and len(base_desc) > 20:
                    self.hole_data[hole]["descriptions"].add(fix_mojibake(base_desc))

        # Convert description sets to lists for easier handling
        for h, hole_data in self.hole_data.items():
            hole_data["descriptions"] = list(hole_data["descriptions"])
            hole_data["insights"] = list(hole_data["insights"])
            # Finalize par/yardage as most common values seen for this hole
            if pars.get(h):
                hole_data["par"] = pars[h].most_common(1)[0][0]
            if yardages.get(h):
                hole_data["yardage"] = yardages[h].most_common(1)[0][0]

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
            r"drive\s*(?:it\s*)?(\d{2,3})\s*yards?",
            r"hit\s*(?:it\s*)?(\d{2,3})\s*yards?",
            r"(\d{2,3})\s*yard\s*drive",
            r"(\d{2,3})\s*yards?",
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
        
        if hole['strategies']:
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
        """Generate hole description by synthesizing multiple source descriptions"""
        if hole_num not in self.hole_data:
            return "No data for that hole."
        
        hole = self.hole_data[hole_num]
        descriptions = hole.get('descriptions', [])
        insights = hole.get('insights', [])

        # Prefer synthesizing from prompt-derived insights when available
        sources = insights if len(insights) >= 1 else descriptions

        if not sources:
            return "A challenging hole at Bethpage Black."

        # If only one source, return it as-is (lightly cleaned)
        if len(sources) == 1:
            return de_template(sanitize_hole_header(clean_bleed(fix_mojibake(sources[0])), hole_num))

        # Multiple sources - ask model to synthesize them
        prompt = (
            f"TASK: description_synthesis\n"
            f"Combine these {len(sources)} observations for Hole {hole_num} at Bethpage Black "
            f"(Par {hole['par']}, {hole['yardage']} yards) into one cohesive narrative:\n\n"
        )

        for i, desc in enumerate(sources, 1):
            prompt += f"{i}. {desc}\n\n"
        
        prompt += "Synthesized description:"
        if self.use_guidance:
            prompt += "\n\nGuidance: " + GUIDANCE

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600).to(self.device)
        expected_hole = hole_num

        # Build a small banned words list to avoid clunky artifacts
        banned_words = [
            "Hints", "Hints:", "Guidance", "Guidance:", "diagram", "Diagram", "Zone", "zone",
            "SuccessOn", "decision sharing", "technical executing", "Permanent damage"
        ]
        bad_words_ids = [self.tokenizer.encode(w, add_special_tokens=False) for w in banned_words]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                bad_words_ids=[ids for ids in bad_words_ids if ids],
            )

        # Decode only the generated tail, not the prompt
        gen_ids = outputs[0, inputs["input_ids"].shape[-1]:].detach().cpu()
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        response = clean_bleed(fix_mojibake(response))
        response = sanitize_hole_header(response, expected_hole)
        response = de_template(response)
        response = sentence_boundary_trim(response)
        
        # If model fails to generate good description, fall back to first available
        if not response or len(response) < 20 or "Strategy:" in response:
            return sources[0]  # Fallback to a source snippet
        
        return response

    def get_strategy_recommendation(self, hole_num, drive_distance):
        """Get strategy recommendation using the trained model"""
        if hole_num not in self.hole_data:
            return "No data for that hole."
        
        # Create prompt similar to training format
        hole = self.hole_data[hole_num]
        # Align with training format
        prompt = (
            "TASK: strategy_selection\n"
            f"Hole {hole_num}, Bethpage Black: Par {hole['par']}, {hole['yardage']} yards. Golfer drives {drive_distance} yards. Best strategy?\n"
            "Respond with: Strategy: <number> yards"
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=250).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0].detach().cpu(), skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        # Extract strategy number
        strategy_match = re.search(r"Strategy:\s*(\d{2,4})\s*(?:yards?)?", response, re.IGNORECASE)
        if not strategy_match:
            # Fallback: any 2–4 digit number
            strategy_match = re.search(r"(\d{2,4})", response)
        if strategy_match:
            strategy_yards = int(strategy_match.group(1))
            
            # Add explanation of choice
            available_cutoffs = [s['cutoff_distance'] for s in hole['strategies'] if isinstance(s.get('cutoff_distance'), int)]
            eligible = [c for c in available_cutoffs if c <= drive_distance]
            
            explanation = f"\nRecommended Strategy: {strategy_yards} yards\n"
            explanation += f"Available strategies: {sorted(available_cutoffs)} yards\n"
            explanation += (
                f"Eligible for {drive_distance}-yard drive: {sorted(eligible) if eligible else ['None - using shortest option']}\n"
            )
            explanation += "Logic: Choose highest available <= drive distance"
            
            return explanation
        else:
            return f"Model response: {response}"
    
    def process_input(self, user_input):
        """Process user input and generate appropriate response"""
        user_input = user_input.strip()
        
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
    parser.add_argument("--adapter_dir", required=False, default="outputs/bethpage-lora/checkpoint-8_hour")
    parser.add_argument("--data-file", required=False, default="data/bethpage_black/train_multitask_case_fixed.jsonl")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", help="Optional one-shot prompt; if provided, runs once and exits", default=None)
    parser.add_argument("--no_desc_guidance", action="store_true", help="Disable extra guidance for description synthesis prompts")
    parser.add_argument("--base-model", default="gpt2", help="Base model name (e.g., gpt2, gpt2-medium). Adapter was trained on gpt2.")
    parser.add_argument("--base-only", action="store_true", help="Use base model without loading the LoRA adapter (for comparison)")
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
