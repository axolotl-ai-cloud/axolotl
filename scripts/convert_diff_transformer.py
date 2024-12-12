"""Test conversion of transformers model attention to differential attention."""
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from axolotl.integrations.diff_transformer.convert import convert_to_diff_attention


def setup_model(
    model_name: str, device: str = "cuda"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def convert_model_attention(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Convert model to use differential attention"""
    try:
        model = convert_to_diff_attention(model)
        return model
    except Exception as exception:
        print(f"Error during model conversion: {exception}")
        raise


def test_inference(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    """Run test inference"""
    # Test prompts
    test_prompts = [
        "The quick brown fox",
    ]

    for prompt in test_prompts:
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            from time import time

            start = time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    num_beams=1,
                    do_sample=False,
                    # temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                    # use_cache=True,
                )
            elasped = time() - start
            print(f"generation time: {elasped}s")

            # Decode
            print(outputs)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}\n")

        except Exception as exception:
            print(f"Error during inference: {str(exception)}")
            raise


def save_converted_model(model: AutoModelForCausalLM, output_dir: str) -> None:
    """Save the converted model"""
    print(f"Saving converted model to {output_dir}")
    model.save_pretrained(output_dir)


def main():
    # Configuration
    model_name = "HuggingFaceTB/SmolLM2-135M"
    # model_name = "openlm-research/open_llama_3b_v2"
    output_dir = "./converted_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load model and tokenizer
        model, tokenizer = setup_model(model_name, device)

        # Print original model info
        print("Original model config:")
        print(f"\t- Hidden size: {model.config.hidden_size}")
        print(f"\t- Number of attention heads: {model.config.num_attention_heads}")

        # Test the original model
        test_inference(model, tokenizer)

        # Convert to differential attention
        model = convert_to_diff_attention(model)
        model.to(model.device)
        print("Model conversion completed")

        # Test the converted model
        test_inference(model, tokenizer)

        # Save converted model
        save_converted_model(model, output_dir)

    except Exception as exception:
        print(f"Error during test: {str(exception)}")
        raise


if __name__ == "__main__":
    main()
