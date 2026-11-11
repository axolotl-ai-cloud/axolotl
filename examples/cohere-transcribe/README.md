# Finetune Cohere Transcribe with Axolotl

[Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) is a 2B-parameter open-weight speech recognition model from Cohere. It pairs a large Conformer audio encoder with a lightweight Transformer decoder and supports 14 languages: Arabic, Chinese, Dutch, English, French, German, Greek, Italian, Japanese, Korean, Polish, Portuguese, Spanish and Vietnamese.

This guide shows how to fine-tune it with Axolotl's speech-to-text (encoder-decoder) support.

## Getting Started

1. Install Axolotl from main following the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build), including the `audio` extra (`librosa` and `soundfile`, required for audio decoding and feature extraction):

    ```bash
    # from a source checkout
    uv pip install -e '.[audio]'

    # or, from the next release onward
    uv pip install 'axolotl[audio]'
    ```

2. Request access to the [model repo](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) and log in with `hf auth login`.

3. Run the fine-tuning example:

    ```bash
    axolotl train examples/cohere-transcribe/cohere-transcribe-03-2026-lora.yaml
    ```

## Dataset format

Each row needs an audio column and a transcript column. Configure them under `asr`:

```yaml
asr:
  audio_column: audio            # a datasets Audio column, or a dict with array/bytes/path
  text_column: raw_transcription
  language: en                   # used when language_column is not set
  language_column:               # optional per-row language code (en, de, ja, ...)
  punctuation: true              # train for punctuated, cased output
```

Audio is decoded, downmixed to mono and resampled to 16 kHz automatically. Write transcripts the way you want the model to output them: with `punctuation: true`, use cased and punctuated text (FLEURS `raw_transcription`, not `transcription`).

## Tips

- Clips longer than 35 seconds or shorter than 20 milliseconds are dropped with a warning when the dataset loads; split long recordings into shorter segments first.
- `sequence_len` caps the decoder length (the 10-token language/task prompt plus the transcript) and must not exceed the decoder limit of 1024; longer transcripts are truncated.
- Prompt tokens (language, punctuation and task tags) are excluded from the loss unless `train_on_inputs: true`.
- Use `attn_implementation: sdpa` or `eager`. Flash attention, sample packing, Liger and Cut Cross Entropy are not supported for this model.
- The encoder's BatchNorm layers stay in eval mode during training so their running statistics are not skewed by small or padded batches.
- Do not set `trust_remote_code`; Axolotl's pinned Transformers version includes the model.
- The model expects the language to be given; it does not detect it.
- For FSDP, set the wrap classes explicitly, since the model's default list names a class that does not exist:

    ```yaml
    fsdp_config:
      transformer_layer_cls_to_wrap: ParakeetEncoderBlock,CohereAsrDecoderLayer
    ```

## Inference

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
model = CohereAsrForConditionalGeneration.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026", dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "./outputs/cohere-transcribe-lora")

audio = load_audio("sample.wav", sampling_rate=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Related Resources

- [Cohere Transcribe model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- [Cohere Transcribe release blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)
- [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
