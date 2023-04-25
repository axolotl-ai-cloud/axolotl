# FAQs

- Can you train StableLM with this? Yes, but only with a single GPU atm. Multi GPU support is coming soon! Just waiting on this [PR](https://github.com/huggingface/transformers/pull/22874)
- Will this work with Deepspeed? That's still a WIP, but setting `export ACCELERATE_USE_DEEPSPEED=true` should work in some cases
