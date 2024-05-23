# FAQs

- Can you train StableLM with this? Yes, but only with a single GPU atm. Multi GPU support is coming soon! Just waiting on this [PR](https://github.com/huggingface/transformers/pull/22874)
- Will this work with Deepspeed? That's still a WIP, but setting `export ACCELERATE_USE_DEEPSPEED=true` should work in some cases
- `Error invalid argument at line 359 in file /workspace/bitsandbytes/csrc/pythonInterface.c`
`/arrow/cpp/src/arrow/filesystem/s3fs.cc:2598:  arrow::fs::FinalizeS3 was not called even though S3 was initialized.`
This could lead to a segmentation fault at exit. Try reinstalling bitsandbytes and transformers from source.
