# LM Eval Harness

Run evaluation on model using the popular lm-evaluation-harness library.

See https://github.com/EleutherAI/lm-evaluation-harness

## Usage

```yaml
plugins:
  - axolotl.integrations.lm_eval.LMEvalPlugin

lm_eval_tasks:
  - gsm8k
  - hellaswag
  - arc_easy

lm_eval_batch_size: # Batch size for evaluation
output_dir: # Directory to save evaluation results
```

## Citation

```bib
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```
