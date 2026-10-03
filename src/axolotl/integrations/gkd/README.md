# On-Policy Distillation (GKD)

On-policy distillation (OPD) has the **student generate its own rollouts** and a
**teacher provide dense, token-level supervision** on those student-visited
states. It is the "missing middle" between GRPO (on-policy, but sparse scalar
reward) and off-policy [KD](../kd/README.md) (dense token-KL, but on fixed
teacher/ground-truth prefixes). The core follows GKD
([2306.13649](https://huggingface.co/papers/2306.13649)).

## Usage

```yaml
plugins:
  - "axolotl.integrations.gkd.GKDPlugin"

gkd_trainer: true
gkd_teacher: meta-llama/Llama-3.1-70B-Instruct   # Axis C: teacher (shared vocab)

gkd_lmbda: 1.0          # Axis B: fraction of steps on student rollouts (1.0 = fully on-policy)
gkd_beta: 1.0           # Axis A: 0 = forward-KL, 1 = reverse-KL, between = JSD
gkd_temperature: 0.9
gkd_max_new_tokens: 256
gkd_seq_kd: false       # for the off-policy fraction, distill teacher-generated sequences

# GKD generates from the prompt, so packing must be off and the prompt must be masked.
sample_packing: false
train_on_inputs: false

datasets:
  - path: your/sft-dataset
    type: chat_template
```

Any standard SFT dataset works — the prompt is recovered from the tokens masked
with `-100`, so no special preprocessing or logprob precomputation is required
(unlike the offline KD plugin).

## The four axes

Every OPD method in the literature is a point in this space; the plugin names each
as a replaceable seam so new papers land as config, not a new trainer.

| Axis | Question | Config | v1 |
|------|----------|--------|----|
| A. Divergence | student scored vs teacher how? | `gkd_beta`, `gkd_divergence` | fwd/rev-KL, JSD (`divergence.py`) |
| B. Rollout | whose prefixes? | `gkd_lmbda`, `gkd_seq_kd` | on-policy fraction (`rollout.py`) |
| C. Teacher | where does supervision come from? | `gkd_teacher` | external HF model (`trainer._resolve_teacher`) |
| D. Weighting | which tokens get the loss? | — | uniform (`trainer._token_weights`, seam for v2) |

## Notes / constraints

- **Shared vocabulary.** Teacher and student must share a tokenizer/vocab; the
  trainer validates `vocab_size` and raises otherwise.
- **Teacher cost.** The full teacher is held in-process and run every step. Budget
  memory accordingly (quantize via `gkd_teacher_init_kwargs`, or use DeepSpeed/FSDP).
- **Stability.** OPD can inflate length or collapse diversity; tune
  `gkd_temperature` and `gkd_beta` (reverse-KL is mode-seeking) to mitigate.
