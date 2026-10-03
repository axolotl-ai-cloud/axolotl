# Public procedural decision mix

`build_public_decision_mix.py` materializes a reproducible local train/dev mix
from `tasksource/procedural-typed-decisions` at revision
`916e6cce365a65c37d58db70c9e369795817651e`. It writes no JevBench, Nimble,
or `unique41088` rows. Generate the public recipe data with:

```bash
python src/axolotl/integrations/decision/scripts/build_public_decision_mix.py ./data/public-procedural
```

Callers who are authorized to hold normalized JevBench and Nimble data can add
each local file with `--protected-normalized`; the generator uses it only for
canonical-state exclusion and never copies it to the output.

## License and attribution

The pinned dataset card declares `Apache-2.0` for the dataset. The separate
tasksource generator repository publishes its code under `CC-BY-4.0`; that is
the code repository's license declaration, not a conflicting dataset-card
license. Preserve the generated contract, the pinned dataset revision, and the
dataset-card and generator-repository links when sharing materialized output.

The recipe uses `nvidia/Nemotron-Labs-Diffusion-8B`; its model card specifies
the NVIDIA Nemotron Open Model License. This document makes no claim about
redistributing model weights or adapters.

## Heldout semantics

When optional normalized JevBench and Nimble heldouts are supplied, the
generator removes any public candidate with the same canonical state. This is
effective across sources. It does not establish group/family disjointness
across those datasets: `hygiene.family_key` and source/group identities include
the source, while no cross-source semantic-group mapping is available. The
contract records canonical-state decontamination only. The public train/dev
outputs are checked for canonical-state, source/family, and source/group
overlap.

No quality or benchmark-generalization claim follows from this recipe. A
cross-source semantic-group mapping, preprocessing, and heldout evaluation are
required before such a claim.

The LoRA targets, rank, alpha, LoRA+ ratio, optimizer, learning rate, batch
shape, diffusion settings, and checkpoint-selection settings match the
default 8B decision recipe. CPU preprocessing at the pinned model revision filtered 640
train rows and 67 dev rows at the 2,048-token logical budget. The prior
temperature-0.5 schedule repeated 304 of the 30,080 usable train rows. This
recipe uses `mixture.temperature: 1.0`, which the reference preprocessing run
confirmed schedules each usable row once: 30,080 draws, or 235 full effective
batches of 128. The prepared development set has 3,005 rows after its 67
logical-budget drops. `max_steps` is therefore 235.

The terminal checkpoint was also the development-selected checkpoint. Under the
frozen two-read heldout protocol it scored 258/324 on Nimble and 15,143/22,773
on JevBench (66.4954%), with JevBench NLL 1.316374, Brier 0.435918, and ECE15
0.166728. The unadapted 8B reader's JevBench ECE15 was 0.135695, so this run
does not support a calibration claim. Its procedural development loss/Brier
improved to 0.289864/0.058297, but that did not translate into matching
private-recipe heldout quality. The development datasets differ, so their
losses are not comparable. The public procedural mix remains a reproducible
format and learning starter, not an exposure-matched reproduction of the
321-step private recipe, a route-quality result, or a calibration claim.
