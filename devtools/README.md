This directory contains example config files that might be useful for debugging. Please see [docs/debugging.qmd](../docs/debugging.qmd) for more information.

## Merging Additional Hole Descriptions

To extend hole descriptions from 2 to 5 variants without editing existing files:

1. Fill `data/bethpage_black/additional_descriptions_template.jsonl` with lines like:
	`{ "hole": 1, "new_description_3": "...", "new_description_4": "...", "new_description_5": "..." }`
2. Save as `additional_descriptions.jsonl` in the same folder (or keep filename and adjust flag).
3. Run the merge + validation script:

	```bash
	python scripts/merge_additional_descriptions.py \
	  --base data/bethpage_black/data_template.json \
	  --additional data/bethpage_black/additional_descriptions.jsonl \
	  --out data/bethpage_black/data_template_5desc_merged.jsonl \
	  --jsonl
	```

Validation failures (non-zero exit):
* Missing any of new_description_3/4/5
* Too short (<40 words variants 3/4, <25 words variant 5) or >170 words
* Starts with boilerplate prefix ("This par", "At ", "Hole ")
* Near-duplicate Jaccard similarity >= 0.80 between any pair of the five

Fix reported issues and rerun. Successful run writes merged JSONL with all five variants per hole.

