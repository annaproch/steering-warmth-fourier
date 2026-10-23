# Four-scenario development pilot

Before another rewrite iteration, run the [judge calibration set](data/calibration_v1/README.md)
to check base correctness, content fidelity, warmth discrimination and A/B position sensitivity.

The committed run `mac_curated_v1` (bf6ae4e) had 0/20 accepted triplets. In 17/20,
moderate and warm tied under the judge; two triplets contained identical text at all
levels. Mean response lengths were 55.85, 73.70 and 84.75 words; 37/60 responses
exceeded the requested ±20% interval in either direction. No drafts were truncated.
All 60 were scored content-preserving, despite a change from "not guaranteed" to
"unlikely" in keyboard advice. The judge also confused rewrite defects with base
correctness. These are dataset and evaluation failures, not manifold results.

V2 keeps those results intact. Its rewrite prompt includes concrete style examples,
permits grounded acknowledgment while preserving substantive content, prohibits
invented emotion and changed uncertainty, and specifies a numerical word budget.
The hard acceptance gate rejects length violations and normalized duplicate variants.
`draft_checks.json` reports these failures without deleting or retrying them.

The judge now evaluates base correctness once per scenario, rewrite fidelity/quality
separately per response, and warmth in three blinded pairwise comparisons. Pair
presentation order is randomized deterministically; intended labels are hidden from
the judge. A/B/tie are valid outcomes. Acceptance requires the warmer response to be
the higher requested level for all three pairs, as well as the existing correctness,
quality and truncation gates. Ties reject the triplet rather than becoming forced ranks.
Absolute `warmth_0_4` stays blank in V2 review CSVs, and `observed_warmth` is null in
accepted records; pairwise results are stored as `warmth_evidence`. This provides
ordinal evidence only, not equal spacing or continuous warmth coordinates.

Pairwise comparison is still an uncalibrated model judgment and can have position
bias. Human audit and, for a larger study, counterbalanced repeats/multiple judges
remain necessary. No real model run has demonstrated that V2 improves acceptance.

## Run on Mac

Activate the existing virtualenv and use a fresh directory:

```sh
source .venv/bin/activate
export RUN=warmth/runs/mac_curated_v2_dev

sh warmth/run_matched.sh seed --run "$RUN" \
  --input warmth/data/curated_smoke_v1.jsonl --per-category 5 --development-pilot

sh warmth/run_matched.sh draft --run "$RUN" --device mps --rewrite-v2

sh warmth/run_matched.sh review --run "$RUN"

sh warmth/run_matched.sh judge --run "$RUN" --device mps \
  --judge-model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/"

sh warmth/run_matched.sh validate --run "$RUN"
```

This takes one training scenario per category from the original deterministic split:
four scenarios, 12 rewrites, four base reviews, 12 content reviews and 12 pair reviews.
Every pilot scenario retains `split=train`. There is no held-out evaluation or useful
manifold estimate in this pilot. Do not proceed to collection or analysis yet.
Drafting and judging resume with the same arguments. Always retain `--rewrite-v2`
when resuming drafting; settings locks prevent switching protocols within a run.
The review and judge stages detect V2 automatically from the saved draft settings.

Inspect `draft_checks.json`, `review.csv`, `pair_review.csv`, and `validation_report.json`.
Manual reviewers can edit the two CSVs, keeping IDs and text intact and adding their
reviewer attribution. `review_key.json` and `pair_review_key.json` reveal the hidden
levels and must not be consulted while rating. Do not select only favorable outputs
for audit. Share all failures as well as successes.

To commit the small pilot for inspection (no activations or weights are present):

```sh
git add -f "$RUN"
git diff --cached --stat
git commit -m "Record revised warmth development pilot"
git push
```

After auditing this pilot, freeze the prompt/rubrics before expanding. Omit
`--development-pilot` in a new run to use the full set. Because the earlier audit
already inspected the original test examples, that 20-scenario set is development
data for future claims; obtain fresh untouched scenarios for confirmatory evaluation.
