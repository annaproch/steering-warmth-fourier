# Directly authored warmth pilot

`data/authored_warmth_v1.jsonl` contains 30 scenarios and **90 complete responses**,
authored directly by the assistant. There is no model generation step or Qwen approval
gate. Each record contains the actual responses as well as their openings and shared
substantive answer. The 10 support, 10 advice and 10 factual-disagreement scenarios
give warmth a natural context. Ordinary factual questions without such a context are
deliberately omitted in this first causal experiment.

Only the short opening changes within each triplet. The remainder of each response
is byte-for-byte identical, including corrections, calculations, advice and uncertainty.
The last response-content token is therefore also identical within a triplet. All 90
openings are unique. Word-count differences within triplets are below 10%; actual
token lengths and positions may still differ. Collection records token counts.

Labels mean **intended** neutral/moderate/warm, not independently measured warmth or
equally spaced points on a manifold. The importer explicitly records that the data
are assistant-authored and not independently validated. `accepted.json` is merely
the existing collector's input filename; it does not signify judge or human approval.
No ratings are invented. Optional blinded review is available in `audit.csv` with
the labels in a separate `audit_key.json`; it is not a prerequisite for collection.

The split is fixed at six train/two validation/two test scenarios per category,
18/6/6 overall. All variants of a scenario stay together. These are scenario holdouts,
not proof of new-topic or new-language-template generalization: authoring style and
broad themes recur across splits. Do not tune on test. Later causal evaluation should
also include fresh prompts without the training openings or copied answers.

This narrow design tests **expressed warmth in an opening**. It may learn empathy
phrases, emotional framing, or sentence structure rather than general warmth. Identical
answer endings may attenuate the signal; last/anchor/mean extraction are all saved
separately. Failure is informative. No spline claim is justified until causal steering
works beyond the authored wording.

## Commands on Mac (run separately)

```sh
source .venv/bin/activate
export RUN=warmth/runs/authored_v1

python warmth/prepare_authored.py --run "$RUN"

sh warmth/run_matched.sh collect --run "$RUN" --device mps --batch-size 2

sh warmth/run_matched.sh analyze --run "$RUN" --split validation --bootstrap 200

sh warmth/run_matched.sh check --run "$RUN" --device mps --method last --layer 16

sh warmth/run_matched.sh steer --run "$RUN" --device mps \
  --split validation --method last --layer 16 --questions 3 --max-tokens 256

sh warmth/run_matched.sh steer --run "$RUN" --device mps \
  --split validation --method anchor --layer 16 --questions 3 --max-tokens 256
```

Preparation requires only standard Python and checks all records. Collection includes
all 90 responses in one pass, capturing three extraction methods at layers 12/16/20;
analysis fits vectors using only training scenarios. Batch size can change on resume;
try 4 if memory permits. Use the same model path and extraction settings throughout.
There is no need to run `seed`, `draft`, `review`, `judge` or `validate` for this authored
workflow. Importing into an existing unrelated run is rejected, and rerunning the same
import preserves audit edits and cached outputs.

Each steering command includes baseline, positive/negative linear shifts, random
controls and warmth prompting. The first three validation prompts cover all three
categories. Expand to `--questions 6` after inspecting the initial output. Watch for
truncation, irrelevant sympathy, repeated openings, changed facts and excessive agreement;
an empathetic preface by itself is not sufficient evidence of useful general control.
Independently assess generated responses before choosing a method/layer. Test selection
and scoring commands are described in `MATCHED_README.md`; do not freeze a selection
before reviewing validation behavior.

To share lightweight outputs after the pilot (omit activation caches and vectors):

```sh
git add -f "$RUN/authored_manifest.json" "$RUN/dataset_checks.json" \
  "$RUN/accepted.json" "$RUN/diagnostics_validation.json" "$RUN/sanity_checks.json"
git add -f "$RUN"/generations_*
git diff --cached --stat
git commit -m "Record authored warmth steering pilot"
git push
```

No model inference or causal result is included with this dataset's implementation.
