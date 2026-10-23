# Judge calibration v1

24 original assistant-authored cases, separate from the generated pilot outputs:

| Task | Cases | Checks |
|---|---:|---|
| Base correctness | 8 | Four true/false pairs: time arithmetic, multiplication, inclusive threshold, counting |
| Content preservation | 8 | Four passes/four failures: exact copy, changed uncertainty, grounded vs invented emotion, conditional advice, factual disagreement |
| Warmth | 8 | Four grounded-care contrasts, three ties, one warmer-but-wrong response |

Every warmth pair is shown in both A/B orders in separate stateless calls, producing
32 total judge calls. Presentation order is shuffled. Expected labels and rationales
are stored in `answer_key.json`, not passed to the judge. The actual prompts contain
only the existing V2 task rubric and case input, with no IDs or intended levels.
Preparation and scoring use only the Python standard library. Judging uses the same
Transformers runner and non-thinking V2 rubrics as the experiment.

These labels are proposed calibration targets, not independent human consensus.
Arithmetic is directly checkable; warmth contrasts and non-identical ties need human
review. The warmer-but-wrong response is deliberately unacceptable on correctness;
the warmth-only judge should still distinguish its caring tone. Warmth is not a
substitute for the separate correctness/content gate.

The set is diagnostic development data designed around observed failures. It is not
a benchmark or a dataset for fitting directions. Similarities to the earlier pilot
are intentional. Do not train steering on it, treat flipped pairs as independent
examples, or repeatedly tune a judge on these labels and report its score as held-out
performance. There are no finely spaced moderate/warm targets here: test obvious
contrasts before asking a judge to resolve subtle ones.

## Run on Mac

From the repository root, after activating the existing virtualenv:

```sh
source .venv/bin/activate
export CAL=warmth/runs/calibration_v1_qwen

python warmth/calibration.py prepare --run "$CAL"

python warmth/calibration.py judge --run "$CAL" --device mps \
  --judge-model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/"

python warmth/calibration.py score --run "$CAL"
```

Judging resumes per presentation. Use a fresh `--run` for another model, token budget,
or rubric. A model need not match the experiment writer: this evaluates the judge in
isolation. Never put the answer key into its prompt. All raw responses, parsed
verdicts, settings and the final `scores.json` are saved. Quality scores are collected
for content cases but not calibrated by this version; only preservation is scored.

Check correctness separately for positive and negative controls; an always-yes judge
must not look successful. Check `position_check.consistent` and `both_correct`:
consistent mistakes are not successes. Any arithmetic failure or failure on an exact
copy/identical pair needs investigation. Warmth mismatches need a text audit, not an
automatic assumption that the proposed answer key is infallible. No aggregate pass
threshold automatically qualifies the model for research use.

To share results:

```sh
git add -f "$CAL"
git diff --cached --stat
git commit -m "Record warmth judge calibration results"
git push
```

No inference results are bundled with the calibration set itself.
