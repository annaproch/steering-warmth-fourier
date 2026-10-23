# Broader strength and first-token comparison

The authored pilot (cad04b8) showed ordered held-out activation projections, but no
convincing monotonic warmth control at layer 16 and strengths ±0.25/±0.5. Random
directions also changed output text. Both methods left the arithmetic answer identical
to baseline at all tested strengths, and 54/84 outputs reached the 256-token cap.

Keep the data, vectors, layer and three validation prompts fixed. Compare the same
broader strengths (-2, -1, +1, +2) with and without steering the last prompt position.
`--steer-first-token` adds the displacement only at the final prompt position during
prefill, then at each generated-token position during decoding. Earlier prompt
positions remain unchanged. Without the flag, prefill is untouched. Baseline and
warmth-prompt controls have no activation intervention in either mode.

Strengths multiply the training mean warm-minus-neutral vector; they are not target
warmth levels. ±2 may degrade output. The two random directions use matching norms
and the same intervention timing. The vector norm and timing flag are saved in each
generation folder's settings. Changing either timing or strengths creates a different
folder. Existing outputs, collection caches and trained vectors remain usable.

## On Mac, after pulling

Run these separately from the repository root:

```sh
source .venv/bin/activate
export RUN=warmth/runs/authored_v1

sh warmth/run_matched.sh check --run "$RUN" --device mps \
  --method anchor --layer 16 --steer-first-token

sh warmth/run_matched.sh steer --run "$RUN" --device mps \
  --split validation --method anchor --layer 16 --questions 3 \
  --strengths -2 -1 1 2 --random-controls 2 --max-tokens 768

sh warmth/run_matched.sh steer --run "$RUN" --device mps \
  --split validation --method anchor --layer 16 --questions 3 \
  --strengths -2 -1 1 2 --random-controls 2 --max-tokens 768 \
  --steer-first-token
```

This is 42 responses per mode (84 total), including baselines, four linear strengths,
eight random controls and warmth prompting per prompt. The larger cap reduces
truncation but does not guarantee completion. Both runs use the same cap to preserve
comparability. The anchor method is chosen for this first comparison because its
held-out direction alignment was stronger than last-token at layer 16; this is a
validation-informed development choice, not a demonstrated causal advantage.

No recollection or reanalysis is needed. Resume by rerunning an identical command;
run one writer per output folder. New intervention code has a new hash, so old
generation folders are preserved. First-token sanity results are written separately
to `sanity_checks_first_token.json`. The timing flag is also locked by `select` for
future test runs; old selection locks lacking that field must not be silently reused.

Inspect warmth directionality, correctness, repetition and truncation against the
random controls. Do not count different wording as successful steering or select a
strength based only on the nicest response. First-token intervention is a mechanistic
test, not a promise that this will work. Keep test prompts untouched.

```sh
git add -f "$RUN/sanity_checks_first_token.json" "$RUN"/generations_*
git diff --cached --stat
git commit -m "Record warmth strength and first-token comparison"
git push
```
