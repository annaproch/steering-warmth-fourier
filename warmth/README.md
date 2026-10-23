# EPITOME emotional-warmth manifold pilot

Uses EPITOME's human Emotional Reactions (ER) labels: 0 absent, 1 weak,
2 strong. Reuses the existing local Goodfire `CubicSpline1D` natural spline
implementation in `vendor/causalab`, commit
`1b6f43a509d2a4ae49fd1bc4252e13889d57e38b`. PCA and centroid preparation are
implemented here because these are unpaired replies, unlike the conciseness
pilot. Existing conciseness code and outputs are unchanged.

Status: public data downloaded, 150-reply pilot prepared, synthetic tests passed.
No Llama activation collection or generation has been run for this experiment.

## Separate commands

Run from `/workspace/project`. Existing workspace dependencies, Goodfire checkout
and downloaded Llama-3.1-8B-Instruct are reused. `run.sh` uses
`/home/agent/venv/bin/python`, with local `.deps` on PYTHONPATH.

```sh
# Already done. Only needed to download the data again; requires network.
sh warmth/download_data.sh

# Already done. Rebuild deterministic 90/30/30 train/validation/test split.
sh warmth/run.sh prepare

# 150 teacher-forced forwards, collecting block outputs 12/16/20.
# CPU by default; resumable per reply. This is the expensive stage.
sh warmth/run.sh collect

# Training-only PCA, three centroids, natural spline, endpoint-line comparison.
sh warmth/run.sh fit

# Validation smoke test: 2 prompts x 13 conditions = 26 generations.
sh warmth/run.sh steer --split validation --questions 2 --max-tokens 128
```

The steering command prints its output folder:
`warmth/runs/pilot/generations_<hash>/`.
Open its `ratings.csv`, keeping the method labels in `outputs.json` hidden
while rating. Fill every row:

- `er_0_2`: no, weak, or strong expressed emotional warmth/concern;
- `coherence_1_5`: 1 unusable, 5 fully coherent;
- `unsupported_claim_0_1`: 1 if the reply introduces an unsupported factual
  assertion, invented personal experience, or unjustified reassurance; else 0;
- `notes`: optional concerns, disagreement with false beliefs, or omitted advice.

Then run, replacing the path with the one printed:

```sh
sh warmth/run.sh score --ratings warmth/runs/pilot/generations_<hash>/ratings.csv
```

These are human ratings, not an automatically validated empathy metric. Use the
[EPITOME rubric](https://github.com/behavioral-data/Empathy-Mental-Health) and
prefer a second independent rater. The original source reply is not a factual
answer key; these scores cannot prove preservation of factual content.

After reviewing validation warmth, fluency, unsupported claims and length:

```sh
# Optional validation strength comparison; separate output folders automatically.
sh warmth/run.sh steer --split validation --questions 5 --strength 0.5

# Lock strength and other settings before evaluating unseen test prompts.
# Example uses the default strength; use the value chosen on validation.
sh warmth/run.sh steer --split test --questions 10 --strength 1 --max-tokens 128

# Fill and score the NEW test ratings.csv as above.
```

No test results should influence layer, rank, strength or token-budget selection.
`--questions` is capped by available unique seeker groups; it can be expanded
without regenerating completed conditions. Changed strength, targets, token cap,
split or fitted artifacts get separate generation folders. Existing ratings are
preserved when resuming. Do not run simultaneous writers on the same run.

For a GPU, add `--device cuda` to `collect` and `steer`. There is no GPU in the
current workspace. Threads default to 8; weights use bfloat16.

## What is fitted

1. Join the three public Reddit annotation files by seeker/reply IDs. Collapse
   consistent duplicates; exclude a key if any file has conflicting annotations.
   Exclude cross-file text mismatches and repeated reply texts.
2. Group seeker IDs and duplicate normalized seeker texts before deterministic
   60/20/20 splitting. This prevents the same seeker context crossing splits;
   author-disjoint splitting is not possible with these files.
3. Within each split, sample equal counts for ER 0/1/2 inside exact IP/EX labels
   and log2 response-word-length bins. Default 30 per ER for training and 10 per
   ER for each evaluation split. Matching reduces, but does not remove, topic
   and length confounding. Manifest records actual counts and source hashes.
4. Use a fixed neutral system message and teacher-force the actual annotated
   reply. Average residual-stream block outputs across **response content tokens
   only**, excluding prompt and end-of-turn tokens. No ER label enters the prompt.
   Skip examples over 768 total tokens; fitting rebalances surviving strata and
   records exclusions. It fails if fewer than 3 replies/class/split survive.
5. Fit centered PCA on training replies only, ranks 2/4/8/16; average projected
   activations by ER; fit Goodfire's natural spline through coordinates 0,1,2.
   Equal spacing is an ordinal-label modeling assumption, not measured distances.
6. Select layer/rank using validation centroid reconstruction error normalized
   for layer scale. Save test geometry only as a descriptive diagnostic. At
   knots, the spline equals the centroids; this cannot prove nonlinear geometry.
   With only endpoints left, an omitted-middle natural spline would be a line,
   so it would not provide an independent spline interpolation test.

## Causal intervention and controls

For targets 0, 0.5, 1.5 and 2, compute a **fixed displacement** from the curve's
ER1 point to the target, reconstructing it into the full hidden dimension.
Compare against the straight ER0-to-ER2 direction and its norm-matched variant,
plus an unsteered baseline. `--strength` scales each displacement.

Add this displacement at the selected layer on each generated response-token
forward pass. The prefill is untouched, so the first generated token is
unsteered. This placement is closer to the teacher-forced response activations
than patching the unrelated prompt boundary, but averaging training tokens still
creates a distribution mismatch that the causal test must resolve.

The model's original response is **not assumed to actually be ER1**: ER1 is only
the reference point for computing displacement. Target coordinates are treatment
settings, not guaranteed output ratings. This pilot does not project live states
onto the curve, enforce a warmth level, or perform ODE integration.

Assess whether held-out outputs show graded warmth without loss of coherence or
more unsupported claims, and whether the spline improves on the linear controls.
Inspect length and cap rates: capped outputs are incomplete. More warmth alone
is not success. Three labels and unpaired replies offer limited evidence for a
one-dimensional manifold, even if the interventions work. Keep anti-sycophancy
training and evaluation as the next experiment; no such ODE is trained here.

## Outputs and reproducibility

Everything is under `warmth/runs/pilot/` (Git-ignored because it contains source
conversation text and large activations):

- `dataset.json`: splits, labels, matching strata, filtering counts, source hashes.
- `collection_config.json`, `cache/<signature>/*.pt`: pooled activations and token
  counts. The signature includes dataset, pipeline code, model path/config,
  tokenizer config, layers and length budget. Do not overwrite model weights in
  place; use a new model directory when changing checkpoints.
- `fits.pt`: PCA means/bases and centroids; splines are rebuilt using Goodfire.
- `geometry_metrics.json`, `selected.json`, `geometry.png`, `fitted_rows.json`.
- `generations_<hash>/`: settings, resumable outputs, blinded rating sheet, scores.

To scale later, use a separate run directory and pass it to **every stage**:

```sh
sh warmth/run.sh prepare --run warmth/runs/larger --train-per-level 100 --eval-per-level 30
sh warmth/run.sh collect --run warmth/runs/larger
sh warmth/run.sh fit --run warmth/runs/larger
sh warmth/run.sh steer --run warmth/runs/larger --split validation --questions 5
```

The manifest reports if exact matching yields fewer samples than requested.
Use the same `--layers`, `--model`, and `--max-length` across collection, fitting
and steering. Configuration changes invalidate caches rather than silently
mixing them. Do not rerun `prepare` over an experiment whose data you edited.

Tests (no model loading):

```sh
MPLCONFIGDIR=/tmp/warmth-mpl PYTHONPATH=.deps /home/agent/venv/bin/python -m unittest discover -s warmth -p 'test_*.py' -v
```

Sources: [EPITOME data and rubric](https://github.com/behavioral-data/Empathy-Mental-Health),
[Goodfire manifold steering code](https://github.com/goodfire-ai/causalab/tree/manifold_steering).

### Batched collection

Collection defaults to `--batch-size 2`. Pending examples are tokenized once,
sorted by total token length, and right-padded within each batch. Attention masks
exclude padding; activation pooling excludes both prompts and padding. Each
example is still cached separately, and a completed cache avoids loading the model.

```sh
sh warmth/run.sh collect --batch-size 4 --threads 8
```

The current environment exposes 9 CPU cores and about 24 GB RAM, with neither
CUDA nor MPS available to PyTorch. Start with 2; try 4 if memory permits. Batch
throughput is printed, but speedup has not been benchmarked on the real model.
GPU hosts can use `--device cuda --batch-size 8` if sufficient GPU memory is
available. Batch size does not change the cache key (minor floating-point changes
are possible). This implementation update changes the pipeline hash, so caches
from the previous code version are not reused.

## Matched response successor

The separate [matched pipeline](MATCHED_README.md) implements three-level drafting, independent review, last-token/suffix-anchor collection, held-out diagnostics, and linear-steering controls. It does not change this EPITOME pilot or reuse its activation cache.
