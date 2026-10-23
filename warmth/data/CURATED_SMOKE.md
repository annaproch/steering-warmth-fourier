# Curated replacement smoke set

`curated_smoke_v1.jsonl` replaces the flawed synthetic 20-scenario seed set. It contains five original, assistant-authored scenarios per category. It is not Warm AI data, an independently human-validated corpus, or a benchmark.

Factual disagreement uses directly checkable arithmetic and an explicit threshold rule. Ordinary questions include an agreement case to avoid equating correctness with disagreement. Emotional support and advice avoid diagnoses, invented reactions, guarantees, and unsupported praise. Base answers are deliberately neutral; invariants preserve claims, recommendations, qualifications and boundaries rather than the user's false belief. All invariants are nonempty strings.

The five examples per category split into three training, one validation and one test scenario (12/4/4 overall). This is enough for a pipeline smoke test, not strong evidence for generalization or curvature. The cases are deliberately simple and do not cover realistic misinformation or adversarial pressure. Some scenarios share broad everyday themes; do not interpret the split as a thematic out-of-distribution evaluation.

Use a **fresh run** because source hashes, drafts and ratings are tied together. Do not overwrite scenarios inside an existing run or reuse its ratings.

On a native Mac, from the project root with the environment activated:

```sh
export RUN=warmth/runs/mac_curated_v1
sh warmth/run_matched.sh seed --run "$RUN" --input warmth/data/curated_smoke_v1.jsonl --per-category 5
sh warmth/run_matched.sh draft --run "$RUN" --device mps
sh warmth/run_matched.sh review --run "$RUN"
```

Importing these scenarios requires no generation. Drafting generates 60 candidate responses and resumes if interrupted. Independently review `review.csv`, or screen with a separately installed compatible model:

```sh
sh warmth/run_matched.sh judge --run "$RUN" --device mps --judge-model /absolute/path/to/independent-instruct-model
sh warmth/run_matched.sh validate --run "$RUN"
```

The judge path is a placeholder, not a bundled model. Judging screens rather than repairs. Review the accepted counts and rejection reasons in `validation_report.json`; filtering can leave insufficient data, especially for the tiny held-out splits. Do not force warmth scores to match the requested levels. Human audit remains necessary before treating model judgments as reliable evidence.

After validation, continue with collection, analysis and steering in `../MATCHED_README.md`, passing `--run "$RUN"` throughout and `--device mps` for model stages. The new source set does not change the pipeline or automatically mark any response as accepted.
