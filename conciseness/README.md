# Conciseness manifold pilot

Status: activation collection and geometry fitting have run. Saved generation
outputs are a partial experiment snapshot; behavioral success has not been established.

Uses Llama-3.1-8B-Instruct, ToneBank questions and saved response variants,
and Goodfire's `CubicSpline1D` implementation. The upstream repository is
cloned into `vendor/causalab`, pinned to
`1b6f43a509d2a4ae49fd1bc4252e13889d57e38b` on `manifold_steering`.

## Run stages separately

From `/workspace/project`:

```sh
# Resume/finish downloading the model, if needed (requires network).
/home/agent/venv/bin/python conciseness/download_model.py

# Already done: prepare the deterministic 18/6/6 question split.
sh conciseness/run.sh prepare

# Collect five requested verbosity levels per question at layers 12/20/28.
# This is 150 prefills, CPU-only; completed questions are cached.
sh conciseness/run.sh collect

# Fit training-only PCA, centroids and natural splines; save plot and metrics.
sh conciseness/run.sh analyze

# Validation donor-patching check: start with two questions (12 generations).
sh conciseness/run.sh gate --questions 2 --max-tokens 128

# After reviewing the gate, test spline/linear interventions on unseen questions.
sh conciseness/run.sh steer --questions 2 --max-tokens 128

# Optional expansion to all six questions in each generation split.
sh conciseness/run.sh gate --questions 6 --max-tokens 128
sh conciseness/run.sh steer --questions 6 --max-tokens 128

# Geometry unit tests (no model required).
PYTHONPATH=.deps /home/agent/venv/bin/python -m unittest discover -s conciseness -p 'test_*.py' -v
```

Collection resumes by question; generation resumes by question/method.
Use a separate output directory/copy of the experiment for protocol changes:
existing cached generations are not recomputed when the token cap changes.

## What this pilot does and does not establish

- The public `withmartian/TONEBANK` release has 200 questions, paired Concise,
  Neutral and other style responses, but no graded concision ratings. Some
  answers appear truncated. Original responses are retained for manual review.
- For this first prospective-control pilot, five **requested verbosity levels**
  supply ordered coordinates. They are not human-rated conciseness labels.
  No response length ranking is mislabeled as information-preserving concision.
- Different instruction wording is used for train, validation, and test. Only
  question-disjoint splits are used. The first run collects the final assistant
  boundary activation, before any answer is generated; it does not encode the
  source answers as if their end states controlled earlier generation.
- PCA is fitted to within-question activation changes, reducing topic effects.
  Ranks 2/4/8/16/32/64 are compared. Selection uses validation geometry only.
  An omitted-middle-level check excludes that level from both centroid and PCA
  fitting. This is a diagnostic with access to multiple prompted conditions,
  not a deployment-time estimator of a new response's conciseness coordinate.
- The donor check replaces one prefill token at one layer. If natural concise
  prompting does not work, or donors do not transfer the style without harming
  content, stop and reconsider the intervention site before scaling.
- Steering adds the fitted displacement from coordinate 2 to 0.5, once at
  the prefill boundary. It compares a spline displacement, a straight-line
  displacement, and a norm-matched straight-line displacement. This is a
  first causal test, not ODE integration or enforced on-manifold motion.
- Judge the text for coverage, correctness, coherence and redundancy. Token
  counts alone are insufficient; outputs at the cap are censored. No automated
  claim of successful conciseness control is made from these metrics.
- A smooth curve can encode instruction wording or length rather than a
  reusable behavior. A positive pilot needs replication with more questions,
  measured concision scores, more paraphrases, and generation-time controls.

## Outputs

- `data/pilot.json`: source responses, question IDs and splits.
- `results/provenance.json`: dataset hash, upstream commit, model and prompts.
- `cache/*.pt`: per-question activations and fitted geometry (Git-ignored).
- `results/geometry_metrics.json`, `selected.json`, `geometry.png`.
- `results/gate_generations.json`, `steer_generations.json`: texts, token counts,
  truncation flags, timing and intervention norms.

Dependencies were installed into workspace-local `.deps`; model weights and
upstream checkout are Git-ignored. No changes were made to upstream code.

## Saved result snapshot

The validation-selected fit uses layer 12 and PCA rank 32. Test spline MSE
is 0.000401 versus 0.000508 for the line. However, when the middle level is
omitted, test spline MSE is 0.001161 versus 0.000996 for the line. These mixed
geometry results do not establish a benefit from nonlinear steering.

- `gate_generations.json`: 19 saved outputs across 4 questions; 14 outputs reached the token cap.
- `steer_generations.json`: 6 saved outputs across 1 questions; 6 outputs reached the token cap.

Outputs are preserved as collected, including incomplete condition coverage.
Review text quality and finish the evaluation before drawing behavioral conclusions.
