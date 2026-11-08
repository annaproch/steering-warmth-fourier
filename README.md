# Activation steering experiments

Research pilots exploring whether directions, curves, and Fourier representations
in language-model activations can control model behavior. The experiments use
Llama 3.1 8B and compare interventions against prompting, linear directions,
random directions, or donor activations, depending on the track.

This repository contains experiment code, small datasets, saved generations,
diagnostics, and research notes. It is a research workspace, not a packaged library.
Ordered activation geometry alone is not evidence of reliable behavioral control.

## What's here

| Directory | Experiment | Where to start |
| --- | --- | --- |
| `fourier-steering/` | Weekday/month representations, Fourier harmonics, steering, and composition controls using the **base** model. Includes early failed protocols and later donor-patching experiments. | [Findings and script guide](fourier-steering/FINDINGS.md), [critical review](fourier-steering/RESEARCH_REVIEW.md) |
| `conciseness/` | Five prompted verbosity levels, training-only PCA and spline fitting, donor checks, and spline-versus-linear steering using the **Instruct** model. | [Experiment guide](conciseness/README.md) |
| `warmth/` | Emotional-warmth experiments: an EPITOME pilot, generated/reviewed matched responses, and a directly authored 30-triplet dataset using the **Instruct** model. | [Recent learnings](warmth/RECENT_LEARNINGS.md), [authored dataset](warmth/AUTHORED_DATASET.md) |
| `CONCISENESS_MANIFOLD_PLAN.md` | Original conciseness experiment design. | [Plan](CONCISENESS_MANIFOLD_PLAN.md) |

The saved conciseness results have mixed geometric evidence and incomplete,
often truncated generation outputs. Reliable causal conciseness control has not
been established. The authored warmth pilot shows transferable activation
differences, but the tested layer-16 last-token and suffix-anchor interventions
have not demonstrated convincing monotonic warmth control beyond random controls.
There is no demonstrated warmth spline advantage or ODE steering implementation.

## Experiments run and results

These are small exploratory pilots, not reliable research findings. The table
summarizes saved runs and notes; incomplete generations and qualitative assessments
are identified explicitly. Follow the links for protocols, controls, and limitations.

| Experiment | What was run | Result |
| --- | --- | --- |
| [Fourier weekday/month steering](fourier-steering/FINDINGS.md) | Harmonic-count and amplitude sweeps, with matched-dimensional PCA and random-direction controls. | Weekday steering reached **4/4 targets** with the full six-dimensional concept subspace at amplitude 1. Fourier and PCA performed essentially identically; random directions achieved **0/4**. Months required larger interventions. No Fourier-specific advantage was established for discrete targets. |
| [Calendar composition controls](fourier-steering/FINDINGS.md) | Weekday and month edits applied together, followed by perturbation-magnitude matching. | Matching improved preservation of the primary answer to **12/12** in each direction. These were separate single-concept readouts, and secondary-edit potency was not reverified after rescaling, so this did **not establish joint control**. |
| [Joint weekday/month donor patching](fourier-steering/results/joint_pilot/SUMMARY.md) | **144 patched trials** across layers 18/24/28, two output orders, and day/month/both edits. | No layer passed the donor gate. Patches could transfer the first answer while leaving the second tied to the source. The planned additive-versus-joint Fourier comparison **was not run**. |
| [Shared-prefix branching](fourier-steering/results/branch_pilot/SUMMARY.md) | **192 interventions**, with separate weekday and month readouts from the edited prefix. | Compact residual patches achieved **0/144** paired successes; last-token cache replacement achieved **0/24**. Full-prefix cache replacement achieved **21/24**, reproducing donor behavior but not establishing a compact controller. No Fourier fit followed. |
| [Conciseness spline pilot](conciseness/README.md) | Five prompted verbosity levels across 30 questions; training-only PCA and spline/line fits; partial donor and steering generations. | Test spline MSE beat the line (**0.000401 vs 0.000508**), but lost when the middle level was omitted (**0.001161 vs 0.000996**). Of 19 saved donor-gate outputs, 14 hit the token cap; **all six steering outputs were capped**. Reliable behavioral control remains unestablished. |
| [EPITOME warmth pilot](warmth/RESULTS_AND_NEXT_STEPS.md) | Activation collection and spline/line fitting on 150 labeled replies, followed by partial steering runs at strengths 1 and 0.25. | Reconstruction gains were minimal; the spline slightly won on validation and lost on test. **All 11 saved generations were truncated**. Interventions often switched to refusal-style responses rather than producing graded warmth; no spline advantage was observed. |
| [Generated warmth datasets](warmth/RECENT_LEARNINGS.md#dataset-construction) | A 20-scenario rewrite run, followed by a revised four-scenario pilot, with automated review. | Accepted **0/20** and **0/4** triplets respectively. Warmth ordering was unreliable; the first run also confounded warmth with verbosity. |
| [Warmth-judge calibration](warmth/RECENT_LEARNINGS.md#judge-calibration) | Qwen3-8B evaluated 24 authored cases in 32 presentations, including reversed A/B warmth pairs. | **6/8** correctness, **7/8** content preservation, and **16/16** clear-warmth presentations; reversal consistency was **8/8 pairs**. These small, authored checks do not justify using the judge as the sole factual/content gate or establish sensitivity to subtle warmth differences. |
| [Authored warmth steering](warmth/RECENT_LEARNINGS.md#causal-steering-results) | **30 triplets / 90 responses**, split 18/6/6 by scenario; activation diagnostics and two steering sweeps totaling **168 responses** on three validation prompts. | Held-out activation differences aligned with training directions, but layer-16 last-token and suffix-anchor steering showed **no convincing monotonic warmth effect beyond random controls**. Larger strengths and first-token steering did not resolve this. The first sweep capped 54/84 responses; the later sweep capped 0/84. |

Interpretation and status cautions:

- Early Fourier claims were corrected after finding fitting/test prompt overlap,
  a month Nyquist-coefficient bug, and invalid interpretations of composition and
  RMSNorm tests. The affected month K=6 results should be disregarded; see the
  [critical review](fourier-steering/RESEARCH_REVIEW.md) and corrected findings.
- Warmth behavior was assessed qualitatively, without completed blinded ratings
  or significance tests. Authored warmth test steering remains untouched.
- The proposed 300-triplet warmth dataset, mean-response steering follow-up,
  and joint Fourier model comparison are plans, not completed results. There is
  no demonstrated nonlinear-steering advantage in these pilots.

## Setup

Run commands from the repository root. Use Python 3.12 or 3.13 and a virtual
environment. The requirements file records the project's CPU-tested package
versions; it is not a complete transitive lockfile.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r warmth/requirements.txt
```

Conciseness and warmth import Goodfire's spline implementation from a pinned
source checkout, including for CLI help and tests. Install it before using them:

```sh
mkdir -p vendor
git clone --branch manifold_steering https://github.com/goodfire-ai/causalab.git vendor/causalab
git -C vendor/causalab checkout 1b6f43a509d2a4ae49fd1bc4252e13889d57e38b
```

If that checkout already exists, verify its revision instead of cloning again.
The full causalab application environment is not needed.

For warmth and conciseness, download the Instruct checkpoint:

```sh
python conciseness/download_model.py
```

This places `NousResearch/Meta-Llama-3.1-8B-Instruct` in
`conciseness/models/llama-3.1-8b-instruct/`. Inference in these pipelines uses
local files, so download the model first. BF16 weights alone occupy roughly
16 GB; allow additional RAM and disk space for loading, activations, and generation.
CPU inference can be slow. Warmth accepts `--device cpu`, `cuda`, or `mps`;
the conciseness pilot runs on CPU. See [native Mac setup](warmth/MAC_SETUP.md)
for Apple Silicon environment and backend checks.

## Run the authored warmth pilot

This workflow uses 30 assistant-authored scenarios and 90 complete responses,
split into 18 training, 6 validation, and 6 test scenarios. Only the opening
changes within each neutral/moderate/warm triplet. These are intended labels,
not independently measured warmth ratings. No drafting or judge step is required.

Use a fresh run directory to keep the committed result snapshots intact:

```sh
export RUN=warmth/runs/authored_local
export DEVICE=cpu  # Change to mps or cuda for a supported accelerator.

# Validate and import the authored dataset; no model needed.
python warmth/prepare_authored.py --run "$RUN"

# Collect last-token, suffix-anchor, and mean-response activations.
sh warmth/run_matched.sh collect --run "$RUN" --device "$DEVICE" --batch-size 1

# Fit training directions and inspect validation geometry.
sh warmth/run_matched.sh analyze --run "$RUN" --split validation --bootstrap 200

# Check intervention behavior before generating the comparison.
sh warmth/run_matched.sh check --run "$RUN" --device "$DEVICE" --method last --layer 16

# Compare baseline, signed linear shifts, random controls, and warmth prompting.
sh warmth/run_matched.sh steer --run "$RUN" --device "$DEVICE" \
  --split validation --method last --layer 16 --questions 3 --max-tokens 768
```

Run each stage separately and inspect its output. Generation folders under
`$RUN/generations_<hash>/` contain `outputs.json`, `settings.json`, and a blinded
`ratings.csv`. Assess warmth, correctness, coherence, and unsupported reassurance;
changed wording alone is not success. Resume with the same command, keep model
and extraction settings consistent, and use one writer per run directory.
Keep test prompts untouched until validation choices are fixed.

Further workflows:

- [Authored dataset design](warmth/AUTHORED_DATASET.md) and [strength/first-token comparison](warmth/STRENGTH_SWEEP.md).
- [Matched pipeline](warmth/MATCHED_README.md): generated drafts, independent review, selection, and scoring.
- [Judge calibration](warmth/data/calibration_v1/README.md) and [revised rewrite protocol](warmth/REWRITE_V2.md).
- [Original EPITOME pilot](warmth/README.md): download data with `sh warmth/download_data.sh`, then use `python warmth/pipeline.py prepare`, `collect`, `fit`, and `steer` as documented there.

## Run the conciseness pilot

After setup and the Instruct download, the committed `conciseness/data/pilot.json`
is enough to start collection. It contains the prepared TONEBANK subset and splits.

```sh
python conciseness/pilot.py collect
python conciseness/pilot.py analyze
python conciseness/pilot.py gate --questions 2 --max-tokens 128
# Review donor-gate outputs before proceeding to steering.
python conciseness/pilot.py steer --questions 2 --max-tokens 128
```

These commands write to fixed `conciseness/cache/` and `conciseness/results/`
directories and may update tracked results. Use a separate checkout for a fresh
comparison. Existing generation caches are not invalidated by changing the token
cap. The example's 128-token cap is a smoke-test budget and may truncate responses.

The optional `prepare` stage rebuilds the split; it requires `pyarrow` and the
upstream TONEBANK parquet at `conciseness/data/tonebank.parquet`, neither of which
is supplied by the setup above. Skip it when using the committed subset.
See the [conciseness guide](conciseness/README.md) for methods and interpretation.

## Run the Fourier experiments

These scripts use `NousResearch/Meta-Llama-3.1-8B`, a separate **base** checkpoint
from the Instruct model above. The core script downloads it through Hugging Face
on first use and runs on CPU:

```sh
python fourier-steering/fourier_steer_real.py
python fourier-steering/basis_control.py
python fourier-steering/compose_matched.py
```

For the later joint weekday/month donor gates, first ensure the base model is in
the Hugging Face cache; these scripts load with `local_files_only=True`:

```sh
python fourier-steering/joint_pilot.py --stage donor
python fourier-steering/branch_pilot.py
```

Read the saved [joint](fourier-steering/results/joint_pilot/SUMMARY.md) and
[shared-prefix](fourier-steering/results/branch_pilot/SUMMARY.md) summaries before
interpreting these experiments. The joint Fourier stage requires a successful
donor gate. Scripts use fixed result/cache paths; some older scripts also contain
original machine paths. The [script guide](fourier-steering/FINDINGS.md#files)
identifies historical experiments, including the flawed initial protocol.

## Lightweight checks

After installing dependencies and the pinned Goodfire checkout, these tests do
not require downloading the full Llama checkpoint. Some use a tiny random model;
the EPITOME dataset check skips when its source data is absent.

```sh
python -m unittest discover -s warmth -p 'test_*.py' -v
python -m unittest discover -s conciseness -p 'test_*.py' -v
python -m unittest discover -s fourier-steering -p 'test_*.py' -v
```

The mathematical audit uses only the Python standard library:

```sh
python fourier-steering/audit_math.py
```

## Saved artifacts and local files

Committed artifacts include small datasets, selected generated responses,
metrics, plots, logs, and some Fourier caches. Model weights, the Goodfire
checkout, and most activation caches are excluded from Git. Although
`warmth/runs/` is ignored, selected historical runs are explicitly tracked.
A clone therefore contains evidence to inspect, but not all files required to
resume the original experiments.

The commands above use the active Python environment. The older
`conciseness/run.sh` and `warmth/run.sh` wrappers hard-code the original container's
interpreter; use the direct Python commands outside that container.
`warmth/run_matched.sh` supports an active virtualenv, a local `.venv`, or
an explicit `PYTHON_BIN`.
