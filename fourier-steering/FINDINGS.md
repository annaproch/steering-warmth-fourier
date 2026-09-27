# Fourier-basis steering of cyclic concepts in Llama 3.1 8B

Exploratory work prompted by Goodfire's *Manifold Steering Reveals the Shared Geometry of
Neural Network Representation and Behavior* ([arXiv 2605.05115](https://arxiv.org/abs/2605.05115)).

## Motivation

Goodfire fit **splines** through activation centroids and steer along the resulting manifold,
arguing that respecting curvature matters: linear steering cuts through off-manifold regions
and produces incoherent output.

For *cyclic* concepts (weekdays, months) the manifold is a closed loop, and the natural basis
for a closed loop is a **Fourier series**, not a spline. Prior work supports the underlying
geometry — Engels et al. ([2405.14860](https://arxiv.org/abs/2405.14860)) found circular
weekday/month representations via PCA, and Nanda et al.
([2301.05217](https://arxiv.org/abs/2301.05217)) found Fourier features in modular arithmetic —
but nobody appears to have used a Fourier basis as the *steering* mechanism for these concepts.

Question: does Fourier-basis steering work, and does it buy anything a spline doesn't?

## Setup

| | |
|---|---|
| Model | Llama 3.1 8B (`NousResearch/Meta-Llama-3.1-8B`, ungated mirror), bfloat16, CPU |
| Layer | 28 of 32 — the layer used in the paper |
| Centroids | **Answer-conditioned**: for each day `d`, average the last-token residual over 6 prompts whose correct next token is `d` (e.g. "The day after Tuesday is" → Wednesday). Same for the 12 months. |
| Fourier fit | `mean`, and coefficient vectors `A_k = (2/n)·Σ_d C[d]·cos(2πkd/n)`, `B_k` likewise with sin |
| Target | `xhat(t) = mean + Σ_{k=1..K} [ A_k·cos(2πkt/n) + B_k·sin(2πkt/n) ]` |
| Intervention | Replace the live state's component in `span{A_k, B_k}` with `xhat(t)`'s component, keep the orthogonal complement. Forward hook on `model.layers[27]`. |
| Amplitude | Scalar `amp` multiplying the injected in-subspace component |

`K=1` reduces to steering on a 2D circle (equivalently PCA-plane polar coordinates).

## Findings

### 1. Answer-conditioned centroids are cleanly decodable at late layers

Held-out nearest-centroid decoding (one prompt per day held out):

| layer | 14 | 20 | 24 | 28 | 31 |
|---|---|---|---|---|---|
| decode | 14% | 86% | **100%** | **100%** | **100%** |

Baseline task accuracy is 90% (38/42 prompts answered correctly), so the model genuinely
performs the task. This protocol matters: an earlier version that took the last token of
sentences like `"Today is Monday."` was fitting the `"."` token, and produced a total null result.

### 2. Harmonics are load-bearing — the circle (K=1) is not enough

Weekdays, hits = target became top-1 prediction (out of 4 targets):

| | amp 1.0 | amp 2.0 | amp 4.0 | centroid variance captured |
|---|---|---|---|---|
| K=1 (circle) | **0/4** | 3/4 | 4/4 | 0.569 |
| K=2 | 3/4 | 4/4 | 4/4 | 0.760 |
| K=3 | **4/4** | 4/4 | 4/4 | 1.000 |

With K=3, steering succeeds at *natural* amplitude where the circle fails completely.
P(target) roughly doubles (K=3/amp=2 gives 0.49–0.55 across all targets; Monday reaches 0.763).

Months are worse and need more harmonics (6 targets):

| | amp 1 | amp 2 | amp 4 | amp 6 | amp 8 | variance |
|---|---|---|---|---|---|---|
| K=1 (circle) | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0.381 |
| K=3 | 0/6 | 1/6 | 1/6 | 4/6 | **6/6** | 0.721 |
| K=4 | 1/6 | 2/6 | 3/6 | 5/6 | **6/6** | 0.807 |
| K=5 | 2/6 | 4/6 | 4/6 | **6/6** | **6/6** | 0.905 |
| K=6 | 1/6 | 3/6 | 3/6 | 5/6 | 5/6 | 1.000 |

The circle **never** works for months at any amplitude tested; K=5 reaches 6/6.
P(September) peaks at 0.904, P(August) at 0.941.

### 3. Harmonics and amplitude substitute for each other

More harmonics → less amplitude needed for the same effect (weekdays: K=3 works at amp 1,
K=1 needs amp 4; months: K=3 needs amp 8, K=5 needs amp 6). Cranking amplitude is
**compensating for a truncated basis** — i.e. pushing off-manifold to make up for
discarding harmonics.

### 4. Using *all* harmonics overfits

For months, K=6 (5/6) underperforms K=5 (6/6), even though K=6 captures 100% of the centroid
variance. April actively degrades as amplitude rises (0.155 at amp 2 → 0.064 at amp 8).
For n=12, k=6 is the Nyquist component (alternating ±1) and is almost certainly fitting noise.
**There is a sweet spot; "more harmonics is better" is false.**

The required number of harmonics tracks how non-circular the concept is, as the power
spectrum predicts: weekday fundamental ≈ 0.56, month fundamental ≈ 0.44.

### 5. Amplitude/phase separation is real but partly an artifact

In polar coordinates on the K=1 circle, phase and radius act as independent knobs:
phase sets **identity** (argmax-day pinned to target for all r ≥ 1; at r = 0 it reverts to the
model's prior, confirming the circle component carries identity), radius sets **confidence**
(P(Monday) 0.16 → 0.63 and day-entropy 1.80 → 0.22 bits monotonically as r goes 1 → 6).

Notably, coherence did **not** degrade off-manifold — total day-mass *rose* (0.29 → 0.65)
rather than collapsing. But see Finding 3: much of the usefulness of the radius dial was
compensating for the truncated K=1 basis, so this is a weaker result than it first appeared.

### 6. Composability works — but only after matching perturbation magnitudes

Steering weekday (K=3, amp 2) and month (K=5, amp 6) simultaneously:

| | before matching ‖Δ‖ | after matching ‖Δ‖ |
|---|---|---|
| weekday under month edit | **1/12**, −79.6% | **12/12**, −18.3% |
| month under weekday edit | 12/12, +7.2% | 12/12, −4.8% |

The month edit was 2.23× larger in norm (54.3 vs 24.4, against a state norm of ~42 — a
perturbation larger than the state itself), so it simply swamped the weekday edit. Matching
‖Δ‖ restores 12/12 in both directions. The dramatic asymmetry was an artifact of amplitude
choice, **not** geometric interference.

### 7. Fundamentals are near-orthogonal; harmonics entangle

Principal angles between the weekday and month subspaces:

| subspaces | angles (deg) |
|---|---|
| 2-dim circles (K=1) | 82.6, 83.2 |
| 6-dim vs 10-dim (K=3 vs K=5) | **51.3**, 58.3, 66.5, 72.6, 80.7, 84.3 |

Two random 6- and 10-dim subspaces in 4096 dimensions would sit at ~87–90°, so 51.3°
(cos ≈ 0.63) is far from chance. **The two concepts look independent at the fundamental and
share structure in the higher harmonics** — plausibly a common "calendar / enumerable
sequence" component. This is the most likely source of the residual 18.3% weekday degradation.

## Corrections made during this work

Recorded because several intermediate claims were wrong, and the errors were informative:

1. **Predicted Llama's weekday circle would be cleaner than GPT-2's. False** — fundamental
   0.56 (Llama, L28) vs 0.66 (GPT-2, L6). A bigger model did not give a cleaner circle.
2. **Claimed the degradation into generic `"the"` reproduced the paper's off-manifold
   incoherence thesis. Wrong** — day-mass rose rather than collapsed, and amplifying fixed the
   incoherence. It was insufficient amplitude, not off-manifold damage.
3. **Reported crosstalk of 2.26% (Finding 6). Invalid** — the control was dead: the month edit
   at that amplitude installed nothing, so "no interference" was vacuous. A non-interference
   result requires proving the *perturbing* edit is potent.
4. **Hypothesised RMSNorm caused the crosstalk. Falsified** — renormalising the composed edit to
   the original norm did not reduce interference (0.065 → 0.070) and damaged the steering itself.
   Norm inflation was only 1.24–1.31×, too small to explain a 25–35% effect.
5. **Basis-mismatch bug**: the circle was fit on end-of-sentence tokens and then applied to
   rotate a mid-sentence day-word token. Different activation populations → the intervention was
   meaningless and every result was null. Caught by a sanity check (injecting zeros must change
   the output; it did, proving the hook worked and the basis was at fault).
6. **Used PCA + polar coordinates instead of an actual Fourier basis.** This is the `K=1`
   special case, and it silently discarded exactly the harmonics that Findings 2–4 show are
   load-bearing.

## Limitations

- **No spline comparison.** Every comparison here is Fourier-K against Fourier-K=1 (my own
  circle), never against the paper's actual TPS/cubic spline method. Claims of Fourier
  *advantage* are therefore unsupported; only the K-ladder results are established.
- **No statistics.** One test prompt per steering direction, 4–6 targets, single seed,
  no error bars.
- **Single-token readout.** Day and month answers compete for the same next-token slot, so
  softmax alone couples them. True independence is not measurable in this design; it needs two
  separate readout positions.
- `"May"` is a confound as a month target (also a common modal verb), inflating its token
  probability for non-calendar reasons.
- The exploratory GPT-2 phase (layer sweeps, list-vs-arithmetic prompt comparisons, induction
  head observations) was lost to a container restart that wiped `/tmp`. Those numbers appear in
  Findings/Corrections but are **not reproducible from this repo**.
- Prompt templates for days and months are formulaically similar, which could itself induce
  shared structure (relevant to Finding 7).

## Open questions

1. **Joint 2D Fourier fit on the 7×12 torus.** Fit both concepts in one product basis; the
   energy in cross terms (both frequencies non-zero) is a direct scalar measure of
   compositionality that a spline surface cannot provide. If cross terms are non-trivial, it
   predicts interference should be *structured* per (day, month) combination rather than uniform —
   a falsifiable test.
2. **Does orthogonalised steering preserve control?** Project the weekday subspace orthogonal to
   the month subspace and steer only in the residual. Zero leakage by construction — but if the
   shared 51° directions carry real identity information, control should weaken. This decides
   whether clean independent control is achievable or the entanglement is load-bearing.
3. **Nuisance controls for Finding 7.** Shuffle day/month label assignments, or refit with
   structurally dissimilar templates, and check whether the subspace overlap survives. Token
   frequency, word length, and "capitalised calendar noun" are all boring explanations that must
   be ruled out.
4. **Which frequencies do downstream layers actually read?** Ablate individual harmonics and
   measure behavioural effect — the Nanda "key frequencies" question, which connects this
   geometry to circuit mechanism. Splines do not hook into this at all.

## Files

| file | what it does |
|---|---|
| `fourier_steer_real.py` | **Core**: Fourier fit + K-harmonic steering; weekday and month K-sweeps (Findings 2–4) |
| `months_deep.py` | Months at K=3…6, full harmonic range (Finding 4) |
| `compose_matched.py` | **Key**: magnitude-matched bidirectional composability + principal angles (Findings 6–7) |
| `compose_fourier.py` | Bidirectional composability with potent Fourier handles (pre-matching) |
| `llama_steer2.py` | Answer-conditioned protocol + hook sanity check (Finding 1) |
| `novel.py` | Amplitude/phase sweep and first composability attempt (Finding 5) |
| `compose2.py` | Month amplitude calibration; showed K=1 month steering fails |
| `norm_crosstalk.py` | RMSNorm hypothesis test (falsified, Correction 4) |
| `llama_full.py` | First pipeline — contains the basis-mismatch bug (Correction 5); kept for the record |
| `results/*.log` | Raw stdout from each run, numbered in execution order |

Scripts are standalone: `python <script>.py` with `torch`, `transformers`, `numpy`.
They download the model from HuggingFace (~16GB) and run on CPU in ~1–2 min per experiment
after weights are cached.
