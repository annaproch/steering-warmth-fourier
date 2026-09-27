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

**Bottom line up front.**

1. **The core method is prior art.** Feucht et al., *Arithmetic in the Wild*
   ([2605.01148](https://arxiv.org/abs/2605.01148)) §4, "Fourier Probes Trained on Addition Can
   Steer Cyclic Tasks", already steers weekdays and months **on Llama-3.1-8B** by computing a
   target phase `θ' = n'(2π/T)`, caching the observed radius and rotating to that phase, with a
   scaling factor α=10. That is this project's method, its phase/radius separation, and its
   large-amplitude requirement — same model, same concepts. They also show the operative periods
   are task-agnostic base-10 ones (2, 5, 10) rather than the concept's own period, which
   reframes the harmonic sweep below.
2. **Fourier ≡ PCA here.** A matched-dimension control (Finding 2b) shows the Fourier basis
   steers identically to plain PCA on the same centroids, so the Fourier framing buys nothing for
   discrete targets.
3. **What survives** is narrow: a 2D circle is far too small a subspace for control (months cannot
   be steered with one at any amplitude; weekdays need amplitude 4), while random subspaces of the
   same dimension fail entirely.
4. **No spline baseline was ever run**, so nothing here supports any claim against the paper's
   method.

> **This document has been corrected after an external review** (`RESEARCH_REVIEW.md`), which
> found a coefficient bug, an invalid falsification, prompt leakage between fitting and testing,
> and several overstated numbers. Corrected claims are marked inline. The review's mathematical
> claims were independently reproduced with `audit_math.py`.

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

### 2. A 2D circle is not enough — you need most of the concept's subspace

> **Superseded in part.** I originally read this as evidence that *harmonics* are
> load-bearing. A control experiment (Finding 2b) shows the operative variable is
> subspace **dimension**, not frequency structure. The empirical numbers below stand;
> the "harmonics" interpretation does not.

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

### 2b. Control: it is dimension, not harmonics

Comparing bases at **matched subspace dimension** — Fourier `span{A_k,B_k}`, top-d PCA of the
centroids, and d random orthonormal directions (`basis_control.py`):

Weekdays, dim 6, amp 1.0:

| basis | Mon | Sat | Sun | Tue | hits |
|---|---|---|---|---|---|
| fourier K=3 | 0.310 | 0.420 | 0.286 | 0.260 | 4/4 |
| **pca d=6** | 0.310 | 0.420 | 0.286 | 0.259 | 4/4 |
| random d=6 | 0.001 | 0.002 | 0.001 | 0.001 | 0/4 |

Identical to three decimals — and necessarily so: 7 centroids, once centred, span exactly 6
dimensions, and Fourier K=3 supplies 2×3 = 6 basis vectors spanning that same space. They are
the same subspace in different coordinates. For months at dim 10, PCA is marginally *better*
than Fourier (amp 6: 0.239/0.409/0.551/0.393 vs 0.193/0.338/0.566/0.373).

**So the Fourier basis buys nothing over PCA for steering to observed category targets.** The
operative variable is how much of the concept's centroid subspace you span.

What survives, and is non-trivial: **random subspaces fail completely** (0/4, P≈0.001, at dims
2, 6 and 10). "More dimensions" only helps if they are *the concept's* dimensions.

Untested, and the only remaining Fourier-specific capability: a Fourier fit provides a
**continuous intrinsic coordinate**, so it can target `t = 2.5` (between Tuesday and Wednesday)
or interpolate smoothly. PCA only reaches observed centroids. Both were evaluated at discrete
indices here, so this experiment says nothing about it.

### 3. Subspace dimension and amplitude substitute for each other

More of the concept subspace → less amplitude needed for the same effect (weekdays: dim 6 works
at amp 1, dim 2 needs amp 4; months: dim 6 needs amp 8, dim 10 needs amp 6). Cranking amplitude
**compensates for a truncated basis**.

Measured off-manifold distance (distance to nearest centroid / mean inter-centroid distance)
confirms the "compensation" is literal: the successful low-amplitude dim-6 weekday setting sits
at 1.05, essentially on the manifold, while amp 4 reaches 2.23 and months reach 4.75.

**But off-manifold is not harmful here**: the best month results (P=0.893) occur at the
*farthest* off-manifold point (4.75). Either off-manifold excursions are benign in this setting,
or top-1 accuracy is the wrong harm metric (an unrelated-token KL measurement would settle it —
see Open Questions).

### 4. ~~Exceeding the span of the centroids overfits~~ — RETRACTED, coefficient bug

For months, K=6 (5/6) underperforms K=5 (6/6), and April degrades as amplitude rises
(0.155 at amp 2 → 0.064 at amp 8). I originally read this as Nyquist-frequency noise.

**This is a bug in `fourier_fit`, not a finding.** The code applies `2/n` to every cosine
coefficient. For even `n`, the `k = n/2` (Nyquist) cosine requires `1/n`, and its sine is
identically zero on the integer sampling grid. So for n=12 the alternating component is
reconstructed at **twice** its true amplitude, and `B_6` is a zero vector — which means
`subspace_basis` hands QR a zero column and gets an arbitrary direction back.

Verified independently (`audit_math.py`): on a pure alternating n=12 signal the current formula
gives max reconstruction error **1.0** (i.e. 2× the signal) versus **1.7e-15** corrected.
Odd n is unaffected (error 6.7e-16), so **weekdays were never affected — only months at K=6.**

The K=6 row should be disregarded until refit with `A_{n/2} = (1/n)·Σ_d C[d]·(−1)^d`, no `B_{n/2}`,
and a rank-revealing SVD with a tolerance instead of QR. Correcting it need not make K=6 better;
it makes the comparison interpretable. The n−1 rank ceiling still holds as a matter of linear
algebra.

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
| 2-dim **PCA planes** (not Fourier fundamentals — see caveat) | 82.6, 83.2 |
| 6-dim vs 10-dim (K=3 vs K=5) | **51.3**, 58.3, 66.5, 72.6, 80.7, 84.3 |

> **Caveat.** The 2-dim figures come from top-2 PCA planes (`novel.py`), *not* from the Fourier
> fundamental subspaces, so "fundamentals are orthogonal, harmonics entangle" is not yet a
> measured comparison — the two rows use different bases. Frequency-specific angles need
> recomputing before that phrasing is justified. Also, a minimum principal angle of 51°, not 0°,
> means the subspaces need not intersect at all: independent latent coordinates can be encoded in
> non-orthogonal directions and recovered with a dual basis, so this is **not** evidence of
> irreducible semantic entanglement. Some of the observed interference may come from the
> sequential-overwrite operator rather than the geometry.

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
4. **Hypothesised RMSNorm caused the crosstalk, then claimed to have falsified it. The test was
   invalid** — renormalising the whole state cannot probe normalisation-mediated interference,
   because RMSNorm is scale-invariant: `RMSNorm(c·x) = RMSNorm(x)` exactly (verified, difference
   0.0 in `audit_math.py`). The experiment establishes only that *that particular remedy* fails
   (interference 0.065 → 0.070, and it damaged the steering itself). RMSNorm causation is neither
   confirmed nor ruled out. In a pre-norm transformer global rescaling also shifts the
   residual-to-branch balance, confounding it further.
5. **Basis-mismatch bug**: the circle was fit on end-of-sentence tokens and then applied to
   rotate a mid-sentence day-word token. Different activation populations → the intervention was
   meaningless and every result was null. Caught by a sanity check (injecting zeros must change
   the output; it did, proving the hook worked and the basis was at fault).
6. **Used PCA + polar coordinates instead of an actual Fourier basis.** This is the `K=1`
   special case, and it silently discarded most of the concept's subspace (Findings 2, 2b).
7. **Attributed the K-ladder result to harmonics. Wrong** — a matched-dimension control
   (Finding 2b) shows Fourier and PCA steer identically, because for n categories both span the
   same (n−1)-dimensional centroid space. I ran the interesting-sounding interpretation for
   several experiments before running the control that distinguished it from the boring one.
   The correct claim is about subspace dimension; "harmonics" was a roundabout way of saying it.

## Limitations

- **No spline comparison.** Every comparison here is against Fourier-K=1 or PCA, never against
  the paper's actual TPS/cubic spline method. Claims of Fourier *advantage* over their method are
  unsupported — and after Finding 2b there is no evidence of Fourier advantage over plain PCA
  either, for discrete targets.
- **Continuous interpolation never tested**, which is the one capability a Fourier fit has that
  PCA-to-centroids does not.
- **No statistics.** One test prompt per steering direction, 4–6 targets, single seed,
  no error bars.
- **Single-token readout.** Day and month answers compete for the same next-token slot, so
  softmax alone couples them. True independence is not measurable in this design; it needs two
  separate readout positions.
- `"May"` is a confound as a month target (also a common modal verb), inflating its token
  probability for non-calendar reasons.
- **The steering test prompts are inside the centroid fitting set.** `"Today is Wednesday.
  Tomorrow is"` is one of the six prompts used to build the *Thursday* centroid, and `"The month
  after February is"` builds the *March* centroid (verified statically in `audit_math.py`). The
  held-out decoding check in Finding 1 does not make the later steering tests held out.
- **Findings 6–7 are preservation tests, not joint control.** A weekday prompt is scored only for
  its weekday and a month prompt only for its month, so no single state was shown to support both
  requested labels at once. "Composability works" overstates it.
- **The magnitude matching is asymmetric.** In `compose_matched.py` only the *perturbing* edit is
  rescaled; each primary edit keeps its own amplitude (weekday 24.4, month 54.3). The match
  constant was also derived from a single Monday edit, and different targets have different norms.
  Potency was not re-verified at the rescaled strengths, so a preserved primary answer could
  simply mean the secondary edit stopped working.
- **Sign convention error**: the month figure reported as "−4.8%" is a 4.8% *increase* in
  P(target), not a degradation.
- The `0.56 / 0.44` "fundamental" power figures quoted under Finding 4's original wording came
  from log 01's sentence-ending activation population, which the corrected protocol replaced —
  they should not be merged with the answer-conditioned results. Projection-into-span variance is
  also not the same quantity as Fourier spectral power.
- Random subspaces are a weak control for anisotropic residual activations; label-permutation
  controls would be stronger, though they cannot change the full centred span.
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
5. **Continuous-coordinate steering** (the open Fourier-specific question from Finding 2b): target
   non-integer `t` and check whether outputs interpolate smoothly between adjacent days, or jump.
   Smooth interpolation would be a capability PCA-to-centroids lacks; a jump would mean the
   continuous coordinate is fiction.
6. **Is off-manifold actually harmful?** Finding 3 shows best results at the farthest off-manifold
   point, contradicting the paper's framing. Measure KL divergence from baseline on
   *unrelated* tokens at matched target-hit rate, comparing low-dim/high-amp against
   high-dim/low-amp. If collateral damage is higher for the off-manifold route, top-1 accuracy was
   simply hiding the cost.

## Files

| file | what it does |
|---|---|
| `fourier_steer_real.py` | **Core**: Fourier fit + K-harmonic steering; weekday and month K-sweeps (Findings 2–4) |
| `basis_control.py` | **Key control**: Fourier vs PCA vs random at matched dimension, plus off-manifold distance (Finding 2b) |
| `months_deep.py` | Months at K=3…6, full harmonic range (Finding 4) |
| `compose_matched.py` | **Key**: magnitude-matched bidirectional composability + principal angles (Findings 6–7) |
| `compose_fourier.py` | Bidirectional composability with potent Fourier handles (pre-matching) |
| `llama_steer2.py` | Answer-conditioned protocol + hook sanity check (Finding 1) |
| `novel.py` | Amplitude/phase sweep and first composability attempt (Finding 5) |
| `compose2.py` | Month amplitude calibration; showed K=1 month steering fails |
| `norm_crosstalk.py` | RMSNorm hypothesis test (falsified, Correction 4) |
| `llama_full.py` | First pipeline — contains the basis-mismatch bug (Correction 5); kept for the record |
| `results/*.log` | Raw stdout from each run, numbered in execution order |
| `RESEARCH_REVIEW.md` | External critical review: prior art, bugs, overstated claims, and a proposed next experiment. Not written by the author of the experiments. |
| `audit_math.py` | Dependency-free reproduction of the review's mathematical claims (Nyquist coefficient, RMSNorm scale-invariance, prompt-set overlap). `python3 audit_math.py` |

Scripts are standalone: `python <script>.py` with `torch`, `transformers`, `numpy`.
They download the model from HuggingFace (~16GB) and run on CPU in ~1–2 min per experiment
after weights are cached.
