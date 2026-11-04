# Recent warmth experiment learnings

## Current conclusion

We have a transferable activation difference between directly authored warmth
variants, but **we have not demonstrated reliable causal warmth control**. The
layer-16 last-token and suffix-anchor interventions tested so far change wording
without a convincing monotonic warmth effect beyond random-direction controls.
There is no demonstrated spline advantage or warmth manifold suitable for combining
with ODE steering yet.

Behavioral assessments below are qualitative inspection of saved generations, not
blinded human ratings or statistical significance tests. Only three validation
prompts have been used for generation: arithmetic correction, disappointment about a
photo-club selection, and preparation-time advice. Test steering remains untouched.

## Dataset construction

1. The first curated-base run (`bf6ae4e`) produced 60 Llama rewrites but accepted
   **0/20 triplets**. Moderate and warm tied on 17/20 scenarios; two triplets had
   identical text at all three levels. Mean lengths increased from roughly
   **56 to 74 to 85 words**, confounding warmth with verbosity. The judge marked
   every response content-preserving despite meaning changes.
2. The revised four-scenario pilot (`565dc21`) improved length compliance to
   **11/12 responses**, with no duplicate variants, but accepted **0/4 triplets**.
   Moderate was judged warmer than the intended warm response in all four cases.
   Neutral responses sometimes used the same collaborative phrases as warm ones.
3. Repeated generator/judge prompt revisions were not an efficient route to the
   first causal experiment. We instead directly authored **30 triplets / 90 complete
   responses** (`8e36f82`), with one shared substantive answer and three short
   warmth openings per scenario. No model-approval gate is required for collection.

The authored dataset has ten scenarios each for emotional support, advice and factual
disagreement, split **18 train / 6 validation / 6 test** by scenario. All 90 openings
are unique. Mean response lengths are **57.1 / 57.0 / 58.5 words**; the largest
within-triplet length ratio is **1.074**. The substantive answer and final response
token ID are identical within each triplet. All sequences fit the collection limit
(maximum checked total length: 158 Llama tokens).

These are intended warmth labels, not independently validated ratings. The design
narrowly isolates expressed warmth in an opening; lexical patterns, emotional framing,
and token positions may still contribute. Scenario holdouts share authoring style and
broad themes, so transfer here is not evidence of broad behavioral generalization.

## Judge calibration

The calibration run (`aa09ad1`) used 24 authored cases and 32 presentations, including
both A/B orders for each warmth pair. Scoring the committed verdicts gave:

| Task | Result |
|---|---:|
| Base correctness | 6/8 |
| Content preservation | 7/8 |
| Clear warmth comparisons | 16/16 presentations |
| Consistency across A/B reversal | 8/8 pairs |

Qwen3-8B rejected the correct `14:20 + 40 + 10 minutes = 15:10` explanation, accepted
`2 + 3 + 1 = 7`, and treated "not guaranteed" changing to "unlikely" as preserving
meaning. The last failure occurred despite an explicit example in the rubric.

Qwen can provisionally help screen **clear warmth contrasts**, but it is not a reliable
sole gate for factual correctness or content fidelity. Passing these eight warmth
pairs does not establish sensitivity to subtle moderate-versus-strong warmth.
The calibration targets were assistant-authored, not independent human consensus;
reversed presentations are not independent samples.

## Activation representation

The authored pilot (`cad04b8`) collected mean-response, last-content-token and
suffix-anchor activations at layers 12, 16 and 20. Directions were fitted exclusively
from the 18 training triplets. At layer 16:

| Extraction | Mean held-out difference cosine with training direction | Mean held-out neutral → moderate → warm coordinate |
|---|---:|---|
| Last token | 0.375 | 0.246 → 0.658 → 0.998 |
| Suffix anchor | 0.550 | 0.247 → 0.562 → 1.058 |

All six validation warm-minus-neutral differences aligned positively with the training
direction for both methods. Mean-response layer 20 also showed positive alignment
(mean cosine **0.581**). These are continuous projections, not classification accuracy
or measured behavioral warmth. Their aggregate ordering is encouraging but does not
establish that every moderate example lies between its endpoints.

There is some reproducible off-line midpoint displacement: layer-16 train/held-out
orthogonal-direction cosines are approximately **0.413 last-token / 0.530 anchor**.
With three authored levels and shared wording conventions, this does not establish
a generalizable nonlinear semantic manifold. Causal utility remains the decisive test.

## Causal steering results

### Initial sweep: ±0.25 and ±0.5 (`cad04b8`)

- Layer 16, last-token and anchor vectors; three validation prompts per method.
- Baseline, signed linear steering, two norm-matched random directions, and explicit
  warmth prompting were compared (84 responses total).
- The arithmetic answer was exactly unchanged across all four linear strengths for
  both extraction methods.
- Support/advice text changed, but there was no convincing warmth progression with
  signed strength. Random controls also changed text and advice.
- **54/84 responses hit the 256-token cap**, including support/advice baselines,
  limiting complete-answer quality assessment.
- Explicit warmth prompting produced clearer warmth changes but sometimes added
  unsupported reassurance. It is a useful positive control, not an ideal target.

### Broader sweep and first-token intervention (`da97585`)

- Anchor vector, layer 16, strengths **−2, −1, +1, +2**, same three prompts and
  random controls. Compared decode-only steering against steering the final prompt
  position as well, so the first generated token could be affected.
- The 768-token cap eliminated truncation: **0/84 capped responses**.
- Implementation sanity checks passed: zero-shift equivalence and baseline equivalence
  after a nonzero intervention. These establish basic hook behavior, not causal warmth.
- **24/42 matched conditions had identical complete text** with and without the
  first-token intervention. This count includes the unsteered controls; it is not a
  standalone measure of intervention effectiveness.
- Arithmetic remained correct, with little added warmth. Support/advice outputs
  varied, but no clear monotonic warmth pattern or advantage over random controls
  emerged from inspection.
- At +2 with first-token steering, a support answer promised the user would be
  "back to creating amazing photos in no time": an example of unwarranted reassurance,
  not evidence of desirable warmth control.

Strengths multiply the training mean difference vector; they are not calibrated
warmth levels. Different text is not sufficient evidence of successful steering.
First-token steering did not resolve the observed weakness at this extraction/layer.

## Next bounded experiment

Keep the authored dataset fixed. Test the **already-collected mean-response vectors
at layers 12, 16 and 20**, on the same validation prompts, with signed strengths,
baseline and norm-matched random controls. Keep token cap and intervention timing
fixed across the comparison. No recollection is required for these vectors.

Rationale: mean-response pooling includes the warmth-bearing opening, whereas the
current anchor vector was extracted after an identical substantive answer and suffix.
This is a plausible representation/intervention mismatch, not a confirmed explanation.
Mean pooling also introduces its own mixture of token identities and positions.

Judge success by a consistent signed warmth change that exceeds random-control
effects while preserving relevance, correctness and reasonable uncertainty. If no
setting shows that signal, reconsider extraction location or the narrowly authored
concept before expanding the dataset or escalating strengths again. If a candidate
works, broaden validation prompts, freeze settings, and only then evaluate held-out
test prompts. Use fresh prompts as well before claiming general warmth control.

Do not fit splines or combine with anti-sycophancy ODE steering until a useful causal
warmth intervention is established. An ordered representation alone is insufficient.
