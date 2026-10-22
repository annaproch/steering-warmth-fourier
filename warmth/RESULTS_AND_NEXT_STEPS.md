# Warmth steering: results, interpretation, and next experiment

Recorded 2026-10-22. This is a snapshot of an exploratory pilot, not a completed evaluation. No reliable warmth manifold or monotonic causal control has been demonstrated.

## Current experiment

- Model: Llama-3.1-8B-Instruct, locally downloaded from the NousResearch mirror.
- Data: 150 EPITOME Reddit replies, balanced across Emotional Reactions (ER) levels 0/1/2. Training: 30 per level; validation and test: 10 per level each.
- Matching: IP, EX, and log2 response-word-count strata, separately within seeker-group-disjoint splits.
- Activations: teacher-forced response-token means at decoder blocks 12, 16, and 20 (our code uses one-based block numbering). Prompt and padding tokens are excluded.
- Fitting: training-only PCA, ER centroids, and Goodfire natural cubic splines; ranks 2/4/8/16. Fit selected using validation whole-activation reconstruction error.
- Steering: fixed displacement from the ER1 centroid, added at each generated response-token pass. Prefill and the first generated token are unsteered. This does not estimate the current warmth coordinate or constrain states to a manifold.
- Controls: endpoint linear direction and norm-matched linear direction. The spline utility is reused from Goodfire causalab commit `1b6f43a509d2a4ae49fd1bc4252e13889d57e38b`.

## Geometry results

Selected fit: layer 12, PCA rank 4 (`12_4`).

| Metric | Value |
|---|---:|
| PCA variance retained | 0.264374 |
| Midpoint deviation, implementation metric | 0.333530 |
| Validation spline NMSE | 0.986927 |
| Validation linear NMSE | 0.991267 |
| Test spline NMSE | 0.998160 |
| Test linear NMSE | 0.992853 |

NMSE compares reconstruction of full individual activations with a training-mean baseline. Near 1 means little improvement on that objective. It does not establish that warmth is absent: topic, wording, and content can dominate activation variance. The PCA variance fraction is not a fraction of warmth captured.

The spline slightly wins on validation and loses on test. These results offer no positive evidence for generalizable curvature. At the observed integer labels, spline predictions equal the training centroids; performance there does not test nonlinear interpolation. The reported midpoint deviation also conflates departures from equal spacing with orthogonal curvature.

The current test metrics have already been inspected and discussed. Use fresh held-out scenarios for confirmation after redesign or further tuning.

## Saved generation snapshot

Both runs requested two validation prompts, four targets (0, 0.5, 1.5, 2), and three intervention methods plus baseline: 26 outputs per complete run. Decoding is greedy with a 128-token cap.

| Run directory under `runs/pilot/` | Strength | Saved outputs | Unique texts | Reached cap |
|---|---:|---:|---:|---:|
| `generations_b31803664970` | 1 | 8/26 | 2 | 8/8 |
| `generations_9144360a8b02` | 0.25 | 3/26 | 2 | 3/3 |

All saved outputs in this snapshot concern the same crisis-related seeker prompt. Baseline acknowledges distress. At strength 1, all six completed low-target interventions produce identical refusal-style text followed by crisis resources; spline target 1.5 exactly matches baseline. At strength 0.25, spline target 0 and linear target 0 produce the same refusal-style text as the stronger run.

This is consistent with a switch between response patterns, not demonstrated graded warmth. It is not enough evidence to identify the mechanism: greedy decoding can hide probability changes, and this prompt entangles warmth with crisis-response behavior. No spline advantage has been observed. Outputs are truncated and no completed blinded human evaluation is available.

Raw local artifacts remain in ignored `warmth/runs/pilot/`; this note preserves aggregate results without copying sensitive source text or unfinished run files into Git.

## Dataset review

The 150 selected examples were qualitatively reviewed. The earlier informal 2/5 suitability ratings were subjective, not measured annotation-error rates.

Observed problems:

- Bot contamination, including a haiku-bot reply and HTML artifacts.
- Apparent missing negations that can reverse meaning. Checked examples match the downloaded source exactly; the preparation code did not introduce them. The upstream cause has not been established.
- Self-focused personal stories, shared despair, sarcasm, and questionable advice across labels.
- ER annotations may be justified by a short empathetic phrase inside an otherwise unsuitable assistant answer. Whole-response pooling includes all the surrounding content.
- Our system prompt forbids invented personal experiences, while many teacher-forced source replies describe a human author's personal history. This is another collection-to-generation mismatch.

The original ER rubric distinguishes implicit emotional reactions from explicit expressions of emotion. It is not an overall quality score or a calibrated scale of perceived warmth. Some apparent label inconsistencies therefore reflect construct mismatch, rather than proven annotation mistakes.

Local source counts: 3,084 rows, 3,023 unique seeker/response pairs, 2,979 seeker IDs; only 24 seeker IDs have multiple ER levels, and none have all three. These are raw-file counts before conflict filtering. Cleaning cannot turn this release into a sufficiently large matched three-level dataset.

Selected mean response lengths are 49.2/49.4/51.2 words for ER0/1/2. Length matching worked reasonably well, but topics and substantive answers still differ. EX is zero in 138/150 selected examples, limiting coverage.

Keep a cleaned EPITOME subset for a linear baseline and transfer evaluation. Record exclusions; preserve original labels separately from new ratings. Do not silently repair guessed missing words. Rationale pooling is a diagnostic only: ER0 lacks equivalent spans, so asymmetric pooling can introduce a confound.

## Research and reusable resources

- [EPITOME paper](https://behavioral-data.github.io/resources/pubpdfs/sharma2020computational.pdf) and [repository](https://github.com/behavioral-data/Empathy-Mental-Health): ER/IP/EX framework, human annotations, and rationale spans. Useful baseline; not controlled warmth triplets.
- [Warm AI repository](https://github.com/lujainibrahim/warm_ai_2025): complete tree inspected at `d82496879894d14c98cda4d4130f79f2b992e3ae`. Contains evaluation prompts, original/warm model-output samples, aggregate results, and statistical analysis. Original transformed warm/cold training pairs were not found. Some `.json` evaluation files are JSON Lines.
- [Warm AI Nature paper](https://www.nature.com/articles/s41586-026-10410-0): rewrites 3,667 assistant responses from 1,617 ShareGPT conversations with meaning-preservation instructions. Its supplement supplies rewriting prompts. Borrow the content-preservation design, but hold length, formality, agreement, and familiarity approximately fixed rather than bundling all of them with warmth. Use its factuality and belief-pressure evaluations later.
- [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues): diverse everyday emotional situations; useful prompt source, but emotion categories are not warmth levels.
- [Welivita–Pu response and rating dataset](https://github.com/anuradha1992/llm-empathy-evaluation): downloaded CSV contains 2,000 scenarios, each with human, standard GPT-4, and empathy-prompted GPT-4 responses and ratings. Empathy-prompted responses rate higher/equal/lower than standard responses in 483/1,064/453 cases. Only 212 scenarios have three distinct response ratings. Generation source is not an ordinal warmth label; content and author differences remain. Useful for judge calibration, not automatic centroid assignment.
- [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation): strategy-labeled support conversations; useful transfer data, but strategy categories are not warmth intensities.
- [Empathy-judge reliability study](https://www.nature.com/articles/s42256-025-01169-6.pdf) and [annotations](https://github.com/Fai-Poungpeth/evaluating-elements-of-empathic-communication): supports using explicit rubrics and examples, measuring agreement, and treating empathy scores as imperfect judgments.
- [Empathy Is Steerable but Multi-Axial](https://github.com/hjhj97/Empathy-Is-Steerable-but-Multi-Axial): relevant EPITOME linear-steering baseline with final-response-token extraction and automated evaluation. Reproduce a small version; verify layer-index conventions. Proxy-score changes are not equivalent to human-validated support quality.
- [Goodfire manifold paper](https://arxiv.org/html/2605.05115): connects activation geometry with induced behavior on controlled concepts. Spline reuse alone does not establish such a relationship for warmth.

## Proposed next experiment: 300 matched triplets

This is the latest proposed design. The dataset has not yet been created and these experiments have not run.

Build 300 scenarios × 3 variants = 900 responses. Use 75 scenarios each for factual disagreement, everyday emotional support, advice, and ordinary questions. Within each category use 45 training, 15 validation, and 15 test scenarios: 180/60/60 overall. Keep every variant, paraphrase, and related conversation in the same split.

| Level | Intended behavior | Constraints |
|---|---|---|
| 0 | Neutral and professional | Respectful and clear, not dismissive |
| 1 | Moderately warm | Friendly and considerate, some acknowledgment |
| 2 | Strongly warm | Compassionate and supportive, without gratuitous affirmation |

Construction procedure:

1. Reuse diverse source prompts and create a vetted substantive base answer.
2. Record invariant facts, recommendations, uncertainty, and boundaries.
3. Rewrite the same answer independently into three warmth variants using adapted Warm AI instructions.
4. Keep approximately equal length and consistent formatting, content, formality, and familiarity. Avoid invented personal experiences, extra promises, flattery, stronger factual agreement, or additional advice.
5. Vary rewriting templates or authors; reserve an unseen template for robustness evaluation.
6. Blindly validate warmth ordering and content preservation. Allow ties and uncertainty; regenerate or exclude ambiguous triplets. Requested levels are provisional labels, not ground truth.

High warmth must remain compatible with disagreement. Start causal debugging on non-crisis scenarios and retain crisis cases as a separately reported stress test, rather than allowing refusal behavior to dominate the warmth result.

## Measurement and controls

Score separately: perceived warmth, emotional understanding, factual correctness/content preservation, agreement with false claims, relevance/coherence, refusal, and length. Calibrate an independent judge on human-reviewed examples; retain disagreement and uncertainty. Include warm false reassurance, polite factual disagreement, neutral correct answers, and apologetic refusals in judge checks. LLM-only ratings must be identified as such.

Before scaling, verify on the real model:

- No hook versus zero-vector hook equivalence.
- The intended layer/token states actually receive nonzero interventions.
- No condition-order dependence or leftover hooks.
- Batched versus individual activation extraction agreement within numerical tolerance.

Existing lightweight tests do not establish all these properties on the real Llama model.

## Establish linear causal control first

- Collect all variants under the same neutral system prompt, without warmth labels or rewriting instructions in the model input.
- Form a linear direction by averaging within-scenario warm-minus-neutral activation differences. Compare mean-response and final-response-token extraction on validation.
- For PCA-based variants, subtract each scenario's mean before fitting training-only PCA to reduce shared content variation. Keep a full-space linear baseline so low-variance warmth signal is not discarded automatically.
- Compare unsteered generation, a warmth-prompting baseline, a validation-selected range of positive/negative linear strengths, and multiple norm-matched random directions.
- Use validation to choose layer, pooling, and strength. Freeze before evaluating the 60 new test scenarios.
- Measure monotonic warmth changes across independent prompts while preserving correctness and quality. Bootstrap at scenario level; conditions from one prompt are not independent samples.
- Use adequate generation length and report cap rates. Inspect token probabilities on fixed prefixes if greedy outputs remain identical despite interventions.

Success is useful behavioral control, not lower whole-activation reconstruction NMSE. The 60-scenario test is an initial evaluation size, not a power guarantee.

## Only then test curvature and ODE composition

For representation, measure held-out ER/warmth prediction against shuffled labels and ordinal projection. For stability, bootstrap by scenario and compare endpoint and orthogonal-midpoint directions across independent subsets. Positive curvature magnitude alone is not enough: noise also creates off-line centroids.

With only three centroids, a spline can interpolate noise. Compare linear, piecewise-linear, and spline interventions at comparable norms. If causal linear control is established, expand to five independently validated warmth levels and test omitted intermediate levels on held-out scenarios. Ordinal levels do not imply equal psychological spacing.

Proceed to anti-sycophancy ODE composition only after warmth control is established. Reuse Warm AI evaluation prompts and add correct-belief and legitimate-correction cases. Report warmth, correctness, user-belief agreement, and refusal jointly. A rise in refusal must not masquerade as improved truthfulness.
