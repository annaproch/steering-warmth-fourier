**Research assessment — 27 September 2026**

There is enough material for an honest exploratory blog post, but the saved results do not yet establish a novel Fourier steering method or successful multi-concept composition. My recommendation is to pursue **frequency-resolved interference and correction in joint cyclic-concept control**. This is a research hypothesis, not a result already demonstrated here.

I reviewed all nine experiment scripts, all nine saved logs, and FINDINGS.md, and checked primary literature. No Llama inference was rerun: the current Python environment lacks NumPy, PyTorch, and Transformers. `audit_math.py` provides dependency-free synthetic checks of several mathematical issues and static checks of prompt overlap. The historical scripts and logs are preserved. Treat this review as a qualification of the stronger claims in FINDINGS.md.

**What the literature already covers**

| Work | Relevance to this project |
|---|---|
| [Engels et al., Not All Language Model Features Are Linear](https://arxiv.org/html/2405.14860v2) | Circular weekday/month features, causal interventions, and continuous circular control predate this project. Their regression analysis also separates input-related nuisance variation from output representations. |
| [Nanda et al., Progress measures for grokking](https://arxiv.org/abs/2301.05217) | Fourier mechanisms and frequency-level causal analysis in modular arithmetic are established foundations. |
| [Karkada et al., Symmetry in language statistics](https://arxiv.org/abs/2602.15029) | Connects representation geometry to statistical symmetries. Useful motivation for testing canonical order against shuffled labels, rather than assuming a Fourier fit proves cyclic structure. |
| [Wurgaft et al., Manifold Steering](https://arxiv.org/html/2605.05115v1) | Fits activation and behavioral manifolds and compares intervened paths. Section 4 already studies factored control on grids/cylinders; Appendix C.2 studies weekday uncertainty along another manifold coordinate. |
| [Feucht et al., Arithmetic in the Wild](https://arxiv.org/html/2605.01148v1) | Essential missing citation. Section 4 and Algorithm 1 explicitly steer cyclic tasks with Fourier probes. The mechanism uses shared base-10 periods, including 2, 5, and 10, rather than only the concept's period. Algorithm 1 has target phases and a radius multiplier. |
| [Scalena et al., Dynamic Activation Composition](https://arxiv.org/abs/2406.17563) | Multi-property steering with dynamically adjusted strengths. Magnitude calibration is already a substantive part of composition research. |
| [Postmus and Abreu, Conceptor Steering](https://arxiv.org/abs/2410.16314) | Soft subspace projections and Boolean composition provide relevant alternatives to summing vectors. |
| [Vu and Nguyen, Angular Steering](https://arxiv.org/abs/2510.26243), [Aparin and Gaintseva, Angle-Norm Decomposition](https://arxiv.org/abs/2606.06735) | Rotation and separation of angular/radial effects are already studied. A generic angle-versus-magnitude story has limited novelty. |
| [Ghasemi et al., ORBIT](https://arxiv.org/abs/2606.22357) | Particularly close composition baseline: joint subspaces, rotations, norm imbalance, and multi-attribute control. |
| [Interference-Aware Composition of Steering Vectors](https://openreview.net/pdf/e13856076696cadb36cf3077e591e29f04424172.pdf) | An anonymous submission proposing dynamic magnitude allocation using interaction information. Relevant prior art, with publication status not established by this review. |
| [Compositional Steering with Steering Tokens](https://arxiv.org/abs/2601.05062) | Evaluates generalization to unseen compositions. Different intervention mechanism, but a useful standard for what a composition claim should demonstrate. |

Other adjacent work includes [The Shape of Beliefs](https://arxiv.org/abs/2602.02315), [Riemannian-Manifold Steering](https://arxiv.org/abs/2605.24942), and [Topological Steering](https://arxiv.org/abs/2609.00597). These concern distributional belief geometry, learned geometry for steering, and topology-based control respectively; they are not direct demonstrations of the proposed joint day/month experiment.

Your centroid DFT is distinct from learned arithmetic Fourier probes. That distinction can support a narrowly stated contribution, but not “nobody has used Fourier features to steer weekdays/months.” I did not establish an exact predecessor for the proposed interference experiment; that is not proof of priority.

**What the saved experiments actually support**

| Experiment | Surviving observation | Interpretation |
|---|---|---|
| `llama_full.py`, log 01 | Fitted sentence-ending states do not successfully steer the earlier weekday token. | Intervention-population mismatch; not evidence against cyclic steering. Its spectrum describes a different activation population. |
| `llama_steer2.py`, log 02 | 38/42 baseline answers correct; 7/7 held-out centroid classifications at layers 24, 28, 31. | Useful protocol sanity check. Seven examples and one held-out template do not establish broad generalization. |
| `novel.py`, log 03 | Radius increases sharpen some PCA-plane interventions; month perturbations are weak. | Exploratory evidence for target-dependent strength effects. Initial non-interference is uninformative when the second handle fails. |
| `compose2.py`, log 04 | PCA month steering fails for July/November over the tested radii; May alone succeeds at high radius. | Good negative calibration result, not a Fourier K=1 sweep. |
| `norm_crosstalk.py`, log 05 | Global norm restoration does not repair the tested interference. | This particular mitigation failed; RMSNorm causation was not falsified. |
| `fourier_steer_real.py`, log 06 | Weekdays at amp=1 improve from 0/4 to 3/4 to 4/4 as K goes 1,2,3. | The cleanest positive pilot. Fourier specificity and held-out robustness remain untested. |
| `months_deep.py`, log 07 | K=5, amp=6 hits 6/6 tested months. | Higher-dimensional intervention works on this prompt. K=6 comparison is compromised by a mathematical bug. |
| `compose_fourier.py`, log 08 | Adding a strong month edit disrupts weekday answers much more than the reverse. | Useful stress test; prompt type, order, strength, and geometry are confounded. |
| `compose_matched.py`, log 09 | Limiting the secondary perturbation preserves the primary top-1 answer in 12/12 trials each way. | Primary-answer robustness, not demonstrated joint composition. |

The more interesting unresolved observation is the overlap of the larger weekday/month subspaces. It motivates experiments; it does not yet identify a shared causal calendar feature.

**Corrections needed before writing a results post**

1. **The month Nyquist coefficient is doubled.** `fourier_fit` uses `2/n` for every cosine. For even n, the k=n/2 cosine requires `1/n`; its sine is identically zero on the sampling grid. For n=12, use A6=(1/12) sum_d C[d](-1)^d and no B6. The current formula doubles the alternating component. On a pure alternating scalar signal, `audit_math.py` reproduces a maximum error of 1, while the corrected reconstruction error is below 2e-15. Consequently, “K=6 overfits noise” is unsupported. Correcting this need not make K=6 better; it makes the comparison interpretable.

2. **QR retains a spurious Nyquist direction.** `subspace_basis` requests 12 columns for twelve centered month centroids, whose meaningful centered rank is at most 11. Floating-point sine residuals and uncentered coefficient fitting can introduce a nuisance direction, or QR can complete a rank-deficient basis. Use explicit Nyquist handling, centered fitting, and rank-revealing SVD with a tolerance. Log actual rank. The logged “dimension 12” should not be treated as twelve meaningful concept directions.

3. **The magnitude matching is only applied to the secondary edit.** In `compose_matched.py`, the first month edit in the reverse-direction test remains at amp=6. Only the following weekday perturbation is rescaled. Also, 24.39 is selected from one Monday edit, while different primary targets can have different norms. Therefore “both edits matched” is not an accurate description of the implemented design.

4. **Potency is not rechecked after rescaling.** Reducing a month perturbation can preserve the weekday because it stopped installing the month. The test must measure both intended changes at the exact deployed strengths. The current setup cannot do this at one next-token slot.

5. **These are preservation tests, not joint-output tests.** A weekday prompt is scored only for its weekday, and a different month prompt only for its month. Both directions are useful, but do not demonstrate a state that simultaneously supports the requested weekday and month. Shared softmax normalization further couples these scores.

6. **The main steering prompts are in the fitting set.** `Today is Wednesday. Tomorrow is` is a training example for Thursday; `The month after February is` is a training example for March. The static audit verifies both. The held-out decoding experiment does not make subsequent steering tests held out. This is a generalization limitation, not evidence of target-answer leakage from model training.

7. **Full Fourier steering is a centroid-subspace baseline at discrete labels.** For seven labels, K=3 exactly reconstructs any seven centroids and spans their centered row space. At integer targets, reordering the labels, refitting the complete basis, and following the same target label leaves the intervention unchanged. No cyclic geometry is needed for that success. A corrected full month fit has the same issue. Fourier's possible advantage lies in truncation, regularization, interpolation, frequency diagnostics, or structured transformations—not full reconstruction alone.

8. **The K sweep changes both deletion and insertion.** The code implements x' = x - P_K(x-mu) + amp*(c_K(t)-mu). Higher K removes more of the original state's representation as well as adding more target structure. Separate these factors: fix a full concept erasure space while varying the injected frequencies, and separately fix the target while varying erasure rank. Compare to equal-rank PCA and report perturbation norms and held-out efficacy curves.

9. **PCA polar steering is not generally Fourier K=1.** The former picks a variance-maximizing plane and empirical angles; the latter fixes canonical label phases. Moreover, A1 cos(theta)+B1 sin(theta) generally traces an ellipse, not a circle, unless A1 and B1 are orthogonal with equal norm. The 82.6/83.2-degree angles come from PCA planes in log 03, not a measured Fourier-fundamental comparison. Recompute frequency-specific angles before making “fundamentals orthogonal, harmonics entangled” claims.

10. **Projection variance is not Fourier spectral power.** Log 06 measures how much centroid variance projects into the span of selected coefficient vectors. It does not measure low-pass reconstruction error or Parseval energy. Different frequency coefficient vectors need not be orthogonal in activation space. The 0.4426 month fundamental figure is from log 01's different, sentence-ending population. These quantities should not be merged.

11. **Several summary numbers overstate the logs.** There is no saved six-target Fourier K=1 month sweep through amp=8: log 06 tests four targets through amp=6, while log 04 tests a PCA method. The matched month probability change is a 4.8% increase, not a 4.8% degradation. Sunday's PCA target probability decreases from radius 4 to 6, and its day-mass also falls; Monday's monotonic trend is not universal.

12. **Whole-state renormalization does not isolate RMSNorm.** Ignoring epsilon, RMSNorm(c*x)=RMSNorm(x) for positive c. Global rescaling therefore does not undo directional competition inside the normalized branch; it also changes the residual-to-branch balance in a pre-norm transformer. Log 05 establishes failure of that rescaling remedy, not absence of normalization-mediated interference.

13. **Subspace overlap is not irreducible semantic interference.** The recorded minimum principal angle is about 51 degrees, not zero: the two fitted spaces need not have any exact intersection. Independent latent coordinates can be encoded in nonorthogonal directions and recovered with a dual basis. Sequential orthogonal overwrites can introduce interference even in that perfectly additive setting. `audit_math.py` illustrates this distinction. Projecting an edit orthogonal to another activation subspace guarantees zero linear projection there, not zero downstream behavioral change.

14. **A few other claims require narrower wording.** Increasing amplitude might compensate for a truncated basis, but might also overcome retained source information or downstream thresholds. Greater concept-token probability does not establish fluent generation or proximity to natural behavioral distributions. The overlap's cause might involve prompt templates, output embeddings, or arithmetic; the current data do not identify it. Isotropic random-subspace comparisons are weak controls for anisotropic residual activations. Label permutations cannot change the full centered centroid span, so use them to test low-frequency ordering, not full-span overlap.

**The experiment I would prioritize**

The question should be: **Can a compact Fourier model predict which day/month edits interfere, and can correcting that structure improve joint steering on unseen combinations at matched standalone efficacy?**

Use a task where both labels genuinely coexist. For example, independently vary a weekday and month in a scheduling record, and request a fixed two-field answer. Start with retrieval to establish the intervention; then use independent weekday and month arithmetic. Do not couple them through real calendar dates. Balance all 7x12 combinations, prompt families, field order, offsets, and irrelevant distractors.

First verify that one common prefix position contains usable information about both labels. Patch that position once before either answer is emitted; remove the hook for subsequent generation. Evaluate both day-first and month-first output orders. Separately, branch from the same edited prefix into day and month readout queries to diagnose retention. Readout probes are supplementary; free generation must demonstrate both correct labels together. Repatching each answer independently would test two local controls, not composition in a shared state.

Split by entire template families and hold out day/month pairs while retaining every individual label in training. Keep validation for K, strength, and regularization selection separate from final testing. Begin with the existing model at L28, then examine a small set of earlier layers once the joint task works. Freeze the data split and evaluation before comparing methods. Report all source/target cases, with baseline-wrong cases and no-op targets separated; bootstrap over independent prompt instances or template groups, not individual correlated target edits.

Fit joint centroids with the decomposition

    h(d,m) = mu + f(d) + g(m) + r(d,m).

Here f is the day main effect, g the month main effect, and r the remaining interaction, with zero marginal means under a balanced design. Fourier-expand f and g first. Add only a small, regularized selection of mixed frequencies if they improve held-out prediction and steering. A full 7x12 DFT can simply memorize 84 centroids; that alone is not a discovery of a smooth torus.

For a joint Fourier model

    h(theta,phi) = sum_(k,l) c_(k,l) exp(i*(k*theta+l*phi)),

the k!=0,l!=0 terms describe non-additivity in this chosen coordinate system. A useful diagnostic is mixed-mode energy divided by total nonconstant energy, evaluated with held-out reconstruction and a noise estimate from independent prompt splits. This is not automatically a measure of behavioral non-compositionality. Additive embeddings can have nonorthogonal directions, and a nonlinear downstream decoder can couple even orthogonal inputs. Conversely, a curved, non-additive embedding can still support independent intrinsic-coordinate control. A spline fit can also be subjected to an interaction analysis; Fourier makes the frequency decomposition convenient, not uniquely possible.

The causal test is to vary or ablate selected mixed modes, then measure whether the predicted pair-specific interference changes. Compare equal-budget mode removals and random/equal-rank controls. A useful result would show a relationship across held-out pairs, followed by an intervention that improves the outcome—not just a visually interesting spectrum.

**Use these baselines before inventing a complicated controller**

| Baseline | What it rules out |
|---|---|
| No intervention and ordinary prompting | Artificial gains from an unnecessarily difficult setup; practical usefulness. |
| Centroid differences and full centroid-subspace replacement | Success from generic label editing. |
| Equal-rank PCA replacement | Extra dimension count as the explanation for Fourier gains. |
| Canonical versus shuffled-order truncated Fourier | A generic low-rank code masquerading as cyclic structure. |
| Periodic cubic splines for each concept and a joint surface | The claimed added value over manifold steering. |
| Sum of deltas computed from the same original state; both sequential overwrite orders | Interference introduced by the editing operator itself. |
| Simultaneous linear constraints / dual-coordinate control | Nonorthogonal projections mistaken for unavoidable semantic entanglement. |
| Additive joint Fourier versus regularized mixed-mode Fourier | Whether interaction terms buy anything beyond independent concept control. |

For simultaneous constraints, stack independently validated linear readouts into W and prescribe their target readings b. The minimum-norm correction is delta=W^dagger*(b-W*x), with regularization if necessary. This removes sequential overwrite order as an explanation, but only guarantees the specified linear readings; behavior still needs testing. If using a generative basis B, its dual B^dagger recovers latent coefficients, which are not generally the same as B-transpose projections. Keep these two interpretations explicit.

Run matched-norm sweeps as diagnostics, but make the main comparison at matched held-out standalone success or target margin. Equal norm does not mean equal semantic strength. Report joint exact-match accuracy, each marginal accuracy, damage to the unedited field, conditional within-concept probabilities, full-vocabulary mass, sequence likelihood, edit norm, and sensitivity to order. Avoid relying on an average that lets improvement on one target hide destruction of another.

When comparing curve parameterizations, hold centroids, intervention subspace, retained residual, target strengths, and evaluation prompts fixed. The original manifold paper's Appendix A.6 describes different replacement conventions for manifold and linear paths; reproduce its protocol separately from this controlled comparison. Most importantly, include intermediate path points: spline and complete Fourier interpolation through the same anchors cannot be distinguished by endpoint reconstruction alone.

**Where magnitude/phase separation can still add value**

Keep three different quantities separate: Fourier phase (which label), a target-profile radius (how far from the fitted center), and the actual edit norm ||x'-x||. Equal target radius does not imply equal edit norm, and scaling all harmonics does not guarantee a pure confidence change.

A stronger experiment preserves the live per-frequency magnitudes and shifts each harmonic by k*delta_theta, rather than replacing every sample with a mean prototype. Recover coefficients using an appropriate dual/learned decoder; QR coordinates alone are not harmonic coordinates. Compare this to fixed-radius target replacement and explicit output-temperature baselines. Test whether within-concept entropy can change while identity, the other concept, and unrelated output quality stay fixed. Even then, call it concentration control unless calibration against known uncertainty is measured.

Another possible Fourier-specific handle is controlled smoothing: convolving a cyclic profile with a wrapped Gaussian damps harmonic k by exp(-k^2*sigma^2/2), separating a phase shift from a bandwidth change in the parameterization. Whether this induces predictable output uncertainty is an empirical question. It could complement composition, but I would not prioritize it before establishing joint control.

**What would make a good blog post**

Today: an exploratory post such as **“When Fourier steering looks compositional—and how to test whether it is.”** Show the positive K sweep, the inert-perturbation trap, the Nyquist error, and the difference between preserving one answer and controlling two. State that Fourier steering and geometric composition have prior art. Do not headline an improvement over Goodfire or successful independent day/month control.

After the proposed experiment: the stronger story would be **“Which frequencies make concept steering interfere?”** The minimum convincing result is robust joint control on unseen combinations plus evidence that frequency-aware selection or mixed-mode correction improves the success-versus-interference tradeoff over the simple baselines. A negative result is also useful if simultaneous linear constraints remove the apparent interference: it would show that the original obstacle came largely from the overwrite operator, not necessarily from the model's concept geometry.

My priority order is: fix/qualify the analysis; establish the shared-state joint task; test additive and simultaneous-constraint controls; measure frequency-specific interference; add mixed terms only if residual failures demand them. The work was close to a useful experimental platform, but the central composition experiment has not yet been run.
