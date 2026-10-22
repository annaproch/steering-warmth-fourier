# Shared-prefix branch pilot

Status: complete

**Decision: stop this shared-prefix-summary setup before Fourier fitting.** None of the compact sites worked. Larger cache replacement validates the readout plumbing, but does not establish a compact shared controller.

| Intervention | All trials: both correct | Clean source/donor trials: both correct |
|---|---:|---:|
| Residual patch: last 1 or 4 tokens, layers 18/24/28 | 0/144 | 0/114 |
| Last-token KV cache, all layers | 0/24 | 0/19 |
| Full-prefix KV cache, all layers | 21/24 | 19/19 |

Clean means both the source and donor answered both readout questions correctly without intervention. There are 19 clean source-target conditions, each reused at six compact residual sites; the 114 compact trials are not 114 independent examples. All 144 residual trials and all 24 last-token-cache trials retained the same greedy label predictions as their source readouts. Small probability changes were not measured.

Unpatched source accuracy was 7/8 paired readouts; natural donor accuracy was 21/24. The month readout sometimes emitted `not` for July, which fails this strict first-token score. Full-cache transplantation exactly reproduced natural donor readouts in 24/24 conditions, including those three donor failures. All 16 identity checks passed.

**Interpretation:** Separate readout branches do not repair the problem. In this task, editing the trailing summary tokens has not established causal access to both stored fields; the readouts continue to use source information. The results are consistent with retrieval from the untouched record tokens, but attention paths were not directly ablated. This does not show that Fourier composability fails, or that no compact intervention exists elsewhere.

**Completed:** 192 donor interventions (144 residual + 24 last-token-cache + 24 full-cache), 32 natural source/donor controls, and 16 identity controls. Each condition has two independent branch readouts. Seven unit tests passed. Runtime of the final model sweep: 273 seconds.

**Not done:** No Fourier model was fit to these branch activations, no held-out Fourier steering comparison, and no spline baseline. Those remain unjustified at the sites tested here. The earlier Fourier fitting scaffold remains available but is not evidence of model performance.

**Recommendation:** End this particular sequence of summary-token trials. For a further composability study, first define a causal intervention at the actual information-bearing positions or a deliberately established shared bottleneck. Alternatively, return to frequency-specific effects in the existing single-concept steering tasks. Do not spend time fitting a 2D surface to these failed intervention sites.

Reproduce with `HF_HUB_OFFLINE=1 /home/agent/venv/bin/python fourier-steering/branch_pilot.py`. Tests: `/home/agent/venv/bin/python -m unittest discover -s fourier-steering -p "test_*pilot.py" -v`.

Aggregated analysis: [analysis.json](analysis.json). Full tables follow.

A single edited prefix cache is copied into two independent branches: one asks for the weekday, the other the month. No readout-position patches. Both scores require the requested labels on the first generated tokens; all labels are verified to be single tokens with a leading space.

Model: Llama 3.1 8B, bfloat16, CPU. Eight source pairs; day-only, month-only, and both-change donors. Prefix labels are at fixed token positions. Compact interventions replace the last 1 or 4 prefix residuals at one of layers 18, 24, 28. Those trailing positions contain only `End of record.`; they exclude the literal label tokens.

Cache controls transplant the last prefix token across all layers, or the entire prefix across all layers. Full-cache replacement is an upper bound, equivalent to giving the readouts the donor context, not evidence of compact concept steering.

| Method | Edit | n | Day correct | Month correct | Both correct |
|---|---|---:|---:|---:|---:|
| natural | baseline | 8 | 8/8 | 7/8 | 7/8 |
| natural | day | 8 | 8/8 | 7/8 | 7/8 |
| natural | month | 8 | 8/8 | 7/8 | 7/8 |
| natural | both | 8 | 8/8 | 7/8 | 7/8 |
| identity_residual | identity | 8 | 8/8 | 7/8 | 7/8 |
| identity_cache | identity | 8 | 8/8 | 7/8 | 7/8 |
| residual_1token_L18 | day | 8 | 0/8 | 7/8 | 0/8 |
| residual_1token_L24 | day | 8 | 0/8 | 7/8 | 0/8 |
| residual_1token_L28 | day | 8 | 0/8 | 7/8 | 0/8 |
| residual_4token_L18 | day | 8 | 0/8 | 7/8 | 0/8 |
| residual_4token_L24 | day | 8 | 0/8 | 7/8 | 0/8 |
| residual_4token_L28 | day | 8 | 0/8 | 7/8 | 0/8 |
| cache_last_all_layers | day | 8 | 0/8 | 7/8 | 0/8 |
| cache_full_all_layers | day | 8 | 8/8 | 7/8 | 7/8 |
| residual_1token_L18 | month | 8 | 8/8 | 0/8 | 0/8 |
| residual_1token_L24 | month | 8 | 8/8 | 0/8 | 0/8 |
| residual_1token_L28 | month | 8 | 8/8 | 0/8 | 0/8 |
| residual_4token_L18 | month | 8 | 8/8 | 0/8 | 0/8 |
| residual_4token_L24 | month | 8 | 8/8 | 0/8 | 0/8 |
| residual_4token_L28 | month | 8 | 8/8 | 0/8 | 0/8 |
| cache_last_all_layers | month | 8 | 8/8 | 0/8 | 0/8 |
| cache_full_all_layers | month | 8 | 8/8 | 7/8 | 7/8 |
| residual_1token_L18 | both | 8 | 0/8 | 0/8 | 0/8 |
| residual_1token_L24 | both | 8 | 0/8 | 0/8 | 0/8 |
| residual_1token_L28 | both | 8 | 0/8 | 0/8 | 0/8 |
| residual_4token_L18 | both | 8 | 0/8 | 0/8 | 0/8 |
| residual_4token_L24 | both | 8 | 0/8 | 0/8 | 0/8 |
| residual_4token_L28 | both | 8 | 0/8 | 0/8 | 0/8 |
| cache_last_all_layers | both | 8 | 0/8 | 0/8 | 0/8 |
| cache_full_all_layers | both | 8 | 8/8 | 7/8 | 7/8 |

Compact eligibility: []. Requires >=50% paired success for each edit type at one fixed site, after all eight source pairs.

Identity residual/cache controls and direct donor readouts are included. Prefix caches are cloned before branching, and checks verify branch evaluation does not mutate the saved shared cache.

Fourier fitting is not part of this donor gate. A full-cache-only success does not justify proceeding with the existing compact Fourier controller.

Artifacts: [raw rows](rows.json), [summary](summary.json), [config](config.json), [code](../../branch_pilot.py).
