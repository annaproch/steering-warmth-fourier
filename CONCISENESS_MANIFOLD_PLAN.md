# Plan: Find a conciseness manifold in Llama 3

**Goal:** Test whether ToneBank’s Concise examples reveal a low-dimensional, curved representation of conciseness in Llama-3.1-8B-Instruct that supports causal steering.

**Method reference:** [Goodfire manifold-steering paper, Appendix A.3](https://arxiv.org/html/2605.05115): collect activations, apply PCA, average by concept value, and fit a spline through the centroids.

1. **Prepare ordered data.** Start with ToneBank prompts and Concise responses, collecting other response variants for the same prompts where available. Rate responses on five ordered conciseness levels, separately checking essential-information coverage. The binary Concise label alone does not define five levels; generate matched variants if coverage is insufficient. Do not treat other tones as automatically non-concise. Split by question into training, validation, and test sets.
2. **Collect activations.** Record residual-stream activations across candidate layers at comparable generation positions. Control for topic, prompt wording, token position, and response length. Use donor-patching checks to identify sites that can causally affect conciseness without destroying content.
3. **Reduce dimensionality.** Fit PCA on training activations only, starting with up to 64 components. Compare smaller ranks on validation data. Inspect within-question differences as a control for topic variation.
4. **Fit the candidate manifold.** Average projected training activations within each ordered conciseness level. Fit a natural cubic spline through these centroids. Use a non-periodic curve: conciseness has no cyclic ordering. Treat the level spacing as an assumption and test sensitivity to it.
5. **Validate geometry and steering.** Compare the spline with a straight-line fit and a PCA-subspace baseline. On held-out questions, test whether moving toward the concise end reduces redundancy and length while preserving information, correctness, and coherence. Select intervention strength on validation data only.

**Success criterion:** Reproducible structure across held-out questions, plus better concision–quality tradeoffs from curved steering than linear baselines at comparable intervention strengths. A spline through centroids or an attractive PCA plot alone is not evidence of a useful manifold.

**Artifacts:** Labeled examples and splits, activation caches, PCA fit, centroids and spline, diagnostic plots, and held-out steering results. Defer ODE composition until conciseness steering works independently.
