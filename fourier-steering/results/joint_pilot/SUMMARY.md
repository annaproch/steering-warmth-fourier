# Joint weekday/month pilot — results

Completed 2026-09-27T13:53:28+00:00.

**Outcome:** No tested layer passed the donor-patching gate. The additive-versus-joint Fourier experiment has not been run. This is a negative result for the chosen shared intervention site, not evidence against Fourier composability.

**Completed:** Implemented the joint task, deterministic 63/21 fitting/held-out pair split, activation caching, one-time donor patching, and additive/joint Fourier fitting and evaluation code. Ran the donor sweep on Llama 3.1 8B at layers 18, 24, 28, with eight source pairs in each of two output orders and three edit types: 144 patched trials, 16 source baselines, and 16 identity-patch checks. All identity checks passed. Four mathematical/data unit tests passed.

**Setup:** bfloat16, CPU, greedy decoding; at most eight generated tokens, stopping after both labels appear. Patch only the final prompt token during prefill, then remove all hooks before continuation. Independent donor conditions are batched. Correctness is exact label matching, not formatting or output-order correctness.

The source record says the meeting is on a weekday in a month. A request specifies the output field order. The prompt includes an opening quote so that the first generated token is a label. An earlier punctuation-predicting prompt was abandoned; its partial results are archived separately and excluded from this table.

| Output order | Layer | Requested edit | Trials | Weekday correct | Month correct | Both correct |
|---|---:|---|---:|---:|---:|---:|
| weekday, month | unpatched | baseline | 8 | 8/8 | 8/8 | 8/8 |
| weekday, month | 18 | day | 8 | 0/8 | 8/8 | 0/8 |
| weekday, month | 24 | day | 8 | 6/8 | 8/8 | 6/8 |
| weekday, month | 28 | day | 8 | 8/8 | 8/8 | 8/8 |
| weekday, month | 18 | month | 8 | 8/8 | 0/8 | 0/8 |
| weekday, month | 24 | month | 8 | 8/8 | 0/8 | 0/8 |
| weekday, month | 28 | month | 8 | 8/8 | 0/8 | 0/8 |
| weekday, month | 18 | both | 8 | 0/8 | 0/8 | 0/8 |
| weekday, month | 24 | both | 8 | 6/8 | 0/8 | 0/8 |
| weekday, month | 28 | both | 8 | 8/8 | 0/8 | 0/8 |
| month, weekday | unpatched | baseline | 8 | 8/8 | 8/8 | 8/8 |
| month, weekday | 18 | day | 8 | 0/8 | 8/8 | 0/8 |
| month, weekday | 24 | day | 8 | 0/8 | 8/8 | 0/8 |
| month, weekday | 28 | day | 8 | 0/8 | 8/8 | 0/8 |
| month, weekday | 18 | month | 8 | 8/8 | 0/8 | 0/8 |
| month, weekday | 24 | month | 8 | 7/8 | 3/8 | 2/8 |
| month, weekday | 28 | month | 8 | 7/8 | 5/8 | 4/8 |
| month, weekday | 18 | both | 8 | 0/8 | 0/8 | 0/8 |
| month, weekday | 24 | both | 8 | 0/8 | 5/8 | 0/8 |
| month, weekday | 28 | both | 8 | 0/8 | 8/8 | 0/8 |

Natural donor answers were correct for 144/144 intervention conditions (each donor is counted once per layer here).

**Interpretation:** At L28 with weekday-first output, the weekday can be transferred, but the month stays at the source value. In month-first output, both-change edits transfer the donor month in 8/8 cases but never the donor weekday. Month-only edits have 4/8 joint success; 5/8 have the target month and 7/8 preserve the weekday. No layer achieves successful both-change transfer in either order. This pattern is consistent with later generation recovering information from the original prompt, but the attention mechanism has not been isolated. A successful single-token next-answer intervention does not establish shared-state multi-field control.

**Gate:** At least 50% joint correctness for every edit type in both orders, at the same layer. Eligible layers: [].

**Prepared but not executed on model data:** The additive K=2 fit (nine coefficients), joint K=2-by-K=2 fit (25 coefficients), training-template activation collection and held-out Fourier steering comparison. The implemented fitting path uses regularized least squares in a common 16-dimensional PCA space; it has not been validated by a successful model run.

**Validation already done:** Both feature designs have full column rank on the 63 training pairs, and all individual labels remain represented. Synthetic held-out recovery tests pass. A pure mixed-frequency signal is exactly recovered by the joint design but not the additive design. These checks validate the design matrices, not model steerability.

**Not done:** No real-model additive-versus-joint Fourier result, no spline comparison, no amplitude sweep, no arithmetic version, and no second-model replication.

**Next decision:** Establish a successful joint causal control using a different intervention site or a shared prefix span before evaluating Fourier composition. Any expanded intervention should retain an actual-donor positive control. The present result does not warrant claiming that joint Fourier steering fails.

Artifacts: [raw donor rows](donor_rows.json), [aggregated donor results](donor_summary.json), [configuration](config.json), [run log](../joint_donor.log), [pilot code](../../joint_pilot.py), [unit tests](../../test_joint_pilot.py).

Commands:

```sh
HF_HUB_OFFLINE=1 /home/agent/venv/bin/python fourier-steering/joint_pilot.py --stage donor
/home/agent/venv/bin/python -m unittest discover -s fourier-steering -p test_joint_pilot.py -v
```
