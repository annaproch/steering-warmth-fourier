"""Bounded shared-prefix donor gate, with independent day/month readouts.

No hooks run on either readout. Both branches start with exactly the same
edited prefix KV cache, copied before the two queries are appended.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from joint_pilot import DAYS, MONTHS, LAYERS, MODEL, Runner, parse

ROOT = Path(__file__).resolve().parent / "results" / "branch_pilot"
SOURCES = [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 1), (0, 5)]
QUERIES = ["\nThe weekday in the record is", "\nThe month in the record is"]


def prefix(d, m):
    return f"Record: weekday={DAYS[d]}; month={MONTHS[m]}.\nEnd of record."


def clone_cache(cache):
    return DynamicCache([(x.keys.clone(), x.values.clone()) for x in cache.layers])


def aggregate(rows):
    result = {}
    for r in rows:
        key = r["method"] + "/" + r["edit"]
        g = result.setdefault(key, dict(n=0, day=0, month=0, both=0))
        g["n"] += 1
        for name in ["day", "month", "both"]:
            g[name] += int(r[name + "_ok"])
    return result


def save(rows, complete=False):
    summary = aggregate(rows)
    methods = sorted({r["method"] for r in rows if r["method"].startswith("residual")})
    eligible = [m for m in methods if complete and all(
        summary[m + "/" + edit]["both"] / summary[m + "/" + edit]["n"] >= 0.5
        for edit in ["day", "month", "both"])]
    (ROOT / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (ROOT / "summary.json").write_text(json.dumps(
        dict(complete=complete, compact_eligible=eligible, counts=summary), indent=2) + "\n")
    lines = ["# Shared-prefix branch pilot", "",
             "Status: " + ("complete" if complete else "running; counts are interim"), "",
             "A single edited prefix cache is copied into two independent branches: "
             "one asks for the weekday, the other the month. No readout-position patches. "
             "Both scores require the requested labels on the first generated tokens; "
             "all labels are verified to be single tokens with a leading space.", "",
             "Model: Llama 3.1 8B, bfloat16, CPU. Eight source pairs; day-only, "
             "month-only, and both-change donors. Prefix labels are at fixed token "
             "positions. Compact interventions replace the last 1 or 4 prefix residuals "
             "at one of layers 18, 24, 28. Those trailing positions contain only "
             "`End of record.`; they exclude the literal label tokens.", "",
             "Cache controls transplant the last prefix token across all layers, or "
             "the entire prefix across all layers. Full-cache replacement is an "
             "upper bound, equivalent to giving the readouts the donor context, "
             "not evidence of compact concept steering.", "",
             "| Method | Edit | n | Day correct | Month correct | Both correct |",
             "|---|---|---:|---:|---:|---:|"]
    for key, s in summary.items():
        method, edit = key.split("/")
        lines.append(f'| {method} | {edit} | {s["n"]} | {s["day"]}/{s["n"]} | '
                     f'{s["month"]}/{s["n"]} | {s["both"]}/{s["n"]} |')
    lines += ["", "Compact eligibility: " + str(eligible) + ". "
              "Requires >=50% paired success for each edit type at one fixed site, "
              "after all eight source pairs.", "",
              "Identity residual/cache controls and direct donor readouts are included. "
              "Prefix caches are cloned before branching, and checks verify branch "
              "evaluation does not mutate the saved shared cache.", "",
              "Fourier fitting is not part of this donor gate. A full-cache-only success "
              "does not justify proceeding with the existing compact Fourier controller.", "",
              "Artifacts: [raw rows](rows.json), [summary](summary.json), "
              "[config](config.json), [code](../../branch_pilot.py).", ""]
    (ROOT / "SUMMARY.md").write_text("\n".join(lines))
    return eligible


class BranchRunner(Runner):
    @torch.inference_mode()
    def prefill(self, ids, specs=(), donor_states=None):
        captured, handles, counts = {}, [], [0] * len(specs)
        for L in LAYERS:
            def hook(module, inp, out, L=L):
                hs = out[0] if isinstance(out, tuple) else out
                captured[L] = hs.detach().clone()
                if not specs:
                    return out
                hs = hs.clone()
                for row, spec in enumerate(specs):
                    if spec.get("layer") == L:
                        span = spec["span"]
                        hs[row, -span:] = donor_states[L][spec["donor"], -span:]
                        counts[row] += 1
                return (hs,) + out[1:] if isinstance(out, tuple) else hs
            handles.append(self.model.model.layers[L - 1].register_forward_hook(hook))
        try:
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        finally:
            for h in handles:
                h.remove()
        assert counts == [int("layer" in s) for s in specs]
        return out.past_key_values, captured

    @torch.inference_mode()
    def read(self, cache):
        # Each original row becomes [day branch, month branch].
        original_length = cache.get_seq_length()
        snapshot = cache.layers[-1].keys.clone()
        branched = clone_cache(cache)
        branched.batch_repeat_interleave(2)
        q = [self.tok.encode(s, add_special_tokens=False) for s in QUERIES]
        assert len(q[0]) == len(q[1])
        n = cache.layers[0].keys.shape[0]
        ids = torch.tensor(q * n)
        out = self.model(input_ids=ids, attention_mask=torch.ones(
            len(ids), original_length + ids.shape[1], dtype=torch.long),
            past_key_values=branched, use_cache=True)
        assert cache.get_seq_length() == original_length
        assert torch.equal(cache.layers[-1].keys, snapshot), "Branch mutated saved cache"
        tokens = out.logits[:, -1].argmax(-1).tolist()
        answers = []
        for i in range(n):
            texts = [self.tok.decode([t]) for t in tokens[2*i:2*i+2]]
            answers.append(dict(day_text=texts[0], month_text=texts[1],
                                prediction=[parse(texts[0])[0], parse(texts[1])[1]]))
        return answers


def result_row(answer, source, target, method, edit, **extra):
    d, m = answer["prediction"]
    return dict(source=source, target=target, method=method, edit=edit,
                day_ok=d == target[0], month_ok=m == target[1],
                both_ok=[d, m] == list(target), **answer, **extra)


@torch.inference_mode()
def main():
    start = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)
    runner = BranchRunner()
    for word in DAYS + MONTHS:
        assert len(runner.tok.encode(" " + word, add_special_tokens=False)) == 1
    (ROOT / "config.json").write_text(json.dumps(dict(
        model=MODEL, revision=runner.model.config._commit_hash, layers=LAYERS,
        sources=SOURCES, spans=[1, 4], queries=QUERIES, torch=torch.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gate="50% paired success for each edit type at one fixed residual patch site"), indent=2))
    rows = []
    for d, m in SOURCES:
        source = (d, m)
        targets = [source, ((d + 3) % 7, m), (d, (m + 5) % 12),
                   ((d + 3) % 7, (m + 5) % 12)]
        labels = ["baseline", "day", "month", "both"]
        seqs = [runner.tok.encode(prefix(*p)) for p in targets]
        assert len({len(x) for x in seqs}) == 1, "Prefix positions differ"
        assert all(x[-4:] == seqs[0][-4:] for x in seqs)
        ids = torch.tensor(seqs)
        donors, donor_states = runner.prefill(ids)
        natural = runner.read(donors)
        for i, answer in enumerate(natural):
            rows.append(result_row(answer, source, targets[i], "natural", labels[i]))
        specs = [dict(method="identity_residual", edit="identity", donor=0, layer=28, span=4),
                 dict(method="identity_cache", edit="identity", donor=0, cache="full")]
        for i in [1, 2, 3]:
            for span in [1, 4]:
                for L in LAYERS:
                    specs.append(dict(method=f"residual_{span}token_L{L}", edit=labels[i],
                                      donor=i, layer=L, span=span))
            for scope in ["last", "full"]:
                specs.append(dict(method=f"cache_{scope}_all_layers", edit=labels[i],
                                  donor=i, cache=scope))
        edited, _ = runner.prefill(ids[:1].repeat(len(specs), 1), specs, donor_states)
        for row, spec in enumerate(specs):
            if "cache" in spec:
                sl = slice(-1, None) if spec["cache"] == "last" else slice(None)
                for dst, src in zip(edited.layers, donors.layers):
                    dst.keys[row, :, sl] = src.keys[spec["donor"], :, sl]
                    dst.values[row, :, sl] = src.values[spec["donor"], :, sl]
        answers = runner.read(edited)
        for spec, answer in zip(specs, answers):
            extra = {}
            if "layer" in spec:
                L, span = spec["layer"], spec["span"]
                delta = donor_states[L][spec["donor"], -span:] - donor_states[L][0, -span:]
                extra = dict(delta_norm=delta.float().norm().item(), scalars_replaced=delta.numel())
            else:
                length = 1 if spec["cache"] == "last" else ids.shape[1]
                extra = dict(scalars_replaced=sum(
                    2 * layer.keys.shape[1] * length * layer.keys.shape[-1] for layer in edited.layers))
            rows.append(result_row(answer, source, targets[spec["donor"]],
                                   spec["method"], spec["edit"], **extra))
            if spec["edit"] == "identity":
                assert answer["prediction"] == natural[0]["prediction"], "Identity failed"
            if spec.get("cache") == "full":
                assert answer["prediction"] == natural[spec["donor"]]["prediction"], "Cache oracle failed"
        save(rows)
        latest = [r for r in rows[-len(specs):] if r["edit"] == "both"]
        print(json.dumps(dict(source=source, natural=natural[0]["prediction"],
                              both_change={r["method"]: r["prediction"] for r in latest},
                              elapsed=round(time.monotonic()-start))), flush=True)
    eligible = save(rows, complete=True)
    print("Complete; compact eligible sites:", eligible, flush=True)


if __name__ == "__main__":
    main()
