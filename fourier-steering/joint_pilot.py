"""Joint weekday/month pilot. Cached Llama; one prefix patch, free generation.

Run with /home/agent/venv/bin/python fourier-steering/joint_pilot.py.
Stage 'donor' is the gate; stage 'fourier' requires a successful donor gate.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Meta-Llama-3.1-8B"
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()
LAYERS = [18, 24, 28]
ROOT = Path(__file__).resolve().parent / "results" / "joint_pilot"
DAY_RE = re.compile(r"\b(" + "|".join(DAYS) + r")\b", re.I)
MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\b", re.I)


def prompt(d, m, template=0, order="dm"):
    fields = "weekday, month" if order == "dm" else "month, weekday"
    record = [f"The meeting is on {DAYS[d]}, in {MONTHS[m]}.",
              f"An appointment falls in {MONTHS[m]} on a {DAYS[d]}.",
              f"The event's month is {MONTHS[m]}; its weekday is {DAYS[d]}."][template]
    return f'{record}\nReturn only the {fields}, separated by a comma.\nAnswer: "'


def parse(text):
    day, month = DAY_RE.search(text), MONTH_RE.search(text)
    return (DAYS.index(day[0].capitalize()) if day else None,
            MONTHS.index(month[0].capitalize()) if month else None)


def features(pairs, joint=False):
    def circle(i, n):
        return [f(2 * np.pi * k * i / n) for k in [1, 2] for f in [np.cos, np.sin]]
    rows = []
    for d, m in pairs:
        a, b = circle(d, 7), circle(m, 12)
        rows.append([1] + a + b + ([x * y for x in a for y in b] if joint else []))
    return np.array(rows)


def heldout(d, m):
    return (m - 2 * d) % 4 == 0  # 21 pairs; all individual labels remain in fitting.


class Runner:
    def __init__(self):
        torch.set_num_threads(8)
        self.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, local_files_only=True).eval()
        self.cache = {}

    @torch.inference_mode()
    def run(self, text, layer=None, vector=None):
        ids = self.tok(text, return_tensors="pt")
        captured, handles, patches = {}, [], []
        for L in LAYERS:
            def hook(module, inp, out, L=L):
                hs = out[0] if isinstance(out, tuple) else out
                captured[L] = hs[0, -1].float().numpy().copy()
                if L != layer:
                    return out
                patches.append(L)
                hs = hs.clone()
                hs[0, -1] = torch.as_tensor(vector, dtype=hs.dtype, device=hs.device)
                return (hs,) + out[1:] if isinstance(out, tuple) else hs
            handles.append(self.model.model.layers[L - 1].register_forward_hook(hook))
        try:
            out = self.model(**ids, use_cache=True)
        finally:
            for h in handles:
                h.remove()
        assert len(patches) == (0 if layer is None else 1)
        generated, mask = [], ids["attention_mask"]
        for step in range(8):
            token = out.logits[:, -1].argmax(-1).reshape(1, 1)
            generated.append(token.item())
            answer = self.tok.decode(generated, skip_special_tokens=True)
            if None not in parse(answer) or token.item() == self.tok.eos_token_id:
                break
            mask = torch.cat([mask, torch.ones_like(token)], dim=1)
            out = self.model(input_ids=token, attention_mask=mask,
                             past_key_values=out.past_key_values, use_cache=True)
        return {"text": answer, "prediction": parse(answer), "tokens": generated}, captured

    def natural(self, text):
        if text not in self.cache:
            key = hashlib.sha256(text.encode()).hexdigest()
            path = ROOT / "cache" / (key + ".npz")
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    result = json.loads(str(z["result"]))
                    self.cache[text] = result, {L: z[str(L)].copy() for L in LAYERS}
            else:
                result, states = self.run(text)
                np.savez_compressed(path, result=json.dumps(result),
                                    **{str(L): v for L, v in states.items()})
                self.cache[text] = result, states
        return self.cache[text]

    @torch.inference_mode()
    def batch_run(self, text, edits):
        """Same source prompt, independent one-time patches in separate batch rows."""
        ids = {k: v.repeat(len(edits), 1) for k, v in self.tok(text, return_tensors="pt").items()}
        handles, counts = [], [0] * len(edits)
        for L in LAYERS:
            def hook(module, inp, out, L=L):
                hs = (out[0] if isinstance(out, tuple) else out).clone()
                for i, (layer, vector) in enumerate(edits):
                    if layer == L:
                        hs[i, -1] = torch.as_tensor(vector, dtype=hs.dtype, device=hs.device)
                        counts[i] += 1
                return (hs,) + out[1:] if isinstance(out, tuple) else hs
            handles.append(self.model.model.layers[L - 1].register_forward_hook(hook))
        try:
            out = self.model(**ids, use_cache=True)
        finally:
            for h in handles:
                h.remove()
        assert counts == [1] * len(edits)
        generated, results, mask = [[] for _ in edits], [None] * len(edits), ids["attention_mask"]
        for step in range(8):
            tokens = out.logits[:, -1].argmax(-1).reshape(-1, 1)
            for i, token in enumerate(tokens[:, 0].tolist()):
                if results[i] is not None:
                    continue
                generated[i].append(token)
                answer = self.tok.decode(generated[i], skip_special_tokens=True)
                if None not in parse(answer) or token == self.tok.eos_token_id or step == 7:
                    results[i] = {"text": answer, "prediction": parse(answer), "tokens": generated[i]}
            if all(r is not None for r in results):
                break
            mask = torch.cat([mask, torch.ones_like(tokens)], dim=1)
            out = self.model(input_ids=tokens, attention_mask=mask,
                             past_key_values=out.past_key_values, use_cache=True)
        return results


def write(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2) + "\n")


def score(result, target):
    a, b = result["prediction"]
    return {"day_ok": a == target[0], "month_ok": b == target[1],
            "joint_ok": (a, b) == tuple(target)}


def summarize(rows, keys):
    groups = {}
    for row in rows:
        key = "/".join(str(row[k]) for k in keys)
        groups.setdefault(key, []).append(row)
    return {key: {"n": len(rs), **{k: sum(r[k] for r in rs) / len(rs)
                                  for k in ["day_ok", "month_ok", "joint_ok"]}}
            for key, rs in groups.items()}


def donor(runner):
    rows = []
    sources = [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 1), (0, 5)]
    for order in ["dm", "md"]:
        for d, m in sources:
            text = prompt(d, m, order=order)
            base, state = runner.natural(text)
            rows.append(dict(order=order, layer=0, edit="baseline", source=[d, m],
                             target=[d, m], **base, **score(base, (d, m))))
            cases = []
            for edit, target in [("day", ((d + 3) % 7, m)),
                                 ("month", (d, (m + 5) % 12)),
                                 ("both", ((d + 3) % 7, (m + 5) % 12))]:
                natural, target_states = runner.natural(prompt(*target, order=order))
                for L in LAYERS:
                    cases.append((L, target_states[L], edit, target, natural))
            results = runner.batch_run(text, [(L, v) for L, v, _, _, _ in cases])
            for (L, v, edit, target, natural), result in zip(cases, results):
                    row = dict(order=order, layer=L, edit=edit, source=[d, m],
                               target=target, donor_prediction=natural["prediction"],
                               delta_norm=float(np.linalg.norm(v - state[L])),
                               **result, **score(result, target))
                    rows.append(row)
            identity, _ = runner.run(text, 28, state[28])
            assert identity["tokens"] == base["tokens"], "Identity patch changed generation"
            write("donor_rows.json", rows)
            print(json.dumps({"completed_source": [d, m], "order": order,
                              "baseline": base["text"], "elapsed": round(time.time() - START)}), flush=True)
    summary = summarize(rows, ["order", "layer", "edit"])
    # Require meaningful transfer of both fields, in both orders, not just the first token.
    eligible = [L for L in LAYERS if all(
        summary[f"{order}/{L}/{edit}"]["joint_ok"] >= 0.5
        for order in ["dm", "md"] for edit in ["day", "month", "both"])]
    gate = {"eligible_layers": eligible, "summary": summary,
            "threshold": "At least 50% joint exact label accuracy for every edit type in both orders"}
    write("donor_summary.json", gate)
    print(json.dumps(gate, indent=2), flush=True)


def fourier(runner):
    gate = json.loads((ROOT / "donor_summary.json").read_text())
    if not gate["eligible_layers"]:
        raise SystemExit("Donor gate failed; do not interpret Fourier fitting as a control test.")
    L = max(gate["eligible_layers"])
    rows, fits = [], []
    train_pairs = [(d, m) for d in range(7) for m in range(12) if not heldout(d, m)]
    test_pairs = [(d, m) for d in range(7) for m in range(12) if heldout(d, m)]
    for order in ["dm", "md"]:
        # Balanced training templates; neither held-out pairs nor template 2 enter fitting.
        Y = np.stack([np.mean([runner.natural(prompt(d, m, t, order))[1][L]
                             for t in [0, 1]], axis=0) for d, m in train_pairs])
        mu = Y.mean(0)
        _, s, vt = np.linalg.svd(Y - mu, full_matrices=False)
        Q = vt[:16].T  # Identical replacement subspace for both fitted models.
        Z = (Y - mu) @ Q
        predictors = {}
        for method in ["additive", "joint"]:
            X = features(train_pairs, method == "joint")
            penalty = np.eye(X.shape[1]) * 0.01 * len(X)
            penalty[0, 0] = 0
            coef = np.linalg.solve(X.T @ X + penalty, X.T @ Z)
            predictors[method] = lambda pair, coef=coef, method=method: (
                mu + features([pair], method == "joint")[0] @ coef @ Q.T)
            fits.append(dict(order=order, method=method, n_coefficients=len(coef),
                             training_mse=float(np.mean((X @ coef - Z) ** 2)),
                             pca_variance=float((s[:16] ** 2).sum() / (s ** 2).sum())))
        for d, m in test_pairs:
            for edit, source in [("day", ((d + 1) % 7, m)),
                                 ("month", (d, (m + 1) % 12)),
                                 ("both", ((d + 1) % 7, (m + 1) % 12))]:
                text = prompt(*source, 2, order)
                base, states = runner.natural(text)
                _, target_states = runner.natural(prompt(d, m, 2, order))
                for method in ["baseline", "donor", "additive", "joint"]:
                    x = states[L]
                    if method == "baseline":
                        result, vec = base, x
                    else:
                        vec = target_states[L] if method == "donor" else (
                            x + Q @ (Q.T @ (predictors[method]((d, m)) - x)))
                        result, _ = runner.run(text, L, vec)
                    rows.append(dict(order=order, layer=L, edit=edit, method=method,
                                     source=source, target=[d, m],
                                     delta_norm=float(np.linalg.norm(vec - x)),
                                     **result, **score(result, (d, m))))
            write("fourier_rows.json", rows)
            print(f"Fourier {order} target {d},{m}: {len(rows)} rows", flush=True)
    write("fourier_summary.json", {"fits": fits, "summary": summarize(rows, ["order", "method", "edit"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["donor", "fourier"], default="donor")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "cache").mkdir(exist_ok=True)
    START = time.time()
    write("config.json", {"model": MODEL, "layers": LAYERS, "torch": torch.__version__,
                          "seed": 0, "generation": "greedy; max 8 tokens; stop once both fields parsed",
                          "fitting_pairs": 63, "heldout_pairs": 21,
                          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    runner = Runner()
    print("Model loaded", flush=True)
    (donor if args.stage == "donor" else fourier)(runner)
