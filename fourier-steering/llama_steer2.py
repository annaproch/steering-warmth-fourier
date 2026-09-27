"""
Corrected experiment matching Goodfire's protocol:
  centroid for day d = last-token activation of prompts whose ANSWER is d
  ("the model is about to say d"), then rotate that state's phase and see
  whether the predicted answer follows around the circle.
Includes a hook sanity check + perturbation magnitude, so a null result
can be distinguished from a broken intervention.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def log(*a):
    print(*a, flush=True)


def prev(d, k=1):
    return DAYS[(DAYS.index(d) - k) % 7]


def nxt(d, k=1):
    return DAYS[(DAYS.index(d) + k) % 7]


def prompts_for(d):
    """Prompts whose correct next word is day d."""
    return [
        f"Today is {prev(d)}. Tomorrow is",
        f"Tomorrow is {nxt(d)}. Today is",
        f"The day after {prev(d)} is",
        f"The day before {nxt(d)} is",
        f"{prev(d,2)}, {prev(d)},",
        f"Yesterday was {prev(d)}, so today must be",
    ]


def last_token_hidden(model, tok, text, layers):
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    return {L: out.hidden_states[L][0, -1, :].float().numpy() for L in layers}


def fit_circle(centroids):
    mean = centroids.mean(axis=0)
    Xc = centroids - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    proj = Xc @ np.stack([e1, e2], axis=1)
    return dict(mean=mean, e1=e1, e2=e2,
                thetas=np.arctan2(proj[:, 1], proj[:, 0]),
                radii=np.linalg.norm(proj, axis=1),
                var2=(S[:2] ** 2).sum() / (S ** 2).sum())


def rotate_to(x, circle, theta, amp_scale=1.0):
    Xc = x - circle["mean"]
    a, b = Xc @ circle["e1"], Xc @ circle["e2"]
    r = np.hypot(a, b) * amp_scale
    return x + (r * np.cos(theta) - a) * circle["e1"] + (r * np.sin(theta) - b) * circle["e2"]


def first_tid(tok, w):
    return tok.encode(" " + w, add_special_tokens=False)[0]


def run_patched(model, ids, pos, layer, vec):
    block = model.model.layers[layer - 1]

    def hook(module, inp, out):
        is_t = isinstance(out, tuple)
        hs = (out[0] if is_t else out).clone()
        hs[0, pos, :] = torch.tensor(vec, dtype=hs.dtype)
        return ((hs,) + out[1:]) if is_t else hs

    h = block.register_forward_hook(hook)
    with torch.no_grad():
        lg = model(**ids).logits[0, -1, :].float()
    h.remove()
    return F.softmax(lg, dim=-1)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    LAYERS = [14, 20, 24, 28, 31]
    day_ids = [first_tid(tok, d) for d in DAYS]
    log(f"loaded ({time.time()-t0:.0f}s)")

    # ---- baseline: can Llama answer these prompts at all? ----
    log("\n=== Baseline accuracy on answer-conditioned prompts ===")
    n_ok = n_tot = 0
    for d in DAYS:
        for p in prompts_for(d):
            ids = tok(p, return_tensors="pt")
            with torch.no_grad():
                probs = F.softmax(model(**ids).logits[0, -1, :].float(), dim=-1)
            pred = tok.decode([int(torch.argmax(probs))]).strip()
            n_ok += pred == d
            n_tot += 1
    log(f"  baseline top-1 correct: {n_ok}/{n_tot} ({n_ok/n_tot:.0%})")

    # ---- centroids: 'about to say day d' states (hold out last prompt) ----
    log("\nbuilding answer-conditioned centroids...")
    train = {L: [] for L in LAYERS}
    test = {L: [] for L in LAYERS}
    for d in DAYS:
        ps = prompts_for(d)
        hs = [last_token_hidden(model, tok, p, LAYERS) for p in ps]
        for L in LAYERS:
            train[L].append(np.mean([h[L] for h in hs[:-1]], axis=0))
            test[L].append(hs[-1][L])
        log(f"  ...{d}")
    train = {L: np.stack(v) for L, v in train.items()}
    test = {L: np.stack(v) for L, v in test.items()}

    # ---- geometry + held-out decoding of these states ----
    log("\n=== Geometry of 'about to say day d' states ===")
    for L in LAYERS:
        c = train[L]
        circ = fit_circle(c)
        n = 7
        X = c - c.mean(axis=0, keepdims=True)
        pw = (np.abs(np.fft.fft(X, axis=0)) ** 2).sum(axis=1)
        fund = (pw[1] + pw[6]) / pw[1:].sum()
        acc = np.mean([int(np.argmin(np.linalg.norm(c - test[L][i], axis=1)) == i) for i in range(n)])
        log(f"  layer {L:>2}: 2D var={circ['var2']:.3f}  fundamental={fund:.3f}  held-out decode={acc:.0%}")

    # ---- steering ----
    test_prompt = "Today is Wednesday. Tomorrow is"   # answer: Thursday
    ids = tok(test_prompt, return_tensors="pt")
    pos = ids["input_ids"].shape[1] - 1               # LAST token, matching centroid population
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    base = F.softmax(out.logits[0, -1, :].float(), dim=-1)
    acts = {L: out.hidden_states[L][0, pos, :].float().numpy() for L in LAYERS}
    log(f"\n=== Steering {test_prompt!r} (patching LAST token, pos {pos}) ===")
    log(f"baseline: {tok.decode([int(torch.argmax(base))])!r} P={base.max().item():.3f}")

    # hook sanity check: zeroing the whole state must change the output
    zero_probs = run_patched(model, ids, pos, 28, np.zeros_like(acts[28]))
    log(f"SANITY (inject zeros @28): top={tok.decode([int(torch.argmax(zero_probs))])!r} "
        f"P(Thursday)={zero_probs[first_tid(tok,'Thursday')].item():.3f}  "
        f"[must differ from baseline 0.49 if hook works]")

    for L in LAYERS:
        circ = fit_circle(train[L])
        x = acts[L]
        pert = np.linalg.norm(rotate_to(x, circ, circ["thetas"][0]) - x) / np.linalg.norm(x)
        n_ok = 0
        log(f"\n--- layer {L} (2D var={circ['var2']:.3f}, |Δ|/|x|={pert:.3f}) ---")
        log(f"{'target':>10} {'top pred':>12} {'P(top)':>7} {'P(target)':>10}  " +
            " ".join(f"{d[:3]:>5}" for d in DAYS))
        for i, (target, th) in enumerate(zip(DAYS, circ["thetas"])):
            vec = rotate_to(x, circ, th)
            p = run_patched(model, ids, pos, L, vec)
            top = int(torch.argmax(p))
            tt = tok.decode([top]).strip()
            ok = tt == target
            n_ok += ok
            log(f"{target:>10} {tt:>12} {p[top].item():>7.3f} {p[day_ids[i]].item():>10.3f}  " +
                " ".join(f"{p[t].item():.3f}" for t in day_ids) + ("  <-- steered!" if ok else ""))
        log(f"  -> {n_ok}/7 targets hit")

    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
