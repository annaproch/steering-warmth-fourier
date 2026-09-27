"""
Full pipeline on Llama 3.1 8B (the model Goodfire used), layer 28 (their layer):
  1. Fourier power spectrum of weekday/month activation centroids
  2. Low-pass reconstruction fidelity + held-out decoding
  3. Phase steering on a non-list arithmetic prompt, swept across layers
Captures all layers' hidden states in one forward pass per prompt to stay cheap.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"
PAPER_LAYER = 28
CACHE = "/home/agent/llama_centroids.npz"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

DAY_TEMPLATES = [
    "Today is {}.", "The meeting is scheduled for {}.", "I will see you on {}.",
    "Yesterday was not {}.",
    "Let's do the days of the week: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, {}",
    "The day after this one is {}.", "She always goes running on {}.", "He was born on a {}.",
]
MONTH_TEMPLATES = [
    "The month is {}.", "My birthday is in {}.", "The event happens every year in {}.",
    "Let's list the months: January, February, March, April, May, June, July, August, September, October, November, {}",
    "It started raining in {}.", "The fiscal year ends in {}.",
]


def log(*a):
    print(*a, flush=True)


def all_layer_last_token(model, tok, text):
    """Return [n_layers+1, hidden] array: last-token hidden state at every layer."""
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    return np.stack([h[0, -1, :].float().numpy() for h in out.hidden_states])


def build_centroids(model, tok, words, templates):
    """[n_layers+1, n_words, hidden] centroids, plus held-out (last template) vectors."""
    per_word_train, per_word_test = [], []
    for w in words:
        stack = [all_layer_last_token(model, tok, t.format(w)) for t in templates[:-1]]
        per_word_train.append(np.mean(stack, axis=0))
        per_word_test.append(all_layer_last_token(model, tok, templates[-1].format(w)))
        log(f"    ...{w}")
    train = np.stack(per_word_train, axis=1)  # [L, n_words, hidden]
    test = np.stack(per_word_test, axis=1)
    return train, test


def power_spectrum(centroids):
    n = centroids.shape[0]
    X = centroids - centroids.mean(axis=0, keepdims=True)
    F_ = np.fft.fft(X, axis=0)
    power = (np.abs(F_) ** 2).sum(axis=1)
    return power / power[1:].sum()


def lowpass(centroids, k_max):
    n = centroids.shape[0]
    mean = centroids.mean(axis=0, keepdims=True)
    F_ = np.fft.fft(centroids - mean, axis=0)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    for k in range(1, k_max + 1):
        mask[k] = True
        mask[(-k) % n] = True
    return np.fft.ifft(F_ * mask[:, None], axis=0).real + mean


def fit_circle(centroids):
    mean = centroids.mean(axis=0)
    Xc = centroids - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    proj = Xc @ np.stack([e1, e2], axis=1)
    return dict(mean=mean, e1=e1, e2=e2,
                thetas=np.arctan2(proj[:, 1], proj[:, 0]),
                var2=(S[:2] ** 2).sum() / (S ** 2).sum())


def patch_additive(x, circle, target_theta):
    Xc = x - circle["mean"]
    a, b = Xc @ circle["e1"], Xc @ circle["e2"]
    r = np.hypot(a, b)
    return x + (r * np.cos(target_theta) - a) * circle["e1"] + (r * np.sin(target_theta) - b) * circle["e2"]


def first_token_id(tok, word):
    return tok.encode(" " + word, add_special_tokens=False)[0]


def find_token_index(tok, text, word):
    enc = tok(text, return_offsets_mapping=True)
    start = text.index(word)
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s <= start < e:
            return i
    raise ValueError("not found")


def inject_at(model, ids, pos_idx, layer, vec):
    block = model.model.layers[layer - 1]

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        hs = (out[0] if is_tuple else out).clone()
        hs[0, pos_idx, :] = torch.tensor(vec, dtype=hs.dtype)
        return (hs,) + out[1:] if is_tuple else hs

    h = block.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1, :].float()
    h.remove()
    return F.softmax(logits, dim=-1)


def main():
    t0 = time.time()
    log("loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    log(f"loaded in {time.time()-t0:.0f}s; n_layers={n_layers}")

    t = time.time()
    _ = all_layer_last_token(model, tok, "Today is Monday.")
    log(f"single forward pass: {time.time()-t:.1f}s")

    # day-name tokenization check
    for d in DAYS[:3]:
        ids_ = tok.encode(" " + d, add_special_tokens=False)
        log(f"  ' {d}' -> {len(ids_)} token(s): {[tok.decode([i]) for i in ids_]}")

    log("\nbuilding day centroids (all layers)...")
    day_train, day_test = build_centroids(model, tok, DAYS, DAY_TEMPLATES)
    log("building month centroids (all layers)...")
    month_train, month_test = build_centroids(model, tok, MONTHS, MONTH_TEMPLATES)
    np.savez_compressed(CACHE, day_train=day_train, day_test=day_test,
                        month_train=month_train, month_test=month_test)
    log(f"cached centroids -> {CACHE}")

    # ---------- 1. power spectrum at the paper's layer ----------
    for name, train in [("Days", day_train), ("Months", month_train)]:
        c = train[PAPER_LAYER]
        n = c.shape[0]
        frac = power_spectrum(c)
        fund = frac[1] + frac[n - 1]
        log(f"\n=== {name} @ layer {PAPER_LAYER}: Fourier power ===")
        log("  " + "  ".join(f"k={k}:{frac[k]:.3f}" for k in range(n)))
        log(f"  fundamental (k=1 + k={n-1}) = {fund:.4f}")
        log(f"  2D-subspace var explained = {fit_circle(c)['var2']:.4f}")

    # ---------- 2. low-pass reconstruction + held-out decoding ----------
    for name, train, test in [("Days", day_train, day_test), ("Months", month_train, month_test)]:
        c, tv = train[PAPER_LAYER], test[PAPER_LAYER]
        n = c.shape[0]
        log(f"\n=== {name} @ layer {PAPER_LAYER}: low-pass reconstruction + held-out decoding ===")
        log(f"{'k_max':>6} {'recon R2':>10} {'decode acc':>11}")
        for k_max in range(0, n // 2 + 1):
            rec = lowpass(c, k_max)
            r2 = 1 - ((c - rec) ** 2).sum() / ((c - c.mean(axis=0)) ** 2).sum()
            acc = np.mean([int(np.argmin(np.linalg.norm(rec - tv[i], axis=1)) == i) for i in range(n)])
            log(f"{k_max:>6} {r2:>10.4f} {acc:>10.1%}")

    # ---------- 3. phase steering on a non-list arithmetic prompt ----------
    base_day = "Wednesday"
    prompt = f"Today is {base_day}. Tomorrow is"
    ids = tok(prompt, return_tensors="pt")
    pos_idx = find_token_index(tok, prompt, base_day)
    day_ids = [first_token_id(tok, d) for d in DAYS]

    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    base_probs = F.softmax(out.logits[0, -1, :].float(), dim=-1)
    hs_all = [h[0, pos_idx, :].float().numpy() for h in out.hidden_states]
    top = int(torch.argmax(base_probs))
    log(f"\n=== Phase steering: {prompt!r} (patching token {pos_idx} = {tok.decode([ids['input_ids'][0,pos_idx].item()])!r}) ===")
    log(f"baseline top pred: {tok.decode([top])!r} P={base_probs[top].item():.3f}")

    for layer in [14, 20, 24, PAPER_LAYER, 31]:
        circle = fit_circle(day_train[layer])
        actual = hs_all[layer]
        n_ok = 0
        log(f"\n--- layer {layer} (2D var={circle['var2']:.3f}) ---")
        log(f"{'target':>10} {'want next':>10} {'top pred':>12} {'P(top)':>7} {'P(want)':>8}  " +
            " ".join(f"{d[:3]:>5}" for d in DAYS))
        for i, (target, theta) in enumerate(zip(DAYS, circle["thetas"])):
            vec = patch_additive(actual, circle, theta)
            p = inject_at(model, ids, pos_idx, layer, vec)
            top = int(torch.argmax(p))
            tt = tok.decode([top]).strip()
            want = DAYS[(i + 1) % 7]
            ok = tt == want
            n_ok += ok
            log(f"{target:>10} {want:>10} {tt:>12} {p[top].item():>7.3f} "
                f"{p[first_token_id(tok, want)].item():>8.3f}  " +
                " ".join(f"{p[t].item():.3f}" for t in day_ids) + ("  <-- correct" if ok else ""))
        log(f"  -> {n_ok}/7 successor predictions correct")

    # ---------- 4. amplitude vs phase: does scaling r change confidence, not identity? ----------
    circle28 = fit_circle(day_train[PAPER_LAYER])
    actual28 = hs_all[PAPER_LAYER]
    wed_theta = circle28["thetas"][DAYS.index(base_day)]
    thu_id = first_token_id(tok, "Thursday")
    log(f"\n=== Amplitude sweep @ layer {PAPER_LAYER} (phase held at {base_day}) ===")
    log(f"{'scale':>6} {'top pred':>12} {'P(top)':>8} {'P(Thursday)':>12}")
    for scale in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        Xc = actual28 - circle28["mean"]
        a, b = Xc @ circle28["e1"], Xc @ circle28["e2"]
        r = np.hypot(a, b) * scale
        vec = actual28 + (r * np.cos(wed_theta) - a) * circle28["e1"] + (r * np.sin(wed_theta) - b) * circle28["e2"]
        p = inject_at(model, ids, pos_idx, PAPER_LAYER, vec)
        top = int(torch.argmax(p))
        log(f"{scale:>6.1f} {tok.decode([top]).strip():>12} {p[top].item():>8.3f} {p[thu_id].item():>12.3f}")

    # ---------- 5. composability: weekday plane vs month plane ----------
    dsub = np.stack([circle28["e1"], circle28["e2"]])
    mc = fit_circle(month_train[PAPER_LAYER])
    msub = np.stack([mc["e1"], mc["e2"]])
    svals = np.linalg.svd(dsub @ msub.T, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(svals, -1, 1)))
    log(f"\n=== Composability @ layer {PAPER_LAYER} ===")
    log(f"  principal angles between weekday-plane and month-plane: "
        f"{angles[0]:.1f} deg, {angles[1]:.1f} deg  (90 = independent)")

    log(f"\ntotal time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
