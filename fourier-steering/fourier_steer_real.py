"""
Actual FOURIER steering (not the k=1 polar special case).

Fit per-concept coefficient vectors from the centroids:
    mean = (1/n) sum_d C[d]
    A_k  = (2/n) sum_d C[d] cos(2 pi k d / n)
    B_k  = (2/n) sum_d C[d] sin(2 pi k d / n)
Target at intrinsic coordinate t, using K harmonics:
    xhat(t) = mean + sum_{k=1..K} [ A_k cos(2 pi k t / n) + B_k sin(2 pi k t / n) ]
Steer by replacing the live activation's component in span{A_k,B_k} with
xhat(t)'s component, keeping the orthogonal complement. K=1 == the old circle.

Question: does K>1 make MONTH steering work, where K=1 completely failed?
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"
LAYER = 28
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def log(*a):
    print(*a, flush=True)


def shift(l, x, k):
    return l[(l.index(x) + k) % len(l)]


def day_prompts(d):
    return [f"Today is {shift(DAYS,d,-1)}. Tomorrow is",
            f"Tomorrow is {shift(DAYS,d,1)}. Today is",
            f"The day after {shift(DAYS,d,-1)} is",
            f"The day before {shift(DAYS,d,1)} is",
            f"{shift(DAYS,d,-2)}, {shift(DAYS,d,-1)},",
            f"Yesterday was {shift(DAYS,d,-1)}, so today must be"]


def month_prompts(m):
    return [f"The month after {shift(MONTHS,m,-1)} is",
            f"The month before {shift(MONTHS,m,1)} is",
            f"{shift(MONTHS,m,-2)}, {shift(MONTHS,m,-1)},",
            f"Last month was {shift(MONTHS,m,-1)}, so this month is",
            f"Next month is {shift(MONTHS,m,1)}, so this month is",
            f"The month right after {shift(MONTHS,m,-1)} on the calendar is"]


def last_hidden(model, tok, t, L):
    ids = tok(t, return_tensors="pt")
    with torch.no_grad():
        return model(**ids, output_hidden_states=True).hidden_states[L][0, -1, :].float().numpy()


def fourier_fit(C):
    """C: [n, hidden] centroids in canonical order. Returns mean, A[k], B[k]."""
    n = C.shape[0]
    d = np.arange(n)
    mean = C.mean(axis=0)
    A, B = {}, {}
    for k in range(1, n // 2 + 1):
        A[k] = (2.0 / n) * (np.cos(2 * np.pi * k * d / n)[:, None] * C).sum(axis=0)
        B[k] = (2.0 / n) * (np.sin(2 * np.pi * k * d / n)[:, None] * C).sum(axis=0)
    return mean, A, B


def fourier_target(mean, A, B, n, t, K):
    x = mean.copy()
    for k in range(1, K + 1):
        x = x + A[k] * np.cos(2 * np.pi * k * t / n) + B[k] * np.sin(2 * np.pi * k * t / n)
    return x


def subspace_basis(A, B, K):
    M = np.stack([v for k in range(1, K + 1) for v in (A[k], B[k])], axis=1)  # hidden x 2K
    Q, _ = np.linalg.qr(M)
    return Q


def steer(x, mean, A, B, n, t, K, amp=1.0):
    """Replace x's component in the K-harmonic subspace with the target's."""
    Q = subspace_basis(A, B, K)
    tgt = fourier_target(mean, A, B, n, t, K)
    xc, tc = x - mean, tgt - mean
    x_in = Q @ (Q.T @ xc)
    t_in = Q @ (Q.T @ tc)
    return x - x_in + amp * t_in


def tid(tok, w):
    return tok.encode(" " + w, add_special_tokens=False)[0]


def run(model, ids, pos, L, vec):
    blk = model.model.layers[L - 1]

    def hook(m, i, out):
        tp = isinstance(out, tuple)
        hs = (out[0] if tp else out).clone()
        hs[0, pos, :] = torch.tensor(vec, dtype=hs.dtype)
        return ((hs,) + out[1:]) if tp else hs

    h = blk.register_forward_hook(hook)
    with torch.no_grad():
        lg = model(**ids).logits[0, -1, :].float()
    h.remove()
    return F.softmax(lg, dim=-1)


def experiment(model, tok, name, words, prompt_fn, test_prompt, targets, amps):
    n = len(words)
    C = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in prompt_fn(w)], axis=0) for w in words])
    mean, A, B = fourier_fit(C)
    ids = tok(test_prompt, return_tensors="pt")
    pos = ids["input_ids"].shape[1] - 1
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    x = out.hidden_states[LAYER][0, pos, :].float().numpy()
    base = F.softmax(out.logits[0, -1, :].float(), dim=-1)
    w_tid = {w: tid(tok, w) for w in words}

    # how much of the between-centroid variance does each K capture?
    log(f"\n{'='*76}\n{name}: test prompt {test_prompt!r}")
    log(f"baseline top={tok.decode([int(torch.argmax(base))])!r} P={base.max().item():.3f}")
    Xc = C - mean
    tot = (Xc ** 2).sum()
    for K in range(1, n // 2 + 1):
        Q = subspace_basis(A, B, K)
        cap = ((Xc @ Q) ** 2).sum() / tot
        log(f"  K={K}: subspace dim={2*K:>2}  centroid variance captured={cap:.3f}")

    for K in [1, 2, 3]:
        if K > n // 2:
            continue
        log(f"\n--- {name}, K={K} harmonics (subspace dim {2*K}) ---")
        log(f"  {'amp':>5} | " + " | ".join(f"{t[:4]:>9}" for t in targets) + " | hits")
        for amp in amps:
            cells, hits = [], 0
            for tgt in targets:
                t_idx = words.index(tgt)
                v = steer(x, mean, A, B, n, t_idx, K, amp)
                p = run(model, ids, pos, LAYER, v)
                top = tok.decode([int(torch.argmax(p))]).strip()
                ok = top == tgt
                hits += ok
                cells.append(f"{p[w_tid[tgt]].item():.3f}{'*' if ok else ' '}")
            log(f"  {amp:>5.1f} | " + " | ".join(f"{c:>9}" for c in cells) + f" | {hits}/{len(targets)}")


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    log(f"loaded ({time.time()-t0:.0f}s)")

    experiment(model, tok, "WEEKDAYS", DAYS, day_prompts,
               "Today is Wednesday. Tomorrow is",
               ["Monday", "Saturday", "Sunday", "Tuesday"],
               [1.0, 2.0, 4.0])

    experiment(model, tok, "MONTHS", MONTHS, month_prompts,
               "The month after February is",
               ["July", "November", "September", "June"],
               [1.0, 2.0, 4.0, 6.0])

    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
