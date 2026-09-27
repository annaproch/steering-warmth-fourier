"""
Control experiment: is it HARMONICS, or just MORE DIMENSIONS?

K=3 Fourier is a 6-dim subspace; K=1 is 2-dim. If a 6-dim subspace from plain
PCA of the centroids steers just as well, the "Fourier/harmonics" framing is
decoration on "use more dimensions". Compare, at matched subspace dimension:
    fourier-K   : span{A_k, B_k}, k=1..K            (2K dims)
    pca-d       : top-d principal components of centroids
    random-d    : d random directions (floor)
Also measure distance from the fitted manifold, to test the claim that high
amplitude + low K lands off-manifold.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from fourier_steer_real import (MODEL_NAME, LAYER, DAYS, MONTHS, day_prompts, month_prompts,
                                last_hidden, fourier_fit, fourier_target, subspace_basis,
                                tid, run, log)

RNG = np.random.default_rng(0)


def pca_basis(C, d):
    Xc = C - C.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:d].T  # hidden x d, orthonormal


def random_basis(hidden, d):
    Q, _ = np.linalg.qr(RNG.normal(size=(hidden, d)))
    return Q


def steer_in_basis(x, Q, mean, target, amp):
    """Replace x's component in span(Q) with the target's, scaled by amp."""
    xc, tc = x - mean, target - mean
    return x - Q @ (Q.T @ xc) + amp * (Q @ (Q.T @ tc))


def dist_from_manifold(v, C):
    """Distance to nearest centroid, normalised by mean inter-centroid distance."""
    d = np.linalg.norm(C - v[None, :], axis=1).min()
    n = C.shape[0]
    pair = [np.linalg.norm(C[i] - C[j]) for i in range(n) for j in range(i + 1, n)]
    return d / np.mean(pair)


def evaluate(model, tok, ids, pos, x, C, mean, A, B, n, words, targets, w_tid, bases, amps):
    """bases: list of (label, Q, target_fn)."""
    for label, Q, tgt_fn in bases:
        log(f"\n  {label} (dim {Q.shape[1]})")
        log(f"  {'amp':>5} | " + " | ".join(f"{t[:4]:>8}" for t in targets) + " | hits | off-manifold")
        for amp in amps:
            cells, hits, dists = [], 0, []
            for tg in targets:
                target_vec = tgt_fn(words.index(tg))
                v = steer_in_basis(x, Q, mean, target_vec, amp)
                p = run(model, ids, pos, LAYER, v)
                top = tok.decode([int(torch.argmax(p))]).strip()
                ok = top == tg
                hits += ok
                cells.append(f"{p[w_tid[tg]].item():.3f}{'*' if ok else ' '}")
                dists.append(dist_from_manifold(v, C))
            log(f"  {amp:>5.1f} | " + " | ".join(f"{c:>8}" for c in cells) +
                f" | {hits}/{len(targets)} | {np.mean(dists):.2f}")


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    log(f"loaded ({time.time()-t0:.0f}s)")

    for name, words, pfn, test_prompt, targets, amps, Ks in [
        ("WEEKDAYS", DAYS, day_prompts, "Today is Wednesday. Tomorrow is",
         ["Monday", "Saturday", "Sunday", "Tuesday"], [1.0, 2.0, 4.0], [1, 3]),
        ("MONTHS", MONTHS, month_prompts, "The month after February is",
         ["July", "November", "September", "August"], [2.0, 6.0, 8.0], [1, 5]),
    ]:
        n = len(words)
        C = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in pfn(w)], axis=0) for w in words])
        mean, A, B = fourier_fit(C)
        w_tid = {w: tid(tok, w) for w in words}
        ids = tok(test_prompt, return_tensors="pt")
        pos = ids["input_ids"].shape[1] - 1
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)
        x = o.hidden_states[LAYER][0, pos, :].float().numpy()
        log(f"\n{'='*84}\n{name}: {test_prompt!r}  baseline="
            f"{tok.decode([int(torch.argmax(F.softmax(o.logits[0,-1,:].float(),dim=-1)))])!r}")
        log(f"  (off-manifold = distance to nearest centroid / mean inter-centroid distance)")

        hidden = x.shape[0]
        bases = []
        for K in Ks:
            Q = subspace_basis(A, B, K)
            bases.append((f"fourier K={K}", Q,
                          lambda i, K=K: fourier_target(mean, A, B, n, i, K)))
            d = 2 * K
            bases.append((f"pca d={d}", pca_basis(C, d), lambda i: C[i]))
            bases.append((f"random d={d}", random_basis(hidden, d), lambda i: C[i]))
        evaluate(model, tok, ids, pos, x, C, mean, A, B, n, words, targets, w_tid, bases, amps)

    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
