"""Months with the full harmonic range K=1..6 (K=3 only captured 0.72 of variance)."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from fourier_steer_real import (MODEL_NAME, LAYER, MONTHS, month_prompts, last_hidden,
                                fourier_fit, subspace_basis, steer, tid, run, log)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    n = len(MONTHS)

    C = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in month_prompts(m)], axis=0) for m in MONTHS])
    mean, A, B = fourier_fit(C)
    prompt = "The month after February is"
    ids = tok(prompt, return_tensors="pt")
    pos = ids["input_ids"].shape[1] - 1
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    x = out.hidden_states[LAYER][0, pos, :].float().numpy()
    base = F.softmax(out.logits[0, -1, :].float(), dim=-1)
    log(f"loaded+fit ({time.time()-t0:.0f}s); baseline top="
        f"{tok.decode([int(torch.argmax(base))])!r} P={base.max().item():.3f}")

    targets = ["July", "November", "September", "June", "April", "August"]
    m_tid = {m: tid(tok, m) for m in MONTHS}
    Xc = C - mean
    tot = (Xc ** 2).sum()

    for K in [3, 4, 5, 6]:
        Q = subspace_basis(A, B, K)
        cap = ((Xc @ Q) ** 2).sum() / tot
        log(f"\n--- MONTHS K={K} (dim {2*K}, variance captured {cap:.3f}) ---")
        log(f"  {'amp':>5} | " + " | ".join(f"{t[:4]:>9}" for t in targets) + " | hits")
        for amp in [1.0, 2.0, 4.0, 6.0, 8.0]:
            cells, hits = [], 0
            for tgt in targets:
                v = steer(x, mean, A, B, n, MONTHS.index(tgt), K, amp)
                p = run(model, ids, pos, LAYER, v)
                top = tok.decode([int(torch.argmax(p))]).strip()
                ok = top == tgt
                hits += ok
                cells.append(f"{p[m_tid[tgt]].item():.3f}{'*' if ok else ' '}")
            log(f"  {amp:>5.1f} | " + " | ".join(f"{c:>9}" for c in cells) + f" | {hits}/{len(targets)}")

    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
