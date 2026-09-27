"""
Composability, done properly: both handles are now potent in a Fourier basis
  weekday: K=3, amp 2  (4/4 alone)
  month:   K=5, amp 6  (6/6 alone)
Test non-interference in BOTH directions. Crosstalk is only meaningful because
each perturbing edit is independently verified to work.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from fourier_steer_real import (MODEL_NAME, LAYER, DAYS, MONTHS, day_prompts, month_prompts,
                                last_hidden, fourier_fit, steer, tid, run, log)

DAY_K, DAY_AMP = 3, 2.0
MON_K, MON_AMP = 5, 6.0


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()

    Cd = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in day_prompts(d)], axis=0) for d in DAYS])
    Cm = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in month_prompts(m)], axis=0) for m in MONTHS])
    dmean, dA, dB = fourier_fit(Cd)
    mmean, mA, mB = fourier_fit(Cm)
    d_tid = {d: tid(tok, d) for d in DAYS}
    m_tid = {m: tid(tok, m) for m in MONTHS}
    log(f"loaded+fit ({time.time()-t0:.0f}s)")

    def state(prompt):
        ids = tok(prompt, return_tensors="pt")
        pos = ids["input_ids"].shape[1] - 1
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)
        return ids, pos, o.hidden_states[LAYER][0, pos, :].float().numpy()

    dids, dpos, dx = state("Today is Wednesday. Tomorrow is")
    mids, mpos, mx = state("The month after February is")

    # ---- direction 1: weekday steering, month coordinate perturbed ----
    log("\n=== 1. WEEKDAY steering (K=3,amp=2), MONTH coordinate perturbed (K=5,amp=6) ===")
    mperts = [None, "July", "September", "August"]
    dtargets = ["Monday", "Saturday", "Sunday", "Tuesday"]
    log(f"  {'day target':>11} | " + " | ".join(f"{str(m)[:9]:>10}" for m in mperts))
    arr, h_alone, h_pert = [], 0, 0
    for dt in dtargets:
        row = []
        for j, mp in enumerate(mperts):
            v = steer(dx, dmean, dA, dB, 7, DAYS.index(dt), DAY_K, DAY_AMP)
            if mp:
                v = steer(v, mmean, mA, mB, 12, MONTHS.index(mp), MON_K, MON_AMP)
            p = run(model, dids, dpos, LAYER, v)
            ok = tok.decode([int(torch.argmax(p))]).strip() == dt
            row.append((p[d_tid[dt]].item(), ok))
            h_alone += ok if j == 0 else 0
            h_pert += ok if j > 0 else 0
        arr.append([v for v, _ in row])
        log(f"  {dt:>11} | " + " | ".join(f"{v:.3f}{'*' if ok else ' '}".rjust(10) for v, ok in row))
    a = np.array(arr)
    log(f"  weekday top-1: alone {h_alone}/4, under month perturbation {h_pert}/12")
    log(f"  mean P(day target) drop from month perturbation = {(a[:,0][:,None]-a[:,1:]).mean():+.4f}")

    # ---- direction 2: month steering, weekday coordinate perturbed ----
    log("\n=== 2. MONTH steering (K=5,amp=6), WEEKDAY coordinate perturbed (K=3,amp=2) ===")
    dperts = [None, "Monday", "Saturday", "Tuesday"]
    mtargets = ["July", "September", "August", "November"]
    log(f"  {'month target':>13} | " + " | ".join(f"{str(d)[:9]:>10}" for d in dperts))
    arr2, h2a, h2p = [], 0, 0
    for mt in mtargets:
        row = []
        for j, dp in enumerate(dperts):
            v = steer(mx, mmean, mA, mB, 12, MONTHS.index(mt), MON_K, MON_AMP)
            if dp:
                v = steer(v, dmean, dA, dB, 7, DAYS.index(dp), DAY_K, DAY_AMP)
            p = run(model, mids, mpos, LAYER, v)
            ok = tok.decode([int(torch.argmax(p))]).strip() == mt
            row.append((p[m_tid[mt]].item(), ok))
            h2a += ok if j == 0 else 0
            h2p += ok if j > 0 else 0
        arr2.append([v for v, _ in row])
        log(f"  {mt:>13} | " + " | ".join(f"{v:.3f}{'*' if ok else ' '}".rjust(10) for v, ok in row))
    a2 = np.array(arr2)
    log(f"  month top-1: alone {h2a}/4, under weekday perturbation {h2p}/12")
    log(f"  mean P(month target) drop from weekday perturbation = {(a2[:,0][:,None]-a2[:,1:]).mean():+.4f}")

    log(f"\n  relative degradation: weekday {((a[:,0][:,None]-a[:,1:]).mean()/a[:,0].mean()):.1%}, "
        f"month {((a2[:,0][:,None]-a2[:,1:]).mean()/a2[:,0].mean()):.1%}")
    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
