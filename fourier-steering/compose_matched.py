"""
Is the asymmetric interference geometric, or just that one edit is bigger?
  (a) measure ||delta|| of each edit and principal angles between the
      6-dim weekday and 10-dim month Fourier subspaces
  (b) re-test both directions with the perturbation magnitudes MATCHED
If matching ||delta|| makes interference symmetric and small, the asymmetry
was an amplitude artifact, not geometry.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from fourier_steer_real import (MODEL_NAME, LAYER, DAYS, MONTHS, day_prompts, month_prompts,
                                last_hidden, fourier_fit, subspace_basis, steer, tid, run, log)

DAY_K, MON_K = 3, 5


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

    # (a) geometry of the two subspaces
    Qd, Qm = subspace_basis(dA, dB, DAY_K), subspace_basis(mA, mB, MON_K)
    sv = np.linalg.svd(Qd.T @ Qm, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(sv, -1, 1)))
    log(f"loaded+fit ({time.time()-t0:.0f}s)")
    log(f"\nweekday subspace dim {Qd.shape[1]}, month subspace dim {Qm.shape[1]}")
    log(f"principal angles: " + ", ".join(f"{a:.1f}" for a in ang) + " deg  (90 = orthogonal)")

    def state(p):
        ids = tok(p, return_tensors="pt")
        pos = ids["input_ids"].shape[1] - 1
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)
        return ids, pos, o.hidden_states[LAYER][0, pos, :].float().numpy()

    dids, dpos, dx = state("Today is Wednesday. Tomorrow is")
    mids, mpos, mx = state("The month after February is")

    # (b) measure ||delta|| at the amplitudes used before
    d_edit = steer(dx, dmean, dA, dB, 7, DAYS.index("Monday"), DAY_K, 2.0)
    m_edit = steer(mx, mmean, mA, mB, 12, MONTHS.index("September"), MON_K, 6.0)
    nd = np.linalg.norm(d_edit - dx)
    nm = np.linalg.norm(m_edit - mx)
    log(f"\n||delta|| weekday(K=3,amp=2) = {nd:.2f}   ||delta|| month(K=5,amp=6) = {nm:.2f}"
        f"   ratio = {nm/nd:.2f}x")
    log(f"||x|| weekday state = {np.linalg.norm(dx):.1f}, month state = {np.linalg.norm(mx):.1f}")

    def scaled_pert(x, mean, A, B, n, idx, K, amp, target_norm):
        """Apply the edit, then rescale the DELTA to a target norm."""
        v = steer(x, mean, A, B, n, idx, K, amp)
        d = v - x
        return x + d * (target_norm / np.linalg.norm(d))

    # matched magnitude = the smaller of the two (the weekday edit)
    MATCH = nd
    log(f"\nmatching both perturbation magnitudes to ||delta||={MATCH:.2f}")

    log("\n=== 1. WEEKDAY steering, MONTH perturbation (magnitude-matched) ===")
    log(f"  {'day target':>11} | " + " | ".join(f"{m:>10}" for m in ['None', 'July', 'September', 'August']))
    a = []
    h0 = hp = 0
    for dt in ["Monday", "Saturday", "Sunday", "Tuesday"]:
        row = []
        for j, mp in enumerate([None, "July", "September", "August"]):
            v = steer(dx, dmean, dA, dB, 7, DAYS.index(dt), DAY_K, 2.0)
            if mp:
                v = scaled_pert(v, mmean, mA, mB, 12, MONTHS.index(mp), MON_K, 6.0, MATCH)
            p = run(model, dids, dpos, LAYER, v)
            ok = tok.decode([int(torch.argmax(p))]).strip() == dt
            row.append((p[d_tid[dt]].item(), ok))
            h0 += ok if j == 0 else 0
            hp += ok if j > 0 else 0
        a.append([v for v, _ in row])
        log(f"  {dt:>11} | " + " | ".join(f"{v:.3f}{'*' if ok else ' '}".rjust(10) for v, ok in row))
    a = np.array(a)
    drop_d = (a[:, 0][:, None] - a[:, 1:]).mean()
    log(f"  weekday top-1: alone {h0}/4, perturbed {hp}/12;  mean drop {drop_d:+.4f} "
        f"({drop_d/a[:,0].mean():.1%} relative)")

    log("\n=== 2. MONTH steering, WEEKDAY perturbation (magnitude-matched) ===")
    log(f"  {'month target':>13} | " + " | ".join(f"{d:>10}" for d in ['None', 'Monday', 'Saturday', 'Tuesday']))
    a2 = []
    g0 = gp = 0
    for mt in ["July", "September", "August", "November"]:
        row = []
        for j, dp in enumerate([None, "Monday", "Saturday", "Tuesday"]):
            v = steer(mx, mmean, mA, mB, 12, MONTHS.index(mt), MON_K, 6.0)
            if dp:
                v = scaled_pert(v, dmean, dA, dB, 7, DAYS.index(dp), DAY_K, 2.0, MATCH)
            p = run(model, mids, mpos, LAYER, v)
            ok = tok.decode([int(torch.argmax(p))]).strip() == mt
            row.append((p[m_tid[mt]].item(), ok))
            g0 += ok if j == 0 else 0
            gp += ok if j > 0 else 0
        a2.append([v for v, _ in row])
        log(f"  {mt:>13} | " + " | ".join(f"{v:.3f}{'*' if ok else ' '}".rjust(10) for v, ok in row))
    a2 = np.array(a2)
    drop_m = (a2[:, 0][:, None] - a2[:, 1:]).mean()
    log(f"  month top-1: alone {g0}/4, perturbed {gp}/12;  mean drop {drop_m:+.4f} "
        f"({drop_m/a2[:,0].mean():.1%} relative)")

    log(f"\nSYMMETRY CHECK  weekday degradation {drop_d/a[:,0].mean():.1%} vs "
        f"month degradation {drop_m/a2[:,0].mean():.1%}")
    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
