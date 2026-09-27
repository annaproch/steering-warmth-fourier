"""
Corrected composability test. Finding A showed edits need enough amplitude to
install at all, so:
  1. calibrate: find the radius at which each edit works ALONE on its own prompt
  2. then test non-interference in BOTH directions at that radius:
       weekday prompt: steer weekday, perturb month -> does weekday survive?
       month   prompt: steer month,   perturb weekday -> does month survive?
A non-interference result only counts if the perturbing edit is itself potent.
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


def shift(lst, x, k):
    return lst[(lst.index(x) + k) % len(lst)]


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


def last_hidden(model, tok, text, layer):
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        return model(**ids, output_hidden_states=True).hidden_states[layer][0, -1, :].float().numpy()


def fit_circle(c):
    mean = c.mean(axis=0)
    Xc = c - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    proj = Xc @ np.stack([e1, e2], axis=1)
    return dict(mean=mean, e1=e1, e2=e2, thetas=np.arctan2(proj[:, 1], proj[:, 0]),
                var2=(S[:2] ** 2).sum() / (S ** 2).sum())


def set_polar(x, circle, theta, amp):
    Xc = x - circle["mean"]
    a, b = Xc @ circle["e1"], Xc @ circle["e2"]
    r = np.hypot(a, b) * amp
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
    d_tid = {d: first_tid(tok, d) for d in DAYS}
    m_tid = {m: first_tid(tok, m) for m in MONTHS}

    day_c = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in day_prompts(d)], axis=0) for d in DAYS])
    month_c = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in month_prompts(m)], axis=0) for m in MONTHS])
    dc, mc = fit_circle(day_c), fit_circle(month_c)
    log(f"loaded+fit ({time.time()-t0:.0f}s) day var={dc['var2']:.3f} month var={mc['var2']:.3f}")

    dprompt, mprompt = "Today is Wednesday. Tomorrow is", "The month after February is"
    dids, mids = tok(dprompt, return_tensors="pt"), tok(mprompt, return_tensors="pt")
    dpos, mpos = dids["input_ids"].shape[1] - 1, mids["input_ids"].shape[1] - 1
    with torch.no_grad():
        dx = model(**dids, output_hidden_states=True).hidden_states[LAYER][0, dpos, :].float().numpy()
        mx = model(**mids, output_hidden_states=True).hidden_states[LAYER][0, mpos, :].float().numpy()

    # ---------- 1. calibrate month amplitude on its OWN prompt ----------
    log("\n=== 1. Calibration: month steering on a month prompt (find potent radius) ===")
    log(f"{'r':>5} " + " ".join(f"{m[:3]:>7}" for m in ["July", "November", "May"]) + "   (P(target), top token)")
    best_r = None
    for r in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        cells, hits = [], 0
        for mt in ["July", "November", "May"]:
            p = run_patched(model, mids, mpos, LAYER, set_polar(mx, mc, mc["thetas"][MONTHS.index(mt)], r))
            top = tok.decode([int(torch.argmax(p))]).strip()
            hits += top == mt
            cells.append(f"{p[m_tid[mt]].item():.3f}{'*' if top==mt else ' '}")
        log(f"{r:>5.1f} " + " ".join(f"{c:>7}" for c in cells) + f"   hits={hits}/3")
        if hits >= 2 and best_r is None:
            best_r = r
    log(f"  -> chosen month radius: {best_r} (* = target became top-1)")
    if best_r is None:
        best_r = 6.0
        log(f"  (no radius gave >=2/3; falling back to {best_r})")

    DAY_R = 4.0  # from finding A: potent for weekday
    log(f"  weekday radius fixed at {DAY_R} (potent per amplitude sweep)")

    # ---------- 2. non-interference: weekday steering under month perturbation ----------
    log("\n=== 2. Weekday steering, month coordinate perturbed (weekday prompt) ===")
    month_pert = [None, "July", "November", "May"]
    day_targets = ["Monday", "Saturday", "Sunday", "Tuesday"]
    tab = {}
    log(f"  {'day target':>11} | " + " | ".join(f"{str(m):>10}" for m in month_pert))
    for dt in day_targets:
        row = []
        for mp in month_pert:
            v = set_polar(dx, dc, dc["thetas"][DAYS.index(dt)], DAY_R)
            if mp:
                v = set_polar(v, mc, mc["thetas"][MONTHS.index(mp)], best_r)
            p = run_patched(model, dids, dpos, LAYER, v)
            top = tok.decode([int(torch.argmax(p))]).strip()
            row.append((p[d_tid[dt]].item(), top == dt))
        tab[dt] = row
        log(f"  {dt:>11} | " + " | ".join(f"{v:.3f}{'*' if ok else ' '}".rjust(10) for v, ok in row))
    arr = np.array([[v for v, _ in tab[d]] for d in day_targets])
    sig = arr[:, 0].max() - arr[:, 0].min()
    xt = np.abs(arr[:, 1:] - arr[:, [0]]).max()
    hits_alone = sum(tab[d][0][1] for d in day_targets)
    hits_pert = sum(ok for d in day_targets for _, ok in tab[d][1:])
    log(f"  weekday top-1 hits: alone {hits_alone}/{len(day_targets)}, "
        f"under month perturbation {hits_pert}/{len(day_targets)*3}")
    log(f"  SIGNAL={sig:.4f}  CROSSTALK={xt:.4f}  crosstalk/signal={xt/sig:.2%}")

    # ---------- 3. reverse direction: month steering under weekday perturbation ----------
    log("\n=== 3. Month steering, weekday coordinate perturbed (month prompt) ===")
    day_pert = [None, "Monday", "Saturday"]
    month_targets = ["July", "November", "May"]
    tab2 = {}
    log(f"  {'month target':>13} | " + " | ".join(f"{str(d):>10}" for d in day_pert))
    for mt in month_targets:
        row = []
        for dp in day_pert:
            v = set_polar(mx, mc, mc["thetas"][MONTHS.index(mt)], best_r)
            if dp:
                v = set_polar(v, dc, dc["thetas"][DAYS.index(dp)], DAY_R)
            p = run_patched(model, mids, mpos, LAYER, v)
            top = tok.decode([int(torch.argmax(p))]).strip()
            row.append((p[m_tid[mt]].item(), top == mt))
        tab2[mt] = row
        log(f"  {mt:>13} | " + " | ".join(f"{v:.3f}{'*' if ok else ' '}".rjust(10) for v, ok in row))
    arr2 = np.array([[v for v, _ in tab2[m]] for m in month_targets])
    sig2 = arr2[:, 0].max() - arr2[:, 0].min()
    xt2 = np.abs(arr2[:, 1:] - arr2[:, [0]]).max()
    h2a = sum(tab2[m][0][1] for m in month_targets)
    h2p = sum(ok for m in month_targets for _, ok in tab2[m][1:])
    log(f"  month top-1 hits: alone {h2a}/{len(month_targets)}, "
        f"under weekday perturbation {h2p}/{len(month_targets)*2}")
    log(f"  SIGNAL={sig2:.4f}  CROSSTALK={xt2:.4f}  crosstalk/signal={xt2/max(sig2,1e-9):.2%}")

    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
