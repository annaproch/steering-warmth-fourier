"""
Two tests the manifold-steering paper does NOT make claims about:

A) AMPLITUDE/PHASE SEPARATION. Is radial distance from the circle centre a
   meaningful control (confidence/intensity) or just off-manifold noise?
   Well-posed: holding phase fixed, sweep radius. Phase should own IDENTITY
   (argmax among the 7 days invariant to radius); radius should own
   CONFIDENCE (P(target)/day-mass and entropy monotone in radius), until it
   breaks down off-manifold (total day-mass collapses).

B) COMPOSABILITY / NON-INTERFERENCE. Weekday and month planes are ~86 deg
   apart geometrically. Causal test: perturb the MONTH coordinate while
   steering the WEEKDAY, and check the weekday outcome is unchanged.
   Crosstalk = spread in P(target day) induced by month target, compared with
   the signal = spread induced by the weekday target itself.
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
    return [
        f"Today is {shift(DAYS,d,-1)}. Tomorrow is",
        f"Tomorrow is {shift(DAYS,d,1)}. Today is",
        f"The day after {shift(DAYS,d,-1)} is",
        f"The day before {shift(DAYS,d,1)} is",
        f"{shift(DAYS,d,-2)}, {shift(DAYS,d,-1)},",
        f"Yesterday was {shift(DAYS,d,-1)}, so today must be",
    ]


def month_prompts(m):
    return [
        f"The month after {shift(MONTHS,m,-1)} is",
        f"The month before {shift(MONTHS,m,1)} is",
        f"{shift(MONTHS,m,-2)}, {shift(MONTHS,m,-1)},",
        f"Last month was {shift(MONTHS,m,-1)}, so this month is",
        f"Next month is {shift(MONTHS,m,1)}, so this month is",
        f"The month right after {shift(MONTHS,m,-1)} on the calendar is",
    ]


def last_hidden(model, tok, text, layer):
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].float().numpy()


def fit_circle(c):
    mean = c.mean(axis=0)
    Xc = c - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    proj = Xc @ np.stack([e1, e2], axis=1)
    return dict(mean=mean, e1=e1, e2=e2,
                thetas=np.arctan2(proj[:, 1], proj[:, 0]),
                var2=(S[:2] ** 2).sum() / (S ** 2).sum())


def set_polar(x, circle, theta, amp_scale):
    """Replace the circle-plane component with (r*amp_scale, theta); keep the rest."""
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


def day_stats(p, day_ids):
    """Return (day_mass, renormalised entropy in bits, argmax-day index)."""
    q = np.array([p[t].item() for t in day_ids])
    mass = q.sum()
    qn = q / mass
    ent = -(qn * np.log2(np.clip(qn, 1e-12, 1))).sum()
    return mass, ent, int(np.argmax(q))


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    day_ids = [first_tid(tok, d) for d in DAYS]
    month_ids = [first_tid(tok, m) for m in MONTHS]
    log(f"loaded ({time.time()-t0:.0f}s)")

    # bases, fit on answer-conditioned states at LAYER
    day_c = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in day_prompts(d)], axis=0) for d in DAYS])
    log("day centroids done")
    month_c = np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in month_prompts(m)], axis=0) for m in MONTHS])
    log("month centroids done")
    dc, mc = fit_circle(day_c), fit_circle(month_c)
    angles = np.degrees(np.arccos(np.clip(
        np.linalg.svd(np.stack([dc["e1"], dc["e2"]]) @ np.stack([mc["e1"], mc["e2"]]).T, compute_uv=False), -1, 1)))
    log(f"day 2D var={dc['var2']:.3f}  month 2D var={mc['var2']:.3f}  "
        f"principal angles={angles[0]:.1f}/{angles[1]:.1f} deg")

    prompt = "Today is Wednesday. Tomorrow is"
    ids = tok(prompt, return_tensors="pt")
    pos = ids["input_ids"].shape[1] - 1
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    x = out.hidden_states[LAYER][0, pos, :].float().numpy()
    base = F.softmax(out.logits[0, -1, :].float(), dim=-1)
    bm, be, bi = day_stats(base, day_ids)
    log(f"\nbaseline: top={tok.decode([int(torch.argmax(base))])!r} "
        f"day-mass={bm:.3f} day-entropy={be:.2f} bits argmax-day={DAYS[bi]}")

    # ---------------- A) amplitude vs phase ----------------
    log("\n" + "=" * 78)
    log("A) AMPLITUDE/PHASE SEPARATION  (does radius control confidence, phase control identity?)")
    log("=" * 78)
    for target in ["Monday", "Saturday", "Sunday"]:
        th = dc["thetas"][DAYS.index(target)]
        log(f"\n  phase fixed at {target}:")
        log(f"  {'r_scale':>8} {'argmax-day':>11} {'P(target)':>10} {'day-mass':>9} {'entropy':>8} {'top token':>14}")
        for s in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
            p = run_patched(model, ids, pos, LAYER, set_polar(x, dc, th, s))
            mass, ent, ai = day_stats(p, day_ids)
            log(f"  {s:>8.1f} {DAYS[ai]:>11} {p[day_ids[DAYS.index(target)]].item():>10.4f} "
                f"{mass:>9.4f} {ent:>8.2f} {tok.decode([int(torch.argmax(p))]).strip()!r:>14}")

    # ---------------- B) composability ----------------
    log("\n" + "=" * 78)
    log("B) COMPOSABILITY  (does perturbing the MONTH coordinate disturb WEEKDAY steering?)")
    log("=" * 78)
    month_targets = [None, "March", "July", "November"]
    day_targets = ["Monday", "Thursday", "Saturday", "Sunday"]
    table = {}
    log(f"\n  {'day target':>11} | " + " | ".join(f"month={str(m):>9}" for m in month_targets))
    for dt in day_targets:
        th = dc["thetas"][DAYS.index(dt)]
        row = []
        for mt in month_targets:
            vec = set_polar(x, dc, th, 1.0)
            if mt is not None:
                vec = set_polar(vec, mc, mc["thetas"][MONTHS.index(mt)], 1.0)
            p = run_patched(model, ids, pos, LAYER, vec)
            row.append(p[day_ids[DAYS.index(dt)]].item())
        table[dt] = row
        log(f"  {dt:>11} | " + " | ".join(f"{v:>15.4f}" for v in row))

    log("\n  interference analysis (P(target day) should depend on day target, not month target):")
    for dt, row in table.items():
        r = np.array(row)
        log(f"    {dt:>10}: no-month={r[0]:.4f}  with-month range=[{r[1:].min():.4f}, {r[1:].max():.4f}]  "
            f"max |shift|={np.abs(r[1:]-r[0]).max():.4f}")
    all_r = np.array([table[d] for d in day_targets])
    signal = all_r[:, 0].max() - all_r[:, 0].min()
    crosstalk = np.abs(all_r[:, 1:] - all_r[:, [0]]).max()
    log(f"\n    SIGNAL  (spread from changing day target)   = {signal:.4f}")
    log(f"    CROSSTALK (max shift from month perturbation) = {crosstalk:.4f}")
    log(f"    crosstalk/signal = {crosstalk/signal:.2%}   (low = composable)")

    # does the month perturbation actually do anything? (control: it must move month outputs)
    log("\n  control: does the month rotation have ANY causal effect? "
        "(month-answer prompt, month steered)")
    mprompt = "The month after February is"
    mids = tok(mprompt, return_tensors="pt")
    mpos = mids["input_ids"].shape[1] - 1
    with torch.no_grad():
        mout = model(**mids, output_hidden_states=True)
    mx = mout.hidden_states[LAYER][0, mpos, :].float().numpy()
    mbase = F.softmax(mout.logits[0, -1, :].float(), dim=-1)
    log(f"    baseline top={tok.decode([int(torch.argmax(mbase))])!r} P={mbase.max().item():.3f}")
    for mt in ["March", "July", "November"]:
        p = run_patched(model, mids, mpos, LAYER, set_polar(mx, mc, mc["thetas"][MONTHS.index(mt)], 1.0))
        top = tok.decode([int(torch.argmax(p))]).strip()
        log(f"    month target {mt:>9}: top={top!r:>14} P(target)={p[month_ids[MONTHS.index(mt)]].item():.4f}"
            + ("   <-- steered!" if top == mt else ""))

    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
