"""
Why did two ~85-degree-orthogonal planes still interfere at ~46%?
Hypothesis: RMSNorm. Orthogonality in the residual basis does not imply
independent control, because the next layer normalises by the WHOLE vector's
norm -- inflating the month component shrinks the effective weekday component.
If that is the mechanism, renormalising the patched state to its original norm
should cut crosstalk sharply, with the weekday effect itself preserved.
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
DAY_R, MONTH_R = 4.0, 6.0


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


def fit_circle(c):
    mean = c.mean(axis=0)
    Xc = c - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ np.stack([Vt[0], Vt[1]], axis=1)
    return dict(mean=mean, e1=Vt[0], e2=Vt[1], thetas=np.arctan2(proj[:, 1], proj[:, 0]))


def set_polar(x, c, th, amp):
    Xc = x - c["mean"]
    a, b = Xc @ c["e1"], Xc @ c["e2"]
    r = np.hypot(a, b) * amp
    return x + (r * np.cos(th) - a) * c["e1"] + (r * np.sin(th) - b) * c["e2"]


def tid(tok, w):
    return tok.encode(" " + w, add_special_tokens=False)[0]


def run(model, ids, pos, L, vec):
    blk = model.model.layers[L - 1]

    def hook(m, i, out):
        t = isinstance(out, tuple)
        hs = (out[0] if t else out).clone()
        hs[0, pos, :] = torch.tensor(vec, dtype=hs.dtype)
        return ((hs,) + out[1:]) if t else hs

    h = blk.register_forward_hook(hook)
    with torch.no_grad():
        lg = model(**ids).logits[0, -1, :].float()
    h.remove()
    return F.softmax(lg, dim=-1)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    d_tid = {d: tid(tok, d) for d in DAYS}
    dc = fit_circle(np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in day_prompts(d)], axis=0) for d in DAYS]))
    mc = fit_circle(np.stack([np.mean([last_hidden(model, tok, p, LAYER) for p in month_prompts(m)], axis=0) for m in MONTHS]))

    prompt = "Today is Wednesday. Tomorrow is"
    ids = tok(prompt, return_tensors="pt")
    pos = ids["input_ids"].shape[1] - 1
    with torch.no_grad():
        x = model(**ids, output_hidden_states=True).hidden_states[LAYER][0, pos, :].float().numpy()
    n0 = np.linalg.norm(x)
    log(f"loaded+fit ({time.time()-t0:.0f}s); ||x||={n0:.1f}")

    # how much do the edits inflate the norm?
    log("\nnorm inflation from each edit:")
    v_d = set_polar(x, dc, dc["thetas"][0], DAY_R)
    v_dm = set_polar(v_d, mc, mc["thetas"][MONTHS.index('July')], MONTH_R)
    log(f"  weekday only      : ||v||/||x|| = {np.linalg.norm(v_d)/n0:.3f}")
    log(f"  weekday + month   : ||v||/||x|| = {np.linalg.norm(v_dm)/n0:.3f}")

    month_pert = [None, "July", "November", "May"]
    day_targets = ["Monday", "Saturday", "Tuesday"]

    for mode in ["raw", "norm-preserved"]:
        log(f"\n=== mode: {mode} ===")
        log(f"  {'day target':>11} | " + " | ".join(f"{str(m):>10}" for m in month_pert))
        arr = []
        hits_alone = hits_pert = 0
        for dt in day_targets:
            row = []
            for j, mp in enumerate(month_pert):
                v = set_polar(x, dc, dc["thetas"][DAYS.index(dt)], DAY_R)
                if mp:
                    v = set_polar(v, mc, mc["thetas"][MONTHS.index(mp)], MONTH_R)
                if mode == "norm-preserved":
                    v = v * (n0 / np.linalg.norm(v))
                p = run(model, ids, pos, LAYER, v)
                ok = tok.decode([int(torch.argmax(p))]).strip() == dt
                row.append((p[d_tid[dt]].item(), ok))
                if j == 0:
                    hits_alone += ok
                else:
                    hits_pert += ok
            arr.append([v for v, _ in row])
            log(f"  {dt:>11} | " + " | ".join(f"{v:.3f}{'*' if ok else ' '}".rjust(10) for v, ok in row))
        a = np.array(arr)
        sig = a[:, 0].max() - a[:, 0].min()
        xt = np.abs(a[:, 1:] - a[:, [0]]).max()
        mean_drop = (a[:, 0][:, None] - a[:, 1:]).mean()
        log(f"  top-1: alone {hits_alone}/{len(day_targets)}, perturbed {hits_pert}/{len(day_targets)*3}")
        log(f"  mean P drop from month perturbation = {mean_drop:+.4f}")
        log(f"  SIGNAL={sig:.4f} CROSSTALK={xt:.4f} crosstalk/signal={xt/sig:.1%}")

    log(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
