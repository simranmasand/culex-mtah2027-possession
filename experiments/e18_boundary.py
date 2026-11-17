"""A3: training changes aimed at under-segmentation (the GRU merges possessions; decoding can't undo it, A1/A2).

    python experiments/e18_boundary.py

Variants of the final no-shape GRU (same config, D16), trained differently:
  bw   loss weight on steps within r steps (0.2 s each) of a label change (H/A/D), e.g. 5 = 5x weight
  cp   change-point head: second output predicting "the state changes at this step", trained with
       weight cp (multitask). Optionally used in decoding: the cost of leaving a state is scaled
       per step by (p_change / mean p_change) ** beta  (beta 0 = head ignored at decoding).
Protocol (same as A1/A2, D13):
  tune    standard hard bench (seed 2027), LOGO, 4-seed ensembles, small decoding grid per variant
  confirm fresh bench (seed 777), hard + easy: the 2 best variants vs the current final (cached in e17/),
          paired episode bootstrap. Keep rule (D4): helps on one variant, no loss on the other.
Results -> experiments/e18/ (results.tsv, summary.json, run.log). No BAR-ATH labels; public 5 min untouched.
"""

import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "starter")]

import bench  # noqa: E402
import methods as M  # noqa: E402
import e9_autoresearch as E  # noqa: E402
import e13_autoresearch as X  # noqa: E402
from e9_confirm import fresh_bench  # noqa: E402

OUT = os.path.join(HERE, "e18")
CACHE = os.path.join(OUT, "cache")
E17_CACHE = os.path.join(HERE, "e17", "cache")
CFG = json.load(open(os.path.join(HERE, "final", "config_d17.json")))   # baseline = final model before D18
TRAIN = CFG["train_config"]
CURRENT = (CFG["decode"]["alpha"], CFG["decode"]["stay"], CFG["decode"]["ha"], 0.0)
SEEDS = (0, 1, 2, 3)
VARIANTS = {
    "bw3_r5": {"bw": 3, "r": 5, "cp": 0.0},
    "bw5_r5": {"bw": 5, "r": 5, "cp": 0.0},
    "bw10_r5": {"bw": 10, "r": 5, "cp": 0.0},
    "bw5_r10": {"bw": 5, "r": 10, "cp": 0.0},
    "cp0.5": {"bw": 1, "r": 5, "cp": 0.5},
    "cp1": {"bw": 1, "r": 5, "cp": 1.0},
    "bw5_cp1": {"bw": 5, "r": 5, "cp": 1.0},
}
DEC_GRID = list(itertools.product([0.3, 0.5, 0.8], [1.0, 2.0, 4.0], [5.0, 10.0, 20.0]))
BETAS = (0.0, 0.5, 1.0)

_full = E.frame_inputs
E.frame_inputs = lambda d, swap=False: np.delete(_full(d, swap), np.r_[slice(9, 15)], axis=1)   # D16
E.TB = None
E.OUT = OUT


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


# -- model + training ------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout, cp):
        super().__init__()
        self.rnn = nn.GRU(n_in, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, 3)
        self.cp_head = nn.Linear(2 * hidden, 1) if cp > 0 else None

    def forward(self, x):
        h = self.rnn(x)[0]
        return self.head(h), (self.cp_head(h)[..., 0] if self.cp_head is not None else None)


def targets(y, r):
    """change[t] = 1 if state differs from t-1 (both labelled), -1 if unknown; near[t] = within r of a change."""
    ch = np.full(len(y), -1.0, dtype=np.float32)
    ok = (y[1:] >= 0) & (y[:-1] >= 0)
    ch[1:][ok] = (y[1:][ok] != y[:-1][ok]).astype(np.float32)
    ch[0] = 0.0 if y[0] >= 0 else -1.0
    idx = np.flatnonzero(ch == 1)
    near = np.zeros(len(y), dtype=bool)
    for i in idx:
        near[max(0, i - r):i + r + 1] = True
    return ch, near


def train(var, games, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    data = []
    for x, y in E.prepared(games):
        ch, near = targets(y, var["r"])
        wt = np.where(near, var["bw"], 1.0).astype(np.float32)
        data.append((x, y, ch, wt))
    model = Net(data[0][0].shape[1], TRAIN["hidden"], TRAIN["layers"], TRAIN["dropout"], var["cp"])
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=1e-4)
    L, B = TRAIN["window"], 32
    per_epoch = max(1, sum(len(d[0]) for d in data) // (L * B))
    ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(TRAIN["epochs"]):
        model.train()
        for _ in range(per_epoch):
            xb, yb, cb, wb = [], [], [], []
            for _ in range(B):
                x, y, ch, wt = data[np.random.randint(len(data))]
                s = np.random.randint(0, len(x) - L)
                xb.append(x[s:s + L]); yb.append(y[s:s + L]); cb.append(ch[s:s + L]); wb.append(wt[s:s + L])
            xb, yb = torch.tensor(np.stack(xb)), torch.tensor(np.stack(yb))
            cb, wb = torch.tensor(np.stack(cb)), torch.tensor(np.stack(wb))
            logits, cpl = model(xb)
            l = ce(logits.reshape(-1, 3), yb.reshape(-1)) * wb.reshape(-1)
            ok = (yb.reshape(-1) >= 0).float()
            loss = (l * ok).sum() / (wb.reshape(-1) * ok).sum().clamp(min=1)
            if cpl is not None:
                m = (cb >= 0).float()
                loss = loss + var["cp"] * (bce(cpl, cb.clamp(min=0)) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def outputs(name, var, tag, games, d):
    """(mean log-probs, mean change prob or None) over SEEDS for held-out game d, cached."""
    path = os.path.join(CACHE, f"{name}_{tag}_{d['name']}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["lp"], (z["cp"] if "cp" in z.files else None)
    tr = [g for g in games if g["name"] != d["name"]]
    Xd = torch.tensor(E.steps(E.frame_inputs(d)))[None]
    lps, cps = [], []
    for s in SEEDS:
        t = time.time()
        model = train(var, tr, s)
        with torch.no_grad():
            lg, cpl = model(Xd)
        lps.append(torch.log_softmax(lg, -1)[0].numpy())
        if cpl is not None:
            cps.append(torch.sigmoid(cpl)[0].numpy())
        log(f"  trained {name} {tag} {d['name']} seed {s} ({time.time() - t:.0f} s)")
    lp = np.mean(lps, axis=0); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    out = {"lp": lp}
    if cps:
        out["cp"] = np.mean(cps, axis=0)
    np.savez(path, **out)
    return lp, out.get("cp")


def current_outputs(tag, d):
    return np.load(os.path.join(E17_CACHE, f"{tag}_{d['name']}.npy")), None


# -- decoding --------------------------------------------------------------------------------------
def frames(a, n):
    le = np.repeat(a, E.SUB, axis=0)
    return np.concatenate([le, np.repeat(le[-1:], n - len(le), axis=0)])


def viterbi_tv(log_e, T, scale):
    """Viterbi with per-frame transitions: off-diagonal probabilities of T multiplied by scale[t]."""
    n = len(log_e)
    off = T.copy(); np.fill_diagonal(off, 0)
    back = np.zeros((n, 3), dtype=np.int8)
    v = log_e[0] + np.log(np.array([0.2, 0.2, 0.6]))
    for t in range(1, n):
        o = np.minimum(off * scale[t], 0.999 / 2)
        Tt = o + np.diag(1 - o.sum(1))
        cand = v[:, None] + np.log(Tt)
        back[t] = cand.argmax(0)
        v = cand.max(0) + log_e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = v.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


class Decoder:
    def __init__(self, games, outs):
        self.games, self.outs, self.T = games, outs, {}

    def method(self, p):
        alpha, stay, ha, beta = p
        cache = {}

        def run(d):
            if d["name"] not in cache:
                key = (d["name"], stay, ha)
                if key not in self.T:
                    self.T[key] = X.transitions([g for g in self.games if g["name"] != d["name"]], stay, ha)
                lp, cp = self.outs[d["name"]]
                n = len(d["ball"])
                le = alpha * frames(lp, n)
                out = np.full(n, "X", dtype="<U1")
                names = np.array(["H", "A", "D"])
                if beta > 0 and cp is not None:
                    sc = frames((cp / cp.mean()) ** beta, n)
                    T = np.exp(self.T[key])
                    for a, b in M._blocks(d["play"]):
                        out[a:b] = names[viterbi_tv(le[a:b], T, sc[a:b])]
                else:
                    for a, b in M._blocks(d["play"]):
                        out[a:b] = names[M.viterbi(le[a:b], self.T[key])]
                cache[d["name"]] = out
            return cache[d["name"]]
        return run

    def score(self, p):
        rows = bench.score(self.method(p), self.games)
        return float(np.mean([r["spellF1"] for r in rows])), int(sum(r["poss_pred"] for r in rows))


def tune(dec, has_cp):
    grid = [g + (b,) for g in DEC_GRID for b in (BETAS if has_cp else (0.0,))]
    res = [(p,) + dec.score(p) for p in grid]
    return max(res, key=lambda r: r[1])


def main():
    os.makedirs(CACHE, exist_ok=True)
    tsv = os.path.join(OUT, "results.tsv")
    with open(tsv, "w") as fh:
        fh.write("variant\tspellF1_hard_tune\tposs\talpha\tstay\tha\tbeta\n")
    hard = bench.bench_games("hard")
    base = Decoder(hard, {d["name"]: current_outputs("tune_hard", d) for d in hard})
    b_sc, b_poss = base.score(CURRENT)
    b_best = tune(base, False)
    log(f"current final: {b_sc:.4f} (poss {b_poss}); best decoding on its outputs {b_best[1]:.4f} (poss {b_best[2]})")
    with open(tsv, "a") as fh:
        fh.write(f"current\t{b_sc:.4f}\t{b_poss}\t" + "\t".join(map(str, CURRENT)) + "\n")
    summary = {"current": {"tune_spellF1": b_sc, "tune_poss": b_poss}, "variants": {}}
    for name, var in VARIANTS.items():
        log(f"== {name} {var} ==")
        outs = {d["name"]: outputs(name, var, "tune_hard", hard, d) for d in hard}
        p, sc, poss = tune(Decoder(hard, outs), var["cp"] > 0)
        summary["variants"][name] = {"var": var, "decode": p, "tune_spellF1": sc, "tune_poss": poss}
        with open(tsv, "a") as fh:
            fh.write(f"{name}\t{sc:.4f}\t{poss}\t" + "\t".join(map(str, p)) + "\n")
        log(f"  {name}: {sc:.4f} (poss {poss}) decode {p}")
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    top = sorted(summary["variants"], key=lambda k: -summary["variants"][k]["tune_spellF1"])[:2]
    summary["confirmed"] = top
    for level in ("hard", "easy"):
        log(f"== confirm on fresh {level} bench (seed 777): {top} ==")
        G = fresh_bench(level)
        tag = f"fresh_{level}"
        cur = Decoder(G, {d["name"]: current_outputs(tag, d) for d in G})
        cs, cposs = cur.score(CURRENT)
        summary.setdefault("confirm", {})[level] = {"current": {"spellF1": cs, "poss": cposs}}
        for name in top:
            v = summary["variants"][name]
            dec = Decoder(G, {d["name"]: outputs(name, v["var"], tag, G, d) for d in G})
            p = tuple(v["decode"])
            s, poss = dec.score(p)
            c = bench.compare(dec.method(p), cur.method(CURRENT), G)
            summary["confirm"][level][name] = {"spellF1": s, "poss": poss, "minus_current": c}
            log(f"  {name}: {s:.4f} (poss {poss}) vs current {cs:.4f} (poss {cposs}); "
                f"diff {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
            json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for name in top:
        cs = [summary["confirm"][lv][name]["minus_current"] for lv in ("hard", "easy")]
        summary["variants"][name]["keep"] = bool(any(c["lo"] > 0 for c in cs) and not any(c["hi"] < 0 for c in cs))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    log(json.dumps({k: summary["variants"][k].get("keep") for k in top}))


if __name__ == "__main__":
    main()
