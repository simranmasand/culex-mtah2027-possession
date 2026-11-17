"""A1 + A2: re-tune decoding for the final no-shape GRU, and blend GRU with HMM evidence.

    python experiments/e17_decode_blend.py

A1  decoding grid for the final model (inputs without team shape, D16), extending the
    possession-change multiplier `ha` past the old grid edge (10) to 20/50/100.
A2  blend per-frame evidence before Viterbi:  w * GRU log-prob + (1 - w) * HMM log-emission
    (HMM = external-only E4 model, renormalised per frame), w in {0.8, 0.6, 0.4}.

Protocol (same as D13):
  tune   standard hard bench (seed 2027), leave-one-game-out, 4-seed GRU ensembles; mean spellF1
  confirm fresh bench (seed 777), hard + easy, 4 seeds: A1 best and A2 best vs the current final
          decoding (w=1, alpha 0.5, stay 4, ha 10), paired episode bootstrap (90% CI)
Keep rule (D4): helps on both variants, or helps on one and is neutral (CI spans 0) on the other.
GRU log-probs are cached in experiments/e17/cache/ so the run can resume.
No BAR-ATH labels; public 5 min untouched.
"""

import itertools
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "starter")]

import bench  # noqa: E402
import methods as M  # noqa: E402
import e9_autoresearch as E  # noqa: E402
import e13_autoresearch as X  # noqa: E402
from e9_confirm import fresh_bench  # noqa: E402

OUT = os.path.join(HERE, "e17")
CACHE = os.path.join(OUT, "cache")
CFG = json.load(open(os.path.join(HERE, "final", "config_d17.json")))   # baseline = final model before D18
TRAIN = CFG["train_config"]
CURRENT = (1.0, CFG["decode"]["alpha"], CFG["decode"]["stay"], CFG["decode"]["ha"])   # (w, alpha, stay, ha)
SEEDS = (0, 1, 2, 3)
GRID_A1 = list(itertools.product([1.0], [0.3, 0.5, 0.8, 1.2], [1.0, 2.0, 4.0, 8.0, 16.0], [5.0, 10.0, 20.0, 50.0, 100.0]))
GRID_A2 = list(itertools.product([0.8, 0.6, 0.4], [0.3, 0.5, 0.8, 1.2], [2.0, 4.0, 8.0], [10.0, 20.0, 50.0]))

_full = E.frame_inputs
E.frame_inputs = lambda d, swap=False: np.delete(_full(d, swap), np.r_[slice(9, 15)], axis=1)   # D16: no shape
E.TB = None
E.OUT = OUT


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def gru_logp(tag, games, d):
    """Mean over SEEDS of per-step GRU log-probs for held-out game d (cached)."""
    path = os.path.join(CACHE, f"{tag}_{d['name']}.npy")
    if os.path.exists(path):
        return np.load(path)
    tr = [g for g in games if g["name"] != d["name"]]
    Xd = torch.tensor(E.steps(E.frame_inputs(d)))[None]
    lps = []
    for s in SEEDS:
        t = time.time()
        model = E.train(TRAIN, tr, s)
        with torch.no_grad():
            lps.append(torch.log_softmax(model(Xd), -1)[0].numpy())
        log(f"  trained {tag} {d['name']} seed {s} ({time.time() - t:.0f} s)")
    lp = np.mean(lps, axis=0)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    np.save(path, lp)
    return lp


def to_frames(lp, n):
    le = np.repeat(lp, E.SUB, axis=0)
    return np.vstack([le, np.repeat(le[-1:], n - len(le), axis=0)])


def evidence(tag, games):
    """{game: (GRU frame log-probs, HMM frame log-emissions renormalised)} for LOGO."""
    ev = {}
    for d in games:
        g = to_frames(gru_logp(tag, games, d), len(d["ball"]))
        tr = [x for x in games if x["name"] != d["name"]]
        h = M.HMM(alpha=1.0).fit(tr).emissions(d)
        h -= np.log(np.exp(h - h.max(1, keepdims=True)).sum(1, keepdims=True)) + h.max(1, keepdims=True)
        ev[d["name"]] = (g, h)
    return ev


def decode(ev_d, d, log_T, w, alpha):
    g, h = ev_d
    le = alpha * (w * g + (1 - w) * h if w < 1 else g)
    out = np.full(len(le), "X", dtype="<U1")
    names = np.array(["H", "A", "D"])
    for a, b in M._blocks(d["play"]):
        out[a:b] = names[M.viterbi(le[a:b], log_T)]
    return out


class Decoder:
    def __init__(self, games, ev):
        self.games, self.ev, self.T = games, ev, {}

    def method(self, params):
        w, alpha, stay, ha = params
        cache = {}

        def run(d):
            if d["name"] not in cache:
                key = (d["name"], stay, ha)
                if key not in self.T:
                    self.T[key] = X.transitions([g for g in self.games if g["name"] != d["name"]], stay, ha)
                cache[d["name"]] = decode(self.ev[d["name"]], d, self.T[key], w, alpha)
            return cache[d["name"]]
        return run

    def score(self, params):
        rows = bench.score(self.method(params), self.games)
        return float(np.mean([r["spellF1"] for r in rows])), int(sum(r["poss_pred"] for r in rows))


def grid_search(dec, grid, name, tsv):
    best = None
    for i, p in enumerate(grid):
        sc, poss = dec.score(p)
        with open(tsv, "a") as fh:
            fh.write(f"{name}\t{p[0]}\t{p[1]}\t{p[2]}\t{p[3]}\t{sc:.4f}\t{poss}\n")
        if best is None or sc > best[1]:
            best = (p, sc, poss)
            log(f"  {name} new best {p} spellF1 {sc:.4f} poss {poss}  ({i + 1}/{len(grid)})")
    return best


def main():
    os.makedirs(CACHE, exist_ok=True)
    tsv = os.path.join(OUT, "grid.tsv")
    if not os.path.exists(tsv):
        with open(tsv, "w") as fh:
            fh.write("search\tw\talpha\tstay\tha\tspellF1_hard_tune\tposs\n")
    log("== tune on standard hard bench (seed 2027) ==")
    hard = bench.bench_games("hard")
    dt = Decoder(hard, evidence("tune_hard", hard))
    cur = dt.score(CURRENT)
    log(f"  current final decoding {CURRENT}: spellF1 {cur[0]:.4f} poss {cur[1]}")
    a1 = grid_search(dt, GRID_A1, "A1", tsv)
    a2 = grid_search(dt, GRID_A2, "A2", tsv)
    summary = {"current": {"params": CURRENT, "tune_spellF1": cur[0], "tune_poss": cur[1]},
               "A1": {"params": a1[0], "tune_spellF1": a1[1], "tune_poss": a1[2]},
               "A2": {"params": a2[0], "tune_spellF1": a2[1], "tune_poss": a2[2]}}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    for level in ("hard", "easy"):
        log(f"== confirm on fresh {level} bench (seed 777) ==")
        G = fresh_bench(level)
        dc = Decoder(G, evidence(f"fresh_{level}", G))
        base = dc.method(CURRENT)
        sb, pb = dc.score(CURRENT)
        summary.setdefault("confirm", {})[level] = {"current": {"spellF1": sb, "poss": pb}}
        for name in ("A1", "A2"):
            p = tuple(summary[name]["params"])
            s, poss = dc.score(p)
            c = bench.compare(dc.method(p), base, G)
            summary["confirm"][level][name] = {"spellF1": s, "poss": poss, "minus_current": c}
            log(f"  {name} {p}: {s:.4f} (poss {poss}) vs current {sb:.4f} (poss {pb}); "
                f"diff {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    for name in ("A1", "A2"):
        cs = [summary["confirm"][lv][name]["minus_current"] for lv in ("hard", "easy")]
        helps = [c["lo"] > 0 for c in cs]
        hurts = [c["hi"] < 0 for c in cs]
        summary[name]["keep"] = bool(any(helps) and not any(hurts))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    log(json.dumps({k: summary[k].get("keep") for k in ("A1", "A2")}))


if __name__ == "__main__":
    main()
