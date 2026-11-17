"""D8 follow-up: multi-seed GRU ensemble (average per-step log-probs over seeds, then Viterbi).

    python experiments/e9_ensemble.py

1. Bench check on the FRESH hard + easy degradation (seed 777, never used by the search): does averaging
   k = 1, 2, 4, 8 seeds raise spellF1 / reduce variance? Leave-one-game-out, vs the HMM.
2. BAR-ATH: 8 seeds trained on all 3 hard bench games -> evaluation/results/nn_e9_ens8.csv + sanity stats.
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "starter")]

import bench  # noqa: E402
import methods as M  # noqa: E402
import e9_autoresearch as E  # noqa: E402
from e9_confirm import fresh_bench, BEST  # noqa: E402

SEEDS = list(range(8))


def step_logp(model, d):
    X = E.steps(E.frame_inputs(d))
    with torch.no_grad():
        return torch.log_softmax(model(torch.tensor(X)[None]), -1)[0].numpy()


def decode(logp, d, log_T, alpha):
    le = np.repeat(logp, E.SUB, axis=0)
    le = np.vstack([le, np.repeat(le[-1:], len(d["ball"]) - len(le), axis=0)]) * alpha
    out = np.full(len(le), "X", dtype="<U1")
    names = np.array(["H", "A", "D"])
    for a, b in M._blocks(d["play"]):
        out[a:b] = names[M.viterbi(le[a:b], log_T)]
    return out


def ensemble_logp(train_games, d, seeds):
    """Mean of per-seed log-probs (geometric mean of probabilities), renormalised."""
    lps = [step_logp(E.train(BEST, train_games, s), d) for s in seeds]
    lp = np.mean(lps, axis=0)
    return lp - np.log(np.exp(lp).sum(1, keepdims=True)), lps


def main():
    out = {"config": BEST, "seeds": SEEDS}
    for level in ("hard", "easy"):
        G = fresh_bench(level)
        hmm = M.hmm_logo(alpha=0.3); hmm.games = G
        cache = {}
        for d in G:
            tr = [g for g in G if g["name"] != d["name"]]
            log_T = M.HMM(alpha=0.3, stay_scale=BEST["stay"]).fit(tr).log_T
            _, lps = ensemble_logp(tr, d, SEEDS)
            cache[d["name"]] = (lps, log_T)
        res = {}
        for k in (1, 2, 4, 8):
            def run(d, k=k):
                lps, log_T = cache[d["name"]]
                lp = np.mean(lps[:k], axis=0)
                lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
                return decode(lp, d, log_T, BEST["alpha"])
            rows = bench.score(run, G)
            c = bench.compare(run, hmm, G)
            res[k] = {"spellF1": float(np.mean([r["spellF1"] for r in rows])),
                      "poss": int(sum(r["poss_pred"] for r in rows)), "vs_hmm": c}
            print(f"{level} ensemble k={k}: spellF1 {res[k]['spellF1']:.4f} poss {res[k]['poss']} "
                  f"vs HMM {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]", flush=True)
        out[level] = res

    # BAR-ATH: train on all three hard games
    hard = bench.bench_games("hard")
    B = bench.barath()
    lp, _ = ensemble_logp(hard, B, SEEDS)
    log_T = M.HMM(alpha=0.3, stay_scale=BEST["stay"]).fit(hard).log_T
    lab = decode(lp, B, log_T, BEST["alpha"])
    from baseline import write_submission
    write_submission(os.path.join(ROOT, "evaluation", "results", "nn_e9_ens8.csv"), lab)
    g = B["play"]; nt = M.features(B)["near_team"]
    alive = np.isin(lab, ["H", "A"]) & g; vis = alive & (nt >= 0)
    out["barath"] = {
        "possessions": int(sum(1 for i in np.flatnonzero(g)[1:] if lab[i] in "HA" and lab[i] != lab[i - 1])),
        "D_share": float(np.mean(lab[g] == "D")), "home_share": float(np.mean(lab[alive] == "H")),
        "agree_nearest": float(np.mean((lab[vis] == "H") == (nt[vis] == 0)))}
    print("BAR-ATH ens8:", out["barath"], flush=True)
    json.dump(out, open(os.path.join(HERE, "e9_nn", "ensemble.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
