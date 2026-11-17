"""D13: overnight GRU search aimed at under-segmentation (too few possessions), bounded and logged.

    python experiments/e13_autoresearch.py --minutes 180

Two-level loop:
  * outer: TRAINING config (arch, size, lr, epochs, window incl. long 1-5 min windows = E11a) — mutated
    from the best so far. Each config is trained with 2 seeds per leave-one-game-out fold on the HARD
    bench; per-step log-probs of the held-out game are cached and averaged over the 2 seeds.
  * inner: DECODING params need no retraining, so a grid is scored on the cached log-probs:
      alpha    emission weight
      stay     scales the probability of leaving a state (higher = more switches)
      ha       extra multiplier on H<->A transitions (the possession-change cost; higher = more changes)
Score = mean hard spellF1 over the 3 held-out games. Fixed evaluation, never edited by the loop.

Final: winner vs the previous best (E9 config, nn_e9_ens8) on the FRESH bench (seed 777), hard + easy,
4-seed ensembles each, paired episode bootstrap. Then an 8-seed BAR-ATH file -> evaluation/results/
nn_best.csv. Keep/replace decision is written to summary.json (`beats_previous_on_both`).
No BAR-ATH labels used; public 5 min untouched.
"""

import argparse
import itertools
import json
import os
import random
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
from e9_confirm import fresh_bench  # noqa: E402

OUT = os.path.join(HERE, "e13_nn")
PREV = json.load(open(os.path.join(HERE, "e9_nn", "summary.json")))["best_config"]

TRAIN_SPACE = {"arch": ["gru"], "hidden": [32, 64, 96], "layers": [1, 2, 3], "dropout": [0.0, 0.1, 0.2],
               "lr": [3e-3, 1e-2, 2e-2], "epochs": [12, 20, 30], "window": [64, 128, 300, 750, 1500]}
DECODE_GRID = {"alpha": [0.3, 0.5, 0.8, 1.2], "stay": [1.0, 2.0, 4.0, 8.0], "ha": [1.0, 3.0, 10.0]}
START = {k: PREV[k] for k in ("arch", "hidden", "layers", "dropout", "lr", "epochs", "window")}
SEARCH_SEEDS = (0, 1)

E.DESCRIPTIONS.update({
    "search13/spellF1_hard": "Best decoded score (whole-possession metric) for this training config, 2-seed average, hard bench, held-out games. Higher = better.",
    "search13/best_spellF1": "Best score found so far in this search.",
    "search13/previous_best_reference": "Score of the previous best GRU config on the same test (the line to beat).",
    "search13/possessions_pred (true 1022)": "Possessions predicted by the best decoding of this config (truth 1,022). The search tries to close this gap.",
    "search13/best_stay": "Chosen 'leave-a-state' scale. Higher = the decoder allows more switches.",
    "search13/best_ha": "Chosen possession-change multiplier (H<->A). Higher = more changes of possession allowed.",
})


def transitions(train_games, stay, ha):
    T = np.exp(M.HMM(alpha=0.3, stay_scale=stay).fit(train_games).log_T)
    T[0, 1] *= ha; T[1, 0] *= ha
    for r in (0, 1):
        off = T[r].sum() - T[r, r]
        if off >= 1:
            T[r] /= T[r].sum()
        else:
            T[r, r] = 1 - off
    return np.log(np.clip(T, 1e-12, 1))


def decode(logp, d, log_T, alpha):
    le = np.repeat(logp, E.SUB, axis=0)
    le = np.vstack([le, np.repeat(le[-1:], len(d["ball"]) - len(le), axis=0)]) * alpha
    out = np.full(len(le), "X", dtype="<U1")
    names = np.array(["H", "A", "D"])
    for a, b in M._blocks(d["play"]):
        out[a:b] = names[M.viterbi(le[a:b], log_T)]
    return out


def held_out_logp(cfg, games, seeds):
    """{game: mean log-prob over seeds} for leave-one-game-out training."""
    out = {}
    for d in games:
        tr = [g for g in games if g["name"] != d["name"]]
        E.MON["fold"] = d["name"]
        lps = []
        for s in seeds:
            model = E.train(cfg, tr, s, val_game=d)
            X = E.steps(E.frame_inputs(d))
            with torch.no_grad():
                lps.append(torch.log_softmax(model(torch.tensor(X)[None]), -1)[0].numpy())
        lp = np.mean(lps, axis=0)
        out[d["name"]] = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    return out


def score_decodes(logps, games, grid):
    """Best decoding params for cached log-probs -> (score, params, possessions, all results)."""
    Ts = {}
    results = []
    for alpha, stay, ha in grid:
        rows = []
        for d in games:
            key = (d["name"], stay, ha)
            if key not in Ts:
                Ts[key] = transitions([g for g in games if g["name"] != d["name"]], stay, ha)
            pred = decode(logps[d["name"]], d, Ts[key], alpha)
            rows.append(bench.score(lambda _d, p=pred: p, [d])[0])
        results.append(((alpha, stay, ha), float(np.mean([r["spellF1"] for r in rows])),
                        int(sum(r["poss_pred"] for r in rows))))
    best = max(results, key=lambda r: r[1])
    return best[1], best[0], best[2], results


def method_from(cfg, dec, train_for, seeds):
    """Bench method: per held-out game, ensemble of `seeds`, decoded with dec=(alpha, stay, ha)."""
    cache = {}

    def run(d):
        if d["name"] not in cache:
            tr = train_for(d)
            lps = []
            for s in seeds:
                model = E.train(cfg, tr, s)
                X = E.steps(E.frame_inputs(d))
                with torch.no_grad():
                    lps.append(torch.log_softmax(model(torch.tensor(X)[None]), -1)[0].numpy())
            lp = np.mean(lps, axis=0); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
            cache[d["name"]] = decode(lp, d, transitions(tr, dec[1], dec[2]), dec[0])
        return cache[d["name"]]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=180)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    E.TB = SummaryWriter(os.path.join(OUT, "tb", time.strftime("%Y%m%d-%H%M%S")))
    E._DESCRIBED.clear(); E.tb_readme()
    tsv = os.path.join(OUT, "results.tsv")
    with open(tsv, "w") as fh:
        fh.write("run\tspellF1_hard\tposs\talpha\tstay\tha\tseconds\tstatus\ttrain_config\n")

    hard = bench.bench_games("hard")
    full_grid = list(itertools.product(*DECODE_GRID.values()))
    rng = random.Random(13)
    t_end = time.time() + args.minutes * 60
    E.MON["t_end"] = t_end
    E.OUT = OUT                      # progress.json lands in e13_nn/
    best_cfg, best_dec, best_score, seen, run_id = None, None, -1.0, set(), 0
    prev_ref = None
    while time.time() < t_end:
        if best_cfg is None:
            cfg = dict(START)
        else:
            cfg = dict(best_cfg)
            for k in rng.sample(list(TRAIN_SPACE), k=rng.choice([1, 1, 2])):
                cfg[k] = rng.choice(TRAIN_SPACE[k])
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key); run_id += 1
        E.MON["run"] = run_id; E.MON["config"] = cfg
        t = time.time()
        try:
            logps = held_out_logp(cfg, hard, SEARCH_SEEDS)
            grid = full_grid if run_id == 1 else rng.sample(full_grid, 20) + ([best_dec] if best_dec else [])
            sc, dec, poss, _ = score_decodes(logps, hard, grid)
            if run_id == 1:
                prev_ref = sc            # START = previous best training config, re-decoded on the grid
            status = "keep" if sc > best_score else "discard"
            if status == "keep":
                best_cfg, best_dec, best_score = cfg, dec, sc
        except Exception as exc:  # noqa: BLE001
            sc, dec, poss, status = 0.0, (0, 0, 0), 0, f"crash:{type(exc).__name__}"
        with open(tsv, "a") as fh:
            fh.write(f"{run_id}\t{sc:.4f}\t{poss}\t{dec[0]}\t{dec[1]}\t{dec[2]}\t{time.time()-t:.0f}\t{status}\t{key}\n")
        print(f"[{run_id}] {sc:.4f} poss {poss} dec {dec} {status} {key}", flush=True)
        E.MON["best"] = best_score; E.MON["last"] = sc
        E.tb_scalar("search13/spellF1_hard", sc, run_id)
        E.tb_scalar("search13/best_spellF1", best_score, run_id)
        E.tb_scalar("search13/previous_best_reference", prev_ref or 0, run_id)
        E.tb_scalar("search13/possessions_pred (true 1022)", poss, run_id)
        E.tb_scalar("search13/best_stay", dec[1], run_id)
        E.tb_scalar("search13/best_ha", dec[2], run_id)
        E.TB.flush(); E._progress()

    # -- final: winner vs previous best on the fresh bench, 4-seed ensembles ----------------------
    summary = {"runs": run_id, "best_train_config": best_cfg, "best_decode": dict(zip(DECODE_GRID, best_dec)),
               "best_hard_search": best_score, "previous_config_redecoded_hard_search": prev_ref}
    prev_dec = (PREV["alpha"], PREV["stay"], 1.0)
    prev_cfg = START
    beats = True
    for level in ("hard", "easy"):
        G = fresh_bench(level)
        tr_for = lambda d, G=G: [g for g in G if g["name"] != d["name"]]
        new = method_from(best_cfg, best_dec, tr_for, (0, 1, 2, 3))
        old = method_from(prev_cfg, prev_dec, tr_for, (0, 1, 2, 3))
        rn, ro = bench.score(new, G), bench.score(old, G)
        c = bench.compare(new, old, G)
        summary[level] = {"new_spellF1": float(np.mean([r["spellF1"] for r in rn])),
                          "new_poss": int(sum(r["poss_pred"] for r in rn)),
                          "old_spellF1": float(np.mean([r["spellF1"] for r in ro])),
                          "old_poss": int(sum(r["poss_pred"] for r in ro)), "new_minus_old": c}
        beats = beats and c["lo"] > -0.005 and c["diff"] > 0
        print(level, summary[level], flush=True)
    summary["beats_previous_on_both"] = bool(beats)

    B = bench.barath()
    lps = []
    for s in range(8):
        model = E.train(best_cfg, hard, s)
        with torch.no_grad():
            lps.append(torch.log_softmax(model(torch.tensor(E.steps(E.frame_inputs(B)))[None]), -1)[0].numpy())
    lp = np.mean(lps, axis=0); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    lab = decode(lp, B, transitions(hard, best_dec[1], best_dec[2]), best_dec[0])
    from baseline import write_submission
    write_submission(os.path.join(ROOT, "evaluation", "results", "nn_best.csv"), lab)
    g = B["play"]; nt = M.features(B)["near_team"]
    alive = np.isin(lab, ["H", "A"]) & g; vis = alive & (nt >= 0)
    summary["barath"] = {
        "possessions": int(sum(1 for i in np.flatnonzero(g)[1:] if lab[i] in "HA" and lab[i] != lab[i - 1])),
        "D_share": float(np.mean(lab[g] == "D")),
        "agree_nearest": float(np.mean((lab[vis] == "H") == (nt[vis] == 0)))}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
