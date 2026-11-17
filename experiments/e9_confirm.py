"""D12: is the E9 GRU win real? Seeds + a fresh bench degradation the search never saw.

    python experiments/e9_confirm.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import bench  # noqa: E402
import methods as M  # noqa: E402
import e9_autoresearch as E  # noqa: E402
from external_metrica import load_game, GAMES  # noqa: E402

BEST = json.load(open(os.path.join(HERE, "e9_nn", "summary.json")))["best_config"]
FRESH_SEED = 777


def fresh_bench(level):
    gaps = bench._barath_gap_lengths()
    out = []
    for g in GAMES:
        raw = load_game(g)
        d = {"players": raw["players"], "ball": raw["ball"], "team_of_slot": raw["team_of_slot"],
             "play": raw["labels"] != "X", "labels": raw["labels"], "name": f"g{g}", "period": raw["period"]}
        out.append(bench.degrade(d, level, np.random.default_rng(FRESH_SEED + g), gaps))
    return out


def main():
    res = {"config": BEST}
    for level in ("hard", "easy"):
        G = fresh_bench(level)
        hmm = M.hmm_logo(alpha=0.3); hmm.games = G
        h = float(np.mean([r["spellF1"] for r in bench.score(hmm, G)]))
        seeds = {}
        for seed in (0, 1, 2, 3):
            m = E.logo(BEST, G, seed=seed)
            rows = bench.score(m, G)
            c = bench.compare(m, hmm, G)
            seeds[seed] = {"spellF1": float(np.mean([r["spellF1"] for r in rows])),
                           "poss": int(sum(r["poss_pred"] for r in rows)), "vs_hmm": c}
            print(f"{level} seed {seed}: GRU {seeds[seed]['spellF1']:.4f} vs HMM {h:.4f}  "
                  f"diff {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]  poss {seeds[seed]['poss']}", flush=True)
        res[level] = {"hmm": h, "seeds": seeds,
                      "gru_mean": float(np.mean([v["spellF1"] for v in seeds.values()])),
                      "gru_sd": float(np.std([v["spellF1"] for v in seeds.values()]))}
    json.dump(res, open(os.path.join(HERE, "e9_nn", "confirm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
