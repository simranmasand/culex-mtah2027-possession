"""E5c: what do the team-shape inputs add to the GRU? (D9)

    python experiments/e5c_ablation.py

Same config with and without the 6 team-shape input columns. Config = the D13 winner if it beat the
previous best on both variants, else the E9 best. Fresh bench (seed 777), easy + hard, 4-seed
ensembles per held-out game, paired episode bootstrap (with-shape minus without-shape).
Results -> experiments/e5c_ablation.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import bench  # noqa: E402
import e9_autoresearch as E  # noqa: E402
import e13_autoresearch as X  # noqa: E402
from e9_confirm import fresh_bench  # noqa: E402

SHAPE_COLS = slice(9, 15)          # frame_inputs: 9 base columns, then the 6 shape features
_full_inputs = E.frame_inputs


def no_shape_inputs(d, swap=False):
    Xf = _full_inputs(d, swap)
    return np.delete(Xf, np.r_[SHAPE_COLS], axis=1)


def pick_config():
    s13 = os.path.join(HERE, "e13_nn", "summary.json")
    if os.path.exists(s13):
        s = json.load(open(s13))
        if s.get("beats_previous_on_both"):
            d = s["best_decode"]
            return s["best_train_config"], (d["alpha"], d["stay"], d["ha"]), "D13 winner"
    p = X.PREV
    return X.START, (p["alpha"], p["stay"], 1.0), "E9 best"


def main():
    cfg, dec, source = pick_config()
    out = {"config": cfg, "decode": dec, "config_source": source}
    seeds = (0, 1, 2, 3)
    for level in ("hard", "easy"):
        G = fresh_bench(level)
        tr_for = lambda d, G=G: [g for g in G if g["name"] != d["name"]]
        E.frame_inputs = _full_inputs
        full = X.method_from(cfg, dec, tr_for, seeds)
        preds_full = {d["name"]: full(d) for d in G}
        E.frame_inputs = no_shape_inputs
        ablated = X.method_from(cfg, dec, tr_for, seeds)
        preds_abl = {d["name"]: ablated(d) for d in G}
        E.frame_inputs = _full_inputs
        mf = lambda d: preds_full[d["name"]]
        ma = lambda d: preds_abl[d["name"]]
        rf, ra = bench.score(mf, G), bench.score(ma, G)
        c = bench.compare(mf, ma, G)
        out[level] = {"with_shape": float(np.mean([r["spellF1"] for r in rf])),
                      "with_shape_poss": int(sum(r["poss_pred"] for r in rf)),
                      "without_shape": float(np.mean([r["spellF1"] for r in ra])),
                      "without_shape_poss": int(sum(r["poss_pred"] for r in ra)),
                      "with_minus_without": c}
        print(level, out[level], flush=True)
    json.dump(out, open(os.path.join(HERE, "e5c_ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
