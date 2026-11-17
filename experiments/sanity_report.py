"""V4: label-free sanity report for BAR-ATH submission files (no ground truth used).

    python experiments/sanity_report.py                       # every CSV in evaluation/results/
    python experiments/sanity_report.py evaluation/results/nn_best.csv ...

Per file, on graded frames only:
  possessions, mean possession length, dead-ball share, home share,
  agreement with the nearest-player team (ball-visible alive frames),
  agreement with Smart Tagging team evidence (phase-tag codes mapped to a team),
  and flags when a number is outside the plausible range.
Plausible ranges come from the bench games and the public window's *rates* (not its labels per frame):
possessions 250-500 (public rate ≈ 360 per match), dead-ball share 15-35%.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "starter")]

FPS = 25
POSS_RANGE = (250, 500)
D_RANGE = (0.15, 0.35)
TAGGED_TEAM = {"BUILD UP", "PROGRESSION", "LONG BALL", "ATTACKING TRANSITION"}
OPPOSITE_TEAM = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION"}


def context():
    from atd_cache import load
    from methods import features
    import load_smart_tagging
    d = load()
    g = d["graded"]
    nt = features({"players": d["players"], "ball": d["ball"], "team_of_slot": d["team_of_slot"], "play": g})["near_team"]
    tag_team = np.full(len(g), -1, dtype=np.int8)            # 0 home, 1 away, -1 no evidence
    side = {"barcelona": 0, "atletic club": 1}
    for e in load_smart_tagging.load():
        code = (e["code"] or "").strip().upper()
        t = side.get((e["tags"].get("Team") or "").strip().lower())
        if t is None:
            continue
        if code in TAGGED_TEAM:
            tag_team[e["start_frame"]:e["end_frame"]] = t
        elif code in OPPOSITE_TEAM:
            tag_team[e["start_frame"]:e["end_frame"]] = 1 - t
    return g, nt, tag_team


def report(path, g, nt, tag_team):
    lab = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy(str)
    idx = np.flatnonzero(g)
    starts = [i for i in idx if lab[i] in "HA" and (i == idx[0] or lab[i] != lab[i - 1] or not g[i - 1])]
    alive = np.isin(lab, ["H", "A"]) & g
    n_alive = alive.sum()
    vis = alive & (nt >= 0)
    tagged = alive & (tag_team >= 0)
    r = {
        "file": os.path.basename(path)[:-4],
        "possessions": len(starts),
        "mean_s": n_alive / max(len(starts), 1) / FPS,
        "D_share": float(np.mean(lab[g] == "D")),
        "X_share": float(np.mean(lab[g] == "X")),
        "home_share": float(np.mean(lab[alive] == "H")) if n_alive else float("nan"),
        "agree_nearest": float(np.mean((lab[vis] == "H") == (nt[vis] == 0))) if vis.any() else float("nan"),
        "agree_tags": float(np.mean((lab[tagged] == "H") == (tag_team[tagged] == 0))) if tagged.any() else float("nan"),
    }
    flags = []
    if not POSS_RANGE[0] <= r["possessions"] <= POSS_RANGE[1]:
        flags.append("possessions " + ("low" if r["possessions"] < POSS_RANGE[0] else "high"))
    if not D_RANGE[0] <= r["D_share"] <= D_RANGE[1]:
        flags.append("D " + ("low" if r["D_share"] < D_RANGE[0] else "high"))
    if r["X_share"] > 0:
        flags.append(f"X on {r['X_share']:.1%} of graded")
    r["flags"] = ", ".join(flags) or "ok"
    return r


def main(paths):
    paths = paths or sorted(glob.glob(os.path.join(ROOT, "evaluation", "results", "*.csv")))
    g, nt, tag_team = context()
    rows = [report(p, g, nt, tag_team) for p in paths]
    df = pd.DataFrame(rows)
    print(f"plausible: possessions {POSS_RANGE[0]}-{POSS_RANGE[1]}, dead-ball share "
          f"{D_RANGE[0]:.0%}-{D_RANGE[1]:.0%}\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
