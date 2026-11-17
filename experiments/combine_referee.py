"""Combine the GRU's labels with the Referee Model's dead-ball calls (A6).

    python experiments/combine_referee.py                       # all variants, report only
    python experiments/combine_referee.py --dead --out F.csv    # (b) only, write the file
    python experiments/combine_referee.py --dead --restart --out F.csv

Inputs: the GRU file (default evaluation/results/nn_a4.csv, the D19 final) and the Referee Model
export (data/cache/referee_dead_ball.csv, from experiments/referee_v3/export_dead_ball.py).

  (b) --dead     where the Referee says dead and the stretch's dead_cause is out / vanished /
                 break / setup -> D. Frames whose cause is only "still" (or before_kickoff) stay
                 with the GRU; X frames stay X.
  (c) --restart  on each Referee restart frame, the restart team gets the following frames until
                 the GRU's own next change of label (or a frame (b) made D).
Afterwards the published pre-kickoff rule (D17) is re-applied, so (c) can never undo it.
Prints, per variant: dead-ball minutes, possessions, share of graded frames changed vs the GRU.
Only counts are printed. No BAR-ATH labels or public ground truth are read.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "evaluation", "results")
DEAD_CAUSES = ("out", "vanished", "break", "setup")
FPS = 25


def combine(gru, dead, cause, restart, restart_team, use_dead=True, use_restart=True):
    """gru: (n,) H/A/D/X labels. Referee columns as in export_dead_ball.py. Returns new labels."""
    out = gru.copy()
    graded = gru != "X"
    made_dead = np.zeros(len(gru), dtype=bool)
    if use_dead:
        made_dead = (dead == 1) & np.isin(cause, DEAD_CAUSES) & graded
        out[made_dead] = "D"
    if use_restart:
        n = len(gru)
        for r in np.flatnonzero(restart == 1).tolist():
            team = restart_team[r]
            if not graded[r] or team not in ("H", "A"):
                continue
            g, j = gru[r], r
            while j < n and gru[j] == g and graded[j] and not made_dead[j]:
                out[j] = team
                j += 1
    return out


def possessions(labels):
    """Runs of H or A (a change of team or a D/X between them ends a run)."""
    lab = np.asarray(labels)
    starts = (lab[1:] != lab[:-1]) & np.isin(lab[1:], ("H", "A"))
    return int(starts.sum() + (lab[0] in ("H", "A")))


def describe(name, labels, gru):
    graded = gru != "X"
    return {"variant": name, "dead_min": float((labels[graded] == "D").sum() / FPS / 60),
            "possessions": possessions(labels),
            "changed_pct": float(100 * (labels[graded] != gru[graded]).mean())}


VARIANTS = {"gru": (False, False), "dead": (True, False), "restart": (False, True), "both": (True, True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru", default=os.path.join(RES, "nn_a4.csv"))
    ap.add_argument("--referee", default=os.path.join(ROOT, "data", "cache", "referee_dead_ball.csv"))
    ap.add_argument("--dead", action="store_true", help="(b) use the Referee's dead-ball calls")
    ap.add_argument("--restart", action="store_true", help="(c) use the Referee's restart team")
    ap.add_argument("--out", help="write the chosen variant (--dead/--restart) to this file")
    args = ap.parse_args()

    sys.path[:0] = [os.path.join(HERE, "final"), HERE, ROOT, os.path.join(ROOT, "starter")]
    from reproduce_final import force_pre_kickoff_dead

    gru = pd.read_csv(args.gru)["label"].astype(str).str.strip().to_numpy()
    ref = pd.read_csv(args.referee, keep_default_na=False)
    cols = (ref["dead"].to_numpy(), ref["dead_cause"].astype(str).to_numpy(),
            ref["restart"].to_numpy(), ref["restart_team"].astype(str).to_numpy())
    assert len(ref) == len(gru), "row counts differ"

    rows = []
    for name, (d, r) in VARIANTS.items():
        lab = combine(gru, *cols, use_dead=d, use_restart=r)
        force_pre_kickoff_dead(lab)
        rows.append(describe(name, lab, gru))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if args.out:
        lab = combine(gru, *cols, use_dead=args.dead, use_restart=args.restart)
        force_pre_kickoff_dead(lab)
        pd.DataFrame({"frame": np.arange(len(lab)), "label": lab}).to_csv(args.out, index=False)
        print(f"wrote {args.out} (dead={args.dead}, restart={args.restart})")


if __name__ == "__main__":
    main()
