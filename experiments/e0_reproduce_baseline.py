"""E0: rebuild baseline_smooth50 from the cached arrays and check it matches the shipped file exactly.

    python experiments/e0_reproduce_baseline.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from atd_cache import load  # noqa: E402
from baseline import smooth, NOBODY  # noqa: E402
from possession_metrics import read_label_csv  # noqa: E402


def nearest_player_codes(d):
    """Vectorised version of starter/baseline.nearest_player_labels."""
    diff = d["players"][:, :, :2] - d["ball"][:, None, :]
    dist = np.einsum("fsk,fsk->fs", diff, diff)
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = dist.argmin(axis=1)
    codes = d["team_of_slot"][nearest].astype(np.int8)
    no_answer = np.isnan(d["ball"][:, 0]) | np.isinf(dist.min(axis=1))
    codes[no_answer] = NOBODY
    return codes


if __name__ == "__main__":
    d = load()
    codes = nearest_player_codes(d)
    for name, window in (("baseline_raw", 0), ("baseline_smooth25", 25),
                         ("baseline_smooth50", 50), ("baseline_smooth125", 125)):
        ours = smooth(codes, window)
        shipped = read_label_csv(os.path.join(ROOT, "evaluation", "results", name + ".csv"))
        shipped = shipped["label"].str.strip().str.upper().to_numpy(dtype=str)
        n_diff = int(np.sum(ours != shipped))
        print("{:<20} frames differing from shipped file: {}".format(name, n_diff))
