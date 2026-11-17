"""Parse the tactical ATD once into numpy arrays indexed by VIDEO frame, cached under data/cache/.

    from experiments.atd_cache import load
    d = load()
    d["players"]       # float32 (199319, 46, 3): x m, y m, speed m/s -- NaN where missing
    d["ball"]          # float32 (199319, 2):     x m, y m             -- NaN where missing
    d["team_of_slot"]  # int8 (46,): 0 = home (H), 1 = away (A)
    d["half"]          # int8 (199319,): 1 / 2 on graded frames, 0 elsewhere
    d["graded"]        # bool (199319,)

Coordinates are converted from the ATD's 0-1 to metres with games.GAMES[...]["pitch"]
(105.3 x 68.0). Row i is video frame i == ATD frame_id i + 1 == submission row i.

The cache lives inside data/, which is gitignored: frame-level tracking derived from the match
must never be committed (RULES.md).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

CACHE_PATH = os.path.join(DATA_DIR, "cache", "atd_tactical.npz")
N_SLOTS = 46


def _parse(raw_path, n_frames, pitch):
    length, width = pitch
    players = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, slots, ball_txt = line.rstrip("\n").split(":")
            row = int(frame_id) - 1
            values = np.array(slots.replace(";", ",").split(","), dtype=np.float32)
            players[row] = values.reshape(N_SLOTS, 3)
            bx, by, _ = ball_txt.split(",")  # ball speed is always NaN in this export
            ball[row] = (float(bx), float(by))
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return players, ball


def build(game=GAME):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    players, ball = _parse(raw_path, n_frames, GAMES[game]["pitch"])
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH, players=players, ball=ball,
             team_of_slot=team_of_each_track(meta_path), half=half)
    return CACHE_PATH


def load(rebuild=False):
    if rebuild or not os.path.exists(CACHE_PATH):
        build()
    with np.load(CACHE_PATH) as npz:
        out = {key: npz[key] for key in npz.files}
    out["graded"] = out["half"] > 0
    return out


if __name__ == "__main__":
    d = load(rebuild="--rebuild" in sys.argv)
    present = ~np.isnan(d["ball"][:, 0])
    print("players", d["players"].shape, "ball", d["ball"].shape)
    print("ball present: all {:.1%}  graded {:.1%}  ({} / {})".format(
        present.mean(), present[d["graded"]].mean(),
        int(present[d["graded"]].sum()), int(d["graded"].sum())))
