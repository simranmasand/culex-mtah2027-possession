"""How wrong is the straight-line fill of missing ball positions, by gap length?

    python tests/selfcheck_gap_fill.py

Needs the real tracking (it reads the same files as make_submission.py). It uses NO labels and
no ground truth: it takes stretches where the ball WAS seen, pretends a few frames in the middle
were missing, fills them the way the model does, and measures how far the fill lands from where
the tracking actually saw the ball. That turns "is 1 s too long to fill?" into a measurement
instead of a guess, and it is allowed because it is learning from the tracking alone.

Prints only summary numbers (metres), never a position, so the output is safe to share.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402
from possession import load, settings, signals  # noqa: E402

GAPS_SECONDS = (0.2, 0.5, 1.0, 2.0)
SAMPLES = 4000
SEED = 0


def main():
    paths = asset_paths(GAME)
    n_frames = source_frames("tactical", GAME)
    fps = GAMES[GAME]["fps"]
    length, width = GAMES[GAME]["pitch"]
    slot_team = load.team_of_each_slot(paths["atd_meta"])
    ball, _, _ = load.read_tracking(paths["atd_raw"], n_frames, len(slot_team))
    ball_m = ball * np.array([length, width], dtype=np.float32)
    seen = ~np.isnan(ball_m[:, 0])
    rng = np.random.RandomState(SEED)

    print("filled position error against the position the tracking actually saw (metres)")
    print("  gap      samples   median    90th    share within 1 m")
    for seconds in GAPS_SECONDS:
        gap = int(round(seconds * fps))
        # a window of gap + 2 frames that is fully seen, so the answer is known
        run = np.convolve(seen.astype(np.int32), np.ones(gap + 2, dtype=np.int32), mode="valid")
        starts = np.flatnonzero(run == gap + 2)
        if len(starts) == 0:
            print("  {:.1f} s    no fully seen stretch that long".format(seconds))
            continue
        starts = rng.choice(starts, size=min(SAMPLES, len(starts)), replace=False)
        errors = []
        for s in starts.tolist():
            trial = ball_m[s:s + gap + 2].copy()
            truth = trial[1:-1].copy()
            trial[1:-1] = np.nan
            filled = signals.fill_short_gaps(trial, gap, length, width)
            errors.append(np.hypot(*(filled[1:-1] - truth).T))
        errors = np.concatenate(errors)
        errors = errors[~np.isnan(errors)]
        print("  {:.1f} s    {:6d}   {:6.2f}   {:6.2f}   {:5.1f}%".format(
            seconds, len(starts), np.median(errors), np.percentile(errors, 90),
            100.0 * np.mean(errors <= 1.0)))
    print("\nsetting in use: FILL_BALL_GAPS_SECONDS = {}".format(settings.FILL_BALL_GAPS_SECONDS))


if __name__ == "__main__":
    main()
