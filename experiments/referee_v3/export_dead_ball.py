"""Write the Referee Model's dead-ball calls as a per-frame table, ready to combine with another
model (for example the GRU).

    python export_dead_ball.py                       # -> evaluation/results/referee_dead_ball.csv
    python export_dead_ball.py --out some/file.csv

Run it from the root of the challenge repository (next to games.py), like make_submission.py.
One row per video frame, 199,319 rows:

    frame          video frame, 0-based (same numbering as the submission file)
    label          the Referee Model's label: H, A, D, or X outside the graded halves
    dead           1 if the Referee Model says the ball is dead, else 0
    dead_cause     why the dead stretch this frame belongs to started:
                     out          ball seen past a touchline or goal line
                     vanished     ball lost from the tracking just after leaving through a line
                     break        everyone walking with the ball still (a whistle)
                     still        ball lying still with nobody near it
                     setup        ball set down for a restart that was then taken
                     before_kickoff   start of a half, before the kick-off
                   empty when the ball is live
    evidence_now   the kinds of dead-ball evidence seen on this exact frame, joined by "+"
                   (a stretch stays dead after its evidence ends, until play restarts)
    restart        1 on the first frame of play after a dead stretch, else 0
    restart_how    on restart frames: kick-off, strike (ball struck from rest with a taker
                   beside it) or safety net (no kick seen; players clearly running again)
    restart_team   on restart frames: H or A, the team that took it

Only counts and labels of our own model are written; no tracking values. The file is as safe to
handle as the submission file itself.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CAUSES = ("out", "vanished", "break", "still", "setup")   # order of precedence on a frame


def dead_ball_columns(labels, report, half_blocks):
    """The columns described above, as a dict of equal-length arrays."""
    n = len(labels)
    parts = report.get("parts", {})
    dead = (labels == "D").astype(np.int8)

    evidence_now = np.full(n, "", dtype=object)
    for cause in CAUSES:
        mask = parts.get(cause)
        if mask is None:
            continue
        hit = np.asarray(mask, dtype=bool)
        evidence_now[hit] = np.where(evidence_now[hit] == "", cause, evidence_now[hit] + "+" + cause)

    dead_cause = np.full(n, "", dtype=object)
    restart = np.zeros(n, dtype=np.int8)
    restart_how = np.full(n, "", dtype=object)
    restart_team = np.full(n, "", dtype=object)
    half_starts = {first for _, first, _ in half_blocks}
    for start, end, how in report.get("dead_reasons", []):
        if start in half_starts:
            cause = "before_kickoff"
        else:
            first = str(evidence_now[start]).split("+")[0]
            cause = first or "out"
        dead_cause[start:end + 1] = cause
        after = end + 1
        if how != "end of half" and after < n and labels[after] in ("H", "A"):
            restart[after] = 1
            restart_how[after] = how
            restart_team[after] = labels[after]
    return {"frame": np.arange(n), "label": labels, "dead": dead, "dead_cause": dead_cause,
            "evidence_now": evidence_now, "restart": restart, "restart_how": restart_how,
            "restart_team": restart_team}


def write_table(path, columns):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    names = list(columns)
    with open(path, "w", newline="\n") as handle:
        handle.write(",".join(names) + "\n")
        for row in zip(*(columns[name] for name in names)):
            handle.write(",".join(str(value) for value in row) + "\n")
    return path


def main():
    from games import GAME, half_video_blocks
    from possession import pipeline, settings

    parser = argparse.ArgumentParser(description="Export the Referee Model's dead-ball calls.")
    parser.add_argument("--model", default=settings.DEFAULT, choices=sorted(settings.PRESETS))
    parser.add_argument("--out", default=os.path.join(HERE, "evaluation", "results",
                                                      "referee_dead_ball.csv"))
    args = parser.parse_args()

    settings.use(args.model)
    labels, report = pipeline.build(GAME)
    blocks = half_video_blocks(GAME)
    columns = dead_ball_columns(labels, report, blocks)
    path = write_table(args.out, columns)

    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    dead = columns["dead"].astype(bool) & graded
    print("\nwrote {} ({} rows)".format(os.path.relpath(path, HERE), len(labels)))
    print("dead ball: {:.1f} min of {:.1f} graded".format(dead.sum() / 1500.0, graded.sum() / 1500.0))
    for cause in ("before_kickoff",) + CAUSES:
        minutes = np.sum(dead & (columns["dead_cause"] == cause)) / 1500.0
        if minutes:
            print("  started by {:15s} {:5.1f} min".format(cause, minutes))
    print("restarts: {}".format(int(columns["restart"].sum())))


if __name__ == "__main__":
    main()
