"""Make the submission file: one label per video frame of the tactical video.

Put this file and the possession/ folder in the root of your copy of the challenge repository
(next to games.py), then:

    python make_submission.py --team YOUR_TEAM_NAME
    python evaluation/validate_submission.py evaluation/results/YOUR_TEAM_NAME.csv
    python evaluation/run_evaluation.py

The match data must already be in ./data, or MTAH_DATA_DIR must point at it (see DATA.md).
Everything printed here describes the output, never the match data itself.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from games import GAME  # noqa: E402  (games.py comes from the challenge repository)
from possession import diagnostics, pipeline, settings  # noqa: E402


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def apply_overrides(pairs, parser):
    """Apply each "NAME=VALUE" to the settings. VALUE is read as a Python number, True/False or
    text, and must be the same kind as the setting it replaces. Returns [(name, value)]."""
    import ast
    done = []
    for pair in pairs:
        name, sep, text = pair.partition("=")
        name, text = name.strip().upper(), text.strip()
        if not sep or not name or not text:
            parser.error("--set needs NAME=VALUE, got {!r}".format(pair))
        if not hasattr(settings, name) or name.startswith("_") or name in (
                "PRESETS", "DEFAULT", "ACTIVE", "V1", "V2", "V3"):
            parser.error("--set: there is no setting called {} in possession/settings.py".format(
                name))
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = text
        old = getattr(settings, name)
        if isinstance(old, bool) != isinstance(value, bool) or \
                not isinstance(value, (type(old), int) if isinstance(old, float) else type(old)):
            parser.error("--set {}: {!r} is not the same kind of value as the current {!r}".format(
                name, value, old))
        if isinstance(old, float):
            value = float(value)
        setattr(settings, name, value)
        done.append((name, value))
    if done:
        settings.ACTIVE = "{} with {}".format(settings.ACTIVE, ", ".join(
            "{}={}".format(name, value) for name, value in done))
    return done


def main():
    parser = argparse.ArgumentParser(description="Make the possession submission file.")
    parser.add_argument("--team", required=True,
                        help="your team name exactly as registered; the file is named after it")
    parser.add_argument("--model", default=settings.DEFAULT, choices=sorted(settings.PRESETS),
                        help="v3 (default) is the final Referee Model; v2 and v1 "
                             "reproduce the earlier versions exactly")
    parser.add_argument("--compare", metavar="CSV",
                        help="another submission file of OUR OWN (e.g. the v1 one) to compare "
                             "with; never a ground truth")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="change one setting from possession/settings.py on top of --model, "
                             "e.g. --set BREAK_PLAYER_SPEED=1.0 (repeat for several)")
    parser.add_argument("--out", default=os.path.join(HERE, "evaluation", "results"),
                        help="folder to write <team>.csv into")
    args = parser.parse_args()

    settings.use(args.model)
    for name, value in apply_overrides(args.set, parser):
        print("setting {} = {!r} (instead of the {} value)".format(name, value, args.model))
    labels, report = pipeline.build(GAME)
    path = write_submission(os.path.join(args.out, "{}.csv".format(args.team)), labels)

    info = pipeline.summary(labels)
    print("\nwrote {}".format(os.path.relpath(path, HERE)))
    print("rows: {} (must be 199319)".format(len(labels)))
    print("frames  H {H}  A {A}  D {D}  X {X}".format(**info["counts"]))
    print("possessions found: {}, average length {:.1f} s".format(
        info["possessions"], info["mean_possession_seconds"]))
    print("stretches: {} dead, {} live ({} live with no team found)".format(
        report["dead_stretches"], report["live_stretches"], report["live_without_team"]))
    print("possession changes: {} from the tracking, {} from phase tags (ball hidden)".format(
        report["switches_from_tracking"], report["switches_from_tags"]))
    from games import half_video_blocks
    blocks = half_video_blocks(GAME)
    print("\nsanity checks on the output (counts only, safe to share):")
    for line in diagnostics.describe(labels, blocks, report):
        print("  " + line)
    if args.compare:
        import numpy as np
        with open(args.compare) as handle:
            other = np.array([row.strip().split(",")[1] for row in handle.readlines()[1:]])
        if len(other) != len(labels):
            print("  --compare file has {} rows, expected {}".format(len(other), len(labels)))
        else:
            print("\nagainst {}:".format(os.path.basename(args.compare)))
            for line in diagnostics.agreement(other, labels, blocks):
                print("  " + line)

    print("\nnext: python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, HERE)))


if __name__ == "__main__":
    main()
