"""Whole-match sanity checks on the model's OUTPUT. Nothing here reads a ground truth.

Everything printed is a count or a duration of our own labels, so it is safe to share (it
contains no tracking rows). It answers "does this look like a football match?" using the two
published yardsticks listed in the README, which is a football reason to prefer one version over another,
unlike a score on the public 5 minutes, which the rules forbid tuning on:

  * a professional match has about 40-45 minutes of dead ball in roughly 100 minutes
    (Premier League 2025-26: 55-58 min in play), so far less dead time means dead-ball
    recall is still too low;
  * under Opta's definition a match has slightly fewer than 200 possessions, about 90-100 per
    team. Far fewer suggests the control rule is too strict; far more, too loose.
"""

import numpy as np

from . import settings
from .dead_ball import _runs

BENCH_DEAD_MINUTES = (38.0, 50.0)   # a little wider than the 40-45 quoted, to allow for rounding
BENCH_POSSESSIONS = (150, 300)      # "just under 200" by Opta's rule; tracking-based may differ


def _percentiles(values, fps):
    if not len(values):
        return "none"
    seconds = np.asarray(values, dtype=float) / fps
    p10, p50, p90 = np.percentile(seconds, [10, 50, 90])
    return "median {:.1f} s (10th {:.1f}, 90th {:.1f})".format(p50, p10, p90)


def _spans(labels, first, last, wanted):
    """Lengths in frames of each run of one of the `wanted` labels within first..last."""
    inside = np.isin(labels[first:last + 1], list(wanted))
    return [b - a + 1 for a, b in _runs(inside)]


def _possession_spans(labels, first, last):
    """[(start, end, how_it_started, how_it_ended)] for each possession (a maximal run of one
    team's label, so H H H A A is two). The two "how" words are 'stoppage' (next to a D),
    'hand-over' (next to the other team) or 'edge' (the start or end of the half, or an X)."""
    def kind(label):
        return {"D": "stoppage", "H": "hand-over", "A": "hand-over"}.get(label, "edge")

    spans = []
    for team in "HA":
        for a, b in _runs(labels[first:last + 1] == team):
            a, b = a + first, b + first
            before = kind(labels[a - 1]) if a > first else "edge"
            after = kind(labels[b + 1]) if b < last else "edge"
            spans.append((a, b, before, after))
    return spans


def _dead_reason_lines(reasons, half_blocks, fps):
    """How each dead stretch ended, and the longest ones. Times are minutes into that half of
    OUR OWN labels (no tracking data), so this is safe to share."""
    lines = []
    for half, first, last in half_blocks:
        mine = [r for r in reasons if first <= r[0] <= last]
        dead = sum(b - a + 1 for a, b, _ in mine) / fps / 60.0
        lead = [b - a + 1 for a, b, how in mine if a == first]
        lines.append("  half {}: {:.1f} min dead; first live play {:.1f} min after the half "
                     "starts".format(half, dead, (lead[0] if lead else 0) / fps / 60.0))
    lines.append("dead stretches by how they ended:")
    for how in ("kick-off", "strike", "safety net", "end of half"):
        lengths = [b - a + 1 for a, b, h in reasons if h == how]
        if lengths:
            lines.append("  {:12s} {:3d} stretches, {:5.1f} min in all, {}, longest {:.0f} s".format(
                how, len(lengths), sum(lengths) / fps / 60.0, _percentiles(lengths, fps),
                max(lengths) / float(fps)))
    lines.append("longest dead stretches (length, where, how it ended):")
    starts = {first: half for half, first, _ in half_blocks}
    for a, b, how in sorted(reasons, key=lambda r: r[0] - r[1])[:8]:
        half = max((h for f, h in starts.items() if f <= a), default=0)
        first = [f for f, h in starts.items() if h == half][0]
        lines.append("  {:6.1f} s   half {} at {:5.1f} min   {}".format(
            (b - a + 1) / float(fps), half, (a - first) / fps / 60.0, how))
    return lines


def describe(labels, half_blocks, report=None, fps=settings.FPS):
    """Lines of text about the labels over the graded halves."""
    lines = []
    graded = sum(last - first + 1 for _, first, last in half_blocks)
    dead_frames = sum(int(np.sum(labels[first:last + 1] == "D")) for _, first, last in half_blocks)
    dead_min = dead_frames / fps / 60.0
    lines.append("model: {}   graded time {:.1f} min".format(
        settings.ACTIVE, graded / fps / 60.0))
    flag = "" if BENCH_DEAD_MINUTES[0] <= dead_min <= BENCH_DEAD_MINUTES[1] else \
        "   <-- outside the usual {:.0f}-{:.0f} min".format(*BENCH_DEAD_MINUTES)
    lines.append("dead ball: {:.1f} min ({:.0f}% of graded time){}".format(
        dead_min, 100.0 * dead_frames / max(graded, 1), flag))

    all_dead, all_spans = [], []
    for half, first, last in half_blocks:
        dead = _spans(labels, first, last, "D")
        spans = _possession_spans(labels, first, last)
        all_dead += dead
        all_spans += spans
        poss = spans
        lines.append("  half {}: {} dead stretches, {} possessions".format(
            half, len(dead), len(poss)))
    all_poss = [b - a + 1 for a, b, _, _ in all_spans]
    lines.append("dead stretches: {}   length {}".format(len(all_dead), _percentiles(all_dead, fps)))
    lines.append("dead stretches under 5 s: {}".format(sum(1 for n in all_dead if n < 5 * fps)))
    flag = "" if BENCH_POSSESSIONS[0] <= len(all_poss) <= BENCH_POSSESSIONS[1] else \
        "   <-- outside the usual {}-{}".format(*BENCH_POSSESSIONS)
    lines.append("possessions: {}   length {}{}".format(
        len(all_poss), _percentiles(all_poss, fps), flag))
    short = sum(1 for n in all_poss if n < 3 * fps)
    lines.append("possessions under 3 s: {} ({:.0f}%)".format(
        short, 100.0 * short / max(len(all_poss), 1)))
    for title, chosen in (("all possessions", all_spans),
                          ("those under 3 s", [sp for sp in all_spans
                                               if sp[1] - sp[0] + 1 < 3 * fps])):
        counts = {}
        for _, _, began, ended in chosen:
            counts[(began, ended)] = counts.get((began, ended), 0) + 1
        parts_text = ", ".join("{}->{}: {}".format(b, e, c) for (b, e), c in sorted(counts.items()))
        lines.append("  {} by (start -> end): {}".format(title, parts_text or "none"))

    if report is not None and report.get("dead_reasons"):
        lines += _dead_reason_lines(report["dead_reasons"], half_blocks, fps)
    if report is not None and report.get("lineup") is not None and settings.KICKOFF_LINEUP_CHECK:
        for half, first, _ in half_blocks:
            frame = report["lineup"].get(half)
            lines.append("  half {}: teams seen lined up in their own halves {}".format(
                half, "by {:.1f} min (no kick-off before this)".format((frame - first) / fps / 60.0)
                if frame is not None else "never in the first {:.0f} min (no limit used)".format(
                    settings.KICKOFF_SEARCH_SECONDS / 60.0)))
    if report is not None and report.get("kickoff_candidates") is not None:
        lines.append("first strikes near the centre spot in each half (kick-off candidates):")
        for half, first, _ in half_blocks:
            found = report["kickoff_candidates"].get(half, [])
            if not found:
                lines.append("  half {}: none at all".format(half))
            for t, tests in found:
                lines.append("  half {} at {:5.1f} min: {}".format(
                    half, (t - first) / fps / 60.0,
                    ", ".join("{} {}".format(name, "yes" if ok else "NO")
                              for name, ok in tests.items())))

    if report is not None and report.get("parts"):
        names = {"out": "ball outside the lines", "vanished": "ball vanished after leaving",
                 "still": "ball lying still, nobody near", "break": "everyone walking, ball still",
                 "setup": "ball set up for a restart"}
        in_half = np.zeros(len(labels), dtype=bool)
        for _, first, last in half_blocks:
            in_half[first:last + 1] = True
        lines.append("frames with each kind of dead-ball evidence (they overlap):")
        for key, text in names.items():
            if key not in report["parts"]:
                continue
            lines.append("  {:32s} {:6.2f} min".format(
                text, int(np.sum(report["parts"][key] & in_half)) / fps / 60.0))
    return lines


def agreement(labels_a, labels_b, half_blocks, fps=settings.FPS):
    """Lines comparing two label arrays of OUR OWN (e.g. v1 against v2) on the graded frames."""
    lines = []
    frames = same = 0
    moved = {}
    for _, first, last in half_blocks:
        a, b = labels_a[first:last + 1], labels_b[first:last + 1]
        frames += len(a)
        same += int(np.sum(a == b))
        for x in "HAD":
            for y in "HAD":
                if x != y:
                    moved[(x, y)] = moved.get((x, y), 0) + int(np.sum((a == x) & (b == y)))
    lines.append("frames labelled the same: {:.1f}%".format(100.0 * same / max(frames, 1)))
    for (x, y), count in sorted(moved.items(), key=lambda item: -item[1])[:4]:
        lines.append("  {} -> {}: {:.1f} min".format(x, y, count / fps / 60.0))
    return lines
