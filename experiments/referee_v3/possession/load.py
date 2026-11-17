"""Read the provided files. Written from the formats documented in the challenge repository
(DATA.md, starter/baseline.py, starter/load_atd.py, starter/load_smart_tagging.py).

Tracking (ATD) raw line:   "<frame_id>:<player>;<player>;...:<ball>"
  each group is "x,y,speed" (players) or "x,y,..." (ball), with NaN where nothing was seen.
  frame_id counts from 1; the submission counts from 0, so video_frame = frame_id - 1.
  x and y are fractions of the pitch (0 to 1), not metres.

Tracking metadata XML: lists the two <Team>s (home first) and one <Player> per track slot, in
the same order as the groups on each raw line, each with the teamId it belongs to.

Smart Tagging XML: <instance>s with <code>, <start>, <end> (seconds from the first video frame)
and <label><group>..</group><text>..</text></label> tags such as Team and Type.
"""

import xml.etree.ElementTree as ET

import numpy as np

from . import settings

HOME, AWAY = 0, 1


def team_of_each_slot(meta_path):
    """Array over track slots: 0 if the slot belongs to the home team, 1 if away.

    Same logic as the organisers' starter/baseline.py: the first <Team> in the file is home.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in the tracking metadata, found {}".format(
            len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("no Player entries in the tracking metadata")
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots],
                    dtype=np.int8)


def read_tracking(raw_path, n_frames, n_slots):
    """Positions (0-1 fractions) of the ball and every track slot, and player speeds, per frame.

    Returns ball (n_frames, 2), players (n_frames, n_slots, 2), speeds (n_frames, n_slots),
    all float32 with NaN where nothing was seen. Row i is video frame i.

    Prints nothing from the file itself: only counts, so the output is safe to share.
    """
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speeds = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    bad_lines = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                bad_lines += 1
                continue
            try:
                frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            except ValueError:
                bad_lines += 1
                continue
            if not 0 <= frame < n_frames:
                bad_lines += 1
                continue

            # Players: "x,y,s;x,y,s;..." -> one flat list of numbers. "NaN" reads as nan.
            # Fast path: every group has exactly three numbers, so convert the line in one go.
            groups = parts[1].split(";")[:n_slots]
            flat = ",".join(groups).split(",")
            if len(flat) == 3 * len(groups):
                try:
                    values = np.array(flat, dtype=np.float32).reshape(-1, 3)
                    count = values.shape[0]
                    players[frame, :count, :] = values[:, :2]
                    speeds[frame, :count] = values[:, 2]
                except ValueError:
                    _parse_groups_slowly(groups, players[frame], speeds[frame])
            else:
                _parse_groups_slowly(groups, players[frame], speeds[frame])

            fields = parts[2].split(",")
            if len(fields) >= 2:
                try:
                    ball[frame, 0] = float(fields[0])
                    ball[frame, 1] = float(fields[1])
                except ValueError:
                    pass

    if bad_lines:
        print("  note: skipped {} tracking lines that did not match the expected format"
              .format(bad_lines))
    return ball, players, speeds


def _parse_groups_slowly(groups, player_row, speed_row):
    """One group at a time, for lines where a group does not have exactly three numbers."""
    for slot, group in enumerate(groups):
        fields = group.split(",")
        if len(fields) < 2:
            continue
        try:
            player_row[slot, 0] = float(fields[0])
            player_row[slot, 1] = float(fields[1])
            if len(fields) > 2:
                speed_row[slot] = float(fields[2])
        except ValueError:
            continue


def _side_of(team_text):
    """'H', 'A' or None from a Smart Tagging Team value ('N/A' and unknown names give None)."""
    text = (team_text or "").strip().lower()
    if any(name in text for name in settings.HOME_TAG_NAMES):
        return "H"
    if any(name in text for name in settings.AWAY_TAG_NAMES):
        return "A"
    return None


def read_tags(tag_path, fps=settings.FPS):
    """Every Smart Tagging instance as {'code', 'start', 'end', 'team', 'type'}.

    code is lower case, start and end are video frames (end is the first frame AFTER the tag, as
    in the organisers' loader), team is 'H', 'A' or None.
    """
    out = []
    for instance in ET.parse(tag_path).getroot().iter("instance"):
        tags = {}
        for label in instance.findall("label"):
            group = (label.findtext("group") or "").strip().lower()
            tags[group] = label.findtext("text")
        try:
            start = int(round(float(instance.findtext("start")) * fps))
            end = int(round(float(instance.findtext("end")) * fps))
        except (TypeError, ValueError):
            continue
        out.append({"code": (instance.findtext("code") or "").strip().lower(),
                    "start": start, "end": end,
                    "team": _side_of(tags.get("team")),
                    "type": (tags.get("type") or "").strip()})
    return out


def set_pieces_from(tags):
    """Set-piece tags only. Used for one thing: naming the restarting team when the tracking
    cannot see who took it (C9). They never override a clear taker in the tracking."""
    return [tag for tag in tags if "set piece" in tag["code"]]


def phases_from(tags):
    """Phase tags turned into 'this team has the ball' hints (E14): [(start, end, 'H' or 'A')].

    Attacking phases name the team with the ball. Defending phases name the team WITHOUT it.
    Transitions, set pieces, and tags with no team (N/A) give no hint.
    """
    other = {"H": "A", "A": "H"}
    hints = []
    for tag in tags:
        if tag["team"] is None or tag["end"] <= tag["start"]:
            continue
        if tag["code"] in settings.PHASES_TEAM_HAS_BALL:
            hints.append((tag["start"], tag["end"], tag["team"]))
        elif tag["code"] in settings.PHASES_OTHER_TEAM_HAS_BALL:
            hints.append((tag["start"], tag["end"], other[tag["team"]]))
    return hints


def read_set_pieces(tag_path, fps=settings.FPS):
    """Set-piece tags straight from the file (kept for convenience)."""
    return set_pieces_from(read_tags(tag_path, fps))
