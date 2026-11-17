"""Per-frame facts the rules are built on. Everything is in metres and metres per second.

For every video frame:
  contact_team   0 = a home player is on the ball, 1 = an away player is, -1 = nobody,
                 -2 = a 50/50 (both teams equally close)
  contact_slot   which track slot is on the ball (-1 if none)
  ball_seen      the tracking itself saw the ball
  ball_visible   the ball was seen, or filled in across a short gap (settings.py)
  ball_xy        ball position in metres (NaN if the ball is not seen or filled in)
  ball_speed     ball speed in m/s from one frame to the next (NaN if unknown)
  ball_speed_smooth  ball speed over the next half second, less affected by jitter (NaN if unknown)
  ball_out       the ball is clearly outside the pitch
  ball_inside    the ball is clearly inside the pitch
  ball_to_centre distance from the ball to the centre spot
  player_speed   average speed of all the players that are seen
"""

import numpy as np

from . import settings

NOBODY = -1
CONTESTED = -2


class Signals(object):
    """A plain bag of per-frame arrays, all of the same length."""

    def __init__(self, **arrays):
        self.__dict__.update(arrays)
        self.n_frames = len(arrays["contact_team"])


def compute(ball, players, speeds, slot_team, pitch, fps=settings.FPS):
    """Turn raw tracking (fractions of the pitch) into the per-frame facts above."""
    length, width = float(pitch[0]), float(pitch[1])
    scale = np.array([length, width], dtype=np.float32)
    ball_m = ball * scale
    players_m = players * scale
    ball_seen = ~np.isnan(ball_m[:, 0])  # seen by the tracking itself
    ball_m = fill_short_gaps(ball_m, int(round(settings.FILL_BALL_GAPS_SECONDS * fps)),
                             length, width, players_m, settings.GAP_SNAP_M,
                             settings.TOUCH_RADIUS_M)
    visible = ~np.isnan(ball_m[:, 0])    # seen, or filled in across a short gap

    home_slots = np.where(slot_team == 0)[0]
    away_slots = np.where(slot_team == 1)[0]
    if len(home_slots) == 0 or len(away_slots) == 0:
        raise ValueError("the tracking metadata must give each team at least one track slot")
    kickoff_shape = teams_in_own_halves(players_m, home_slots, away_slots, length, width)
    lined_up = teams_in_own_halves(players_m, home_slots, away_slots, length, width,
                                   need_players=True)

    with np.errstate(invalid="ignore"):
        # Distance from every tracked player to the ball. A missing player or ball is "infinitely
        # far", so it can never be the nearest.
        diff = players_m - ball_m[:, None, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        dist = np.where(np.isnan(dist), np.inf, dist)
        del diff

        d_home = dist[:, home_slots].min(axis=1)
        d_away = dist[:, away_slots].min(axis=1)
        slot_home = home_slots[dist[:, home_slots].argmin(axis=1)]
        slot_away = away_slots[dist[:, away_slots].argmin(axis=1)]
        del dist

        home_nearer = d_home <= d_away
        d_near = np.where(home_nearer, d_home, d_away)
        d_far = np.where(home_nearer, d_away, d_home)
        near_team = np.where(home_nearer, 0, 1).astype(np.int8)
        near_slot = np.where(home_nearer, slot_home, slot_away)

        on_ball = visible & (d_near <= settings.TOUCH_RADIUS_M)
        if settings.CONTEST_RULE == "radius":
            # B4: a player from each team within reach of the ball is a 50/50
            clearly_one_team = d_far > settings.CONTEST_RADIUS_M
        else:
            clearly_one_team = (d_far - d_near) >= settings.DUEL_MARGIN_M

    contact_team = np.full(len(ball), NOBODY, dtype=np.int8)
    contact_team[on_ball & clearly_one_team] = near_team[on_ball & clearly_one_team]
    contact_team[on_ball & ~clearly_one_team] = CONTESTED
    contact_slot = np.where(on_ball & clearly_one_team, near_slot, -1).astype(np.int16)

    # Ball speed from one frame to the next. Impossible jumps are tracking glitches.
    ball_speed = np.full(len(ball), np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        step = ball_m[1:] - ball_m[:-1]
        ball_speed[1:] = np.sqrt((step * step).sum(axis=1)) * fps
        ball_speed[ball_speed > settings.MAX_BALL_SPEED] = np.nan

        x, y = ball_m[:, 0], ball_m[:, 1]
        out = settings.OUT_OF_PLAY_MARGIN_M
        ball_out = visible & ((x < -out) | (x > length + out) | (y < -out) | (y > width + out))
        inner = settings.RESTART_INSIDE_MARGIN_M
        ball_inside = visible & (x >= inner) & (x <= length - inner) & \
            (y >= inner) & (y <= width - inner)
        ball_to_centre = np.sqrt((x - length / 2.0) ** 2 + (y - width / 2.0) ** 2)

    player_speed = _average_player_speed(speeds, players_m, fps)
    ball_speed_smooth = smooth_speed(ball_m, settings.SMOOTH_SPEED_FRAMES, fps)

    return Signals(contact_team=contact_team, contact_slot=contact_slot, ball_xy=ball_m,
                   ball_seen=ball_seen, ball_visible=visible, ball_speed=ball_speed,
                   ball_speed_smooth=ball_speed_smooth, ball_out=ball_out,
                   ball_inside=ball_inside, ball_to_centre=ball_to_centre,
                   player_speed=player_speed, d_home=d_home, d_away=d_away,
                   slot_home=slot_home, slot_away=slot_away, kickoff_shape=kickoff_shape,
                   lined_up=lined_up,
                   pitch_length=length, pitch_width=width)


def teams_in_own_halves(players_m, home_slots, away_slots, length, width=None,
                        need_players=False):
    """True on frames where the two teams stand in opposite halves, as at a kick-off (Law 8).

    Only players ON the pitch count: the track slots also hold substitutes, who warm up along the
    touchlines in either half. Either team may be on either side (they change ends at half-time),
    so both ways round are tried. A player within KICKOFF_LINE_TOLERANCE_M of the halfway line
    counts as on either side, and up to KICKOFF_MAX_WRONG_SIDE players may be on the wrong side
    (tracking errors). With too few players on the pitch to judge, the answer is True: we only
    rule out what we can see is wrong. With need_players=True it is False instead: "the teams
    are SEEN lined up", as used by the Law 8 line-up check.
    """
    half = length / 2.0
    tol = settings.KICKOFF_LINE_TOLERANCE_M
    with np.errstate(invalid="ignore"):
        x = players_m[:, :, 0]
        y = players_m[:, :, 1]
        on_pitch = (x >= 0.0) & (x <= length)
        if width is not None:
            on_pitch &= (y >= 0.0) & (y <= width)
        left = on_pitch & (x < half - tol)
        right = on_pitch & (x > half + tol)
    seen = on_pitch
    home_left, home_right = left[:, home_slots].sum(axis=1), right[:, home_slots].sum(axis=1)
    away_left, away_right = left[:, away_slots].sum(axis=1), right[:, away_slots].sum(axis=1)
    wrong = np.minimum(home_right + away_left, home_left + away_right)
    enough = (seen[:, home_slots].sum(axis=1) >= settings.KICKOFF_MIN_PLAYERS) & \
        (seen[:, away_slots].sum(axis=1) >= settings.KICKOFF_MIN_PLAYERS)
    ok = wrong <= settings.KICKOFF_MAX_WRONG_SIDE
    return (enough & ok) if need_players else (~enough | ok)


def smooth_speed(ball_m, window, fps=settings.FPS):
    """Ball speed in m/s from where the ball is now to where it is `window` frames later.

    Tracking jitter makes the frame-to-frame speed of a ball lying still read several m/s, while
    the net movement over about half a second is small. Looking forward (not centred) means a
    ball that has come to rest reads as still from the first frame it is at rest, so a stoppage
    is not found late. NaN when either end is not on record.
    """
    n = len(ball_m)
    speed = np.full(n, np.nan, dtype=np.float32)
    if window <= 0 or n <= window:
        return speed
    with np.errstate(invalid="ignore"):
        step = ball_m[window:] - ball_m[:n - window]
        speed[:n - window] = np.sqrt((step * step).sum(axis=1)) * fps / float(window)
    return speed


def fill_short_gaps(ball_m, max_frames, length, width, players_m=None, snap_m=0.0,
                    owner_radius_m=1.5):
    """Fill gaps of up to max_frames missing frames with a straight line between the sightings
    on either side, but only when both sightings are on the pitch. Longer gaps stay missing.

    With players_m given and snap_m > 0: if a player who was NOT on the ball at either
    sighting (further than owner_radius_m from it) comes within snap_m of the straight line, the
    ball is put at that player at his closest moment, and the gap is filled with two straight
    lines through him instead: the hidden touch is kept. One snap per gap, the closest player.
    """
    filled = ball_m.copy()
    if max_frames <= 0:
        return filled
    seen = np.flatnonzero(~np.isnan(ball_m[:, 0]))
    if len(seen) < 2:
        return filled
    missing = np.diff(seen) - 1
    snapping = players_m is not None and snap_m > 0
    for k in np.flatnonzero((missing > 0) & (missing <= max_frames)).tolist():
        a, b = int(seen[k]), int(seen[k + 1])
        pa, pb = ball_m[a], ball_m[b]
        if not (_on_pitch(pa, length, width) and _on_pitch(pb, length, width)):
            continue
        filled[a + 1:b] = _line(pa, pb, b - a)
        if snapping:
            touch = _hidden_touch(filled, players_m, a, b, snap_m, owner_radius_m)
            if touch is not None:
                f, spot = touch
                filled[f] = spot
                if f - a > 1:
                    filled[a + 1:f] = _line(pa, spot, f - a)
                if b - f > 1:
                    filled[f + 1:b] = _line(spot, pb, b - f)
    return filled


def _line(p0, p1, steps):
    """The points strictly between p0 and p1 on a straight line cut into `steps` equal steps."""
    fraction = (np.arange(1, steps, dtype=np.float32) / float(steps))[:, None]
    return p0 + (p1 - p0) * fraction


def _hidden_touch(filled, players_m, a, b, snap_m, owner_radius_m):
    """(frame, player position) of the closest non-owner player to the filled line, or None."""
    with np.errstate(invalid="ignore"):
        def dist(frame_rows, points):
            d = np.sqrt(((frame_rows - points[:, None, :]) ** 2).sum(axis=2))
            return np.where(np.isnan(d), np.inf, d)

        owners = (dist(players_m[[a, b]], filled[[a, b]]) <= owner_radius_m).any(axis=0)
        d = dist(players_m[a + 1:b], filled[a + 1:b])     # (gap frames, slots)
    d[:, owners] = np.inf
    if d.size == 0 or not np.isfinite(d).any():
        return None
    i, slot = np.unravel_index(int(np.argmin(d)), d.shape)
    if d[i, slot] > snap_m:
        return None
    return a + 1 + int(i), players_m[a + 1 + int(i), slot].copy()


def _on_pitch(point, length, width):
    return 0.0 <= point[0] <= length and 0.0 <= point[1] <= width


def _average_player_speed(speeds, players_m, fps):
    """Mean speed of the players seen on each frame, NaN if nobody is seen.

    Uses the speeds in the tracking file. If that column turns out to be empty, falls back to
    speeds worked out from positions, ignoring jumps too fast to be a person (identity switches).
    """
    valid = ~np.isnan(speeds)
    if not valid.any():
        speeds = np.full(speeds.shape, np.nan, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            step = players_m[1:] - players_m[:-1]
            speeds[1:] = np.sqrt((step * step).sum(axis=2)) * fps
            speeds[speeds > 12.0] = np.nan
        valid = ~np.isnan(speeds)

    total = np.where(valid, speeds, 0.0).sum(axis=1)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
