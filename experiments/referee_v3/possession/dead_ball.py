"""Rules C8 and C9: when the ball is dead, when play restarts, and who restarts it.

Each half is cut into alternating stretches:

  ("dead", first_frame, last_frame)
  ("live", first_frame, last_frame, restarting_team)   team is 0 home, 1 away, or None

A half begins dead (waiting for the kick-off) and goes live at the kick-off.
The ball goes dead when there is evidence of a stoppage (see dead_evidence): the ball is clearly
out of the pitch, or has vanished near a line, or lies still with nobody near it, or everyone
slows to walking pace and the ball stops (a whistle for a foul, offside, injury, VAR...), or
(v3) it sits on a spot being set up for a restart that is then taken.
It comes back to life when the ball is struck inside the pitch (v3: from rest, with a taker
beside it, or tapped off the centre spot), or, as a safety net, once players are clearly
running again.

No frame numbers of this match are written into the code: the kick-offs, like every other
restart, are found from the tracking.
"""

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import settings


def _runs(mask):
    """[(first, last)] of each block of consecutive True values in a boolean array."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist()))


def _join_and_filter(mask, bridge, shortest):
    """Blocks of True in `mask`, with blocks separated by at most `bridge` frames joined into
    one (the gap is filled in) and blocks shorter than `shortest` frames dropped."""
    joined = []
    for first, last in _runs(mask):
        if joined and first - joined[-1][1] - 1 <= bridge:
            joined[-1][1] = last
        else:
            joined.append([first, last])
    out = np.zeros(len(mask), dtype=bool)
    for first, last in joined:
        if last - first + 1 >= shortest:
            out[first:last + 1] = True
    return out


def _ball_speed(sig):
    """The ball speed used to judge "is the ball still?", smoothed from v2 on so jitter does not fool it."""
    if settings.SMOOTH_BALL_SPEED:
        return sig.ball_speed_smooth
    return sig.ball_speed


def break_in_play(sig, fps=settings.FPS):
    """True on frames inside a visible break: players at walking pace and the ball still."""
    speed = _ball_speed(sig)
    with np.errstate(invalid="ignore"):
        players_slow = sig.player_speed < settings.BREAK_PLAYER_SPEED
        ball_still = (~sig.ball_visible) | np.isnan(speed) | (speed < settings.BREAK_BALL_SPEED)
    slow = players_slow & ball_still

    shortest = int(round(settings.BREAK_MIN_SECONDS * fps))
    bridge = int(round(settings.BREAK_BRIDGE_SECONDS * fps))
    return _join_and_filter(slow, bridge, shortest)


def vanished_after_exit(sig, fps=settings.FPS):
    """True on frames where the ball has dropped out of the tracking after leaving through,
    or right beside, a line, and stays gone for VANISH_MIN_SECONDS (so it went out of play).

    A ball last seen heading back into the pitch is not counted: it is more likely hidden by
    players than out. Judged in hindsight, so the first unseen frame is the one that is marked.
    """
    out = np.zeros(sig.n_frames, dtype=bool)
    if not settings.VANISH_NEAR_LINE:
        return out
    length, width = sig.pitch_length, sig.pitch_width
    need = int(round(settings.VANISH_MIN_SECONDS * fps))
    back = settings.TOUCH_WINDOW_FRAMES
    for first, last in _runs(~sig.ball_visible):
        if first == 0 or last - first + 1 < need:
            continue
        x, y = (float(v) for v in sig.ball_xy[first - 1])
        # Distance to each line and the direction that points OUT of the pitch there.
        sides = ((x, (-1.0, 0.0)), (length - x, (1.0, 0.0)),
                 (y, (0.0, -1.0)), (width - y, (0.0, 1.0)))
        gap, normal = min(sides, key=lambda side: side[0])
        if gap > settings.VANISH_NEAR_LINE_M:
            continue
        earlier = sig.ball_xy[first - 1 - back] if first - 1 - back >= 0 else None
        if earlier is not None and not np.isnan(earlier[0]):
            vx, vy = (x - float(earlier[0])) * fps / back, (y - float(earlier[1])) * fps / back
            if vx * normal[0] + vy * normal[1] < -settings.VANISH_MAX_INWARD_SPEED:
                continue  # heading back inside: hidden, not out
        out[first:last + 1] = True
    return out


def out_of_play_evidence(sig, fps=settings.FPS):
    """True on frames where the ball is out of play: outside the lines (and staying out), or
    vanished just after leaving through a line."""
    raw = sig.ball_out.copy()
    if settings.OUT_MIN_SECONDS <= 0 and not settings.VANISH_NEAR_LINE:
        return raw  # v1: every frame outside the margin counts
    raw |= vanished_after_exit(sig, fps)
    if settings.OUT_MIN_SECONDS <= 0:
        return raw
    bridge = int(round(settings.OUT_BRIDGE_SECONDS * fps))
    shortest = max(1, int(round(settings.OUT_MIN_SECONDS * fps)))
    return _join_and_filter(raw, bridge, shortest)


def still_ball_evidence(sig, fps=settings.FPS):
    """True on frames inside a spell where the ball stays put and no player is near it.

    For every 2 s window: enough of it is seen, the ball's positions all fit inside a small box,
    and no player ever comes within a metre. A stretch of such windows is marked from the first
    frame of the first window to the last frame of the last.
    """
    n = sig.n_frames
    need = max(1, int(round(settings.STILL_BALL_MIN_SECONDS * fps)))
    covered = np.zeros(n, dtype=bool)
    if not settings.STILL_BALL or n < need:
        return covered
    x = np.where(sig.ball_visible, sig.ball_xy[:, 0], np.nan).astype(np.float64)
    y = np.where(sig.ball_visible, sig.ball_xy[:, 1], np.nan).astype(np.float64)
    nearest = np.minimum(sig.d_home, sig.d_away).astype(np.float64)
    wx, wy, wn = (sliding_window_view(a, need) for a in (x, y, nearest))
    seen = np.isfinite(wx).sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)   # all-unseen windows are all-NaN
        spread = np.maximum(np.nanmax(wx, axis=1) - np.nanmin(wx, axis=1),
                            np.nanmax(wy, axis=1) - np.nanmin(wy, axis=1))
        valid = (seen >= settings.STILL_BALL_MIN_SEEN * need) &             (spread <= settings.STILL_BALL_RANGE_M) & (wn.min(axis=1) > settings.STILL_BALL_CLEAR_M)
    starts = np.zeros(n + 1, dtype=np.int64)
    starts[np.flatnonzero(valid)] = 1
    running = np.cumsum(starts)[:n]
    behind = np.concatenate((np.zeros(need, dtype=np.int64), running[:n - need]))
    return (running - behind) > 0


def _windows(sig, size):
    """For every window of `size` frames (window i = frames i .. i+size-1), over the frames where
    the ball is on record: how many there are, how far apart they spread (the larger side of the
    box around them, in metres) and their mean position. Arrays of length n - size + 1."""
    x = np.where(sig.ball_visible, sig.ball_xy[:, 0], np.nan).astype(np.float64)
    y = np.where(sig.ball_visible, sig.ball_xy[:, 1], np.nan).astype(np.float64)
    wx, wy = sliding_window_view(x, size), sliding_window_view(y, size)
    seen = np.isfinite(wx).sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)   # all-unseen windows are all-NaN
        spread = np.maximum(np.nanmax(wx, axis=1) - np.nanmin(wx, axis=1),
                            np.nanmax(wy, axis=1) - np.nanmin(wy, axis=1))
        mean_x, mean_y = np.nanmean(wx, axis=1), np.nanmean(wy, axis=1)
    return seen, spread, mean_x, mean_y


def _in_penalty_area(sig, x, y):
    with np.errstate(invalid="ignore"):
        deep = (x <= settings.PENALTY_AREA_DEPTH_M) | \
            (x >= sig.pitch_length - settings.PENALTY_AREA_DEPTH_M)
        return deep & (np.abs(y - sig.pitch_width / 2.0) <= settings.PENALTY_AREA_HALF_WIDTH_M)


def _taker_team(sig, frame):
    """The team of the player nearest the ball, within RESTART_TAKER_RADIUS_M, in the few frames
    up to `frame` (0 home, 1 away), or None if nobody is that close."""
    best = None
    for f in range(max(0, frame - settings.RESTART_TAKER_LOOKBACK_FRAMES),
                   min(sig.n_frames, frame + 1)):
        if not sig.ball_visible[f]:
            continue
        d_home, d_away = float(sig.d_home[f]), float(sig.d_away[f])
        nearest = min(d_home, d_away)
        if nearest <= settings.RESTART_TAKER_RADIUS_M and (best is None or nearest < best[0]):
            best = (nearest, 0 if d_home <= d_away else 1)
    return None if best is None else best[1]


def _strike_starts(struck):
    return np.flatnonzero(struck & ~np.concatenate(([False], struck[:-1])))


def setup_evidence(sig, fps=settings.FPS, struck=None):
    """True on the set-up of a restart we can recognise, even if the stoppage before it
    was missed. A set-up is the ball lying on one spot for RESTART_SETUP_SECONDS, outside the
    penalty areas, then struck by a player beside it, with no opponent of his within
    SETUP_OPPONENT_CLEAR_M of the ball in the last second (Laws 13 and 15 keep opponents back).
    The whole stretch the ball lay there is marked, up to the frame before the strike.
    """
    n = sig.n_frames
    out = np.zeros(n, dtype=bool)
    size = int(round(settings.RESTART_SETUP_SECONDS * fps))
    if not settings.RESTART_BACKTRACK or n <= size:
        return out
    if struck is None:
        struck = ball_struck(sig, fps)
    seen, spread, mean_x, mean_y = _windows(sig, size)
    with np.errstate(invalid="ignore"):
        valid = (seen >= settings.STILL_BALL_MIN_SEEN * size) & \
            (spread <= settings.STILL_BALL_RANGE_M) & ~_in_penalty_area(sig, mean_x, mean_y)
    chain_start = np.full(len(valid), -1, dtype=np.int64)
    for first, last in _runs(valid):
        chain_start[first:last + 1] = first
    check = int(round(settings.SETUP_OPPONENT_CHECK_SECONDS * fps))

    for t in _strike_starts(struck).tolist():
        window = t - size
        if window < 0 or not valid[window]:
            continue
        team = _taker_team(sig, t)
        if team is None:
            continue
        opponents = sig.d_away if team == 0 else sig.d_home
        recent = slice(max(0, t - check), t)
        if (opponents[recent][sig.ball_visible[recent]] <= settings.SETUP_OPPONENT_CLEAR_M).any():
            continue  # an opponent right on the ball: live play, not a set piece
        out[int(chain_start[window]):t] = True
    return out


def _settled_before(sig, fps=settings.FPS):
    """True where the ball was at rest (within a 1 m box) in the half second before the frame,
    or was barely seen then, so we cannot tell."""
    n = sig.n_frames
    ready = np.ones(n, dtype=bool)
    size = max(1, int(round(settings.RESTART_SETTLE_SECONDS * fps)))
    if n > size:
        seen, spread, _, _ = _windows(sig, size)
        with np.errstate(invalid="ignore"):
            at_rest = (seen < settings.STILL_BALL_MIN_SEEN * size) | \
                (spread <= settings.RESTART_SETTLE_RANGE_M)
        ready[size:] &= at_rest[:n - size]   # frame t looks at frames t-size .. t-1
    return ready


def restart_ready(sig, fps=settings.FPS):
    """True on frames where a strike can be a restart: the ball was at rest just before
    (or was barely seen, so we cannot tell) and a player is beside it. All True when off."""
    ready = np.ones(sig.n_frames, dtype=bool)
    if settings.RESTART_SETTLED_CHECK:
        ready &= _settled_before(sig, fps)
    if settings.RESTART_NEEDS_TAKER:
        ready &= _taker_near(sig)
    return ready


def kickoff_candidates(sig, first, last, struck, fps=settings.FPS, limit=6):
    """For the first `limit` strikes near the centre spot in frames first..last:
    [(frame, {test: passed})], whether or not each test is switched on. Used by diagnostics.py
    to say why a half's kick-off was or was not accepted."""
    settled = _settled_before(sig, fps)
    taker = _taker_near(sig)
    found = []
    for t in _strike_starts(struck[first:last + 1]).tolist():
        t += first
        if not _near_centre(sig, t):
            continue
        shape = getattr(sig, "kickoff_shape", None)
        found.append((t, {
            "ball at rest before": bool(settled[t]),
            "taker within {} m".format(settings.RESTART_TAKER_RADIUS_M): bool(taker[t]),
            "teams in own halves": True if shape is None else bool(shape[max(t - 1, 0)]
                                                                    or shape[t]),
        }))
        if len(found) >= limit:
            break
    return found


def kickoff_nudge(sig, fps=settings.FPS, struck=None):
    """Soft taps: True on frames t where a ball at rest near the centre spot (in the half
    second before t) is, half a second later (on average over 6 frames), at least
    KICKOFF_NUDGE_M from where it lay, with a player beside it. Judged on averages so that
    tracking jitter cannot fake it. Hard kick-offs are left to the strike, which is exact."""
    n = sig.n_frames
    out = np.zeros(n, dtype=bool)
    settle = max(1, int(round(settings.RESTART_SETTLE_SECONDS * fps)))
    later, after = settle, 6
    if not settings.KICKOFF_NUDGE or n <= settle + later + after:
        return out
    seen, spread, spot_x, spot_y = _windows(sig, settle)
    seen_after, _, then_x, then_y = _windows(sig, after)
    t = np.arange(settle, n - later - after + 1)
    before, ahead = t - settle, t + later
    with np.errstate(invalid="ignore"):
        at_rest = (seen[before] >= settings.STILL_BALL_MIN_SEEN * settle) & \
            (spread[before] <= settings.RESTART_SETTLE_RANGE_M)
        centre = np.hypot(spot_x[before] - sig.pitch_length / 2.0,
                          spot_y[before] - sig.pitch_width / 2.0) <= settings.KICKOFF_RADIUS_M
        moved = (seen_after[ahead] >= after // 2) & \
            (np.hypot(then_x[ahead] - spot_x[before], then_y[ahead] - spot_y[before])
             >= settings.KICKOFF_NUDGE_M)
    out[t] = at_rest & centre & moved
    # A hard kick-off is timed by its strike instead, which is exact: no nudge in the stretch
    # looked at here if the ball is struck during it.
    if struck is None:
        struck = ball_struck(sig, fps)
    total = np.concatenate(([0], np.cumsum(struck)))
    upto = np.minimum(np.arange(n) + later + after, n)
    out &= (total[upto] - total[np.arange(n)]) == 0
    return out & _taker_near(sig)


def _taker_near(sig):
    """True where a player was within RESTART_TAKER_RADIUS_M of the ball in the last few frames."""
    n = sig.n_frames
    with np.errstate(invalid="ignore"):
        near = sig.ball_visible & (np.minimum(sig.d_home, sig.d_away)
                                   <= settings.RESTART_TAKER_RADIUS_M)
    back = settings.RESTART_TAKER_LOOKBACK_FRAMES
    total = np.concatenate(([0], np.cumsum(near)))
    start = np.maximum(np.arange(n) - back, 0)
    return (total[np.arange(n) + 1] - total[start]) > 0


def dead_evidence(sig, fps=settings.FPS, parts=None, struck=None):
    """True on every frame where there is evidence the ball is not in play.

    If `parts` is a dict it is filled with one boolean array per kind of evidence ("out",
    "vanished", "still", "break", "setup"), which diagnostics.py uses to say what found the stops.
    """
    breaks = break_in_play(sig, fps)
    out = out_of_play_evidence(sig, fps)
    still = still_ball_evidence(sig, fps)
    setup = setup_evidence(sig, fps, struck)
    if parts is not None:
        parts.update({"out": out, "vanished": vanished_after_exit(sig, fps) & out,
                      "still": still, "break": breaks, "setup": setup})
    return breaks | out | still | setup


def ball_struck(sig, fps=settings.FPS):
    """True on frames where the ball is inside the pitch and moving fast for a few frames.

    In v2 it must also be going somewhere: a ball lying still whose tracked position jitters
    reads as fast frame after frame but covers no ground, so the distance actually travelled
    must be at least RESTART_STRAIGHTNESS of the distance walked frame by frame.
    """
    with np.errstate(invalid="ignore"):
        fast = sig.ball_inside & (sig.ball_speed >= settings.RESTART_BALL_SPEED)
    k = settings.RESTART_SUSTAIN_FRAMES
    n = sig.n_frames
    total = np.concatenate(([0], np.cumsum(fast)))
    out = np.zeros(n, dtype=bool)
    if n < k:
        return out
    out[:n - k + 1] = (total[k:] - total[:-k]) == k

    if settings.RESTART_STRAIGHTNESS > 0 and n > k:
        steps = np.nan_to_num(sig.ball_speed, nan=0.0) / float(fps)
        walked_total = np.concatenate(([0.0], np.cumsum(steps)))
        walked = walked_total[k:] - walked_total[:-k]            # frames t .. t+k-1
        net = np.full(n - k + 1, np.nan)
        with np.errstate(invalid="ignore"):
            move = sig.ball_xy[k:] - sig.ball_xy[:n - k]          # position t+k-1 minus t-1
            net[1:] = np.sqrt((move * move).sum(axis=1))
            straight = net >= settings.RESTART_STRAIGHTNESS * walked
        out[:n - k + 1] &= straight
    return out


def restarting_team(sig, frame, set_pieces, fps=settings.FPS):
    """Rule C9: the team that takes the restart has the ball. 0 home, 1 away, None unknown.

    First choice: the player clearly nearest the ball as it is struck (the taker).
    Second choice: the team on an automatic set-piece tag around that moment.
    """
    best = None
    first = max(0, frame - settings.RESTART_TAKER_LOOKBACK_FRAMES)
    for f in range(first, min(sig.n_frames, frame + 1)):
        if not sig.ball_visible[f]:
            continue
        d_home, d_away = float(sig.d_home[f]), float(sig.d_away[f])
        nearest = min(d_home, d_away)
        clear = abs(d_home - d_away) >= settings.DUEL_MARGIN_M
        if nearest <= settings.RESTART_TAKER_RADIUS_M and clear:
            if best is None or nearest < best[0]:
                best = (nearest, 0 if d_home < d_away else 1)
    if best is not None:
        return best[1]

    window = int(round(settings.RESTART_TAG_WINDOW_SECONDS * fps))
    nearby = []
    for tag in set_pieces:
        if tag["team"] is None:
            continue
        if tag["start"] - window <= frame <= tag["end"] + window:
            # how far the restart is from the tag's interval (0 if inside it)
            gap = max(tag["start"] - frame, frame - tag["end"], 0)
            nearby.append((gap, tag["team"]))
    if nearby:
        return 0 if min(nearby)[1] == "H" else 1
    return None


def play_resumed(sig, fps=settings.FPS):
    """resume_from[t] = s: by frame t the players have been running since frame s, for
    RESUME_MIN_SECONDS, with the ball never out. -1 on every other frame.

    v1 marked only the single frame at which the running had lasted long enough. v2 marks every
    frame from then until the running stops, so the stoppage can be judged old enough later.
    """
    with np.errstate(invalid="ignore"):
        running = (sig.player_speed >= settings.RESUME_PLAYER_SPEED) & ~sig.ball_out
    need = int(round(settings.RESUME_MIN_SECONDS * fps))
    resume_from = np.full(sig.n_frames, -1, dtype=np.int64)
    for first, last in _runs(running):
        if last - first + 1 >= need:
            if settings.SAFETY_NET_PERSISTENT:
                resume_from[first + need - 1:last + 1] = first
            else:
                resume_from[first + need - 1] = first
    return resume_from


def _near_centre(sig, t):
    before, now = sig.ball_to_centre[max(t - 1, 0)], sig.ball_to_centre[t]
    near = [d for d in (before, now) if not np.isnan(d)]
    return bool(near) and min(near) <= settings.KICKOFF_RADIUS_M


def _kickoff_shape(sig, t):
    """the teams stand in opposite halves just before (or at) frame t."""
    shape = getattr(sig, "kickoff_shape", None)
    if not settings.KICKOFF_HALVES_CHECK or shape is None:
        return True
    return bool(shape[max(t - 1, 0)] or shape[t])


def _ball_lets_net_fire(sig):
    """a function (a, b) -> may the safety net restart play somewhere in frames a..b?
    Yes if the ball was never seen then (we cannot tell), or was seen moving inside the pitch."""
    if not settings.SAFETY_NET_BALL_MOVING:
        return lambda a, b: True
    with np.errstate(invalid="ignore"):
        moving = sig.ball_inside & (sig.ball_speed_smooth >= settings.BREAK_BALL_SPEED)
    seen_total = np.concatenate(([0], np.cumsum(sig.ball_visible)))
    moving_total = np.concatenate(([0], np.cumsum(moving)))

    def allows(a, b):
        if seen_total[b + 1] - seen_total[a] == 0:
            return True
        return moving_total[b + 1] - moving_total[a] > 0
    return allows


def _run_ends(mask):
    """ends[t] = last frame of the block of True containing t, or -1 where mask[t] is False."""
    ends = np.full(len(mask), -1, dtype=np.int64)
    for first, last in _runs(mask):
        ends[first:last + 1] = last
    return ends


def split_half(sig, first, last, set_pieces, fps=settings.FPS, evidence=None, struck=None,
               resume_from=None, starts=None, reasons=None):
    """Cut one half (video frames first..last) into dead and live stretches.

    The half starts dead and goes live at the kick-off: the ball struck (or, in v3, nudged) near
    the centre spot with the teams in opposite halves. If no such kick-off is found anywhere in
    the half (v3: within its first KICKOFF_SEARCH_SECONDS), it is tried again accepting any
    restart, so a kick-off the tracking missed cannot leave the half dead until a later one.

    evidence is dead_evidence(sig); struck is ball_struck(sig); resume_from is play_resumed(sig);
    starts is restart_starts(sig, struck). Each is worked out here if not given.

    If `reasons` is a list, one (dead_start, dead_end, how_it_ended) is added to it per dead
    stretch, how_it_ended being "kick-off", "strike", "safety net" or "end of half". That is
    what diagnostics.py uses to say why play stays dead.
    """
    if struck is None:
        struck = ball_struck(sig, fps)
    if evidence is None:
        evidence = dead_evidence(sig, fps, struck=struck)
    if resume_from is None:
        resume_from = play_resumed(sig, fps)
    if starts is None:
        starts = restart_starts(sig, struck, fps)

    not_before = lineup_frame(sig, first, last, fps)
    why = []
    stretches = _split(sig, first, last, set_pieces, fps, evidence, starts, resume_from,
                       strict_kickoff=True, reasons=why, not_before=not_before)
    lives = [stretch[1] for stretch in stretches if stretch[0] == "live"]
    search = int(round(settings.KICKOFF_SEARCH_SECONDS * fps))
    too_late = bool(lives) and search > 0 and lives[0] > first + search
    if not lives or too_late:
        why = []
        stretches = _split(sig, first, last, set_pieces, fps, evidence, starts, resume_from,
                           strict_kickoff=False, reasons=why, not_before=not_before)
    if reasons is not None:
        reasons.extend(why)
    return stretches


def lineup_frame(sig, first, last, fps=settings.FPS):
    """Law 8: the first frame at which the teams have been SEEN lined up in their own halves for
    KICKOFF_LINEUP_SECONDS, within the kick-off search window of this half. The opening kick-off
    cannot come before it. None if the check is off or no line-up is seen in the window."""
    lined = getattr(sig, "lined_up", None)
    if not settings.KICKOFF_LINEUP_CHECK or lined is None:
        return None
    search = int(round(settings.KICKOFF_SEARCH_SECONDS * fps))
    end = last if search <= 0 else min(last, first + search)
    need = max(1, int(round(settings.KICKOFF_LINEUP_SECONDS * fps)))
    for a, b in _runs(lined[first:end + 1]):
        if b - a + 1 >= need:
            return first + a + need - 1
    return None


def restart_starts(sig, struck, fps=settings.FPS):
    """Frames that can restart play: a strike from a ball at rest with a taker beside it,
    or a soft tap at the centre spot. In v1 and v2 this is every strike."""
    return (struck & restart_ready(sig, fps)) | kickoff_nudge(sig, fps, struck)


def _split(sig, first, last, set_pieces, fps, evidence, struck, resume_from, strict_kickoff,
           reasons=None, not_before=None):
    """`struck` is the frames that can restart play (restart_starts, or ball_struck in v1/v2).
    `not_before` (lineup_frame) is the earliest frame the opening kick-off can be, or None."""
    if reasons is None:
        reasons = []
    not_before = first if not_before is None else not_before
    min_dead = int(round(settings.MIN_DEAD_SECONDS * fps))
    safety_dead = int(round(settings.SAFETY_NET_MIN_DEAD_SECONDS * fps))
    run_end = _run_ends(evidence)
    net_allowed = _ball_lets_net_fire(sig)
    stretches = []
    dead, dead_start, waiting_for_kickoff = True, first, True
    live_start, team = None, None
    ignore_until = -1   # after a restart inside a stretch of evidence, that stretch is not new

    for t in range(first, last + 1):
        if not dead:
            if evidence[t] and t > ignore_until:
                stretches.append(("live", live_start, t - 1, team))
                dead, dead_start = True, t
            continue

        restart, how = None, None
        strict = waiting_for_kickoff and strict_kickoff
        if waiting_for_kickoff and t < not_before:
            continue   # Law 8: the teams have not lined up yet
        if struck[t]:
            if strict:
                if _near_centre(sig, t) and _kickoff_shape(sig, t):
                    restart, how = t, "kick-off"
            elif waiting_for_kickoff or t - dead_start >= min_dead:
                restart, how = t, "kick-off" if waiting_for_kickoff else "strike"
        if restart is None and not strict and resume_from[t] >= 0:
            run_start = int(resume_from[t])
            if settings.SAFETY_NET_PERSISTENT:
                earliest = max(dead_start + 1, not_before) if waiting_for_kickoff \
                    else dead_start + safety_dead
                candidate = max(run_start, earliest)
                if candidate <= t and candidate > dead_start and net_allowed(candidate, t):
                    restart, how = candidate, "safety net"
            elif run_start > dead_start and \
                    (waiting_for_kickoff or run_start - dead_start >= min_dead):
                restart, how = run_start, "safety net"  # players running since then: play on
        if restart is None:
            continue

        if restart > dead_start:
            stretches.append(("dead", dead_start, restart - 1))
            reasons.append((dead_start, restart - 1, how))
        dead, waiting_for_kickoff = False, False
        live_start, team = restart, restarting_team(sig, restart, set_pieces, fps)
        ignore_until = max(int(run_end[min(restart, last)]), int(run_end[t]))

    if dead:
        stretches.append(("dead", dead_start, last))
        reasons.append((dead_start, last, "end of half"))
    else:
        stretches.append(("live", live_start, last, team))
    return stretches
