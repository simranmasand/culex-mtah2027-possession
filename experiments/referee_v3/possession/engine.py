"""Rules A1-A3, B4-B7 and E13: who has the ball while it is alive.

How it works, in football terms:

* After a restart, the team that took it has the ball (C9).
* Possession stays with that team until the OTHER team gains control (B4: a loose ball, a
  50/50, a clearance nobody controls, a blocked or parried shot all leave it where it was).
* The other team's first touch starts a "challenge". From then on we count, for that team:
  - touches     each new spell of one of its players being on the ball
  - seconds     time on the ball, plus time the ball spends travelling between its players
                (A2: a ball in the air stays with the team that kicked it)
  - passes      the ball moving at least 5 m from one of its players to a different one
* If the team that had the ball touches it again, the challenge is cancelled: it never had
  control (B5: a touch, a deflection or an interception without control is not a change).
* When touches, seconds and passes add up to control (A1, see settings.py), possession
  switches. E13: the switch is placed at the moment control began, the first touch of the spell.

When the ball cannot be seen:

* Short gaps (up to 1 s) are already filled in with a straight line (signals.py), so the rules
  above simply carry on.
* During a longer hidden spell, the automatic phase tags are used (E14). If they clearly name one
  team as having the ball, and it is not the team we had, possession switches to that team at
  the moment the tags start saying so. If they name the team we already had, any half-finished
  challenge by the other team is dropped. If they are silent or disagree, nothing changes.
"""

import numpy as np

from . import settings
from .dead_ball import _runs

LABEL = {0: "H", 1: "A"}


def phase_masks(hints, n_frames, fps=settings.FPS):
    """Two arrays over all frames: tags say home has the ball / tags say away has it.

    The first and last PHASE_TRIM_SECONDS of every tag are left out: phase tags lag
    the real turnovers, so their edges are the least reliable part.
    """
    trim = int(round(settings.PHASE_TRIM_SECONDS * fps))
    home = np.zeros(n_frames, dtype=bool)
    away = np.zeros(n_frames, dtype=bool)
    for start, end, team in hints:
        start, end = max(0, start + trim), min(n_frames, end - trim)
        if end > start:
            (home if team == "H" else away)[start:end] = True
    return home, away


def hidden_spell_hints(sig, first, last, phase, fps=settings.FPS):
    """[(frame, team)]: for each long spell with the ball unseen where the tags clearly agree,
    the team they name and the first frame they name it."""
    if phase is None:
        return []
    home, away = phase
    shortest = int(round(settings.PHASE_MIN_HIDDEN_SECONDS * fps))
    need = int(round(settings.PHASE_MIN_EVIDENCE_SECONDS * fps))
    hints = []
    for a, b in _runs(~sig.ball_visible[first:last + 1]):
        if b - a + 1 < shortest:
            continue
        start, end = first + a, first + b
        n_home = int(home[start:end + 1].sum())
        n_away = int(away[start:end + 1].sum())
        for team, mine, theirs, mask in ((0, n_home, n_away, home), (1, n_away, n_home, away)):
            if mine >= need and theirs < settings.PHASE_MAX_CONFLICT * mine:
                hints.append((start + int(np.argmax(mask[start:end + 1])), team))
    return hints


def is_real_touch(sig, start, end, fps=settings.FPS):
    """False only for a short spell where the ball is seen to carry straight on past the player.

    A ball flying past (often over his head) puts a player within reach for a few frames without
    a touch. A real touch changes the ball's velocity: a kick, a trap, a block, a header. A long
    spell is a carry and always counts; so does any spell around which the ball is not seen.
    """
    if not settings.TOUCH_FILTER:
        return True
    if end - start + 1 >= int(round(settings.TOUCH_CARRY_SECONDS * fps)):
        return True
    k = settings.TOUCH_WINDOW_FRAMES
    if start - k < 0 or end + k >= sig.ball_xy.shape[0]:
        return True
    p_before, p_start = sig.ball_xy[start - k], sig.ball_xy[start]
    p_end, p_after = sig.ball_xy[end], sig.ball_xy[end + k]
    if np.isnan(p_before[0]) or np.isnan(p_start[0]) or np.isnan(p_end[0]) \
            or np.isnan(p_after[0]):
        return True
    v_in = (p_start - p_before) * fps / k
    v_out = (p_after - p_end) * fps / k
    if float(np.hypot(*(v_out - v_in))) >= settings.TOUCH_MIN_SPEED_CHANGE:
        return True
    speed_in, speed_out = float(np.hypot(*v_in)), float(np.hypot(*v_out))
    if min(speed_in, speed_out) >= 2.0:
        cosine = float(np.dot(v_in, v_out)) / (speed_in * speed_out)
        turn = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if turn >= settings.TOUCH_MIN_TURN_DEGREES:
            return True
    return False


def contact_episodes(sig, first, last, fps=settings.FPS):
    """[(team, slot, start, end)]: spells of one player clearly on the ball, in order.

    A blip of up to JITTER_GAP_FRAMES frames does not split a spell in two. A short spell where
    the ball just flies past the player is not a touch and is left out (is_real_touch).
    """
    teams = sig.contact_team[first:last + 1]
    slots = sig.contact_slot[first:last + 1]
    episodes = []
    current = None
    for i in np.flatnonzero(teams >= 0).tolist():
        frame, team, slot = first + i, int(teams[i]), int(slots[i])
        if (current is not None and current[0] == team and current[1] == slot
                and frame - current[3] - 1 <= settings.JITTER_GAP_FRAMES):
            current[3] = frame
        else:
            if current is not None:
                episodes.append(tuple(current))
            current = [team, slot, frame, frame]
    if current is not None:
        episodes.append(tuple(current))
    return [ep for ep in episodes if is_real_touch(sig, ep[2], ep[3], fps)]


def control_progress(passes, held_frames, touches, fps=settings.FPS):
    """How close a team is to control: 1.0 or more means it has control (rule A1).

    Any one measure reaching its target is enough on its own. Partial evidence is combined with
    weights that favour passes over seconds over touches.
    """
    p = passes / float(settings.CONTROL_PASSES)
    s = held_frames / (settings.CONTROL_SECONDS * fps)
    t = touches / float(settings.CONTROL_TOUCHES)
    combined = settings.WEIGHT_PASSES * p + settings.WEIGHT_SECONDS * s + \
        settings.WEIGHT_TOUCHES * t
    return max(p, s, t, combined)


def _ball_distance(sig, frame_a, frame_b):
    a, b = sig.ball_xy[frame_a], sig.ball_xy[frame_b]
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def possession_in_stretch(sig, first, last, restart_team, fps=settings.FPS, phase=None,
                          stats=None):
    """[(frame, team)]: the moments possession starts or changes inside one live stretch.

    phase is the pair from phase_masks(), or None to ignore the tags. stats, if given, is a dict
    whose 'switches_from_tracking' and 'switches_from_tags' counts are increased.
    """
    episodes = contact_episodes(sig, first, last, fps)
    max_transit = int(round(settings.MAX_TRANSIT_SECONDS * fps))

    # Touches (ball seen) and tag hints (ball hidden) in time order. They never share a frame.
    events = [(start, "touch", (team, slot, start, end))
              for team, slot, start, end in episodes]
    events += [(frame, "hint", team)
               for frame, team in hidden_spell_hints(sig, first, last, phase, fps)]
    events.sort(key=lambda event: event[0])

    holder = restart_team
    switches = [] if holder is None else [(first, holder)]
    challenge = None

    for frame, kind, payload in events:
        if kind == "hint":
            team = payload
            if holder is None:
                holder = team
                switches.append((first, holder))
            elif team != holder:
                holder = team
                switches.append((frame, holder))
                if stats is not None:
                    stats["switches_from_tags"] = stats.get("switches_from_tags", 0) + 1
            challenge = None
            continue

        team, slot, start, end = payload
        if holder is None:
            # Nobody could tell who restarted: the first team on the ball took the restart.
            holder = team
            switches.append((first, holder))
            continue

        if team == holder:
            challenge = None  # the team with the ball touched it again: no change (B4, B5)
            continue

        if challenge is None:
            challenge = {"start": start, "touches": 0, "held": 0, "passes": 0, "last": None}

        last_ep = challenge["last"]
        if last_ep is not None:
            gap = start - last_ep[3] - 1
            if gap <= max_transit:
                challenge["held"] += gap  # ball travelling between team-mates
            if slot != last_ep[1] and \
                    _ball_distance(sig, last_ep[3], start) >= settings.PASS_MIN_DISTANCE_M:
                challenge["passes"] += 1
        challenge["touches"] += 1

        # Walk through this spell on the ball and find the first frame of control, if any.
        held_before = challenge["held"]
        length = end - start + 1
        confirmed_at = None
        for k in range(1, length + 1):
            if control_progress(challenge["passes"], held_before + k,
                                challenge["touches"], fps) >= 1.0:
                confirmed_at = start + k - 1
                break
        challenge["held"] = held_before + length
        challenge["last"] = (team, slot, start, end)

        if confirmed_at is not None:
            switch_at = challenge["start"] if settings.BACKDATE_SWITCH else confirmed_at
            holder = team
            switches.append((switch_at, holder))
            challenge = None
            if stats is not None:
                stats["switches_from_tracking"] = stats.get("switches_from_tracking", 0) + 1

    return switches


def fill_labels(labels, first, last, switches):
    """Write H / A into labels[first..last] from the switch points. Leaves X if none."""
    switches = sorted(switches)
    for i, (frame, team) in enumerate(switches):
        until = switches[i + 1][0] if i + 1 < len(switches) else last + 1
        labels[max(frame, first):min(until, last + 1)] = LABEL[team]
