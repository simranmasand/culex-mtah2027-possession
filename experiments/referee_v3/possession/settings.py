"""Every number the model uses, in one place, in plain English.

Each setting names the football rule it comes from (A1, B4, C8 and so on: the numbering of the
rules the team wrote down before any code existed). None of these numbers was chosen by looking
at the match data or the public ground truth. They come from football sense, the Laws of the Game
and published studies (see the README for the reading list). That is what the competition rules
ask for: change a number only for a football reason, never because the public 5-minute score
went up.

Three presets exist:

    v1  the first version, exactly as scored on the public 5 minutes. Kept so its file can be
        regenerated and compared against.
    v2  v1 with tighter dead-ball and touch rules from published studies.
    v3  the final rules model (the "Referee Model"), active by default. Stricter restarts,
        restart set-ups, the Law 8 kick-off checks and the team's own control rule A1
        (3 s / 3 touches / 7 passes, which v2 had loosened).

`use("v1")`, `use("v2")` or `use("v3")` switches every setting that differs between them.
Anything not listed in the presets at the bottom is the same in all three.
"""

import sys

FPS = 25  # frames per second of the tactical video. 25 frames = 1 second.

# ------------------------------------------------------------------------------------------------
# A1 / B5: when does the other team "have" the ball?
# A team takes possession only once it has CONTROL. Control is any ONE of these on its own:
CONTROL_SECONDS = 3.0   # keeping the ball for this long (v2: 1.5)
CONTROL_TOUCHES = 3     # this many touches (v2: 2)
CONTROL_PASSES = 7      # this many passes between team-mates (v2: 1)
#
# These are the team's own answer to A1 ("at least 3-4 s, 3-4 touches or 7-10 passes"). v2 had
# lowered them to Opta's "one controlled action"; v3 goes back to A1, which avoids many short
# false switches.
#
# ...or a MIX of them. Each one is turned into "how far along" it is (1 of 3 touches = 0.33),
# then they are added together with these weights. Passes count most, then seconds, then touches,
# exactly as the team asked. When the weighted total reaches 1.0 the team has control.
WEIGHT_PASSES = 1.0
WEIGHT_SECONDS = 0.8
WEIGHT_TOUCHES = 0.6

# ------------------------------------------------------------------------------------------------
# How a "touch" and a "pass" are recognised in tracking data (there is no touch data, only
# positions, so these are how we translate the football idea into geometry).
TOUCH_RADIUS_M = 1.5       # v1: 2.0. A player within this distance of the ball is "on the ball".
                           # Published methods use 0.5-1.5 m (Vidal-Codina et al.).
JITTER_GAP_FRAMES = 3      # a blip shorter than this does not split one touch into two
PASS_MIN_DISTANCE_M = 5.0  # the ball must travel at least 5 m between team-mates to be a pass
MAX_TRANSIT_SECONDS = 3.0  # ball travelling between two team-mates for up to 3 s still counts
                           # as "keeping the ball" (A2: ball in the air stays with the kicker)
#
# B4: a 50/50 is nobody's ball. How a 50/50 is recognised:
#   "margin"  (v1, v2) both teams are close and the nearest players are within DUEL_MARGIN_M of
#             each other's distance.
#   "radius"  (v3) a player from EACH team is within CONTEST_RADIUS_M of the ball. This
#             is the published "duel zone" (Vidal-Codina et al.: two opponents within 1 m).
CONTEST_RULE = "radius"
CONTEST_RADIUS_M = 1.0
DUEL_MARGIN_M = 0.5        # the "margin" rule, and how much nearer a restart taker must be than
                           # the nearest opponent to decide which team took it
#
# A ball flying past a player (often high over his head, which flat tracking cannot see) puts him
# "within 1.5 m" for a few frames without him touching it. A short spell on the ball only counts
# as a touch if the ball visibly changed speed or direction, as published detectors require
#. A long spell (a carry) always counts. If the ball is not seen around the spell,
# it counts: we only drop a spell we can positively see was a fly-past.
TOUCH_FILTER = True
TOUCH_CARRY_SECONDS = 0.6       # a spell this long is a carry, no check needed
TOUCH_WINDOW_FRAMES = 5         # ball velocity is measured over 5 frames before and after
TOUCH_MIN_SPEED_CHANGE = 3.0    # m/s change in ball velocity that proves a touch...
TOUCH_MIN_TURN_DEGREES = 25.0   # ...or a turn of this much while the ball is moving at 2+ m/s

# ------------------------------------------------------------------------------------------------
# E13: possession switches at the moment the new team got control.
# True: once we are sure they have control, move the switch back to their FIRST touch of that
# spell. False: switch only at the moment we became sure.
BACKDATE_SWITCH = True

# ------------------------------------------------------------------------------------------------
# C8: when is the ball dead?
OUT_OF_PLAY_MARGIN_M = 0.3  # v1: 0.5. Ball more than this outside the touchline or goal line.
                            # The ball is about 0.11 m across, so fully out is ~0.11 m.
#
# A real stoppage keeps the ball out for many seconds. A ball that only LOOKS out (a high ball
# projected off the pitch by flat tracking, or one jittery reading) is back in within a moment.
# So the ball only counts as out if it stays out for this long (in hindsight, so the dead stretch
# still starts at the first frame it went out), with gaps shorter than OUT_BRIDGE_SECONDS ignored
# (the ball lying on the line flickers in and out).
OUT_MIN_SECONDS = 1.0           # v1: 0 (a single frame was enough)
OUT_BRIDGE_SECONDS = 0.5
#
# Published work treats a ball that disappears from the data as a dead-ball signal (Vidal-Codina
# et al.). If the ball was last seen within VANISH_NEAR_LINE_M of a line, was
# not heading back into the pitch, and then stays unseen for VANISH_MIN_SECONDS, it went out.
VANISH_NEAR_LINE = True         # v1: False
VANISH_NEAR_LINE_M = 1.5
VANISH_MIN_SECONDS = 2.0
VANISH_MAX_INWARD_SPEED = 1.0   # m/s towards the pitch that still counts as "not coming back"
#
# A ball that stays within about a metre of one spot for 2 s, with nobody within 1 m of it, is not
# in play: real play never leaves the ball alone like that. Catches stoppages
# where players keep jogging. Judged by where the ball stays, not by its speed, so tracking jitter
# and missing frames do not break it up: at least STILL_BALL_MIN_SEEN of the window must be seen.
STILL_BALL = True               # v1: False
STILL_BALL_RANGE_M = 1.0        # the ball's positions all fit inside a box this big
STILL_BALL_CLEAR_M = 1.0
STILL_BALL_MIN_SECONDS = 2.0
STILL_BALL_MIN_SEEN = 0.6
#
# Frame-to-frame ball speed is dominated by tracking jitter: a ball lying still can read several
# m/s. Whether the ball is "still" is therefore judged from where it is now and where it is half
# a second later (v1 used the single-frame speed).
SMOOTH_BALL_SPEED = True        # v1: False
SMOOTH_SPEED_FRAMES = 12        # compare the position now with the one 12 frames (0.48 s) later
#
# "Break in play" (a whistle for a foul, offside, injury, VAR...). We cannot hear the whistle, so
# we look for what a break looks like: everyone slows down AND the ball is not moving.
BREAK_PLAYER_SPEED = 1.0    # average player speed below this (walking pace). v3 also tried 1.5 m/s,
                            # but over the whole match that doubled this rule's dead time (19 to
                            # 36 min) and pushed total dead time past the 40-45 min a real match
                            # has: at 1.5 m/s slow build-up play looks like a stoppage. So the
                            # final v3 keeps 1.0 m/s.
BREAK_BALL_SPEED = 1.0      # ...and the ball moving slower than 1 m/s (or not seen)...
BREAK_MIN_SECONDS = 3.0     # ...for at least 3 seconds in a row
BREAK_BRIDGE_SECONDS = 0.4  # v1: 0. A flicker shorter than this (a jittery ball reading) does
                            # not end a break.
#
# Working backwards from a restart. A free kick, corner or throw-in has a recognisable
# set-up: the ball sits on one spot for a few seconds, the taker beside it, and the other team
# kept back (Law 13: 9.15 m at free kicks and corners; Law 15: 2 m at throw-ins), then it is
# struck. When we find that pattern, the set-up time was dead even if we missed the stoppage
# itself. To keep live play out of it (a defender standing on the ball, a keeper holding it):
#   * the ball must sit within a 1 m box (STILL_BALL_RANGE_M) for RESTART_SETUP_SECONDS,
#   * no player of the team that then strikes it may have an OPPONENT within
#     SETUP_OPPONENT_CLEAR_M of the ball during the last second of the set-up,
#   * the spot must be outside both penalty areas, where a keeper holding the ball is live play
#     (goal kicks are found anyway, because the ball went out over the goal line first).
RESTART_BACKTRACK = True            # v1, v2: False
RESTART_SETUP_SECONDS = 3.0
SETUP_OPPONENT_CLEAR_M = 2.0        # the smallest distance the Laws give (throw-ins)
SETUP_OPPONENT_CHECK_SECONDS = 1.0
PENALTY_AREA_DEPTH_M = 16.5         # Law 1
PENALTY_AREA_HALF_WIDTH_M = 20.16   # Law 1: 40.32 m wide
#
# C8 / C9: when does play restart?
RESTART_BALL_SPEED = 3.0       # the ball is struck: faster than 3 m/s...
RESTART_SUSTAIN_FRAMES = 5     # v1: 3. ...for this many frames in a row...
RESTART_STRAIGHTNESS = 0.7     # v1: 0 (off). ...and actually travelling: the net distance covered
                               # is at least this share of the path walked. A still ball whose
                               # tracked position jitters has speed but goes nowhere.
RESTART_INSIDE_MARGIN_M = 0.3  # ...while inside the pitch
MIN_DEAD_SECONDS = 3.0         # v1, v2: 2. No restart within 3 s of the stoppage. Even
                               # the quickest team averages 12.7 s on a throw-in (Opta).
#
# a restart is taken from a ball at rest, by a taker beside it. Without these checks a
# ball rolled or passed back to the taker during the stoppage looks like the restart itself.
#   * in the RESTART_SETTLE_SECONDS before the strike, the ball's positions fit in a 1 m box
#     (if the ball is barely seen then, we give it the benefit of the doubt);
#   * a player is within RESTART_TAKER_RADIUS_M of the ball as it is struck.
RESTART_SETTLED_CHECK = True   # v1, v2: False
RESTART_SETTLE_SECONDS = 0.5
RESTART_SETTLE_RANGE_M = 1.0
RESTART_NEEDS_TAKER = True     # v1, v2: False
#
# Kick-off (Law 8): the ball is on the centre mark and every player is in his own half.
KICKOFF_RADIUS_M = 3.0         # the ball is struck within this distance of the centre spot
KICKOFF_HALVES_CHECK = True    # v1, v2: False. Also check the two teams are in opposite halves
                               #, so a stray kick near the centre cannot start a half.
KICKOFF_LINE_TOLERANCE_M = 1.0 # a player this close to the halfway line counts as on either side
KICKOFF_MAX_WRONG_SIDE = 2     # players allowed on the wrong side (tracking errors, slot swaps)
KICKOFF_MIN_PLAYERS = 6        # with fewer players seen than this per team, skip the check
#
# The half's OPENING kick-off comes within a few minutes of the start of its video block (the
# teams walk out and line up). If the tracking does not show it, a later kick-off (after a goal)
# must not be taken for it, or everything before that goal is labelled dead. So the strict
# kick-off is only looked for in the first KICKOFF_SEARCH_SECONDS; if none is found there, the
# half starts at the first restart or the safety net instead. 0 = search the whole half (v1, v2).
KICKOFF_SEARCH_SECONDS = 300.0
#
# Law 8: before the opening kick-off both teams line up in their own halves. When the kick-off
# itself is not seen, the fallback (first restart or safety net) is therefore not allowed before
# the tracking shows the teams lined up (at least KICKOFF_MIN_PLAYERS of each team on the pitch,
# at most KICKOFF_MAX_WRONG_SIDE on the wrong side) for KICKOFF_LINEUP_SECONDS in a row, within
# the search window above. If the line-up is never seen there, there is no such limit.
KICKOFF_LINEUP_CHECK = True    # v1, v2: False
KICKOFF_LINEUP_SECONDS = 3.0
#
# Soft taps: a kick-off is often a gentle tap that never reaches 3 m/s. A ball at rest
# within KICKOFF_RADIUS_M of the centre spot that is then nudged KICKOFF_NUDGE_M away, with a
# player beside it, also restarts play. Only at the centre spot: anywhere else a ball being nudged
# into place for a free kick would look the same.
KICKOFF_NUDGE = True           # v1, v2: False
KICKOFF_NUDGE_M = 0.5
#
# Safety net, for when the kick of a restart is not seen (the ball is often missing): play has
# clearly resumed once the players are running again, on average at least 2 m/s, for 3 seconds,
# with the ball not out. The restart is then placed where that running started. Without this, one
# missed restart would leave the rest of the half "dead".
RESUME_PLAYER_SPEED = 2.0
RESUME_MIN_SECONDS = 3.0
#
# v1 could only use the net at the single moment the running had lasted 3 s. v2 lets it fire at
# any moment while the running continues, once the stoppage is SAFETY_NET_MIN_DEAD_SECONDS old.
# v3 waits 30 s, since average stoppages last 15-37 s (Opta) and players shuffling for
# a corner or free kick can look like running; and if the ball is seen in that time it must be
# inside the pitch and moving at some point (a ball seen lying still means play has not resumed).
SAFETY_NET_PERSISTENT = True        # v1: False
SAFETY_NET_MIN_DEAD_SECONDS = 30.0  # v2: 10. (only used when SAFETY_NET_PERSISTENT)
SAFETY_NET_BALL_MOVING = True       # v1, v2: False
#
# C9: who restarts? The player nearest the ball when it is struck (the taker)...
RESTART_TAKER_RADIUS_M = 1.5        # v1, v2: 3.0
RESTART_TAKER_LOOKBACK_FRAMES = 5
# ...or, if no taker is clear, the team named on an automatic set-piece tag near that moment.
RESTART_TAG_WINDOW_SECONDS = 5.0
HOME_TAG_NAMES = ("barcelona",)            # how the tags spell the home club (DATA.md)
AWAY_TAG_NAMES = ("atletic", "athletic")   # the tags spell it "Atletic Club" (DATA.md)

# ------------------------------------------------------------------------------------------------
# When the ball cannot be seen (about 4 frames in 10)
#
# 1. Short gaps: over a second or less a ball travels in a roughly straight line, so a gap of up
#    to 1 s between two sightings inside the pitch is filled with a straight line between them.
FILL_BALL_GAPS_SECONDS = 1.0   # 0 switches this off
#
#    the main error of a straight line is a touch hidden inside the gap. If another
#    player (not the one on the ball at either end of the gap) comes within GAP_SNAP_M of the
#    line, the ball is put at that player at his closest moment and the line bends there, so the
#    touch is seen (Kim et al. do the same). The team chose 0.5 m, the tightest published
#    possession zone, so only a player right on the ball's path counts. 0 switches it off.
GAP_SNAP_M = 0.5               # v1, v2: 0
#
# 2. Longer gaps (E14): the automatic phase tags say which team is attacking. A phase where a team
#    builds up, progresses, creates chances, plays a long ball, scores or has the ball in the
#    final third or box means THAT team has the ball. "Defending in ... third" means the OTHER
#    team has it. Transitions say nothing new: the team that had the ball keeps it until control
#    switches (E14). Set pieces are already used for restarts.
USE_PHASE_TAGS = True
PHASES_TEAM_HAS_BALL = ("build up", "progression", "creating chances", "ball in final third",
                        "ball in the box", "players in the box", "long ball", "goals")
PHASES_OTHER_TEAM_HAS_BALL = ("defending in attacking third", "defending in middle third",
                              "defending in defensive third")
#
# A hidden spell only changes possession when the tags clearly agree, because they are automatic
# and can be wrong:
PHASE_MIN_HIDDEN_SECONDS = 1.0    # the ball has been unseen for at least 1 s
PHASE_MIN_EVIDENCE_SECONDS = 1.0  # tags name the same team for at least 1 s of that spell
PHASE_MAX_CONFLICT = 0.25         # and tags naming the other team cover under a quarter as much
#
# phase tags lag real turnovers (Li and Link 2026), so the first and last second of
# each tag is ignored. A tag shorter than twice this is ignored completely.
PHASE_TRIM_SECONDS = 1.0          # v1, v2: 0

# ------------------------------------------------------------------------------------------------
# Tracking clean-up
MAX_BALL_SPEED = 40.0  # m/s. Faster than any real kick, so a jump like that is a tracking glitch

# ------------------------------------------------------------------------------------------------
# Presets. Every setting whose value differs between any two versions is listed in all three.
V1 = {
    "CONTROL_SECONDS": 3.0, "CONTROL_TOUCHES": 3, "CONTROL_PASSES": 7,
    "TOUCH_RADIUS_M": 2.0, "TOUCH_FILTER": False, "CONTEST_RULE": "margin",
    "OUT_OF_PLAY_MARGIN_M": 0.5, "OUT_MIN_SECONDS": 0.0,
    "VANISH_NEAR_LINE": False, "STILL_BALL": False, "SMOOTH_BALL_SPEED": False,
    "BREAK_PLAYER_SPEED": 1.0, "BREAK_BRIDGE_SECONDS": 0.0, "RESTART_BACKTRACK": False,
    "RESTART_SUSTAIN_FRAMES": 3, "RESTART_STRAIGHTNESS": 0.0, "MIN_DEAD_SECONDS": 2.0,
    "RESTART_SETTLED_CHECK": False, "RESTART_NEEDS_TAKER": False, "RESTART_TAKER_RADIUS_M": 3.0,
    "KICKOFF_HALVES_CHECK": False, "KICKOFF_NUDGE": False, "KICKOFF_SEARCH_SECONDS": 0.0,
    "KICKOFF_LINEUP_CHECK": False,
    "SAFETY_NET_PERSISTENT": False, "SAFETY_NET_MIN_DEAD_SECONDS": 10.0,
    "SAFETY_NET_BALL_MOVING": False,
    "GAP_SNAP_M": 0.0, "PHASE_TRIM_SECONDS": 0.0,
}
V2 = dict(V1, **{
    "CONTROL_SECONDS": 1.5, "CONTROL_TOUCHES": 2, "CONTROL_PASSES": 1,
    "TOUCH_RADIUS_M": 1.5, "TOUCH_FILTER": True,
    "OUT_OF_PLAY_MARGIN_M": 0.3, "OUT_MIN_SECONDS": 1.0,
    "VANISH_NEAR_LINE": True, "STILL_BALL": True, "SMOOTH_BALL_SPEED": True,
    "BREAK_BRIDGE_SECONDS": 0.4,
    "RESTART_SUSTAIN_FRAMES": 5, "RESTART_STRAIGHTNESS": 0.7,
    "SAFETY_NET_PERSISTENT": True,
})
V3 = dict(V2, **{
    "CONTROL_SECONDS": 3.0, "CONTROL_TOUCHES": 3, "CONTROL_PASSES": 7,
    "CONTEST_RULE": "radius",
    "BREAK_PLAYER_SPEED": 1.0,
    "KICKOFF_LINEUP_CHECK": True,       # Law 8: no opening kick-off before the teams line up
    "RESTART_BACKTRACK": True,
    "RESTART_SETTLED_CHECK": True, "RESTART_NEEDS_TAKER": True,
    "RESTART_TAKER_RADIUS_M": 1.5, "KICKOFF_NUDGE": True,
    "SAFETY_NET_MIN_DEAD_SECONDS": 30.0, "SAFETY_NET_BALL_MOVING": True,
    "KICKOFF_HALVES_CHECK": True,
    "KICKOFF_SEARCH_SECONDS": 300.0,      # opening kick-off within 5 min of the half's start
    "MIN_DEAD_SECONDS": 3.0,
    "GAP_SNAP_M": 0.5,
    "PHASE_TRIM_SECONDS": 1.0,
})
PRESETS = {"v1": V1, "v2": V2, "v3": V3}
DEFAULT = "v3"
ACTIVE = DEFAULT   # which preset use() last applied


def use(name):
    """Switch every preset-controlled setting to the named preset ("v1", "v2" or "v3")."""
    if name not in PRESETS:
        raise ValueError("unknown preset {!r}; have {}".format(name, sorted(PRESETS)))
    module = sys.modules[__name__]
    for key, value in PRESETS[name].items():
        setattr(module, key, value)
    module.ACTIVE = name
    return name
