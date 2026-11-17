"""Checks the model on a MADE-UP mini match, so it can be tested without any real match data.

    python tests/test_synthetic.py

Two home players (track slots 0 and 1) and two away players (slots 2 and 3). The story:

  frames    0-49   waiting for kick-off, ball on the centre spot           -> D
  frames   50-299  home kicks off and dribbles                             -> H
  frames  300-310  an away player gets one quick touch (a deflection)      -> still H (B5)
  frames  311-400  home has it again                                       -> H
  frames  401-600  away intercepts and keeps it for 8 seconds              -> A from 401 (E13)
  frames  601-640  away kicks it out over the touchline                    -> D once it is out
  frames  641-703  home gets ready to take the throw-in                    -> D
  frames  704-799  home throws in and keeps it                             -> H (C9)
  frames  800-900  whistle: everyone stops, ball still                     -> D (C8)
  frames  901-999  away takes the free kick and keeps it                   -> A (C9)
  frames 1000-1099 after the final whistle                                 -> X
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import dead_ball, engine, load, pipeline, settings, signals  # noqa: E402

import contextlib  # noqa: E402


@contextlib.contextmanager
def preset(name):
    """Run a block of checks under settings preset `name`, then put back whichever was active."""
    before = settings.ACTIVE
    settings.use(name)
    try:
        yield
    finally:
        settings.use(before)


L, W = 105.3, 68.0
N = 1100
HALF = [(1, 0, 999)]
CENTRE = np.array([L / 2, W / 2])


def build_match(quick_out=False):
    """quick_out: the ball crosses the touchline 20 frames earlier (by frame ~619 instead of
    ~639), so the throw-in comes 3.4 s after it went out rather than 2.6 s. v3 needs at least
    3 s of dead time before a restart; real throw-ins average 17.7 s."""
    ball = np.full((N, 2), np.nan)
    players = np.full((N, 4, 2), np.nan)
    speeds = np.full((N, 4), 3.0)

    def put(slot, f0, f1, xy):
        players[f0:f1 + 1, slot] = xy

    def move_ball(f0, f1, p0, p1):
        for f in range(f0, f1 + 1):
            a = (f - f0) / float(f1 - f0)
            ball[f] = np.array(p0) + (np.array(p1) - np.array(p0)) * a

    # Default spots, far from the action.
    put(0, 0, N - 1, (30.0, 10.0))
    put(1, 0, N - 1, (30.0, 58.0))
    put(2, 0, N - 1, (80.0, 20.0))
    put(3, 0, N - 1, (80.0, 48.0))

    # 0-49: waiting for kick-off. Home slot 0 stands at the ball, everyone walking.
    ball[0:50] = CENTRE
    put(0, 0, 74, CENTRE - (0.3, 0.0))
    put(1, 0, 74, (42.65, 34.0))
    speeds[0:50] = 0.5

    # 50-74: kick-off pass from the centre to home slot 1.
    move_ball(49, 74, CENTRE, (43.15, 34.0))

    # 75-299: home slot 1 dribbles towards the away goal at 3 m/s.
    for f in range(75, 300):
        x = 42.65 + (f - 75) * 0.12
        players[f, 1] = (x, 34.0)
        ball[f] = (x + 0.5, 34.0)
    x1 = 42.65 + (299 - 75) * 0.12

    # 300-400: slot 1 stops with the ball. 300-310 an away player nicks it briefly.
    ball[300:401] = (x1 + 0.5, 34.0)
    put(1, 300, 400, (x1, 34.0))
    put(1, 300, 310, (x1 - 2.5, 34.0))
    put(2, 300, 310, (x1 + 0.8, 34.0))

    # 401-600: away slot 2 intercepts and dribbles back the other way for 8 s.
    for f in range(401, 601):
        b = (x1 + 0.5 - (f - 401) * 0.12, 34.0)
        ball[f] = b
        players[f, 2] = (b[0] + 0.3, 34.0)
        players[f, 1] = (b[0] + 6.0, 34.0)
    b600 = ball[600].copy()

    # 601-640: away kicks it out over the touchline (y beyond 68 m).
    if quick_out:
        move_ball(600, 620, b600, (b600[0], 70.0))
        ball[621:641] = (b600[0], 68.8)
    else:
        move_ball(600, 640, b600, (b600[0], 70.0))
    put(2, 601, 640, (b600[0] + 0.3, 34.0))

    # 641-700: ball held just outside the line, home slot 1 ready to throw.
    ball[641:701] = (b600[0], 68.8)
    put(1, 641, 700, (b600[0], 69.0))
    put(0, 641, 730, (b600[0], 60.0))

    # 701-730: throw-in to home slot 0.
    move_ball(700, 730, (b600[0], 68.8), (b600[0], 60.5))

    # 731-799: home slot 0 dribbles.
    for f in range(731, 800):
        x = b600[0] + (f - 731) * 0.12
        players[f, 0] = (x, 60.0)
        ball[f] = (x + 0.5, 60.0)

    # 800-900: whistle. Everyone walks, the ball sits still, away slot 2 stands over it.
    ball[800:901] = (70.0, 30.0)
    speeds[800:901] = 0.5
    put(2, 800, 900, (70.3, 30.0))
    put(0, 800, 900, (60.0, 30.0))
    put(3, 800, 999, (85.0, 30.0))

    # 901-925: away free kick to slot 3, who keeps it until the end.
    move_ball(900, 925, (70.0, 30.0), (84.5, 30.0))
    ball[926:1000] = (84.5, 30.0)

    # 1000-1099: after the final whistle, nothing is tracked.
    ball[1000:] = np.nan
    players[1000:] = np.nan

    scale = np.array([L, W])
    return ball / scale, players / scale, speeds


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok  " + message)


def test_story():
    print("made-up match, the whole story")
    ball, players, speeds = build_match()
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    sig = signals.compute(ball.astype(np.float32), players.astype(np.float32),
                          speeds.astype(np.float32), slot_team, (L, W))
    labels, _ = pipeline.label_frames(sig, HALF, [], N)

    first_out = int(np.flatnonzero(sig.ball_out)[0])
    check(set(labels[0:50]) == {"D"}, "before kick-off is D")
    check(labels[50] == "H", "home has it from the kick-off")
    check(set(labels[300:311]) == {"H"}, "one quick away touch does not change possession")
    check(labels[400] == "H", "home still has it just before the interception")
    check(set(labels[401:first_out]) == {"A"},
          "away has it from its first touch (switch moved back to the interception)")
    check(set(labels[first_out:704]) == {"D"}, "ball out of play is D until the throw-in")
    check(set(labels[704:800]) == {"H"}, "home has it from the moment it throws in")
    check(set(labels[800:901]) == {"D"}, "a break in play (whistle) is D")
    check(set(labels[901:1000]) == {"A"}, "away has it from its free kick")
    check(set(labels[1000:]) == {"X"}, "after the final whistle is X")

    info = pipeline.summary(labels)
    check(info["possessions"] == 4, "four possessions in the story (H, A, H, A)")


def test_switch_timing():
    print("switch timing (E13)")
    ball, players, speeds = build_match()
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    sig = signals.compute(ball.astype(np.float32), players.astype(np.float32),
                          speeds.astype(np.float32), slot_team, (L, W))
    saved = settings.BACKDATE_SWITCH
    try:
        settings.BACKDATE_SWITCH = False
        labels, _ = pipeline.label_frames(sig, HALF, [], N)
    finally:
        settings.BACKDATE_SWITCH = saved
    # 3 seconds of control = 75 frames, so the late switch lands at 401 + 74 = 475.
    check(labels[474] == "H" and labels[475] == "A",
          "without moving the switch back, it lands when control is confirmed (3 s later)")


def test_missed_kickoff():
    print("safety net: a half whose kick-off is never seen near the centre")
    ball, players, speeds = build_match()
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    sig = signals.compute(ball.astype(np.float32), players.astype(np.float32),
                          speeds.astype(np.float32), slot_team, (L, W))
    # Pretend the half starts at frame 641, while the ball is out for the throw-in: no kick-off
    # from the centre ever happens, so the first restart found must be accepted instead.
    stretches = dead_ball.split_half(sig, 641, 999, [], settings.FPS)
    check(stretches[0] == ("dead", 641, 703), "dead until the first restart that is found")
    check(stretches[1][:2] == ("live", 704) and stretches[1][3] == 0,
          "goes live at the throw-in, taken by home")


class FakeSignals(object):
    """Just enough of a Signals object to drive the possession engine directly."""

    def __init__(self, n):
        self.contact_team = np.full(n, -1, dtype=np.int8)
        self.contact_slot = np.full(n, -1, dtype=np.int16)
        self.ball_xy = np.zeros((n, 2), dtype=np.float32)
        self.ball_visible = np.ones(n, dtype=bool)

    def touch(self, team, slot, f0, f1, x):
        self.contact_team[f0:f1 + 1] = team
        self.contact_slot[f0:f1 + 1] = slot
        self.ball_xy[f0:f1 + 1] = (x, 30.0)

    def hide(self, f0, f1):
        self.ball_visible[f0:f1 + 1] = False
        self.ball_xy[f0:f1 + 1] = np.nan


def test_gap_filling():
    print("filling short gaps in the ball track")
    ball = np.full((20, 2), np.nan, dtype=np.float32)
    ball[0] = (10.0, 10.0)
    ball[4] = (14.0, 10.0)    # 3 missing frames between two sightings on the pitch
    ball[9] = (200.0, 10.0)   # 4 missing frames, but this sighting is off the pitch
    ball[19] = (20.0, 10.0)   # 9 missing frames: longer than the limit of 5
    filled = signals.fill_short_gaps(ball, 5, L, W)
    check(abs(filled[2, 0] - 12.0) < 1e-4, "a short gap is filled with a straight line")
    check(np.isnan(filled[6, 0]), "a gap ending off the pitch is left empty")
    check(np.isnan(filled[14, 0]), "a long gap is left empty")
    check(np.isnan(ball[2, 0]), "the original track is not changed")


def test_hidden_ball():
    print("long spells with the ball hidden: phase tags (E14)")
    n = 200

    sig = FakeSignals(n)
    sig.touch(0, 0, 10, 14, 50.0)
    sig.hide(20, 99)
    stats = {}
    phase = engine.phase_masks([(40, 91, "A")], n)
    switches = engine.possession_in_stretch(sig, 0, n - 1, 0, phase=phase, stats=stats)
    check(switches == [(0, 0), (40, 1)], "tags saying away has it switch possession when they start")
    check(stats.get("switches_from_tags") == 1, "the switch is counted as coming from the tags")

    phase = engine.phase_masks([(40, 91, "A"), (50, 81, "H")], n)
    switches = engine.possession_in_stretch(sig, 0, n - 1, 0, phase=phase)
    check(switches == [(0, 0)], "tags that disagree change nothing")

    sig = FakeSignals(n)
    sig.hide(20, 39)
    phase = engine.phase_masks([(20, 40, "A")], n)
    switches = engine.possession_in_stretch(sig, 0, n - 1, 0, phase=phase)
    check(switches == [(0, 0)], "a hidden spell shorter than 1 s is left alone")

    sig = FakeSignals(n)
    sig.touch(1, 2, 10, 14, 50.0)   # away starts a challenge...
    sig.hide(20, 99)
    sig.touch(1, 2, 110, 114, 50.0)
    sig.touch(1, 2, 130, 134, 50.0)
    switches = engine.possession_in_stretch(sig, 0, n - 1, 0)
    check(switches == [(0, 0), (10, 1)], "without tags, three away touches take control")
    phase = engine.phase_masks([(30, 71, "H")], n)
    switches = engine.possession_in_stretch(sig, 0, n - 1, 0, phase=phase)
    check(switches == [(0, 0)],
          "...but tags saying home still has it cancel that challenge, so no switch")


def test_touches_and_passes():
    print("touches, passes and cancelled challenges (A1, B4, B5)")
    sig = FakeSignals(100)
    sig.touch(1, 2, 10, 14, 50.0)  # away touch
    sig.touch(1, 3, 20, 24, 60.0)  # away pass to a team-mate 10 m away
    sig.touch(1, 2, 30, 34, 50.0)  # and back: third touch = control
    switches = engine.possession_in_stretch(sig, 0, 99, 0)
    check(switches == [(0, 0), (10, 1)], "three away touches take control, dated to the first")

    sig = FakeSignals(100)
    sig.touch(1, 2, 10, 14, 50.0)
    sig.touch(0, 0, 16, 18, 52.0)  # home touches it again: the challenge is cancelled
    sig.touch(1, 2, 20, 24, 50.0)
    sig.touch(1, 3, 30, 34, 60.0)
    switches = engine.possession_in_stretch(sig, 0, 99, 0)
    check(switches == [(0, 0)], "a home touch in between cancels the away challenge")

    check(engine.control_progress(0, 75, 1) >= 1.0, "3 seconds on the ball is control")
    check(engine.control_progress(7, 0, 0) >= 1.0, "7 passes is control")
    check(engine.control_progress(0, 0, 3) >= 1.0, "3 touches is control")
    check(engine.control_progress(1, 50, 2) >= 1.0, "1 pass + 2 s + 2 touches add up to control")
    check(engine.control_progress(0, 12, 1) < 1.0, "one half-second touch is not control")


def test_file_reading():
    print("reading files in the documented formats")
    folder = tempfile.mkdtemp()
    meta = os.path.join(folder, "meta.xml")
    raw = os.path.join(folder, "raw.txt")
    tags = os.path.join(folder, "tags.xml")
    with open(meta, "w") as handle:
        handle.write('<Root><Teams><Team id="T1"><Name>Home FC</Name></Team>'
                     '<Team id="T2"><Name>Away FC</Name></Team></Teams><Players>'
                     '<Player id="P1" teamId="T1"/><Player id="P2" teamId="T1"/>'
                     '<Player id="P3" teamId="T2"/><Player id="P4" teamId="T2"/>'
                     '</Players></Root>')
    with open(raw, "w") as handle:
        handle.write("1:0.5,0.5,1.2;0.4,0.4,2.0;NaN,NaN,NaN;0.6,0.6,0.1:0.5,0.5,0.0\n")
        handle.write("2:0.5,0.5,1.2;0.4,0.4,2.0;0.7,0.7,1.0;0.6,0.6,0.1:NaN,NaN,NaN\n")
        handle.write("3:0.5,0.5;0.4,0.4,2.0;0.7,0.7,1.0;0.6,0.6,0.1:0.1,0.2,0.0\n")
    with open(tags, "w") as handle:
        handle.write('<file><ALL_INSTANCES>'
                     '<instance><ID>1</ID><start>10.0</start><end>15.0</end>'
                     '<code>SET PIECES</code><label><group>Team</group><text>Barcelona</text>'
                     '</label><label><group>Type</group><text>Corner Kick</text></label>'
                     '</instance>'
                     '<instance><ID>2</ID><start>20.0</start><end>25.0</end>'
                     '<code>SET PIECES</code><label><group>Team</group><text>N/A</text>'
                     '</label></instance>'
                     '<instance><ID>3</ID><start>30.0</start><end>40.0</end>'
                     '<code>BUILD UP</code><label><group>Team</group><text>Atletic Club</text>'
                     '</label></instance>'
                     '<instance><ID>4</ID><start>50.0</start><end>55.0</end>'
                     '<code>DEFENDING IN MIDDLE THIRD</code><label><group>Team</group>'
                     '<text>Barcelona</text></label></instance>'
                     '<instance><ID>5</ID><start>60.0</start><end>65.0</end>'
                     '<code>DEFENSIVE TRANSITION</code><label><group>Team</group>'
                     '<text>Barcelona</text></label></instance>'
                     '</ALL_INSTANCES></file>')

    slot_team = load.team_of_each_slot(meta)
    check(slot_team.tolist() == [0, 0, 1, 1], "track slots get the right team (home first)")
    ball, players, speeds = load.read_tracking(raw, 4, 4)
    check(abs(ball[0, 0] - 0.5) < 1e-6 and np.isnan(ball[1, 0]), "ball read, NaN kept as missing")
    check(np.isnan(players[0, 2, 0]) and abs(players[1, 2, 0] - 0.7) < 1e-6,
          "players read, frame_id 1 becomes video frame 0")
    check(abs(players[2, 0, 0] - 0.5) < 1e-6 and np.isnan(speeds[2, 0]),
          "a group with only x,y still reads (slow path)")
    check(np.isnan(ball[3, 0]), "a frame missing from the file stays missing")
    pieces = load.read_set_pieces(tags, 25)
    check(len(pieces) == 2, "only SET PIECES tags are kept")
    check(pieces[0]["start"] == 250 and pieces[0]["team"] == "H", "tag time and team read")
    check(pieces[1]["team"] is None, "N/A team is treated as unknown")
    hints = load.phases_from(load.read_tags(tags, 25))
    check(hints == [(750, 1000, "A"), (1250, 1375, "A")],
          "Athletic building up = A; Barcelona defending = A; transitions give no hint")


# ------------------------------------------------------------------------------------------------
# v2 behaviours. Small made-up situations built straight into the per-frame facts.

def make_sig(ball_m, players_m=None, speed=3.0):
    """Signals for a made-up passage: ball_m is (n, 2) metres (NaN = unseen). Two home and two
    away players sit far from everything unless players_m (n, 4, 2) says otherwise."""
    n = len(ball_m)
    if players_m is None:
        players_m = np.zeros((n, 4, 2))
        players_m[:, 0] = (10.0, 5.0)
        players_m[:, 1] = (10.0, 60.0)
        players_m[:, 2] = (95.0, 5.0)
        players_m[:, 3] = (95.0, 60.0)
    scale = np.array([L, W])
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    speeds = np.full((n, 4), speed, dtype=np.float32)
    return signals.compute((ball_m / scale).astype(np.float32),
                           (players_m / scale).astype(np.float32), speeds, slot_team, (L, W))


def test_presets():
    print("presets (v1 and v2 are reproducible, v3 is the default)")
    check(settings.DEFAULT == "v3", "v3 is the default")
    check(all(getattr(settings, key) == value for key, value in settings.V3.items()),
          "the settings in the file are the v3 preset")
    check(set(settings.V1) == set(settings.V2) == set(settings.V3),
          "every preset sets the same list of settings")
    with preset("v1"):
        check(settings.CONTROL_PASSES == 7 and settings.TOUCH_RADIUS_M == 2.0
              and not settings.STILL_BALL, "use('v1') brings back the first version's numbers")
    with preset("v2"):
        check(settings.CONTROL_PASSES == 1 and not settings.RESTART_BACKTRACK,
              "use('v2') brings back the second version's numbers")
    check(settings.CONTROL_PASSES == 7 and settings.RESTART_BACKTRACK,
          "and v3 is back afterwards, with the team's own control rule ")
    try:
        settings.use("v4")
        check(False, "an unknown preset is refused")
    except ValueError:
        check(True, "an unknown preset is refused")


def test_v2_control():
    print("v2 control: the first controlled action (A1, B5)")
    check(engine.control_progress(1, 0, 0) >= 1.0, "one completed pass is control")
    check(engine.control_progress(0, 0, 2) >= 1.0, "two touches are control")
    check(engine.control_progress(0, 38, 1) >= 1.0, "1.5 s on the ball is control")
    check(engine.control_progress(0, 5, 1) < 1.0, "a single quick touch is still not control")
    sig = FakeSignals(100)
    sig.touch(1, 2, 10, 14, 50.0)   # away intercepts...
    sig.touch(1, 3, 22, 26, 60.0)   # ...and passes 10 m to a team-mate: a controlled action
    switches = engine.possession_in_stretch(sig, 0, 99, 0)
    check(switches == [(0, 0), (10, 1)],
          "an interception plus a completed pass flips possession, dated to the interception")
    sig = FakeSignals(100)
    sig.touch(1, 2, 10, 14, 50.0)   # a lone touch (a tackle or deflection)...
    sig.touch(0, 0, 18, 22, 52.0)   # ...and home has it again: no change (B5)
    check(engine.possession_in_stretch(sig, 0, 99, 0) == [(0, 0)],
          "a lone away touch changes nothing")


def test_touch_filter():
    print("a ball flying past a player is not a touch (G1)")
    n = 60
    ball = np.zeros((n, 2), dtype=np.float32)
    for f in range(n):
        ball[f] = (30.0 + 0.4 * f, 30.0)        # 10 m/s straight line, nobody touches it
    sig = FakeSignals(n)
    sig.ball_xy = ball
    check(not engine.is_real_touch(sig, 20, 27), "a short spell with the ball carrying straight on")
    stopped = ball.copy()
    stopped[28:] = stopped[27]                    # the ball is stopped dead: a trap
    sig.ball_xy = stopped
    check(engine.is_real_touch(sig, 20, 27), "a short spell where the ball stops is a touch")
    turned = ball.copy()
    for f in range(28, n):
        turned[f] = (turned[27][0], turned[27][1] + 0.4 * (f - 27))   # 90 degree deflection
    sig.ball_xy = turned
    check(engine.is_real_touch(sig, 20, 27), "a short spell where the ball turns is a touch")
    sig.ball_xy = ball
    check(engine.is_real_touch(sig, 10, 30), "a long spell (a carry) always counts")
    hidden = ball.copy()
    hidden[15:19] = np.nan
    sig.ball_xy = hidden
    check(engine.is_real_touch(sig, 20, 27), "unseen ball around the spell: benefit of the doubt")
    with preset("v1"):
        sig.ball_xy = ball
        check(engine.is_real_touch(sig, 20, 27), "v1 does not filter")


def test_out_of_play_evidence():
    print("ball out of play: it has to stay out (C8)")
    n = 400
    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    ball[100:103, 1] = 69.0                      # three jittery frames beyond the touchline
    sig = make_sig(ball)
    check(not dead_ball.out_of_play_evidence(sig).any(), "a three-frame glitch outside is ignored")
    with preset("v1"):
        check(dead_ball.out_of_play_evidence(sig)[100:103].all(), "(v1 would have stopped play)")

    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    ball[150:300, 1] = 69.0
    ball[200:203, 1] = 67.0                      # lying on the line, flickering back in
    sig = make_sig(ball)
    out = dead_ball.out_of_play_evidence(sig)
    check(out[150:300].all() and not out[149] and not out[300],
          "a ball out for seconds is out from the first frame it left")


def test_vanished_ball():
    print("ball vanishes after leaving through a line")
    n = 400
    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    for f in range(100, 150):                    # rolls towards the touchline at 5 m/s...
        ball[f] = (50.0, 57.8 + (f - 100) * 0.2)  # ...last seen at y = 67.6, 0.4 m inside it
    ball[150:260] = np.nan                       # ...and is not seen for 4.4 s
    ball[260:] = (50.0, 66.0)
    sig = make_sig(ball)
    gone = dead_ball.vanished_after_exit(sig)
    check(gone[150:260].all() and not gone[149] and not gone[260],
          "unseen for seconds just after heading out: out of play")

    ball2 = ball.copy()
    ball2[100:140] = (50.0, 30.0)
    for f in range(140, 150):                    # same last sighting, but coming back inside
        ball2[f] = (50.0, 67.6 + (149 - f) * 0.2)
    sig = make_sig(ball2)
    check(not dead_ball.vanished_after_exit(sig).any(), "heading back into the pitch: just hidden")

    ball3 = ball.copy()
    ball3[100:150] = (50.0, 30.0)
    sig = make_sig(ball3)
    check(not dead_ball.vanished_after_exit(sig).any(), "vanished in mid-pitch: not out")

    ball4 = ball.copy()
    ball4[170:] = (50.0, 66.0)                   # back after only 20 unseen frames (0.8 s)
    sig = make_sig(ball4)
    check(not dead_ball.vanished_after_exit(sig).any(), "gone for under 2 s: not enough")


def test_still_ball():
    print("a ball lying still with nobody near it is not in play")
    n = 300
    rng = np.random.RandomState(1)
    ball = np.tile(np.array([50.0, 30.0]), (n, 1)) + rng.normal(0.0, 0.15, (n, 2))
    sig = make_sig(ball)
    check(float(np.nanmedian(sig.ball_speed)) > 1.0,
          "(the jitter really does look fast frame to frame)")
    still = dead_ball.still_ball_evidence(sig)
    check(still.all(), "still and alone for seconds: dead evidence from the first frame")
    patchy = ball.copy()
    patchy[rng.rand(n) < 0.4] = np.nan                 # the ball is unseen on 4 frames in 10
    check(dead_ball.still_ball_evidence(make_sig(patchy))[:240].all(),
          "...even with the ball unseen on 4 frames in 10")

    players = np.zeros((n, 4, 2))
    players[:, 0] = (10.0, 5.0)
    players[:, 1] = (10.0, 60.0)
    players[:, 2] = (50.5, 30.0)                     # an away player right beside it
    players[:, 3] = (95.0, 60.0)
    sig_near = make_sig(ball, players)
    check(not dead_ball.still_ball_evidence(sig_near).any(), "a player beside the ball: nothing")

    moving = np.array([[40.0 + 0.1 * f, 30.0] for f in range(n)])
    check(not dead_ball.still_ball_evidence(make_sig(moving)).any(), "a rolling ball is not still")
    with preset("v1"):
        check(not dead_ball.still_ball_evidence(sig).any(), "v1 has no such rule")


def test_strike_needs_movement():
    print("a restart is a ball going somewhere, not a jittering one (C9)")
    n = 200
    rng = np.random.RandomState(2)
    still = np.tile(np.array([50.0, 30.0]), (n, 1)) + rng.normal(0.0, 0.2, (n, 2))
    sig = make_sig(still)
    check(not dead_ball.ball_struck(sig)[10:190].any(), "a jittery ball at rest is not a strike")
    with preset("v1"):
        check(dead_ball.ball_struck(sig)[10:190].any(), "(v1 could be fooled by it)")
    kick = np.tile(np.array([50.0, 30.0]), (n, 1))
    for f in range(100, n):
        kick[f] = (50.0 + 0.4 * (f - 100), 30.0)     # struck at 10 m/s from frame 100
    struck = dead_ball.ball_struck(make_sig(kick))
    check(struck[101] and not struck[100], "a real kick is found, at the first frame it moves")


class FakeDeadSignals(object):
    """Just enough of a Signals object to drive dead_ball._split."""

    def __init__(self, n):
        self.n_frames = n
        self.ball_visible = np.zeros(n, dtype=bool)
        self.d_home = np.full(n, np.inf)
        self.d_away = np.full(n, np.inf)


def test_safety_net():
    print("safety net: play is back once players run, even if no kick was seen (C9)")
    n = 700
    sig = FakeDeadSignals(n)
    evidence = np.zeros(n, dtype=bool)
    evidence[100:600] = True                          # a long stoppage (evidence from frame 100)
    struck = np.zeros(n, dtype=bool)
    # Half's kick-off found at frame 0 would normally come first; here the half is already dead
    # from frame 100 so we drive _split from a live start: make frame 0..99 a live stretch.
    struck[0] = True
    resume = np.full(n, -1, dtype=np.int64)
    resume[110 + 74:400] = 110                        # players running from frame 110 to 399

    stretches = dead_ball._split(sig, 0, n - 1, [], 25, evidence, struck, resume, False)
    live = [st for st in stretches if st[0] == "live"]
    check(len(live) == 2 and live[0][1] == 0 and live[0][2] == 99,
          "play, then a stoppage from the first frame of evidence")
    check(live[1][1] == 350,
          "restart placed once the stoppage is 10 s old while players are still running")
    check(live[1][2] == n - 1, "and the rest of that stoppage's evidence does not stop play again")

    resume_once = np.full(n, -1, dtype=np.int64)
    resume_once[110 + 74] = 110
    with preset("v1"):
        stretches = dead_ball._split(sig, 0, n - 1, [], 25, evidence, struck, resume_once, False)
    check(sum(1 for st in stretches if st[0] == "live") == 1,
          "v1: a quick restart was never recovered (the old limitation)")


def test_story_v2():
    print("the whole made-up match under v2")
    ball, players, speeds = build_match()
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    sig = signals.compute(ball.astype(np.float32), players.astype(np.float32),
                          speeds.astype(np.float32), slot_team, (L, W))
    labels, _ = pipeline.label_frames(sig, HALF, [], N)
    first_out = int(np.flatnonzero(sig.ball_out)[0])
    check(set(labels[0:50]) == {"D"}, "before kick-off is D")
    check(set(labels[300:311]) == {"H"}, "one quick away touch does not change possession")
    check(set(labels[401:first_out]) == {"A"}, "away has it from its first touch")
    check(set(labels[first_out:704]) == {"D"}, "ball out of play is D until the throw-in")
    check(set(labels[704:800]) == {"H"}, "home has it from the throw-in")
    check(set(labels[800:901]) == {"D"}, "a break in play is D")
    check(set(labels[901:1000]) == {"A"}, "away has it from its free kick")
    check(set(labels[1000:]) == {"X"}, "after the final whistle is X")
    check(pipeline.summary(labels)["possessions"] == 4, "four possessions (H, A, H, A)")


# ------------------------------------------------------------------------------------------------
# v3 behaviours (the final Referee Model).

def test_v3_control():
    print("v3 control: back to the team's own rule A1 ")
    check(engine.control_progress(1, 0, 0) < 1.0, "one completed pass alone is not control")
    check(engine.control_progress(0, 0, 2) < 1.0, "two touches alone are not control")
    check(engine.control_progress(0, 75, 1) >= 1.0, "3 seconds on the ball is control")
    check(engine.control_progress(0, 0, 3) >= 1.0, "3 touches is control")
    check(engine.control_progress(7, 0, 0) >= 1.0, "7 passes is control")


def _one_ball(home_m, away_m, n=10):
    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    players = np.zeros((n, 4, 2))
    players[:, 0] = (50.0 - home_m, 30.0)
    players[:, 1] = (10.0, 60.0)
    players[:, 2] = (50.0 + away_m, 30.0)
    players[:, 3] = (95.0, 60.0)
    return make_sig(ball, players)


def test_contested():
    print("a 50/50 is a player from each team within 1 m of the ball (B4)")
    check(int(_one_ball(0.4, 0.9).contact_team[5]) == signals.CONTESTED,
          "home 0.4 m and away 0.9 m from the ball: contested")
    check(int(_one_ball(0.4, 1.3).contact_team[5]) == 0,
          "home 0.4 m, away 1.3 m: home is on the ball")
    with preset("v1"):
        check(int(_one_ball(0.4, 0.95).contact_team[5]) == 0,
              "(v1's 0.5 m margin rule gave that 0.4 / 0.95 m case to home)")
    check(int(_one_ball(0.4, 0.95).contact_team[5]) == signals.CONTESTED,
          "v3 calls the same 0.4 / 0.95 m case a 50/50")


def test_gap_snap():
    print("a player right on the path of a hidden ball touched it")
    n = 11
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[0] = (40.0, 30.0)
    ball[10] = (50.0, 30.0)
    players = np.full((n, 4, 2), np.nan, dtype=np.float32)
    players[:, 0] = (39.5, 30.0)      # the passer, beside the ball at the last sighting
    players[:, 1] = (50.3, 30.0)      # the receiver, beside it at the next sighting
    players[:, 2] = (45.0, 30.3)      # an away player 0.3 m from the ball's straight path
    players[:, 3] = (95.0, 60.0)
    filled = signals.fill_short_gaps(ball, 25, L, W, players, 0.5, 1.5)
    check(np.allclose(filled[5], (45.0, 30.3)), "the ball is put at the player on its path")
    check(np.allclose(filled[3], (43.0, 30.18)) and np.allclose(filled[8], (48.0, 30.12)),
          "and the gap is two straight lines through him")
    players[:, 2] = (45.0, 31.0)      # 1 m off the path: too far for a 0.5 m snap
    filled = signals.fill_short_gaps(ball, 25, L, W, players, 0.5, 1.5)
    check(np.allclose(filled[5], (45.0, 30.0)), "a player 1 m off the path changes nothing")
    players[:, 2] = (95.0, 5.0)
    players[:, 0] = (41.0, 30.0)      # the passer runs along behind the ball
    filled = signals.fill_short_gaps(ball, 25, L, W, players, 0.5, 1.5)
    check(np.allclose(filled[5], (45.0, 30.0)), "the passer and receiver themselves never snap")
    check(np.allclose(signals.fill_short_gaps(ball, 25, L, W)[5], (45.0, 30.0)),
          "without players (v1, v2) it is the plain straight line")


def test_kickoff_halves():
    print("a kick-off needs the teams in opposite halves (Law 8)")
    players = np.full((3, 16, 2), np.nan)
    home, away = np.arange(8), np.arange(8, 16)
    players[:, home] = (30.0, 34.0)
    players[:, away] = (80.0, 34.0)
    players[1, home[:4]] = (80.0, 34.0)          # four home players in the other half
    players[2, home[3:]] = np.nan                 # only three home players seen
    shape = signals.teams_in_own_halves(players, home, away, L)
    check(bool(shape[0]), "both teams in their own halves")
    check(not shape[1], "four players on the wrong side: not a kick-off")
    check(bool(shape[2]), "too few players seen to judge: not ruled out")
    mirrored = signals.teams_in_own_halves(players[:, ::-1], home, away, L)
    check(bool(mirrored[0]), "either team may be on either side (they change ends)")

    squad = np.full((1, 22, 2), np.nan)
    home, away = np.arange(11), np.arange(11, 22)
    squad[0, home[:8]] = (30.0, 34.0)
    squad[0, away[:8]] = (80.0, 34.0)
    squad[0, home[8:]] = (80.0, -3.0)            # three home subs warming up beyond the touchline
    squad[0, away[8:]] = (30.0, 71.0)            # ...and three away subs, in the other half
    check(bool(signals.teams_in_own_halves(squad, home, away, L, W)[0]),
          "substitutes warming up off the pitch are not counted")


def test_restart_from_rest():
    print("a restart is struck from a ball at rest, with a taker beside it")
    n = 200
    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    for f in range(100, n):
        ball[f] = (50.0 + 0.4 * (f - 100), 30.0)     # struck at 10 m/s from frame 100
    players = np.zeros((n, 4, 2))
    players[:, 0] = (49.5, 30.0)                      # the taker
    players[:, 1] = (10.0, 60.0)
    players[:, 2] = (95.0, 5.0)
    players[:, 3] = (95.0, 60.0)
    sig = make_sig(ball, players)
    struck = dead_ball.ball_struck(sig)
    starts = dead_ball.restart_starts(sig, struck)
    check(bool(struck[101] and starts[101]), "ball at rest, taker beside it: a restart")
    check(not starts[130], "a ball already rolling for a second cannot be the restart")
    alone = players.copy()
    alone[:, 0] = (10.0, 5.0)
    sig = make_sig(ball, alone)
    check(not dead_ball.restart_starts(sig, dead_ball.ball_struck(sig)).any(),
          "nobody beside the ball: not a restart")
    with preset("v2"):
        sig = make_sig(ball, alone)
        check(dead_ball.restart_starts(sig, dead_ball.ball_struck(sig))[101],
              "(v2 took any strike)")


def test_kickoff_nudge():
    print("a soft kick-off tap restarts play")
    n = 200
    ball = np.tile(CENTRE, (n, 1))
    for f in range(100, 150):
        ball[f] = CENTRE + (0.04 * (f - 100), 0.0)   # nudged forward at 1 m/s, never 3 m/s
    ball[150:] = ball[149]
    players = np.zeros((n, 4, 2))
    players[:, 0] = CENTRE - (0.3, 0.0)
    players[:, 1] = (40.0, 34.0)
    players[:, 2] = (70.0, 20.0)
    players[:, 3] = (70.0, 48.0)
    sig = make_sig(ball, players)
    check(not dead_ball.ball_struck(sig).any(), "(the tap is too soft to count as a strike)")
    stretches = dead_ball.split_half(sig, 0, n - 1, [])
    live = [st for st in stretches if st[0] == "live"]
    check(len(live) >= 1 and 95 <= live[0][1] <= 105 and live[0][3] == 0,
          "the half goes live at the tap (within 0.2 s), home kicking off")


def test_setup_backtrack():
    print("working back from a restart: its set-up was dead")
    n = 300
    ball = np.zeros((n, 2))
    for f in range(100):
        ball[f] = (20.0 + 0.3 * f, 30.0)              # played forward...
    ball[100:200] = (50.0, 30.0)                      # ...then set down for 4 s...
    for f in range(200, n):
        ball[f] = (50.0 + 0.4 * (f - 199), 30.0)      # ...and struck at 10 m/s
    players = np.zeros((n, 4, 2))
    players[:, 0] = ball - (0.5, 0.0)
    players[100:, 0] = (49.5, 30.0)                   # the taker beside the ball
    players[:, 1] = (10.0, 60.0)
    players[:, 2] = (95.0, 5.0)
    players[:, 3] = (95.0, 60.0)
    setup = dead_ball.setup_evidence(make_sig(ball, players))
    check(setup[100:200].all() and not setup[90] and not setup[200],
          "the ball lying on its spot before the strike is dead")

    pressed = players.copy()
    pressed[:, 2] = (51.0, 30.0)                      # an opponent 1 m from the ball
    check(not dead_ball.setup_evidence(make_sig(ball, pressed)).any(),
          "an opponent right on the ball: live play, not a set piece")
    in_box = ball.copy()
    in_box[100:200] = (8.0, 34.0)
    in_box[200:] = in_box[200:] - (42.0, -4.0)
    in_box_players = players.copy()
    in_box_players[100:, 0] = (7.5, 34.0)
    check(not dead_ball.setup_evidence(make_sig(in_box, in_box_players))[100:200].any(),
          "inside a penalty area (a keeper holding the ball): not used")
    with preset("v2"):
        check(not dead_ball.setup_evidence(make_sig(ball, players)).any(), "(v2 has no such rule)")


def test_break_speed():
    print("a break is still everyone below 1 m/s (1.5 m/s was tried and dropped)")
    n = 200
    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    players = np.zeros((n, 4, 2))
    players[:, 0] = (50.5, 30.0)                      # someone stands by the ball
    players[:, 1] = (10.0, 60.0)
    players[:, 2] = (95.0, 5.0)
    players[:, 3] = (95.0, 60.0)
    check(not dead_ball.break_in_play(make_sig(ball, players, speed=1.2)).any(),
          "players drifting at 1.2 m/s: not a break (could be slow build-up play)")
    check(dead_ball.break_in_play(make_sig(ball, players, speed=0.8)).all(),
          "everyone walking at 0.8 m/s around a still ball: a break")
    saved = settings.BREAK_PLAYER_SPEED
    try:
        settings.BREAK_PLAYER_SPEED = 1.5               # the threshold that was tried and dropped
        check(dead_ball.break_in_play(make_sig(ball, players, speed=1.2)).all(),
              "(at 1.5 m/s the 1.2 m/s drift would count)")
    finally:
        settings.BREAK_PLAYER_SPEED = saved


def test_safety_net_v3():
    print("safety net waits 30 s, and a ball seen lying still blocks it")
    n = 1600
    sig = FakeDeadSignals(n)
    sig.ball_inside = np.zeros(n, dtype=bool)
    sig.ball_speed_smooth = np.zeros(n)
    evidence = np.zeros(n, dtype=bool)
    evidence[100:1500] = True
    struck = np.zeros(n, dtype=bool)
    struck[0] = True
    resume = np.full(n, -1, dtype=np.int64)
    resume[110 + 74:1400] = 110                       # players running from frame 110

    def restarts():
        stretches = dead_ball._split(sig, 0, n - 1, [], 25, evidence, struck, resume, False)
        return [st[1] for st in stretches if st[0] == "live"][1:]

    check(restarts() == [850], "ball never seen: restart once the stoppage is 30 s old")
    sig.ball_visible[:] = True
    sig.ball_inside[:] = True
    check(restarts() == [], "ball seen lying still the whole time: play has not resumed")
    sig.ball_speed_smooth[1000:] = 5.0
    check(restarts() == [850], "ball seen moving from frame 1000: play had resumed")


def test_phase_trim():
    print("the first and last second of each phase tag are ignored")
    home, _ = engine.phase_masks([(100, 200, "H"), (250, 290, "H")], 300)
    check(not home[100:125].any() and home[125:175].all() and not home[175:200].any(),
          "a 4 s tag is used for its middle 2 s")
    check(not home[250:290].any(), "a tag under 2 s long is ignored")
    with preset("v1"):
        home, _ = engine.phase_masks([(100, 200, "H")], 300)
        check(home[100:200].all(), "(v1 used every frame of a tag)")


def test_unseen_opening_kickoff():
    print("an unseen opening kick-off is not replaced by one after a goal much later")
    n = 1000
    ball = np.full((n, 2), np.nan)
    for f in range(60, 800):
        ball[f] = (30.0 + 0.2 * ((f - 60) % 200), 20.0)   # play on, far from the centre spot
    ball[800:850] = CENTRE                                # a kick-off after a goal...
    for f in range(850, n):
        ball[f] = CENTRE + (0.4 * (f - 849), 0.0)        # ...struck from the centre
    players = np.zeros((n, 4, 2))
    players[:, 0] = ball - (0.5, 0.0)
    players[:60, 0] = (29.5, 20.0)
    players[800:, 0] = CENTRE - (0.5, 0.0)
    players[:, 1] = (40.0, 34.0)
    players[:, 2] = (70.0, 20.0)
    players[:, 3] = (70.0, 48.0)
    sig = make_sig(ball, players)
    saved = settings.KICKOFF_SEARCH_SECONDS
    try:
        settings.KICKOFF_SEARCH_SECONDS = 10.0            # a 10 s search, to fit this short test
        live = [st[1] for st in dead_ball.split_half(sig, 0, n - 1, []) if st[0] == "live"]
        check(bool(live) and live[0] < 100, "the half starts with the first play that is seen")
        settings.KICKOFF_SEARCH_SECONDS = 0.0             # v1 / v2: search the whole half
        live = [st[1] for st in dead_ball.split_half(sig, 0, n - 1, []) if st[0] == "live"]
        check(bool(live) and live[0] >= 850,
              "(searching the whole half waits for the later kick-off: the old problem)")
    finally:
        settings.KICKOFF_SEARCH_SECONDS = saved


def test_lineup_check():
    print("no opening kick-off before the teams have lined up (Law 8)")
    n, S = 1200, 16
    ball = np.full((n, 2), np.nan)
    for f in range(0, 300):
        ball[f] = (30.0 + 0.2 * (f % 100), 20.0)          # warm-up passes before the half starts
    players = np.zeros((n, S, 2))
    rng = np.random.RandomState(3)
    players[:300] = rng.uniform((5.0, 5.0), (100.0, 63.0), (300, S, 2))   # all over the pitch
    players[300:, :8] = (40.0, 34.0)                       # from frame 300 lined up: home left...
    players[300:, 8:] = (65.0, 34.0)                       # ...away right
    players[:, 0] = np.where(np.isnan(ball), players[:, 0], ball - (0.5, 0.0))
    scale = np.array([L, W])
    slot_team = np.array([0] * 8 + [1] * 8, dtype=np.int8)
    speeds = np.full((n, S), 3.0, dtype=np.float32)        # players moving throughout
    sig = signals.compute((ball / scale).astype(np.float32), (players / scale).astype(np.float32),
                          speeds, slot_team, (L, W))
    saved = settings.KICKOFF_SEARCH_SECONDS
    try:
        settings.KICKOFF_SEARCH_SECONDS = 20.0             # fit the short test
        check(dead_ball.lineup_frame(sig, 0, n - 1) == 300 + 74,
              "teams seen lined up for 3 s from frame 300")
        live = [st[1] for st in dead_ball.split_half(sig, 0, n - 1, []) if st[0] == "live"]
        check(bool(live) and live[0] >= 374, "the half does not start during the warm-up")
        with preset("v2"):
            settings.KICKOFF_SEARCH_SECONDS = 20.0
            live = [st[1] for st in dead_ball.split_half(sig, 0, n - 1, []) if st[0] == "live"]
            check(bool(live) and live[0] < 300, "(without the check it started during the warm-up)")
    finally:
        settings.KICKOFF_SEARCH_SECONDS = saved


def test_story_v3():
    print("the whole made-up match under v3")
    ball, players, speeds = build_match(quick_out=True)
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    sig = signals.compute(ball.astype(np.float32), players.astype(np.float32),
                          speeds.astype(np.float32), slot_team, (L, W))
    labels, _ = pipeline.label_frames(sig, HALF, [], N)
    first_out = int(np.flatnonzero(sig.ball_out)[0])
    check(set(labels[0:50]) == {"D"}, "before kick-off is D")
    check(set(labels[50:300]) == {"H"}, "home has it from the kick-off")
    check(set(labels[300:401]) == {"H"},
          "home standing on the ball, then pressed, is not taken for a set piece")
    check(set(labels[401:first_out]) == {"A"}, "away has it from its first touch")
    check(set(labels[first_out:704]) == {"D"}, "ball out of play is D until the throw-in")
    check(set(labels[704:800]) == {"H"}, "home has it from the throw-in")
    check(set(labels[800:901]) == {"D"}, "a break in play is D")
    check(set(labels[901:1000]) == {"A"}, "away has it from its free kick")
    check(set(labels[1000:]) == {"X"}, "after the final whistle is X")
    check(pipeline.summary(labels)["possessions"] == 4, "four possessions (H, A, H, A)")


def test_dead_ball_export():
    print("the dead-ball export for combining with another model")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import export_dead_ball
    ball, players, speeds = build_match(quick_out=True)
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    sig = signals.compute(ball.astype(np.float32), players.astype(np.float32),
                          speeds.astype(np.float32), slot_team, (L, W))
    labels, report = pipeline.label_frames(sig, HALF, [], N)
    cols = export_dead_ball.dead_ball_columns(labels, report, HALF)
    check(all(len(v) == N for v in cols.values()), "one row per frame in every column")
    check((cols["dead"] == (labels == "D")).all(), "dead is 1 exactly where the label is D")
    check(set(cols["dead_cause"][0:50]) == {"before_kickoff"}, "the wait for kick-off is marked")
    check(cols["dead_cause"][650] == "out" and cols["dead_cause"][850] in ("break", "setup"),
          "the throw-in stoppage is 'out', the whistle is a break or a set-up")
    check(cols["restart"][704] == 1 and cols["restart_team"][704] == "H"
          and cols["restart_how"][704] == "strike", "the throw-in restart: home, struck")
    check(cols["restart"][901] == 1 and cols["restart_team"][901] == "A",
          "the free-kick restart: away")
    check(int(cols["restart"].sum()) == 3, "three restarts: kick-off, throw-in, free kick")


def run_v1_tests():
    with preset("v1"):
        test_touches_and_passes()
        test_story()
        test_switch_timing()
        test_missed_kickoff()
        test_hidden_ball()


def run_v2_tests():
    with preset("v2"):
        test_v2_control()
        test_touch_filter()
        test_out_of_play_evidence()
        test_vanished_ball()
        test_still_ball()
        test_strike_needs_movement()
        test_safety_net()
        test_story_v2()


if __name__ == "__main__":
    test_file_reading()
    test_gap_filling()
    test_presets()
    print("\n--- the first version (v1) still behaves as it did ---")
    run_v1_tests()
    print("\n--- version 2 still behaves as it did ---")
    run_v2_tests()
    print("\n--- version 3 ---")
    test_v3_control()
    test_contested()
    test_gap_snap()
    test_kickoff_halves()
    test_restart_from_rest()
    test_kickoff_nudge()
    test_setup_backtrack()
    test_break_speed()
    test_safety_net_v3()
    test_phase_trim()
    test_unseen_opening_kickoff()
    test_lineup_check()
    test_story_v3()
    test_dead_ball_export()
    print("\nall checks passed")
