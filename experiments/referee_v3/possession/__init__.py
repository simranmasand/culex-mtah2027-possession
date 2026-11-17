"""Frame-level ball possession from tracking data, built on the team's football rules.

settings.py  every number, in plain English
load.py      read the tracking and the set-piece tags
signals.py   per-frame facts: who is on the ball, ball speed, ball out, players slowing
dead_ball.py split each half into dead-ball and live stretches, and find who restarts
engine.py    inside a live stretch, decide who has the ball using the control rules
pipeline.py  glue: data in, one label per video frame out
diagnostics.py  whole-match sanity checks on the output (no labels used)
"""
