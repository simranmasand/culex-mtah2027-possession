# Referee Model (KOone v3)

> **In this repository** the Referee Model lives in `experiments/referee_v3/` and is run in place (no copying into
> the repository root): `cd experiments/referee_v3` and prefix each command below with `PYTHONPATH=../..`
> (see the top-level README §4). It is the team's logic-based reference model; the submitted file is the GRU's.


A rules-only possession model for More Than a Hack 2027, Phase 1. For every frame of the tactical
video it gives H (Barcelona), A (Athletic), D (dead ball) or X (outside the halves). Nothing is
learned from data: every rule comes from the Laws of the Game, the team's own possession rules or
a published study, and no number was set by looking at the match or the public ground truth.

It is meant to sit next to the GRU model. The GRU is better at following possession when the ball
cannot be seen; the Referee Model is better at saying when the ball is out of play and who
restarts it. `export_dead_ball.py` writes those calls out frame by frame so the two can be
combined.

## What's in the folder

| Path | What it is |
| --- | --- |
| `possession/settings.py` | Every threshold, with the rule or source it comes from, and the v1/v2/v3 presets |
| `possession/load.py` | Reads the tracking (`_atd.xml`, `_atd.txt`) and the Smart Tagging XML |
| `possession/signals.py` | Per-frame facts: who is on the ball, ball speed, ball out, gap filling |
| `possession/dead_ball.py` | Dead and live stretches, restarts, and who takes them |
| `possession/engine.py` | Who has the ball during live play (control rule, phase tags) |
| `possession/pipeline.py` | Puts it all together |
| `possession/diagnostics.py` | Whole-match sanity checks printed after a run |
| `make_submission.py` | Writes `<team>.csv` |
| `export_dead_ball.py` | Writes the dead-ball calls as a per-frame table |
| `tests/test_synthetic.py` | A made-up mini match with a known answer, plus one check per rule. No data needed |
| `tests/selfcheck_gap_fill.py` | Measures the ball gap-fill error on the real tracking (no labels used) |
| `bench/metrica_bench.py` | Scores v1, v2, v3 and the organisers' baseline on the three Metrica sample games |

## Setup

1. Copy everything in this folder into the root of the challenge repository, next to `games.py`.
2. Put the match data in `./data`, or set `MTAH_DATA_DIR` (see `DATA.md` in the challenge repo).
   Only the tactical tracking is needed; the Smart Tagging XML is used if it is there. The video
   is not used.
3. Python 3.9 or later with numpy and pandas (the challenge repo's `requirements.txt`).

## Running it

```
python tests/test_synthetic.py                    # about 10 s, no data needed
python make_submission.py --team KOone_v3         # about 2 min, writes evaluation/results/KOone_v3.csv
python export_dead_ball.py                        # writes evaluation/results/referee_dead_ball.csv
python bench/metrica_bench.py --data <sample-data>/data
```

`make_submission.py` options:

- `--model v1|v2|v3` picks a preset (v3 by default). v1 and v2 regenerate the earlier files exactly.
- `--set NAME=VALUE` changes one setting for a trial run, for example `--set BREAK_PLAYER_SPEED=1.2`.
  Write trials with `--out evaluation/trials` so the scorer doesn't pick them up.
- `--compare other.csv` prints how many frames agree with another submission file (for example
  the GRU's) and the largest disagreements.

After each run it prints label-free checks: dead-ball minutes, possessions, each half, how every
dead stretch ended, and the longest stoppages. A real match has about 40 to 45 minutes of dead
ball and a little under 200 possessions (Opta); those two numbers were used to pick between
versions, never the public 5-minute score.

## How it decides

**Dead ball (Law 9).** A stretch goes dead on any of these:

| Evidence | Rule |
| --- | --- |
| Out | Ball more than 0.3 m past a line, staying out for 1 s |
| Vanished | Ball last seen within 1.5 m of a line, not heading back in, then unseen for 2 s |
| Break | Players averaging under 1 m/s with the ball still for 3 s (a whistle) |
| Still | Ball within a 1 m box for 2 s with nobody within 1 m |
| Set-up | Ball sitting on a spot for 3 s, outside the penalty areas, then struck by a player beside it with no opponent within 2 m (Laws 13 and 15) |

**Restarts.** Play restarts when the ball is struck from rest (3 m/s for 0.2 s, travelling in a
straight line) with a taker within 1.5 m, at least 3 s after the stoppage. The taker's team gets
the ball; if no taker is clear, the Smart Tagging set-piece tag is used. If no kick is seen, a
safety net restarts play once players have been running for 3 s, but only after 30 s of dead time.

**Kick-offs (Law 8).** The ball is struck or tapped at the centre spot with the teams in their own
halves (substitutes off the pitch are ignored). A half's opening kick-off is only looked for in
its first 5 minutes; if the tracking doesn't show it, play starts at the first restart after the
teams are seen lined up.

**Possession.** The team that restarted keeps the ball until the other team gains control: 3 s on
the ball, 3 touches or 7 passes, or a weighted mix of the three (passes count most). The change is
then dated back to that team's first touch. A player is on the ball within 1.5 m, and only if the
ball visibly changed speed or direction (so a ball flying past doesn't count). A player from each
team within 1 m is a 50/50 and changes nothing.

**Hidden ball.** Gaps of up to 1 s are filled with a straight line, bent through any player within
0.5 m of its path. Longer gaps use the Smart Tagging phase tags ("build up", "defending in middle
third" and so on), ignoring the first and last second of each tag.

## Versions

| Version | What changed |
| --- | --- |
| v1 | First rules model. Out of play at 0.5 m, breaks, restarts on any strike, control at 3 s / 3 touches / 7 passes |
| v2 | Tighter out rule, vanished and still ball, touch filter, smoother speeds, looser control (1.5 s / 2 touches / 1 pass) |
| v3 | Stricter restarts, restart set-ups, kick-off checks (Law 8), 50/50 radius, gap bending, trimmed phase tags, control back to 3 s / 3 / 7 |

Two things were tried in v3 and changed after the full-match checks: a 1.5 m/s break threshold
(dropped: it doubled the break rule's dead time to 36 minutes) and a kick-off search over the
whole half (it took a kick-off after a goal for the opening one and left 35 minutes of the second
half dead; now limited to the first 5 minutes).

## Results

**BAR-ATH, whole match, no labels used:**

| Check | v1 | v2 | v3 | Real matches |
| --- | --- | --- | --- | --- |
| Dead ball | 32.3 min | 38.5 min | 41.6 min | 40 to 45 min |
| Possessions | 251 | 288 | 223 | a little under 200 |
| Possessions, half 1 / half 2 | 130 / 121 | 146 / 142 | 110 / 113 | about even |

**BAR-ATH public 5 minutes (17 true possessions; a check, not used to choose):** v1 spellF1 0.542,
v2 0.533. The final v3 has not been scored there yet.

**Metrica sample games** (`bench/metrica_bench.py`, mean of the three games, spellF1). Labels are
built from Metrica's event data by the script; "ball hidden" removes the ball in long random gaps,
closer to the match feed than the clean files:

| Model | Clean tracking | Ball hidden | Possessions found (truth 375) |
| --- | --- | --- | --- |
| Organisers' baseline (smooth50) | 0.402 | 0.203 | 502 / 294 |
| v1 | 0.499 | 0.176 | 230 / 86 |
| v2 | 0.612 | 0.237 | 247 / 101 |
| v3 | 0.538 | 0.207 | 204 / 95 |

On clean tracking v3 finds 80 to 91% of the dead time, depending on the game. With the ball hidden it is no better than
the baseline, because it merges possessions it cannot see change hands. That is the case for
combining it with the GRU rather than submitting it alone.

## Known limits

- Thresholds come from the Laws and published studies, not from data, so some suit this tracking
  feed less well than others.
- When the ball is hidden for long spells and no tag covers it, whoever had the ball keeps it.
- The second-half kick-off isn't visible in the tracking; the model infers it from the line-up and
  may be a little early or late.
- Quick free kicks taken before the ball settles can be missed and fall to the 30 s safety net.

## Rules to keep to

- Never paste match data, rows or tracebacks that quote the data into an AI chat. Counts and
  scores printed by these scripts are fine.
- Keep `data/` out of git.
- Don't change a setting because the public 5-minute score went up. Change it for a football
  reason, or because it helped on external data such as the Metrica games.

## Sources

- IFAB Laws of the Game, Laws 1, 8, 9, 12, 13 and 15: https://www.theifab.com/laws/latest/
- Vidal-Codina et al. (2022), event detection from tracking data: https://arxiv.org/abs/2202.00804
- Link and Hoernig (2017), individual ball possession in soccer: https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225
- Bradley et al. (2010), speed bands in elite football
- Opta / The Analyst on possessions and ball-in-play time: https://theanalyst.com/articles/possessions-and-sequences-in-football
- Kim et al. (2023), ball trajectory inference: https://arxiv.org/abs/2306.08206
- Metrica Sports sample data (commit e706dd5): https://github.com/metrica-sports/sample-data
