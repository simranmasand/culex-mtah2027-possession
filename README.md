# CuleX — More than a Hack 2027, Phase 1: frame-level ball possession

This repository regenerates our submission `CuleX.csv` (one label per video frame of BAR-ATH: `H`, `A`, `D`, `X`)
from the provided match data plus one public external dataset. It contains code only: **no match data** is included.

We built **two independent models** and keep them separate. Both regenerate their files from this repository:

| | Referee Model v3 (logic) | GRU + Viterbi (learned) — **submitted** |
|---|---|---|
| Idea | Rules from the Laws of the Game and our written possession rules; nothing fitted to data | Network trained on public Metrica games degraded to look like BAR-ATH's tracking |
| Code | `experiments/referee_v3/` | `experiments/` (entry point `experiments/final/reproduce_final.py`) |
| Inputs | Tactical tracking + Smart Tagging | Tactical tracking + Metrica sample data (training) |
| Output | `evaluation/results/referee_v3.csv` | `CuleX.csv` |
| Runtime | ~5 s | ~45 min (CPU) |
| Bench spellF1, hard / easy | 0.448 / 0.526 | **0.684 / 0.817** |
| Dead ball: recall / precision (bench, hard) | **0.90** / 0.54 | 0.82 / **0.82** |

The Referee Model finds more of the real stoppages (especially long ones), but also calls many live-play ball gaps
dead, which splits possessions. The GRU learns from context how much a ball gap means a stoppage, and is far better on
the challenge metric (+0.24 spellF1, 90% CI [+0.21, +0.27]). Details: `experiments/FINDINGS.md` (section A7) and the
technical document `CuleX.pdf`.

**GRU in one paragraph.** A bidirectional GRU reads per-frame features computed from the automatic tracking (ball
visibility and gap length, nearest player's team and distance, player speeds, ball off-pitch) at 5 Hz and outputs H/A/D
probabilities; a Viterbi decoder with a 3-state transition matrix turns them into possession runs. The network is
trained only on the Metrica Sports public sample games (3 matches, labels derived by our code from Metrica's event data),
degraded to look like BAR-ATH's tracking (ball hidden in long gaps, player dropouts; 3 random degradations per game).
The loss weights steps near a possession change ×5 and adds a change-point head, which fixes the main failure of plain
training (merging short possessions). 8 seeds are averaged.

**Referee Model in one paragraph.** Each half is cut into dead and live stretches. Dead: ball out (and staying out
1 s), ball vanished just after leaving through a line, everyone walking with the ball still (a whistle), ball lying
still with nobody near, or a restart being set up. Play restarts on a strike from rest with a taker beside it, or a
kick-off tap at the centre spot with the teams in their own halves (Law 8). The restarting team has the ball until
the other team gains control (3 s on the ball, 3 touches or 7 passes); short ball gaps are filled with a straight
line, long ones use the Smart Tagging phase tags. Every threshold and its source: `experiments/referee_v3/possession/settings.py`.

No BAR-ATH labels are used by either model; the public 5-minute window was used only to check the file format and
frame alignment, never to train, tune or choose.

## Repository layout

```
README.md                  this file (start here)
LICENSE-CODE               MIT licence for our code
experiments/
  README.md                map of every script (what is needed to regenerate, what is evaluation, what is history)
  final/                   the submitted GRU: reproduce_final.py, config.json, pinned requirements, reference manifest
  referee_v3/              the logic-based Referee Model (not submitted; published for comparison)
  FINDINGS.md              research log: every experiment, its result and the decisions taken
  eval_table.md            summary table of all models
  e*/  *.py                pipeline modules, experiments and their saved summaries
evaluation/                organisers' scorer + format validator (unchanged)
starter/, games.py         organisers' loaders (unchanged; used by our code)
data/                      not committed: match bundle + Metrica sample data go here (see §1)
CHALLENGE.md, DATA.md, RULES.md, SUBMISSION.md, ...   organisers' documents (unchanged); their README: README_ORGANISERS.md
```

**Quick start:** §1 data → §2 install → §3 regenerate `CuleX.csv` (one command). §4 regenerates the Referee Model.

## 1. Data (not in this repository)

1. **Match bundle** — unpack the tactical bundle you received at enrolment into `data/tactical/` as described in
   the organisers' `DATA.md`. The GRU reads the tactical ATD tracking; the Referee Model reads the tracking and the
   Smart Tagging XML.
2. **External training data — Metrica Sports sample data** (public, https://github.com/metrica-sports/sample-data),
   needed by the GRU and by the bench evaluations:

   ```bash
   git clone https://github.com/metrica-sports/sample-data data/external/metrica_sample
   git -C data/external/metrica_sample checkout e706dd506b360d69d9d123d5b8026e7294b13996
   ```

   We use Sample Games 1–3 (tracking + event data). Credit: *Metrica Sports sample data*. Per its README, use
   responsibly and acknowledge the source.

`data/` is gitignored; derived caches are written to `data/cache/` (also gitignored).

## 2. Install

Tested with Python 3.11.18 on macOS (arm64), CPU only.

```bash
conda create -n mtah python=3.11.18 -y && conda activate mtah
pip install -r experiments/final/requirements-final.txt
```

The Referee Model needs only numpy and pandas (already in that file).

## 3. Regenerate the submission (GRU)

```bash
python experiments/final/reproduce_final.py --rebuild-cache --out CuleX.csv
python evaluation/validate_submission.py CuleX.csv
```

- Runtime: about 45 min on a laptop CPU (Apple M-series; 8 seeds, ~5 min each, plus cache building).
- Expected output: sha256 `7be53b3e1bdc8542a3c22d40c6adc6a10a98a1276942bfc7f3da84faae173ca3`,
  label counts H 69,151 / A 55,177 / D 34,529 / X 40,462 (199,319 frames). Reference manifest:
  `experiments/final/manifest_reference.json`.
- The script writes `experiments/final/manifest.json` (config, library versions, sha256, label counts).
- `--check FILE` compares the regenerated labels with an existing file frame by frame (`frames_differing: 0` = exact).
- The model is fully defined by `experiments/final/config.json` (network, inputs, training loss, extra degradations,
  decoding, seeds, pre-kickoff rule).

**Fast route without retraining (recommended on another machine).** The 8 trained networks are in
`experiments/final/weights/` (seed_0.pt … seed_7.pt, 5.5 MB, trained on the Metrica sample data only):

```bash
python experiments/final/reproduce_final.py --rebuild-cache --weights experiments/final/weights --out CuleX.csv
```

This skips training (about 1–2 min) and gives the same file (verified: same sha256, 0 frames differing). It still needs the
Metrica data, because the decoder's transition matrix is fitted on those games.

**Determinism.** Training is seeded and runs on CPU; we reproduced the file bit-for-bit 6 times on the same machine (one of
them from a clean copy of this repository in a fresh environment, all caches rebuilt, 43 min; the last one while saving the
shipped weights). Floating-point results of CPU training can differ across CPU architectures / BLAS builds; the
`--weights` route avoids training and is the most stable way to regenerate the file elsewhere. If your sha256 differs,
please compare with `--check` against our submitted file.

What the pipeline does:

1. Parses the raw ATD and the Metrica games into caches (`experiments/atd_cache.py`, `experiments/external_metrica.py`).
2. Builds the degraded Metrica training bench (`experiments/bench.py`; fixed seeds) plus 2 extra degradations per game.
3. Trains the GRU per seed (`experiments/e18_boundary.py` loss; `experiments/e9_autoresearch.py` features/steps).
4. Viterbi-decodes each graded half of BAR-ATH (`experiments/e13_autoresearch.py`, `experiments/methods.py`).
5. Sets `D` on the graded frames before the 2nd-half kick-off — the organisers publish that these 2,934 frames
   (110,808–113,741) are all dead ball (FAQ.md).
6. Writes `frame,label`; `X` only on the ungraded frames (before kick-off, halftime, after the final whistle).

## 4. Regenerate the Referee Model

Run from its folder, with the repository root on the Python path (it uses the organisers' `games.py`):

```bash
cd experiments/referee_v3
PYTHONPATH=../.. python tests/test_synthetic.py      # ~10 s, made-up match with a known answer, no data needed
PYTHONPATH=../.. python make_submission.py --team referee_v3 --out ../../evaluation/results
PYTHONPATH=../.. python export_dead_ball.py --out ../../data/cache/referee_dead_ball.csv   # dead-ball cause per frame
cd ../..
python evaluation/validate_submission.py evaluation/results/referee_v3.csv
```

(Windows: `set PYTHONPATH=..\..` first.) Expected: sha256 `83581bdf1aa39b1cd1a0b7bdb20acc6a7f7687f7f2dc8a6e45c5ba773e603cfc`,
H 62,126 / A 34,346 / D 62,385 / X 40,462; 223 possessions, 41.6 min dead ball. The rules are deterministic.
More on the model, its versions (v1–v3) and its sources: `experiments/referee_v3/README.md`.

## 5. Evaluations behind the technical document

All bench numbers use the Metrica games degraded to BAR-ATH-like tracking, leave-one-game-out, a fresh degradation
seed never used for tuning, and a paired episode bootstrap (90% CI). They need the Metrica data; the GRU rows retrain
the networks (several hours on CPU; results are cached under `experiments/*/cache/`).

| Script | What it produces |
|---|---|
| `experiments/e22_referee_vs_gru.py` | Referee vs GRU side by side: spellF1, dead-ball precision/recall, stoppages found by length → `experiments/e22/summary.md` |
| `experiments/e21_referee_combo.py` + `experiments/combine_referee.py` | Combining the Referee's dead-ball / restart calls with the GRU (tested, not kept: it lowers spellF1) → `experiments/e21/summary.json` |
| `experiments/eval_table.py` | Summary table of every model (bench + label-free BAR-ATH stats + public check) → `experiments/eval_table.md`; `--md-only` rebuilds it from `eval_table.json` |
| `experiments/sanity_report.py` | Label-free BAR-ATH checks for any submission file (possessions, dead share, agreement with nearest player / tags) |
| `experiments/figures/dead_ball_by_minute.py` | Ball coverage per minute vs where each model calls dead ball (BAR-ATH, no labels) |
| `experiments/figures/feature_engineering.py` | Feature-engineering figures |
| `experiments/e9_*.py`, `e13_*.py`, `e17_*.py` … `e20_*.py` | The GRU's development steps (search, ensembles, loss, data variety), see FINDINGS |
| `experiments/referee_v3/bench/metrica_bench.py` | The Referee Model's own Metrica bench (its own event-derived labels; `--labels` to use others) |


## 6. Compliance

- No hand labels on BAR-ATH, no BAR-ATH ground truth, no human-produced data about BAR-ATH.
- The public 5-minute ground truth was used only as a correctness check, never for training, tuning or model selection.
- Training labels (GRU) come only from the Metrica sample games (labelled by Metrica; converted to per-frame H/A/D by
  our code). The Referee Model has no training data: its thresholds come from the Laws of the Game and published
  studies.
- The GRU does not use Smart Tagging, homography or video. The Referee Model uses the automatic Smart Tagging tags
  (set-piece team, phase tags), which are part of the provided bundle.

## Licence and credits

Code: MIT licence (`LICENSE-CODE`), team CuleX. Organiser starter code and evaluation: see `LICENSE` (organisers' licence).
External data: Metrica Sports sample data, https://github.com/metrica-sports/sample-data.
