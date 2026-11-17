# Findings — tactical feed, BAR-ATH

Exploration notebooks are not published (their outputs show match tracking). Cache: `experiments/atd_cache.py`.

> **How to read this file.** It is the team's research log, in the order the work happened. Labels: E = exploration
> experiments, V = evaluation tools, A = later GRU improvements, D = decisions, Q = review items. Key sections for the
> technical document: *Evaluation summary*, *A7 (Referee Model vs GRU)*, *D19 (final model)*, *Architecture choice*,
> *For the technical document — limitations / feature engineering*, *D17 (pre-kickoff rule)*, *X policy*.

## E0 — pipeline check
- Rebuilt all four shipped baselines from our cached arrays: **0 frames differ** from the shipped
  files (`experiments/e0_reproduce_baseline.py`). Parsing and `video_frame = frame_id − 1` confirmed.

## E1 — data
**Integrity**
- 199,319 records; 158,857 graded frames; mean 20.5 tracked players on graded frames.
- Ball present on 59.3% of graded frames (94,214) — matches DATA.md.

**Ball gaps are few and long, not flicker**
- 150 gaps on graded frames, median ≈ 10.7 s, p90 ≈ 34 s, longest ≈ 3.7 min.
- Only 0.3% of missing frames sit in gaps ≤ 5 s → interpolation recovers almost nothing.
  Gaps ≤ 30 s hold 56% of missing frames; the rest are even longer.
- ⇒ Ownership during gaps has to come from **team shape / context**, not ball interpolation (E5 ↑ priority, E3-interpolation ↓).

**Ball**
- Speed (finite diff): median 5.1 m/s, p90 14.3, p99 24.6.
- 3,949 graded frames with the ball outside the pitch (x ∈ [−6.9, 117], y up to 101 m — some are
  likely aerial-ball projection artefacts).
- Nearest-player distance: median 1.9 m, p75 4.1 m, p90 7.0 m.
- Nearest-team flips on 1.9% of consecutive visible frames → raw labels fragment heavily.

**Players**
- 46 slots: 24 home, 22 away. 2,218 tracklets, median lifetime 13.6 s (p10 2 s) — heavy fragmentation.
- >11 tracked players on 8.1% (home) / 4.7% (away) of graded frames (duplicates); <10 on ~14%.
- Direction of play: half 1 home mean x 49 m / away 61 m; half 2 home 56 m / away 44 m.

**Smart Tagging**
- 638 tags, 82% of graded frames covered by at least one; team: Barcelona 314, Athletic 288, N/A 36.
- SET PIECES types: throw-in 26, corner 12, goal kick 10, free kick 5, kick-off 6.

**Public 5 min, qualitative (no fitting)**
- `D` stretches: two of four have the ball missing 80–93% of the time; one has the ball outside
  the pitch 28% of the time; one has the ball fully visible but players slow.
- Mean player speed ≈ 1.1–1.5 m/s in `D` vs 2.2–3.5 m/s in live play.
- Possessions as short as 0.8 s exist — don't impose a long minimum duration.

## Implications for next experiments
1. **E2 dead ball**: combine long ball gap + low mean player speed + ball out of pitch + Smart
   Tagging set pieces. Strong, cheap signals.
2. **E5 before E3**: the 41% missing is mostly in long gaps → team-shape classifier is the main lever.
3. **E4 HMM** to absorb the 2% frame-level flicker.

## E1b — Smart-Tagging-only baseline
Notebook `experiments/02_smarttag_baseline.ipynb` → `evaluation/results/smarttag_naive.csv` (valid).
- Inputs at prediction time: Smart Tagging only. Constants calibrated label-free on tracking.
- Per-code weight = logit(share of frames where the tagged team is nearest the ball); mirrored
  pairs de-duplicated (350 → 238 team tags); box/final-third codes weight 0.
- Restart ≈ set-piece window start + 6 s (kick-off + 3.5 s), from speed/ball-visibility profiles.
  Team evidence after the restart is weak (agreement 0.54).
- Untagged stretches ≥ 8 s → `D` (≥ 20 s gaps: players 1.16 m/s vs 1.86 in tagged play).
- Full match: 153 possessions, mean 28.6 s (baseline_smooth50: 594, 7.2 s). Agrees with the
  nearest-player team on 64% of ball-visible alive frames (smooth50: 86%).
- Public 5 min (check only): macroF1 0.516, spellF1 0.256, 8 possessions vs 17 true. Too coarse:
  phases span several possessions, so short possessions are invisible.

## V1 (part 1) — Metrica public sample data imported
Source: github.com/metrica-sports/sample-data @ e706dd5 → `data/external/metrica_sample/` (gitignored).
Terms: use responsibly, **acknowledge "Metrica Sports sample data" in the technical document**.
Loader + labels: `experiments/external_metrica.py` (cache `data/cache/metrica_g{1,2,3}.npz`).

Labels from events (other games → allowed): possession = team of latest PASS/CARRY/SET PIECE/
RECOVERY/SHOT (loose/contested ball stays with last team); dead from BALL OUT / FAULT RECEIVED /
goal / shot out / offside / hand ball to the next SET PIECE, only if no live event in between.

| game | frames | possessions | mean / median s | <1 s | D share | ball present |
|---|---|---|---|---|---|---|
| 1 | 145,006 | 344 | 12.7 / 5.9 | 6% | 24.9% | 60.9% |
| 2 | 141,156 | 309 | 13.7 / 7.2 | 6% | 25.0% | 59.0% |
| 3 | 143,761 | 369 | 12.6 / 6.1 | 7% | 19.2% | 67.8% |
(BAR-ATH public window: mean 14.2 s, D 20%.)

- Ball is blanked on ~90–100% of `D` frames and 16–23% of alive frames; gaps are long (median
  5–8 s), like BAR-ATH. ⚠️ On the bench "ball missing ⇒ dead" is stronger than on BAR-ATH (where
  some `D` has the ball visible) — the degradation step must break that shortcut.
- Players: always 22 tracked, no fragmentation/duplicates → degradation must add both.
- Nearest-player team agrees with event labels on 79–82% of ball-visible alive frames.

**Bench reproduces the full-match ranking of the baselines** (spellF1):

| window | g1 | g2 | g3 | BAR-ATH full | BAR-ATH public |
|---|---|---|---|---|---|
| raw | 0.116 | 0.093 | 0.087 | 0.049 | 0.038 |
| ±25 | 0.371 | 0.359 | 0.339 | 0.200 | 0.287 |
| **±50** | **0.421** | **0.421** | **0.384** | **0.243** | 0.326 |
| ±125 | 0.293 | 0.294 | 0.318 | 0.224 | **0.424** |

All three games pick ±50, like the private full match; the public window picks ±125 (wrong).
Absolute scores are higher (clean tracking) — use the bench for **ranking**, not to predict the score.

## V1/V3/E2/E4 — bench runs (2026-11-14)
Bench: `experiments/bench.py` (easy/hard degradation, seeded); methods: `experiments/methods.py`.
spellF1 = mean over 3 games; Δ = paired episode bootstrap (198 episodes), 90% CI.

| method | easy | hard | BAR-ATH poss | BAR-ATH D share | public spellF1 (check) |
|---|---|---|---|---|---|
| baseline50 | 0.349 | 0.284 | 594 | 0.32 | 0.326 |
| E2 rules (gap ≥ 1 s or off pitch → D) | 0.427 | 0.339 | 493 | 0.43 | 0.354 |
| **E4 HMM, external-only, LOGO** | **0.553** | **0.434** | 373 | 0.40 | 0.400 |

- Speed-based dead-ball rules hurt on both variants (flag slow live play as dead).
- HMM vs rules on hard: +0.095 [+0.074, +0.117]; vs baseline50: +0.150 [+0.131, +0.170].
- HMM hyperparameters: α (emission temper) matters (0.3 ≫ 0.1, 1.0); stay_scale flat 0.25–4.
- ⚠️ BAR-ATH D share ≈ 40% vs 19–25% expected → over-calling D inside live-play ball gaps. Next: E5.

## E5 — attacking direction + team shape (2026-11-14)
- Direction per half from the goalkeeper end = BAR-ATH metadata (Barcelona left_to_right in half 1).
- Shape score (bench-trained logistic, antisymmetric): H/A accuracy 0.78 alive, 0.67 when ball missing (hard LOGO).
- As an HMM channel: spellF1 unchanged (hard −0.008 [−0.029, +0.014]); macroF1 +0.035; fewer possessions.
- Bench also over-predicts D (32–37% vs 23%); a D offset lowers spellF1 → D share is not the bottleneck.
- BAR-ATH shape HMM: 269 possessions, D 38%, agrees with nearest team 74.5% (vs 86% without shape).

## D1 — architecture decision (2026-11-14)
**Chosen:** HMM (H/A/D, Viterbi per half) as backbone; try a per-frame gradient-boosted classifier
as its emission model (E5b). Sequence nets (RNN/GRU/TCN/transformers) only as an overnight stretch (E9).

Why not RNNs / more sophisticated architectures as the main line:
- **Small effective sample**: 3 bench games ≈ 1,000 possessions. 430k frames, but neighbouring
  frames are near-duplicates, so a flexible model mostly memorises 3 games.
- **Domain shift**: bench = clean tracking we degraded ourselves + event-derived labels; BAR-ATH =
  automatic tracking + corrected-tracking labels. Flexible models learn the degradation artefacts;
  an HMM with a few dozen parameters is far less exposed.
- **The metric needs segmentation anyway**: spellF1 rewards whole possessions; raw net outputs
  flicker and would still need Viterbi/HSMM on top.
- **Time**: ~1 day for a trustworthy LOGO-validated net vs a few hours for HMM upgrades with a
  clearer expected gain, three days before the deadline.
- **Evidence so far**: richer features (team shape, E5) raised frame accuracy but not spellF1 —
  the bottleneck is segmentation, not per-frame expressiveness.

## E9 — NN autoresearch (2026-11-14, 45 min, 38 configs, 0 crashes)
`experiments/e9_autoresearch.py`; log `experiments/e9_nn/results.tsv`, `summary.json`, TensorBoard `e9_nn/tb/`.
Inputs: HMM features + team shape + direction-normalised ball position at 5 Hz, team-swap
augmentation; output log-probs decoded by Viterbi with HMM transitions. Selection on hard LOGO spellF1.

**Best: bidirectional GRU, 2 layers × 32 hidden, lr 0.01, 12 epochs, window 64 steps (12.8 s),
emission weight 0.4, stay 1.0.**

| | hard spellF1 | easy spellF1 | Δ vs HMM hard | Δ vs HMM easy |
|---|---|---|---|---|
| E4 HMM | 0.434 | 0.553 | — | — |
| **E9 GRU** | **0.525** | **0.628** | **+0.094 [+0.062, +0.127]** | **+0.074 [+0.046, +0.101]** |

- **This contradicts the D1 prior** (small data → nets not worth it): on the bench the GRU beats the
  HMM by a clear margin on both variants.
- Caveats: (1) selection bias — best of 38 configs chosen on the same hard LOGO score; easy is only a
  partial independent check (same games). (2) One seed. (3) Best config sits on the **edge of the
  search space** (max emission weight 0.4, max epochs 12, max lr 0.01) → more headroom likely.
  (4) **Under-segments**: 626 vs 1,022 bench possessions; on BAR-ATH 243 possessions (HMM 373) and
  public check 9 vs 17 possessions (spellF1 0.508 — check only). BAR-ATH D share 0.22 (closer to the
  expected 20–25% than the HMM's 0.40).

## D12 — GRU confirmation (2026-11-14)
`experiments/e9_confirm.py` → `e9_nn/confirm.json`. Best E9 config retrained with 4 seeds, scored on a
**fresh bench degradation (seed 777) the search never saw**, leave-one-game-out, vs the HMM.

| | HMM | GRU mean ± sd (4 seeds) | Δ per seed (90% CI) |
|---|---|---|---|
| hard | 0.434 | **0.493 ± 0.016** | +0.050 … +0.086, every CI > 0 |
| easy | 0.561 | **0.622 ± 0.008** | +0.053 … +0.071, every CI > 0 |

- **The win holds**: 8/8 runs beat the HMM with intervals clear of 0.
- The search score was optimistic: hard ≈ 0.49 on fresh data vs 0.525 in the search (winner's curse ≈ 0.03).
- Seed spread on hard (sd 0.016) is similar in size to the margins between configs → average several
  seeds (ensemble) for any submission.
- Still under-segments (612–654 vs 1,022 possessions on hard).
- **D1 prior revised**: with Viterbi on top and team-swap augmentation, a small bi-GRU generalises across
  games on this bench better than the hand-built HMM emissions.

## For the technical document — limitations & data notes

### Label definition mismatch between the bench and BAR-ATH (D5)
- **Bench labels** (Metrica public sample games) are derived from **event data**: possession = team of
  the latest controlling event (pass, carry, recovery, set piece, shot); dead ball = from ball out /
  foul / goal / offside to the next set piece.
- **BAR-ATH ground truth** is derived from **corrected tracking**: `D` where the corrected tracking
  marks the ball dead, otherwise the team it names as owner (CHALLENGE.md).
- The two can draw the line differently on loose balls, deflections and the exact start of a
  stoppage, so **absolute bench scores do not predict the BAR-ATH score**.
- Why the bench is still used: it ranks the published baselines in the same order as the full-match
  ground truth (the public 5 min does not), and its possession statistics are close to BAR-ATH's
  public window (mean 12.6–13.7 s vs 14.2 s; dead ball 19–25% vs 20%).
- Mitigation: the bench is used only to **rank** methods (paired episode bootstrap, easy + hard
  degradation); every candidate is also checked on BAR-ATH with a label-free sanity report
  (possession count, mean duration, dead-ball share, agreement with the nearest-player team).

### External data credit and training-data disclosure (D6)
- **External data:** Metrica Sports public sample data (3 anonymised matches, tracking + events),
  https://github.com/metrica-sports/sample-data, commit `e706dd5`. Terms: use responsibly and
  acknowledge the source in anything public.
- **Who labelled it / how much:** labels derived algorithmically from Metrica's own event data for
  those 3 matches (~430k frames, ~1,000 possessions). No human labelling by us.
- **Used for:** training (HMM emissions, team-shape score, GRU) and validation (leave-one-game-out bench).
- **Provided match (BAR-ATH):** no hand labels of any kind; the public 5-minute ground truth was used
  only as a pipeline correctness check, never for training or tuning (RULES.md).

## E11 — growing match context (idea logged 2026-11-14, not run yet)
- Current models: Viterbi uses whole halves but remembers only the current state; the bi-GRU runs over the
  whole half at prediction time but was trained on 12.8 s windows, so it can't exploit older context.
- Flaws: look-back-only discards legal future context (offline task); long history adds little about the
  current owner; 6 bench halves are too few to learn minute-scale patterns; early-error propagation;
  long-sequence training cost; long-run statistics transfer worst from clean bench to BAR-ATH.
- Plan: (b) match-context inputs, (c) test-time self-adaptation on BAR-ATH confident frames (algorithmic
  pseudo-labels, allowed), (a) long-window stateful GRU overnight. No full-history transformer.

### Bench degradation ("damage") and data augmentation
Code: `experiments/bench.py` (degradation), `experiments/methods.py`, `experiments/team_shape.py`,
`experiments/e9_autoresearch.py` (augmentation). All randomness is seeded (bench: seed 2027 + game;
fresh confirmation bench: 777 + game) so every variant is reproducible.

**Why:** the Metrica sample tracking is clean (22 players always, ball blanked almost only during
dead time), whereas BAR-ATH is automatic (≈20.5 players tracked, duplicates on ≈6% of frames, ball
missing on 41% of graded frames in long gaps, ball sometimes visible during dead time). Models
validated on clean data would over-trust signals that BAR-ATH doesn't provide.

**Preprocessing (not damage):** Metrica coordinates scaled to metres (105 × 68); player speed computed
from frame-to-frame displacement × 25, smoothed over 5 frames (BAR-ATH provides speed directly).

**Easy variant** (player-tracking damage only):
1. **Dropouts:** per slot, gaps start at exponentially distributed intervals (mean 60 s) and last an
   exponential duration (mean 4 s) → ≈20.6 players tracked on average (BAR-ATH 20.5).
2. **Duplicates:** at random moments (≈6% of frames covered, mean episode 2 s) a random player is
   copied, with 0.5 m Gaussian position jitter, into a spare slot of the same team.

**Hard variant** = easy + ball damage:
3. **Ball visible in dead time:** in 50% of dead-ball stretches the ball is shown frozen at the
   position where it next reappears (within 10 s) — mimics a ball sitting at the restart spot.
4. **Ball hidden in live play:** extra gaps are inserted at random live-play frames, with lengths
   drawn from BAR-ATH's own observed ball-gap lengths, until 41% of played frames lack the ball
   (BAR-ATH's rate).

**Effect** (baseline50 spellF1): clean 0.41 → easy 0.35 → hard 0.28 (BAR-ATH full match 0.24).
Both variants rank methods the same way; decisions require a gain on both (D4).

**Data augmentation (training only):**
- **Team swap:** every training game is also used with home and away swapped (labels H↔A, features
  mirrored). Used in the HMM emission counts, the team-shape score (built antisymmetric, no
  intercept) and the GRU. Prevents learning a home/away bias from 3 games.
- **Random windows:** the GRU trains on randomly placed windows (64–256 steps at 5 Hz) sampled from
  both halves of each training game.
- **Direction normalisation:** positions and team-shape features are expressed relative to each team's
  attacking direction per half (from the metadata on BAR-ATH, the goalkeeper end on the bench), so
  ends switching at half-time doesn't look like a different situation.
- **Seeds:** the final GRU is confirmed over 4 training seeds; submissions should average seeds.

## Q13 — 8-seed GRU ensemble (2026-11-14)
`experiments/e9_ensemble.py` → `e9_nn/ensemble.json`. Per-step log-probs averaged over seeds, then Viterbi.
Fresh bench degradation (seed 777), leave-one-game-out, vs HMM (paired 90% CI).

| seeds averaged | hard spellF1 | Δ vs HMM hard | easy spellF1 | Δ vs HMM easy |
|---|---|---|---|---|
| 1 | 0.485 | +0.050 [+0.021, +0.079] | 0.632 | +0.071 [+0.042, +0.099] |
| 2 | 0.515 | +0.080 | 0.623 | +0.062 |
| 4 | 0.514 | +0.081 | 0.649 | +0.086 |
| **8** | **0.520** | **+0.087 [+0.059, +0.114]** | **0.663** | **+0.101 [+0.074, +0.127]** |

- Averaging seeds helps on both variants (+0.035 hard, +0.031 easy from 1 → 8 seeds).
- BAR-ATH (`nn_e9_ens8.csv`, valid): **215 possessions**, D share 0.24, home share 0.58, agrees with the
  nearest-player team on 76% of ball-visible alive frames (HMM 86%). Public check (not selection):
  macroF1 0.774 (best of all files), spellF1 0.523, **9 vs 17 possessions**.
- ⚠️ **Under-segmentation is now the main weakness**: ~600 vs 1,022 possessions on the bench, 215 on
  BAR-ATH (≈360 expected from the public window's rate). Top target for D13 (switch-cost / duration term).

### Figure: key moments from Smart Tagging
`experiments/figures/key_moments_smart_tagging.png` — goals, kick-offs, corners, free kicks, goal kicks and
throw-ins on a match-minute axis, coloured by tagged team (hollow = team not tagged), with attacking
direction per half as a text annotation. Set-piece times are estimated restarts (window start + 6 s;
kick-offs + 3.5 s). Smart Tagging contains **no penalty** tag. It tags **4 goals** (BAR 23', ATH 31',
ATH 42', BAR 74'), each followed by a "Kick Off (After Goal)" — tags are automatic, so check against the
video before quoting a score. Team colours validated for colour-blind separation.

### Figure: territory and key moments (combined)
`experiments/figures/territory_and_key_moments.png` — top: each team's average player position (1-min
rolling median), expressed towards the goal Barcelona attacks (2nd half flipped so "up" always means
play in Athletic's half); bottom: Smart Tagging set pieces; vertical lines = tagged goals across both.
Reading: both lines move together (players follow the ball), so the top panel shows *where play is*
(territory), not who has the ball. Sharp dips of both lines to deep positions line up with goals and
kick-offs (teams resetting into their own halves), i.e. restarts are visible in the tracking itself.

## D13 — overnight GRU search against under-segmentation (2026-11-15, 3 h, 20 configs, 0 crashes)
`experiments/e13_autoresearch.py` → `e13_nn/` (results.tsv, summary.json, TensorBoard run `e13_overnight`).
- **Winner:** bi-GRU 3×64, dropout 0.1, lr 0.003, 20 epochs, window 12.8 s; decoding: emission weight 0.5,
  leave-state scale 4, H↔A change multiplier 10. Search score 0.577 hard (previous config re-decoded: 0.528).
- **Fresh bench (seed 777, never searched), 4-seed ensembles, paired episode bootstrap:**
  hard **0.573 vs 0.514**, Δ +0.058 [+0.041, +0.078]; easy **0.699 vs 0.649**, Δ +0.052 [+0.033, +0.070].
  Confirmation score ≈ search score (0.573 vs 0.577) → little selection optimism this time.
- **Under-segmentation reduced, not solved:** bench possessions 763 hard / 841 easy (previous 602 / 734; truth 1,022).
- **Where the gain came from:** most of it is decoding (cheaper possession changes: stay 4, ha 10); the
  re-decoded old config already gained +0.05 in-search. Bigger/deeper nets and lower lr added the rest.
  Long windows (E11a, 300–1,500 steps) never made the top 8 → long context doesn't help here.
- **Edge of search space:** ha = 10 is the grid maximum → a follow-up could try 20–50.
- **BAR-ATH `nn_best.csv` (8 seeds, valid):** 250 possessions, mean 18.8 s, D 0.26, home 0.61, agrees with
  nearest team 0.755, with Smart Tagging 0.869; no sanity flags (previous file flagged "possessions low").
  Public check (not selection): spellF1 **0.600**, macroF1 0.780, 11 vs 17 possessions — consistent with the bench.

## Evaluation summary — all submission files (rebuilt 2026-11-16 after D19, one protocol)
Generated by `python experiments/eval_table.py` → `experiments/eval_table.md` / `eval_table.json` (no hand-typed numbers; cite this table).
All bench numbers on the **fresh** bench (seed 777, never tuned on); all GRU rows 4-seed ensembles (submission files use 8).
Final file: **`nn_a4.csv`** (D19). Earlier finals kept as rows: `nn_a3` (D18), `nn_final` (D17).

| File | Method | Bench hard | Bench easy | BAR-ATH poss | D share | Agree nearest | Agree tags | Public spellF1 / macroF1 / poss |
|---|---|---|---|---|---|---|---|---|
| nn_a4 | **Final (D19).** nn_a3 trained on 3 random hard degradations instead of 1 (A4); change-point head also used in decoding; pre-kickoff = D | 0.684 | 0.817 | 316 | 0.22 | 0.76 | 0.84 | 0.640 / 0.858 / 18 |
| nn_a3 | D18: nn_final's network trained with loss ×5 within 1 s of a label change + change-point head (A3); pre-kickoff = D | 0.646 | 0.804 | 346 | 0.31 | 0.76 | 0.84 | 0.491 / 0.851 / 18 |
| nn_final | D17: bi-GRU 3×64, no team-shape inputs, Viterbi; 2nd-half pre-kickoff = D (D17) | 0.554 | 0.741 | 288 | 0.29 | 0.76 | 0.86 | 0.573 / 0.866 / 13 |
| nn_noshape | nn_final before the pre-kickoff rule (same model) | 0.554 | 0.741 | 291 | 0.28 | 0.76 | 0.86 | 0.573 / 0.866 / 13 |
| nn_best | bi-GRU 3×64 with team-shape inputs (D13) | 0.573 | 0.699 | 250 | 0.26 | 0.75 | 0.87 | 0.600 / 0.780 / 11 |
| nn_e9_ens8 | bi-GRU 2×32 (E9) | 0.514 | 0.649 | 216 | 0.24 | 0.76 | 0.87 | 0.523 / 0.774 / 9 |
| nn_e9 | bi-GRU 2×32 (E9), 1 seed | 0.493 ± 0.016 | 0.622 ± 0.008 | 244 | 0.22 | 0.75 | 0.87 | 0.508 / 0.687 / 9 |
| hmm_ext_hard | HMM, hand-binned emissions (E4) | 0.433 | 0.561 | 373 | 0.40 | 0.86 | 0.79 | 0.400 / 0.720 / 20 |
| hmm_ext_hard_shape | HMM + team-shape channel (E5) | 0.452 | 0.554 | 270 | 0.38 | 0.75 | 0.85 | 0.413 / 0.766 / 13 |
| rules_e2 | dead-ball rules + smoothing (E2) | 0.327 | 0.423 | 493 | 0.43 | 0.86 | 0.75 | 0.353 / 0.678 / 26 |
| baseline_smooth50 | organisers' baseline (nearest player, 50-frame vote) | 0.290 | 0.350 | 594 | 0.32 | 0.86 | 0.74 | 0.326 / 0.621 / 26 |
| smarttag_naive | 🏷️ Smart Tagging only (E1b) | † | † | 153 | 0.31 | 0.64 | 0.98 | 0.256 / 0.516 / 8 |

Bench: fresh degradation (seed 777, never searched or tuned on), leave-one-game-out, mean spellF1 over the 3 Metrica games (1,022 true possessions). GRU rows = 4-seed ensembles (submission files use 8); nn_e9 = mean ± sd of 4 single seeds. nn_noshape and nn_final have the same bench score (the pre-kickoff rule has no bench equivalent). \* package run on tracking only (no Smart Tagging on the bench); † not testable on the bench. 🏷️ = uses Smart Tagging.
BAR-ATH: label-free output stats on graded frames (expected ≈360 possessions, 15–35% dead ball). Public 5 min: correctness check only, 17 true possessions, noise ≈ ±0.04 — never used for selection.

Final model vs references on the same fresh bench (paired episode bootstrap, 90% CI, 198 episodes):

| | Final − nn_a3 (D18) | Final − nn_final (D17) | Final − HMM | Final − baseline50 |
|---|---|---|---|---|
| hard | +0.039 [+0.021, +0.055] | +0.129 [+0.105, +0.152] | +0.251 [+0.223, +0.277] | +0.394 [+0.365, +0.423] |
| easy | +0.014 [-0.003, +0.031] | +0.077 [+0.058, +0.098] | +0.256 [+0.231, +0.279] | +0.468 [+0.445, +0.492] |

Notes: smarttag_naive's tag agreement (0.98) is circular — it is built from the tags. On the fresh bench the HMM + shape channel
scores slightly above the plain HMM on hard (0.452 vs 0.433) and below on easy (0.554 vs 0.561) — still mixed, as in Q10.
**Public window** (check only, never used for selection): nn_final 0.573 (13 possessions) → nn_a3 0.491 (18) → nn_a4 0.640 (18);
truth 17 possessions. These swings (±0.08) are larger than the bench differences they would suggest and go in both directions,
which is what 17 possessions of noise looks like; the decisions rest on the fresh bench (198 episodes per variant) and the
label-free BAR-ATH checks. macroF1 stays 0.85–0.87 for all three, so frame alignment is fine.

## V4 — label-free sanity report (2026-11-14; table refreshed 2026-11-16 after D19)
`python experiments/sanity_report.py [files…]` (default: every CSV in `evaluation/results/`). No ground truth.
Flags: possessions outside 250–500 (public-window rate ≈ 360/match), dead-ball share outside 15–35%, any X on graded frames.

| file | poss | mean s | D | home | agree nearest | agree tags | flags |
|---|---|---|---|---|---|---|---|
| **nn_a4 (final, D19)** | 316 | 15.7 | 0.22 | 0.56 | 0.76 | 0.84 | ok |
| nn_a3 (D18) | 346 | 12.8 | 0.31 | 0.62 | 0.76 | 0.84 | ok |
| nn_final | 288 | 15.6 | 0.29 | 0.61 | 0.76 | 0.86 | ok |
| nn_noshape | 291 | 15.6 | 0.28 | 0.61 | 0.76 | 0.86 | ok |
| nn_best | 250 | 18.8 | 0.26 | 0.61 | 0.75 | 0.87 | ok |
| nn_e9_ens8 | 216 | 22.4 | 0.24 | 0.58 | 0.76 | 0.87 | possessions low |
| nn_e9 | 244 | 20.2 | 0.22 | 0.64 | 0.75 | 0.87 | possessions low |
| hmm_ext_hard | 373 | 10.2 | 0.40 | 0.62 | 0.86 | 0.79 | D high |
| hmm_ext_hard_shape | 270 | 14.5 | 0.38 | 0.60 | 0.75 | 0.85 | D high |
| rules_e2 | 493 | 7.4 | 0.43 | 0.59 | 0.86 | 0.75 | D high |
| baseline_smooth50 | 594 | 7.2 | 0.32 | 0.57 | 0.86 | 0.74 | possessions high |
| smarttag_naive | 153 | 28.6 | 0.31 | 0.64 | 0.64 | (0.98, built from tags) | possessions low |

- The GRU agrees *most* with the Smart Tagging team evidence (0.87 vs HMM 0.79) while agreeing *less*
  with the nearest-player team (0.76 vs 0.86): it follows phases of play, not momentary ball proximity —
  consistent with it merging short possessions. The A3 loss (nn_a3) splits more: 346 possessions (closest of the GRUs to ≈360),
  mean 12.8 s, tag agreement slightly lower (0.84) as it follows short possessions inside phases.

## E5c — GRU ablation without team-shape inputs (2026-11-15)
`experiments/e5c_ablation.py` → `e5c_ablation.json`. Same D13 winner config and decoding, fresh bench (seed 777),
4-seed ensembles, paired episode bootstrap (with shape minus without).
| level | with shape | without shape | Δ (with − without) [90% CI] | possessions (with / without; truth 1,022) |
|---|---|---|---|---|
| hard | 0.573 | 0.554 | +0.018 [−0.004, +0.044] | 763 / 680 |
| easy | 0.699 | 0.741 | **−0.041 [−0.058, −0.023]** | 841 / 859 |

- Mixed result. On easy, the shape inputs **hurt** significantly. On hard they help a little, but the CI crosses 0.
- Shape inputs push possession counts up on hard only (+83), so they don't clearly address under-segmentation.
- Decision rule (D4): keep a component only if it helps on both variants. It doesn't clearly help on either.
- Not yet applied: `experiments/final/config.json` still uses the full inputs. Dropping them is a one-line config change; then `reproduce_final.py` must be rerun.

## For the technical document — feature engineering (2026-11-15)
Script: `experiments/figures/feature_engineering.py` → `experiments/figures/fe_*.png`, numbers in `fe_numbers.json`.
Label-conditioned panels use the Metrica bench (hard variant, which the final model trains on). BAR-ATH panels use no labels. The public 5-minute ground truth is not read.

### Inputs to the final model
The tracking feed gives positions only: ball state and ball-owning team are empty, and the ball is missing on 41% of graded frames. So every input is derived from player and ball positions (`methods.features`, `e9_autoresearch.frame_inputs`).

| # | Input | How it is calculated | What it tells the model |
|---|---|---|---|
| 1–3 | Nearest team (home / away / ball not seen) | Euclidean distance from the ball to every observed track slot; team of the closest slot (`team_of_slot` from the ATD metadata). "Not seen" when the ball is missing | Who is probably on the ball |
| 4 | Nearest distance | Distance to that closest slot, clipped to 20 m, ÷ 20. Set to 20 m when the ball is missing | How much to trust input 1–3 |
| 5 | Ball-gap length | Every frame inside a run of missing ball gets the run's length in seconds; 0 when visible. Stored as log(1 + s) ÷ 4 | Long gaps make dead ball more likely |
| 6 | Ball off pitch | Ball visible and more than 0.5 m outside the 105.3 × 68 m pitch | Throw-ins, goal kicks, corners |
| 7 | Player speed | Mean speed of all observed players, gaps filled forward/backward, 1-s centred rolling mean, ÷ 4 | Play slows down at stoppages |
| 8 | Ball x, attack-normalised | (x − 52.65) ÷ 52.65, multiplied by +1 or −1 so positive always means "towards the goal home attacks". Direction per half from each team's deepest player; matches the metadata (Barcelona left to right in the first half) | Field position without the half-time switch |
| 9 | Ball y | (y − 34) ÷ 34 | Wing vs centre |

**Processing after the features:**
- **5 Hz steps.** Each block of 5 frames (0.2 s) is averaged into one step. The model runs on steps, and the output is repeated back to 25 fps.
- **Team-swap augmentation.** Every training game is also fed in with home and away swapped (inputs 1–2, 8 and the labels flipped). The model can't learn "home has the ball more", only the evidence.
- **Missing values.** Ball inputs become 0 when the ball is missing; input 3 and input 5 tell the model why.

**Tested and dropped:** six team-shape inputs (home minus away, measured along each team's own attack: mean / most advanced / deepest forward position, width, length, forward velocity). They clearly hurt on the easy bench (E5c, −0.041) and didn't clearly help on hard, so the final model doesn't use them (D16).

### Figures
**`fe_timeline.png` — what the model sees over two minutes of BAR-ATH.** Five strips: nearest team, nearest distance, ball-gap length, player speed, attack-normalised ball x. Shows the two typical ball gaps (~10 s), players slowing to ~1 m/s before a gap (a likely stoppage), and ball x crossing from home's attacking half to its own half as possession changes.

**`fe_ball_gaps.png` — the missing ball.** Left: gap lengths weighted by time. BAR-ATH misses the ball on 40.7% of graded frames, in 150 gaps with median 10.7 s; 99.7% of missing time is in gaps longer than 5 s, so interpolation can't recover it. The hard bench matches this (41.5% missing; clean Metrica 37.4%). Right: share of frames that are dead ball by gap length. Visible ball: 17% dead; gaps under 1 s: 0%; gaps over 2 s: 25–41%. A missing ball roughly doubles the chance of dead ball but doesn't decide it, which is why it's an input and not a rule. *Caveat:* on the hard bench, where the ball is hidden is partly our degradation choice (ball hidden in live play to match BAR-ATH; frozen ball shown in half of dead stretches), so this panel reflects the bench design too.

**`fe_nearest_player.png` — how far to trust the nearest player.** Left: on live, ball-visible bench frames, the nearest team is the team in possession 76% of the time overall, 96% when the closest player is within 1 m (half of all frames), and no better than a coin flip beyond about 4 m (passes in flight, loose balls). Right: during dead balls the nearest player is further away (median 2.6 m vs 1.0 m live). This is why distance is an input next to the team.

**`fe_player_speed.png` — players slow down when the ball is dead.** Median mean player speed 0.98 m/s in dead ball vs 1.88 m/s live. The distributions overlap (slow build-up play), so a speed threshold alone flagged slow live play as dead (rejected in E2), but as a soft input it helps.

**`fe_team_shape.png` — the dropped team-shape signal.** Difference in mean forward position (home minus away, each along its own attack). It leans the right way (median +10.0 m when home has the ball, −11.7 m when away has it) but the sign is right only 63% of the time. Too weak to add over the ball-based inputs, consistent with E5c.

### Figure: ball coverage vs dead-ball calls per minute (2026-11-16, after D19)
`experiments/figures/dead_ball_by_minute.png` (`python experiments/figures/dead_ball_by_minute.py [files…]`, default nn_a4 nn_a3
nn_final; numbers in `dead_ball_by_minute.json`). Top: % of graded frames with a tracked ball per minute of BAR-ATH (graded mean
59%). Below, one panel per model: % of graded frames called dead ball (grey bars) with % ball missing overlaid (blue line). No labels used.
Caption: *The final model (nn_a4) ties dead-ball calls to ball gaps far less than the earlier finals: per-minute correlation 0.21
vs 0.48 (nn_a3) and 0.43 (nn_final); dead-ball share 22% vs 31% / 29%.* Long gaps that the earlier models called mostly dead
(e.g. minutes 47–49 and ~107) are mostly live for nn_a4, while the pre-kickoff minutes (73–75, D17 rule) stay 100% dead in all
three. Consistent with A4's aim (training on more varied gap patterns so a missing ball is not read as a stoppage), and better on
the fresh bench, but it cannot be verified on BAR-ATH without labels — the largest open uncertainty of the final file.

### Final file after D16 (2026-11-15)
`python experiments/final/reproduce_final.py --out evaluation/results/nn_noshape.csv` (config `inputs: no_shape`), 815 s, valid,
sha256 `b3b8fb3e…c998c95` (full hash in `experiments/final/manifest.json`).
BAR-ATH: 291 possessions (nn_best 250), mean 15.6 s, D 0.28, agrees with nearest team 0.765, with Smart Tagging 0.862, no flags.
Public check (not selection): spellF1 0.573 (nn_best 0.600), macroF1 0.866 (0.780), 13 vs 17 possessions. The spellF1 difference is inside the
public window's noise (17 possessions, band ≈ 0.04) and goes the other way from the easy bench, so it doesn't change the D16 decision.

## A1 + A2 — decoding re-tune and GRU+HMM blend for the final model (2026-11-15, ~45 min)
`experiments/e17_decode_blend.py` → `e17/` (grid.tsv, summary.json, run.log). Final no-shape GRU (D16), tuned on the
standard hard bench (LOGO, 4 seeds), confirmed on the fresh bench (seed 777) vs the current final decoding
(emission weight 0.5, leave-state 4, possession-change 10), paired episode bootstrap.

| | Tuned setting (w, emission weight, leave-state, possession-change) | Tune (hard) | Fresh hard Δ [90% CI] | Fresh easy Δ [90% CI] | Kept? |
|---|---|---|---|---|---|
| current final | (1, 0.5, 4, 10) | 0.569 (678 poss) | — (0.554, 680 poss) | — (0.741, 859 poss) | — |
| A1 decode re-tune | (1, 0.8, 1, 5) | 0.574 (672) | +0.000 [−0.003, +0.004] | −0.003 [−0.008, +0.001] | no |
| A2 GRU+HMM blend | (0.8, 0.3, 4, 20) | 0.573 (706) | +0.001 [−0.008, +0.011] | −0.000 [−0.011, +0.009] | no |

- **Both neutral.** Decoding is a flat plateau for this model: 56 of 100 A1 settings score within 0.009 of the best.
- **Decoding can't fix under-segmentation.** Across the sensible settings, possessions stay at about 615–720 (truth 1,022), even
  with possession changes made 100× cheaper. The GRU's outputs are confident enough that the decoder rarely overrides them,
  so the merging happens inside the network. The remaining lever is training (A3), not decoding.
- Extreme settings (high leave-state × high possession-change) break the transition matrix and score 0; harmless, excluded by the max.
- Final config unchanged.

## A3 — training against merged possessions (2026-11-15, ~4 h, 7 variants)
`experiments/e18_boundary.py` → `e18/` (results.tsv, summary.json, run.log); head-to-head `e18_head_to_head.py` → `e18/head_to_head.json`.
Same network, inputs and decoder as the final model (bi-GRU 3×64, no team shape, Viterbi); only the **training loss** changes:
- **bw** = boundary weighting: steps within ±r steps (0.2 s each) of a label change (H↔A, into/out of D) count bw× in the loss.
  Plain cross-entropy lets the network ignore the ~2% of steps near a change, so merging short possessions is cheap for it.
- **cp** = change-point head: a second output predicts "the state changes here", trained jointly (weight cp). It could also scale
  the leave-state cost per step in decoding (beta); the tuned beta was 0 for every variant, so the head only acts as a training signal.

Protocol as A1/A2: tune on the standard hard bench (LOGO, 4 seeds, decoding re-tuned per variant over emission weight 0.3/0.5/0.8 ×
leave-state 1/2/4 × possession-change 5/10/20), the top 2 confirmed on the fresh bench (seed 777) vs the current final, paired episode bootstrap.

| Variant | Decoding (emission w, leave-state, poss-change) | Tune hard (poss) | Fresh hard Δ vs final [90% CI] (poss) | Fresh easy Δ vs final [90% CI] (poss) | Kept? |
|---|---|---|---|---|---|
| current final | (0.5, 4, 10) | 0.569 (678) | — 0.554 (680) | — 0.741 (859) | — |
| bw3_r5 (×3, ±1 s) | (0.5, 1, 20) | 0.634 (836) | | | not confirmed |
| **bw5_r5** (×5, ±1 s) | (0.3, 2, 20) | **0.646** (859) | **+0.074** [+0.052, +0.094] → 0.629 (842) | **+0.056** [+0.038, +0.076] → 0.798 (1,007) | ✅ |
| bw10_r5 (×10, ±1 s) | (0.3, 1, 5) | 0.637 (917) | | | not confirmed |
| bw5_r10 (×5, ±2 s) | (0.3, 2, 20) | 0.638 (855) | | | not confirmed |
| cp0.5 (head only) | (0.5, 4, 20) | 0.577 (702) | | | not confirmed |
| cp1 (head only) | (0.3, 4, 20) | 0.578 (699) | | | not confirmed |
| **bw5_cp1** (×5, ±1 s + head) | (0.3, 4, 20) | 0.644 (871) | **+0.091** [+0.071, +0.110] → 0.646 (803) | **+0.063** [+0.043, +0.084] → 0.804 (1,017) | ✅ |

Head-to-head on the fresh bench, bw5_cp1 − bw5_r5: hard **+0.017 [+0.002, +0.032]**, easy +0.007 [−0.007, +0.021] → bw5_cp1
passes the keep rule against bw5_r5 too.

- **Biggest gain of the project after the GRU itself.** Both confirmed variants beat the final model on both bench variants with
  CIs well clear of 0; the gain (+0.06 to +0.09) is larger than D13's (+0.05 to +0.06).
- **It fixes the diagnosed weakness.** Fresh-bench possessions 680 → 803–842 (hard) and 859 → 1,007–1,017 (easy), truth 1,022.
  A1/A2 showed decoding can't add possessions; changing what the network is penalised for does.
- **Boundary weighting does the work; the change-point head alone doesn't** (cp0.5/cp1 ≈ +0.01 on tune, possessions ~700).
  Added on top of ×5 weighting, the head gives a small extra gain on hard (+0.017) — it helps as an auxiliary training signal.
- ×3–×10 and ±1–2 s all land within 0.012 on tune → a broad optimum, not a lucky setting.
- Caveats: the choice between the two confirmed variants uses the fresh bench (tune had them tied, 0.646 vs 0.644), so bw5_cp1's
  fresh score is slightly optimistic; its margin over the current final (+0.09 / +0.06) is far larger than that selection effect.
  BAR-ATH effect not yet measured (needs the 8-seed file + sanity report).
- **Adopted as the final model (D18, below).**

## D18 — final model switched to A3 bw5_cp1 (2026-11-15)
- `experiments/final/config.json`: adds `train_loss: {bw: 5, r: 5, cp: 1}` and decode (emission weight 0.3, leave-state 4,
  possession-change 20); network, inputs (no team shape), seeds 0–7, training bench (hard) and pre-kickoff rule unchanged.
  `reproduce_final.py` trains with `e18_boundary.train` when `train_loss` is set (same code as the A3 experiment).
  Previous config kept as `config_d17.json` (e17/e18 baselines are pinned to it).
- Output `evaluation/results/nn_a3.csv`: valid; sha256 19d316a3…3511; 847 s on CPU; H 68,593 / A 41,737 / D 48,527 / X 40,462
  (X only on the 40,462 ungraded frames); pre-kickoff rule changed 1,932 frames to D.
- BAR-ATH sanity: 346 possessions (was 288; expected ≈360), mean 12.8 s, D share 0.31, no flags.
- Fresh bench vs previous final: hard +0.091 [+0.071, +0.110], easy +0.063 [+0.043, +0.084].
- Public 5-min check: 0.491 / macroF1 0.851 / 18 possessions (previous 0.573 / 0.866 / 13) — see the Evaluation summary note.
- Reproducibility: second clean run of `reproduce_final.py --check evaluation/results/nn_a3.csv` → identical sha256, **0 frames differing** (829 s; `experiments/final/manifest.json`, first run in `manifest_a3_run1.json`).
- Submission order: **`nn_a3.csv`**, fallbacks `nn_final.csv`, `nn_noshape.csv`, `nn_best.csv`, `hmm_ext_hard.csv`.

## A4 — more varied training data for the A3 model (2026-11-15, ~4 h, 3 variants)
`experiments/e19_data_mix.py` → `e19/` (results.tsv, summary.json, run.log); head-to-head `e19_head_to_head.py` → `e19/head_to_head.json`.
Base = the D18 final (A3 bw5_cp1). Only the training data / length changes; the held-out game never appears in training in any
degradation; extra degradation seeds (5000+g, 6000+g) never overlap the fresh-bench seed (777+g).

| Variant | Decoding (emission w, leave-state, poss-change, cp-beta) | Tune hard (poss) | Fresh hard Δ vs D18 [90% CI] (poss) | Fresh easy Δ vs D18 [90% CI] (poss) | Kept? |
|---|---|---|---|---|---|
| D18 final (bw5_cp1) | (0.3, 4, 20, 0) | 0.644 (871) | — 0.646 (803) | — 0.804 (1,017) | — |
| **draws3**: + 2 extra random degradations of the same level | (0.3, 1, 5, 0.5) | **0.699** (839) | **+0.039** [+0.021, +0.055] → 0.684 (814) | +0.014 [−0.003, +0.031] → 0.817 (977) | ✅ |
| mix: + the other degradation level (hard + easy copies) | (0.3, 1, 20, 0.5) | 0.673 (896) | **+0.043** [+0.028, +0.060] → 0.688 (833) | +0.002 [−0.017, +0.020] → 0.805 (986) | ✅ |
| ep30: 30 epochs instead of 20 | (0.3, 4, 20, 0) | 0.667 (857) | | | not confirmed (3rd on tune) |

Head-to-head on the fresh bench, draws3 − mix: hard −0.004 [−0.019, +0.010], easy +0.012 [−0.003, +0.028] → tied.

- **More degradation variety helps.** Seeing the same games with different ball-gap and dropout patterns teaches the network
  what is noise, not signal: +0.04 on the fresh hard bench, no loss on easy. More epochs alone help less on tune (0.667) than
  more varied data (0.673–0.699), so the gain is from variety, not just longer training.
- With more varied training, the tuned decoder now **uses the change-point head** (beta 0.5: leaving a state is cheaper where the
  head expects a change) — in A3 the head was a training signal only.
- Preferred variant: **draws3** — best on the tuning bench (the pre-set selection rule), tied with mix head-to-head, slightly better
  on easy, and keeps training inside one degradation level (same as the final model's `train_bench: hard`).
- Cost: 3× training data → about 45 min CPU for the 8-seed file (was 14 min).
- **Adopted as the final model (D19, below).**

## D19 — final model switched to A4 draws3 (2026-11-16)
- `experiments/final/config.json` (= `config_a4.json`): D18 model + `train_extra_draws: [5000, 6000]` (2 extra random hard
  degradations of the 3 Metrica games) + decode (emission weight 0.3, leave-state 1, possession-change 5, change-point beta 0.5).
  Previous configs: `config_a3.json` (D18), `config_d17.json` (D17).
- Output `evaluation/results/nn_a4.csv`: valid; sha256 7be53b3e…3ca3; 2,537 s CPU (3× training data);
  H 69,151 / A 55,177 / D 34,529 / X 40,462 (X only on ungraded frames); pre-kickoff rule changed 1,964 frames to D.
- Reproducibility: second clean run with `--check` → identical sha256, **0 frames differing** (2,634 s; `manifest_a4_check.json`).
- Fresh bench vs nn_a3: hard +0.039 [+0.021, +0.055], easy +0.014 [−0.003, +0.031]; vs nn_final (D17): hard +0.129, easy +0.077.
- BAR-ATH sanity: 316 possessions (nn_a3 346; expected ≈360), mean 15.7 s, no flags. **Biggest behaviour change of any switch:**
  dead-ball share 0.31 → 0.22 and home share 0.62 → 0.56. Both inside the plausible range (15–35% dead; public window 19–25% dead).
- Public 5-min check: 0.640 / macroF1 0.858 / 18 possessions (nn_a3 0.491 / 0.851 / 18).
- Submission order: **`nn_a4.csv`**, fallbacks `nn_a3.csv`, `nn_final.csv`, `nn_best.csv`, `hmm_ext_hard.csv`.

## A5 — even more degradation variety (2026-11-17, overnight, 2 variants) — not kept

Base = A4 draws3 (the D19 final). Same network, loss and protocol (tune on the standard hard bench, LOGO, 4 seeds;
confirm on the fresh bench only if a variant beats the base on tune by > 0.005).

| Variant | Training data | Tune spellF1 (hard) | Possessions (truth 1,022) | Decode (alpha, stay, ha, beta) |
|---|---|---|---|---|
| **draws3 (final)** | bench level + 2 extra random degradations | **0.699** | 839 | (0.3, 1, 5, 0.5) |
| draws5 | bench level + 4 extra random degradations | 0.689 | 846 | (0.3, 2, 10, 0.5) |
| draws3_mix | draws3 + the same 3 draws at the other level | 0.693 | 846 | (0.3, 1, 10, 0.5) |

Neither variant beat draws3, so nothing went to the fresh bench and the final model is unchanged (D19, `nn_a4`).
Reading: the gain from degradation variety levelled off after 3 copies. Code: `experiments/e20_more_variety.py`;
results in `experiments/e20/`. **Config frozen 2026-11-17 at D19.**

## A7 — two approaches side by side: Referee Model (logic) and GRU (learned) (2026-11-17)

**For the technical document.** The team built two independent models and keeps them separate (A6 showed that
pasting one's dead-ball calls into the other hurts, so they are compared, not merged):

- **Referee Model v3** (`experiments/referee_v3/`, colleague): logic only. Every threshold comes from the Laws of the Game
  (Laws 8, 9, 13, 15), the team's written possession rules (A1–E14) or a published study; nothing is fitted to data.
  Dead ball = ball out (staying out 1 s), vanished near a line, everyone walking with the ball still (whistle), ball
  lying still with nobody near, or a restart set-up; play restarts on a strike from rest with a taker beside it (or a
  kick-off tap at the centre spot, teams in own halves); possession changes on control (3 s / 3 touches / 7 passes);
  Smart Tagging phase tags carry possession through long ball gaps. Regenerates exactly
  (`referee_v3.csv`, sha256 83581bdf…3cfc; identical on 2 runs here and from the clean public-repo copy in a fresh
  environment; ~5 s).
- **GRU** (`experiments/`, final D19): learned from the Metrica games, degraded to look like BAR-ATH's tracking.

**Summary (fresh bench, hard; full tables below).**

| | Referee Model (logic) | GRU (learned, submitted) |
|---|---|---|
| spellF1 (challenge metric) | 0.448 | **0.684** (+0.24, 90% CI [+0.21, +0.27]) |
| macroF1 | 0.719 | **0.831** |
| Dead ball: share of true dead frames found (recall) | **0.90** | 0.82 |
| Dead ball: share of its dead calls that are right (precision) | 0.54 | **0.82** |
| Stoppages ≥ 30 s found | **98%** | 83% |
| Stoppages 15–30 s found | **94%** | 83% |
| Stoppages 1–5 s found | **61%** | 46% |
| Dead share of frames (truth 23%) | 38% | 23% |
| Possessions (truth 1,022) | 509 | 814 |
| BAR-ATH (no labels): dead ball, possessions | 41.6 min (39%), 223 | 23.0 min (22%), 316 |
| BAR-ATH: per-minute r(dead ball, ball missing) | 0.68 | 0.21 |
| Public 5 min (check only, never used to choose) | 0.514 | 0.640 |

Same Metrica bench games, same metrics (`experiments/e22_referee_vs_gru.py` → `experiments/e22/summary.md`):

| Bench | Model | spellF1 | macroF1 | Poss (truth) | D precision | D recall | D F1 | Stoppages found | Called stoppages real | Start delay (s) | Dead share (truth) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_hard | Referee (logic) | 0.433 | 0.712 | 510 (1022) | 0.54 | 0.87 | 0.66 | 0.86 of 196 | 0.58 of 254 | -0.0 | 0.38 (0.23) |
| tune_hard | GRU (learned, final) | 0.698 | 0.822 | 839 (1022) | 0.76 | 0.81 | 0.78 | 0.81 of 196 | 0.79 of 200 | +0.0 | 0.25 (0.23) |
| fresh_hard | Referee (logic) | 0.448 | 0.719 | 509 (1022) | 0.54 | 0.90 | 0.68 | 0.84 of 196 | 0.60 of 239 | -0.0 | 0.38 (0.23) |
| fresh_hard | GRU (learned, final) | 0.684 | 0.831 | 814 (1022) | 0.82 | 0.82 | 0.82 | 0.75 of 196 | 0.79 of 187 | +0.0 | 0.23 (0.23) |
| fresh_easy | Referee (logic) | 0.526 | 0.753 | 597 (1022) | 0.55 | 0.90 | 0.69 | 0.86 of 196 | 0.61 of 261 | -0.1 | 0.38 (0.23) |
| fresh_easy | GRU (learned, final) | 0.817 | 0.865 | 977 (1022) | 0.81 | 0.81 | 0.81 | 0.89 of 196 | 0.83 of 212 | +0.0 | 0.23 (0.23) |

Stoppages found, by true stoppage length (fresh hard):

| Length | n | Referee | GRU |
|---|---|---|---|
| 1-5s | 28 | 0.61 | 0.46 |
| 5-15s | 80 | 0.80 | 0.76 |
| 15-30s | 47 | 0.94 | 0.83 |
| 30-infs | 41 | 0.98 | 0.83 |

spellF1, GRU − Referee (paired episode bootstrap, 90% CI): tune_hard +0.263 [+0.227, +0.299]; fresh_hard +0.236 [+0.207, +0.265]; fresh_easy +0.290 [+0.261, +0.321]

BAR-ATH (no labels): GRU dead 23.0 min, Referee dead 41.6 min, both 15.7 min; 68% of the GRU's dead frames are also Referee-dead; label agreement 60%. Referee-only dead minutes by cause: before_kickoff 0.0, break 10.1, out 10.3, setup 0.1, still 5.5.

**Reading.**
- The Referee Model is the better *dead-ball detector by recall*: it finds 90 % of dead frames (GRU 82 %) and more
  of the true stoppages, especially long ones (≥ 30 s: 98 % vs 83 %; 15–30 s: 94 % vs 83 %) and the very short ones
  (1–5 s: 61 % vs 46 %). Its rules see a stoppage as soon as the ball leaves the pitch or play stops.
- Its cost is precision: about half its dead-ball frames are live play (D precision 0.54, GRU 0.82; it calls 38 %
  of frames dead against a true 23 %). Most of that comes from long ball gaps in live play, which the rules read as
  stoppages; on BAR-ATH the Referee-only dead time is mostly "break" (10.1 min) and "out" (10.3 min, incl. a ball
  vanishing near a line). Those false stoppages split possessions, so on the challenge metric the GRU is far ahead:
  spellF1 +0.24 (hard) / +0.29 (easy), CIs well above 0.
- The figure `figures/dead_ball_by_minute.png` shows the same on BAR-ATH without labels: the Referee's dead-ball
  calls track the minutes where the ball is missing (per-minute r = 0.68), the GRU's much less (r = 0.21).
- How the GRU improves on the logic: it learns *how much* a ball gap means a stoppage from the context (player
  speeds, nearest-player distance, gap length, what happens either side), instead of a fixed rule, and its
  possession-change cost is learned from the data. The logic model stays as the interpretable reference: every call
  it makes can be traced to a Law or rule (`export_dead_ball.py` gives the cause per frame).
- **Decision D20 (2026-11-17, user): submit the GRU only** (`CuleX.csv` = `nn_a4.csv`, sha256 7be53b3e…3ca3). The
  Referee Model is not submitted and not combined with the GRU (A6); it is published in the same repository as the
  logic-based reference, with one command to regenerate its own file, so the evaluators can check both and the
  comparison above.

## A6 — GRU + Referee Model (colleague's rules model v3) dead-ball calls (2026-11-17) — not kept

Combination (`experiments/combine_referee.py`): start from the GRU labels; (b) `dead`: frames the Referee calls dead
with cause out / vanished / break / setup -> D; (c) `restart`: after each Referee restart, the restart team until the
GRU's next label change; `both`. The published pre-kickoff rule is re-applied afterwards. Bench check
(`experiments/e21_referee_combo.py`): the D19 GRU's cached LOGO outputs vs the Referee run on the same degraded
Metrica games (tracking only; no tags exist there). Nothing tuned. spellF1, paired episode bootstrap vs GRU (90% CI):

| Variant | Tune hard | Fresh hard | Fresh easy | Dead share (truth 0.23) |
|---|---|---|---|---|
| GRU alone (final) | 0.698 | 0.684 | 0.817 | 0.23–0.25 |
| Referee alone | −0.263 [−0.299, −0.227] | −0.236 [−0.265, −0.207] | −0.290 [−0.321, −0.261] | 0.38 |
| dead | −0.046 [−0.060, −0.033] | −0.042 [−0.058, −0.028] | −0.056 [−0.070, −0.043] | 0.38–0.41 |
| restart | −0.002 [−0.007, +0.002] | −0.005 [−0.011, +0.001] | −0.003 [−0.006, −0.000] | 0.23–0.25 |
| both | −0.047 [−0.060, −0.034] | −0.045 [−0.062, −0.030] | −0.055 [−0.069, −0.042] | 0.38–0.40 |

Reading: the Referee calls ~38% of frames dead on the bench (truth 23%) and ~39% on BAR-ATH (41.6 of 105.9 min), so
its dead-ball calls cut real possessions; restart-team information is neutral to slightly negative. No variant
beats the GRU on any bench, so the final file is unchanged (D19). On BAR-ATH the `dead` variant would have moved
dead time from 23.0 to 43.5 min and changed 19.3% of graded frames.

## Architecture choice — why the final model stays a bi-GRU + Viterbi (2026-11-15)
For the technical document. Backbone: a bidirectional GRU (a gated *recurrent* network reading 0.2-s steps forward and
backward in time; offline task, so future context is allowed) gives per-step H/A/D probabilities; Viterbi with a
3-state transition matrix turns them into possession runs. What else was considered, and why it wasn't pursued:

| Alternative | Evidence / reason | Decision |
|---|---|---|
| Rules + smoothing (E2) | Bench hard 0.339; too many possessions (1,269 vs 1,022) | superseded |
| HMM with hand-binned emissions (E4) | Bench hard 0.434; kept as fallback and reference | fallback |
| Gradient-boosted per-frame classifier → HMM (E5b, D10/D11) | Does the job of the GRU's per-step output without sequence context | closed, superseded by GRU (D14) |
| TCN (dilated 1-D convolutions) | Tried in the E9 search: best 0.488 vs GRU 0.525 on the same score | not pursued |
| Long-context GRU, 1–5 min windows (E11a) | In the D13 search; never in the top 8 configurations | not pursued |
| Transformer | Needs far more data than 6 training halves (≈1,000 possessions); attention over long context also failed to help as E11a | not pursued |
| Player-level network (DeepSets / graph net over all players) | Learning from raw positions needs more data; the hand-made team-shape summary of the same information didn't help (E5c, D16) | not pursued |
| GRU + CRF (transition costs learned jointly with the network, sequence-level loss) | Most principled fix for merged possessions, since decoding can't undo the network's confident outputs (A1/A2). Roughly half a day; uncertain gain on 6 halves | A3 showed training changes do move the count; CRF remains the next step after the deadline (time) |
| Change-point head / boundary-weighted loss (A3) | Cheapest way to push the network itself to split possessions. Boundary weighting ×5 (+ change-point head) beats the final model on fresh hard +0.091 and easy +0.063, possessions 680 → 803 (truth 1,022) | **works** — proposed as the new final (D18) |

Common thread: with 3 bench games, the limit is data, not model capacity, so the work went into the bench (degradation, paired
bootstrap), decoding and seed ensembles rather than larger architectures. Credit: training data is Metrica Sports sample data.

## D17 — second-half pre-kickoff frames are dead ball (2026-11-15)
Published rule (FAQ.md, DATA.md): the 2nd half is graded from video frame 110,808 but kicks off at 113,742; those 2,934
graded frames are all `D`. The 1st half is graded from its kick-off (11,185), so it has no such stretch. Halftime itself
(90,652–110,807) is ungraded; all our files write `X` there, as the docs ask.
- `nn_noshape.csv` wrote 1,657 D and **1,277 H** on those frames. Every tracking-only file had the same issue
  (862–2,894 non-D frames); only files using Smart Tagging set-piece windows (e.g. smarttag_naive) had none.
- Why the model misses it: Metrica has no halftime frames, and only one of the six bench halves (game 1, 2nd half) has a
  long pre-kickoff stretch, so the model never learned "players walking into position = dead ball".
- Fix: after decoding, `reproduce_final.py` sets D on graded frames between each half's first graded frame and its kick-off
  (kick-off frames from `starter/load_smart_tagging.KICKOFF_VIDEO_FRAMES`; config `pre_kickoff_dead: true`).
  Uses published match boundaries only, no labels.
- Result `evaluation/results/nn_final.csv`: exactly the 1,277 frames changed, nothing else; valid; 288 possessions, D 0.29.
  Bench and public-window scores unchanged (both lie outside the affected frames). Expected effect on the full-match
  score: up to 1,277 frames (0.8%) corrected for macroF1; one fewer phantom possession at the 2nd-half start for spellF1.

## X (abstention) policy — for the technical document
Our files write `X` only on the 40,462 ungraded frames (before kick-off, halftime, after the final whistle) and never on graded
frames. On graded frames `X` counts as a miss in macroF1 (the tie-break), while spellF1 (the ranking metric) replaces it
with the previous label, so abstaining only pays when a guess would split a possession. Viterbi already smooths those
stretches, so the abstention policy (E7) was not pursued.
- **Reproducibility check (2026-11-15):** `python experiments/final/reproduce_final.py` regenerated `nn_final.csv` from raw data
  with **0 differing frames**; sha256 `71bfc6bf9f512807482c10376bb3a89fbe3ac56e3373147906d465e5056c9f87` matches; valid;
  851 s on CPU (Python 3.11.18, numpy 2.4.6, torch 2.15.0). Manifest: `experiments/final/manifest.json`.
