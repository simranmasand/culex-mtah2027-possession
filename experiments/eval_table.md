| File | Method | Bench hard | Bench easy | BAR-ATH poss | D share | Agree nearest | Agree tags | Public spellF1 / macroF1 / poss |
|---|---|---|---|---|---|---|---|---|
| nn_a4 | **Final (D19).** nn_a3 trained on 3 random hard degradations instead of 1 (A4); change-point head also used in decoding; pre-kickoff = D | 0.684 | 0.817 | 316 | 0.22 | 0.76 | 0.84 | 0.640 / 0.858 / 18 |
| referee_v3 | 🏷️ **Referee Model v3** (logic only: Laws of the Game rules for dead ball, restarts, control; Smart Tagging phase tags in long ball gaps) | 0.448* | 0.526* | 223 | 0.39 | 0.76 | 0.84 | 0.514 / 0.725 / 11 |
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

Bench: fresh degradation (seed 777, never searched or tuned on), leave-one-game-out, mean spellF1 over the 3 Metrica games (1,022 true possessions). GRU rows = 4-seed ensembles (submission files use 8); nn_e9 = mean ± sd of 4 single seeds. nn_noshape and nn_final have the same bench score (the pre-kickoff rule has no bench equivalent). \* Referee Model run on tracking only (no Smart Tagging on the bench); † not testable on the bench. 🏷️ = uses Smart Tagging.
BAR-ATH: label-free output stats on graded frames (expected ≈360 possessions, 15–35% dead ball). Public 5 min: correctness check only, 17 true possessions, noise ≈ ±0.04 — never used for selection.

Final model vs references on the same fresh bench (paired episode bootstrap, 90% CI, 198 episodes):

| | Final − Referee (logic) | Final − nn_a3 (D18) | Final − nn_final (D17) | Final − HMM | Final − baseline50 |
|---|---|---|---|---|---|
| hard | +0.236 [+0.207, +0.265] | +0.039 [+0.021, +0.055] | +0.129 [+0.105, +0.152] | +0.251 [+0.223, +0.277] | +0.394 [+0.365, +0.423] |
| easy | +0.290 [+0.261, +0.321] | +0.014 [-0.003, +0.031] | +0.077 [+0.058, +0.098] | +0.256 [+0.231, +0.279] | +0.468 [+0.445, +0.492] |
