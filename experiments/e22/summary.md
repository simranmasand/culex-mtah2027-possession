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
