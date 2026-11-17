# experiments/ — what each file is

Scripts keep the experiment numbers used in `FINDINGS.md` (E = exploration, A = later GRU improvements, D = decisions).
Start with the top README; this page is the map.

## 1. Submitted model (GRU) — needed to regenerate `CuleX.csv`

| File | Role |
|---|---|
| `final/reproduce_final.py` | **Entry point.** Builds caches, trains 8 seeds, decodes BAR-ATH, writes the file + manifest |
| `final/config.json` | The final model's full definition (D19). `config_a3.json`, `config_d17.json` = earlier finals |
| `final/requirements-final.txt` | Pinned dependencies |
| `final/manifest_reference.json` | Expected sha256, label counts, library versions |
| `atd_cache.py` | Parses the BAR-ATH tracking (ATD) into a cache under `data/cache/` |
| `external_metrica.py` | Loads the Metrica sample games and derives per-frame H/A/D labels from their events |
| `bench.py` | Degrades the Metrica games to look like BAR-ATH's tracking; scoring + paired episode bootstrap |
| `team_shape.py`, `methods.py` | Per-frame features, Viterbi decoding, HMM/rules baselines |
| `e9_autoresearch.py` | GRU network, input features, 5 Hz steps, training loop |
| `e13_autoresearch.py` | Transition matrix fitted on the bench, decoding |
| `e18_boundary.py` | Training loss: ×5 weight near possession changes + change-point head (A3) |
| `e19_data_mix.py` | Extra random degradations of the training games (A4) |
| `e9_confirm.py` | The fresh bench (degradation seed 777, never used for tuning) |

## 2. Logic-based model (Referee Model v3) — not submitted, published for comparison

| File | Role |
|---|---|
| `referee_v3/` | Self-contained rules model (its own README). Run from that folder with `PYTHONPATH=../..` (see top README §4) |

## 3. Evaluation (numbers and figures in the technical document)

| File | Output |
|---|---|
| `eval_table.py` | `eval_table.md` / `.json`: every model on the fresh bench + label-free BAR-ATH stats + public check. `--md-only` rebuilds from the json; the public column needs `python evaluation/run_evaluation.py` first |
| `e22_referee_vs_gru.py` | `e22/summary.md`: Referee vs GRU, dead-ball precision/recall, stoppages by length (A7) |
| `e21_referee_combo.py`, `combine_referee.py` | `e21/summary.json`: combining the two models (A6, not kept) |
| `sanity_report.py` | Label-free checks of any BAR-ATH submission file (V4) |
| `figures/feature_engineering.py`, `figures/dead_ball_by_minute.py` | Figures (written to `figures/`, not committed: they are drawn from match tracking) |

## 4. Development history (each step's script and saved summary)

| Step | Script | Saved result |
|---|---|---|
| E0 reproduce the organisers' baseline | `e0_reproduce_baseline.py` | — |
| E9 first GRU search, ensembles | `e9_autoresearch.py`, `e9_confirm.py`, `e9_ensemble.py` | `e9_nn/` |
| D13 GRU search against merged possessions | `e13_autoresearch.py` | `e13_nn/` |
| E5c team-shape ablation | `e5c_ablation.py` | `e5c_ablation.json` |
| A1/A2 decoding re-tune, GRU+HMM blend | `e17_decode_blend.py` | `e17/` |
| A3 boundary-weighted loss | `e18_boundary.py`, `e18_head_to_head.py` | `e18/` |
| A4 more varied training data | `e19_data_mix.py`, `e19_head_to_head.py` | `e19/` |
| A5 even more variety (not kept) | `e20_more_variety.py` | `e20/` |

Model outputs cached by these scripts go to `*/cache/` (gitignored). Re-running them retrains networks (hours on CPU).
