"""Evaluation summary for the technical document, built from result files only (no hand-typed numbers).

    python experiments/eval_table.py        # -> experiments/eval_table.md + eval_table.json

One protocol for every bench number: FRESH bench (degradation seed 777, never used for any search or
tuning), leave-one-game-out, mean spellF1 / macroF1 over the 3 Metrica games, hard + easy.
GRU rows: 4-seed ensembles (cached outputs from e19 for the final A4 model, e18 for nn_a3, e17 for nn_final; e13 summary for nn_best and
nn_e9_ens8's config; e9 confirm.json mean ± sd of single seeds for nn_e9). Submission files use 8 seeds.
Paired episode bootstrap (90% CI) of the final model vs the HMM and vs baseline50 on the same bench.
BAR-ATH columns: label-free stats from sanity_report.py. Public columns: evaluation/scoreboard.csv
(run `python evaluation/run_evaluation.py` first) - a correctness check, never used for selection.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")]

import bench  # noqa: E402
import methods as M  # noqa: E402
from e9_confirm import fresh_bench  # noqa: E402


def mean_rows(rows):
    return {"spellF1": float(np.mean([r["spellF1"] for r in rows])),
            "macroF1": float(np.mean([r["macroF1"] for r in rows])),
            "poss": int(sum(r["poss_pred"] for r in rows))}


def main():
    import e17_decode_blend as A        # previous final (D17) cached 4-seed outputs + decoder (no-shape inputs)
    import e18_boundary as A3           # final model (D18 = A3 bw5_cp1) cached 4-seed outputs + decoder
    E9 = A.E
    E9.frame_inputs = lambda d, swap=False: np.delete(A._full(d, swap), np.r_[9:15], axis=1)   # undo double wrap
    s18 = json.load(open(os.path.join(A3.OUT, "summary.json")))["variants"]["bw5_cp1"]
    import e19_data_mix as A4           # final model (D19 = A4 draws3) cached 4-seed outputs
    dec19 = tuple(json.load(open(os.path.join(A4.OUT, "summary.json")))["variants"]["draws3"]["decode"])
    res = {"hard": {}, "easy": {}, "compare": {}}
    for level in ("hard", "easy"):
        G = fresh_bench(level)
        hmm = M.hmm_logo(alpha=0.3); hmm.games = G
        hmm_s = M.hmm_logo(alpha=0.3, shape=True); hmm_s.games = G
        dec = A.Decoder(G, A.evidence(f"fresh_{level}", G))
        prev = dec.method(A.CURRENT)
        d3 = A3.Decoder(G, {d["name"]: A3.outputs("bw5_cp1", s18["var"], f"fresh_{level}", G, d) for d in G})
        a3 = d3.method(tuple(s18["decode"]))
        d4 = A3.Decoder(G, {d["name"]: A4.outputs("draws3", "fresh", level, G, d) for d in G})
        final = d4.method(dec19)
        methods = {"nn_a4": final, "nn_a3": a3, "nn_final": prev, "hmm_ext_hard": hmm, "hmm_ext_hard_shape": hmm_s,
                   "rules_e2": M.rules(window=50, min_gap_s=1.0, slow=np.inf),
                   "baseline_smooth50": M.baseline(50)}
        for name, m in methods.items():
            res[level][name] = mean_rows(bench.score(m, G))
            print(level, name, res[level][name], flush=True)
        res["compare"][level] = {"final_minus_prev": bench.compare(final, a3, G),
                                 "final_minus_d17": bench.compare(final, prev, G),
                                 "final_minus_hmm": bench.compare(final, hmm, G),
                                 "final_minus_baseline50": bench.compare(final, methods["baseline_smooth50"], G)}
        print(level, res["compare"][level], flush=True)
    e13 = json.load(open(os.path.join(HERE, "e13_nn", "summary.json")))
    conf = json.load(open(os.path.join(HERE, "e9_nn", "confirm.json")))
    for level in ("hard", "easy"):
        res[level]["nn_best"] = {"spellF1": e13[level]["new_spellF1"], "poss": e13[level]["new_poss"]}
        res[level]["nn_e9_ens8"] = {"spellF1": e13[level]["old_spellF1"], "poss": e13[level]["old_poss"]}
        res[level]["nn_e9"] = {"spellF1": conf[level]["gru_mean"], "sd": conf[level]["gru_sd"]}
    write_md(res)
    json.dump(res, open(os.path.join(HERE, "eval_table.json"), "w"), indent=1)


def add_referee(res):
    """Referee Model (logic-based model v3) bench rows come from experiments/e22_referee_vs_gru.py (run separately)."""
    s = json.load(open(os.path.join(HERE, "e22", "summary.json")))
    for lv in ("hard", "easy"):
        r = s[f"fresh_{lv}"]["referee_v3"]
        res[lv]["referee_v3"] = {"spellF1": r["spellF1"], "macroF1": r["macroF1"], "poss": r["poss"]}
        res["compare"][lv]["final_minus_referee"] = s[f"fresh_{lv}"]["gru_minus_referee"]


def write_md(res):
    import sanity_report as S
    add_referee(res)
    files = ["nn_a4", "referee_v3", "nn_a3", "nn_final", "nn_noshape", "nn_best", "nn_e9_ens8", "nn_e9", "hmm_ext_hard", "hmm_ext_hard_shape",
             "rules_e2", "baseline_smooth50", "smarttag_naive"]
    desc = {
        "nn_a4": "**Final (D19).** nn_a3 trained on 3 random hard degradations instead of 1 (A4); change-point head also used in decoding; pre-kickoff = D",
        "nn_a3": "D18: nn_final's network trained with loss ×5 within 1 s of a label change + change-point head (A3); pre-kickoff = D",
        "nn_final": "D17: bi-GRU 3×64, no team-shape inputs, Viterbi; 2nd-half pre-kickoff = D (D17)",
        "nn_noshape": "nn_final before the pre-kickoff rule (same model)",
        "nn_best": "bi-GRU 3×64 with team-shape inputs (D13)",
        "nn_e9_ens8": "bi-GRU 2×32 (E9)",
        "nn_e9": "bi-GRU 2×32 (E9), 1 seed",
        "hmm_ext_hard": "HMM, hand-binned emissions (E4)",
        "hmm_ext_hard_shape": "HMM + team-shape channel (E5)",
        "referee_v3": "🏷️ **Referee Model v3** (logic only: Laws of the Game rules for dead ball, restarts, control; Smart Tagging phase tags in long ball gaps)",
        "rules_e2": "dead-ball rules + smoothing (E2)",
        "baseline_smooth50": "organisers' baseline (nearest player, 50-frame vote)",
        "smarttag_naive": "🏷️ Smart Tagging only (E1b)",
    }
    g, nt, tags = S.context()
    sb = pd.read_csv(os.path.join(ROOT, "evaluation", "scoreboard.csv")).set_index("submission")
    lines = ["| File | Method | Bench hard | Bench easy | BAR-ATH poss | D share | Agree nearest | Agree tags | Public spellF1 / macroF1 / poss |",
             "|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        path = os.path.join(ROOT, "evaluation", "results", f + ".csv")
        r = S.report(path, g, nt, tags)
        bh, be = [], []
        for lv, out in (("hard", bh), ("easy", be)):
            key = "nn_final" if f == "nn_noshape" else f
            v = res[lv].get(key)
            if v is None:
                out.append("†")
            elif "sd" in v:
                out.append(f"{v['spellF1']:.3f} ± {v['sd']:.3f}")
            else:
                star = "*" if f == "referee_v3" else ""
                out.append(f"{v['spellF1']:.3f}{star}")
        p = sb.loc[f]
        lines.append(f"| {f} | {desc[f]} | {bh[0]} | {be[0]} | {r['possessions']} | {r['D_share']:.2f} | "
                     f"{r['agree_nearest']:.2f} | {r['agree_tags']:.2f} | "
                     f"{p['spell_f1_mean']:.3f} / {p['macro_f1']:.3f} / {int(p['poss'])} |")
    c = res["compare"]
    notes = [
        "",
        "Bench: fresh degradation (seed 777, never searched or tuned on), leave-one-game-out, mean spellF1 over the 3 Metrica games "
        "(1,022 true possessions). GRU rows = 4-seed ensembles (submission files use 8); nn_e9 = mean ± sd of 4 single seeds. "
        "nn_noshape and nn_final have the same bench score (the pre-kickoff rule has no bench equivalent). "
        "\\* Referee Model run on tracking only (no Smart Tagging on the bench); † not testable on the bench. 🏷️ = uses Smart Tagging.",
        "BAR-ATH: label-free output stats on graded frames (expected ≈360 possessions, 15–35% dead ball). "
        "Public 5 min: correctness check only, 17 true possessions, noise ≈ ±0.04 — never used for selection.",
        "",
        "Final model vs references on the same fresh bench (paired episode bootstrap, 90% CI, 198 episodes):",
        "",
        "| | Final − Referee (logic) | Final − nn_a3 (D18) | Final − nn_final (D17) | Final − HMM | Final − baseline50 |",
        "|---|---|---|---|---|---|",
    ]
    for lv in ("hard", "easy"):
        cells = [c[lv][k] for k in ("final_minus_referee", "final_minus_prev", "final_minus_d17", "final_minus_hmm", "final_minus_baseline50")]
        notes.append(f"| {lv} | " + " | ".join(f"{x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for x in cells) + " |")
    open(os.path.join(HERE, "eval_table.md"), "w").write("\n".join(lines + notes) + "\n")
    print("\n".join(lines + notes))


if __name__ == "__main__":
    if "--md-only" in sys.argv:          # rebuild the table from eval_table.json (+ e22) without re-running the bench
        res = json.load(open(os.path.join(HERE, "eval_table.json")))
        write_md(res)
        json.dump(res, open(os.path.join(HERE, "eval_table.json"), "w"), indent=1)
    else:
        main()
