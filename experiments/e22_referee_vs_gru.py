"""A7: the two approaches side by side — Referee Model (logic, rules from the Laws of the Game) vs GRU (learned).

    python experiments/e22_referee_vs_gru.py      # -> experiments/e22/summary.json + summary.md

Bench (Metrica sample games, degraded; tune hard + fresh hard + fresh easy), both models on the same games:
  Referee = experiments/referee_v3, preset v3, tracking only (no Smart Tagging on the bench), nothing fitted.
  GRU     = the D19 final (A4 draws3), cached 4-seed LOGO outputs (e19/cache), its tuned decoding.
Metrics: spellF1 / macroF1 / possessions (the challenge metrics, evaluation/possession_metrics.py);
dead ball, frame level: precision / recall / F1 of D;
dead ball, stoppage level: a true stoppage (D run >= 1 s) counts as found when >= 50 % of it is called D; a
called stoppage counts as real when >= 50 % of it is truly D; recall split by stoppage length;
start delay: for found stoppages, first D call within [start - 5 s, end] minus the true start (median, s);
paired episode bootstrap of spellF1 (GRU - Referee).
BAR-ATH (label-free): how the two files' dead-ball calls overlap, and the Referee's stated cause where the GRU
says live. No BAR-ATH labels; public 5 min untouched.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import e21_referee_combo as R  # noqa: E402  (imports the referee_v3 package first)

B, A, bench, fresh_bench = R.B, R.A, R.bench, R.fresh_bench
OUT = os.path.join(HERE, "e22")
FPS = 25
BUCKETS = ((1, 5), (5, 15), (15, 30), (30, 10 ** 6))


def runs(mask):
    p = np.concatenate(([False], mask, [False]))
    e = np.flatnonzero(p[1:] != p[:-1])
    return list(zip(e[0::2], e[1::2]))          # [start, end)


def dead_metrics(gt, pred):
    g, p = gt == "D", pred == "D"
    tp = int((g & p).sum())
    prec, rec = tp / max(p.sum(), 1), tp / max(g.sum(), 1)
    true_st = [(a, b) for a, b in runs(g) if b - a >= FPS]
    pred_st = [(a, b) for a, b in runs(p) if b - a >= FPS]
    found = [p[a:b].mean() >= 0.5 for a, b in true_st]
    real = [g[a:b].mean() >= 0.5 for a, b in pred_st]
    delays = []
    for (a, b), f in zip(true_st, found):
        if f:
            lo = max(0, a - 5 * FPS)
            first = lo + int(np.argmax(p[lo:b]))
            delays.append((first - a) / FPS)
    by_len = {}
    for lo_s, hi_s in BUCKETS:
        sel = [f for (a, b), f in zip(true_st, found) if lo_s * FPS <= b - a < hi_s * FPS]
        by_len[f"{lo_s}-{hi_s if hi_s < 10 ** 6 else 'inf'}s"] = {"n": len(sel), "found": float(np.mean(sel)) if sel else None}
    return {"D_precision": prec, "D_recall": rec, "D_F1": 2 * prec * rec / max(prec + rec, 1e-9),
            "dead_share": float(p.mean()), "stoppages_true": len(true_st), "stoppages_called": len(pred_st),
            "stoppage_recall": float(np.mean(found)), "stoppage_precision": float(np.mean(real)) if real else 0.0,
            "recall_by_length": by_len, "start_delay_median_s": float(np.median(delays)) if delays else None}


def run(kind, level):
    G = bench.bench_games(level) if kind == "tune" else fresh_bench(level)
    outs = {d["name"]: B.outputs("draws3", kind, level, G, d) for d in G}
    gru = A.Decoder(G, outs).method(R.DEC)
    ref_lab = {d["name"]: R.referee_columns(d)[1] for d in G}
    ref = lambda d: ref_lab[d["name"]]
    res = {}
    for name, m in (("referee_v3", ref), ("gru_final", gru)):
        rows = bench.score(m, G)
        gt = np.concatenate([d["labels"] for d in G])
        pr = np.concatenate([m(d) for d in G])
        res[name] = {"spellF1": float(np.mean([r["spellF1"] for r in rows])),
                     "macroF1": float(np.mean([r["macroF1"] for r in rows])),
                     "poss": int(sum(r["poss_pred"] for r in rows)),
                     **dead_metrics(gt, pr)}
    res["truth"] = {"poss": int(sum(r["poss_true"] for r in bench.score(gru, G))),
                    "dead_share": float(np.mean(np.concatenate([d["labels"] == "D" for d in G])))}
    res["gru_minus_referee"] = bench.compare(gru, ref, G)
    return res


def barath():
    res_dir = os.path.join(ROOT, "evaluation", "results")
    gru = pd.read_csv(os.path.join(res_dir, "nn_a4.csv"))["label"].astype(str).to_numpy()
    ref = pd.read_csv(os.path.join(res_dir, "referee_v3.csv"))["label"].astype(str).to_numpy()
    exp = pd.read_csv(os.path.join(ROOT, "data", "cache", "referee_dead_ball.csv"), keep_default_na=False)
    graded = gru != "X"
    gd, rd = (gru == "D") & graded, (ref == "D") & graded
    cause = exp["dead_cause"].astype(str).to_numpy()
    only_ref = rd & ~gd
    return {"graded_min": float(graded.sum() / FPS / 60),
            "gru_dead_min": float(gd.sum() / FPS / 60), "referee_dead_min": float(rd.sum() / FPS / 60),
            "both_dead_min": float((gd & rd).sum() / FPS / 60),
            "share_of_gru_dead_also_referee": float((gd & rd).sum() / max(gd.sum(), 1)),
            "share_of_referee_dead_also_gru": float((gd & rd).sum() / max(rd.sum(), 1)),
            "referee_only_dead_min_by_cause": {c: float((only_ref & (cause == c)).sum() / FPS / 60)
                                              for c in sorted(set(cause[only_ref]))},
            "label_agreement": float((gru[graded] == ref[graded]).mean())}


def write_md(s):
    L = ["| Bench | Model | spellF1 | macroF1 | Poss (truth) | D precision | D recall | D F1 | Stoppages found | "
         "Called stoppages real | Start delay (s) | Dead share (truth) |", "|" + "---|" * 12]
    for key in ("tune_hard", "fresh_hard", "fresh_easy"):
        r = s[key]
        for name, lab in (("referee_v3", "Referee (logic)"), ("gru_final", "GRU (learned, final)")):
            m = r[name]
            L.append(f"| {key} | {lab} | {m['spellF1']:.3f} | {m['macroF1']:.3f} | {m['poss']} ({r['truth']['poss']}) | "
                     f"{m['D_precision']:.2f} | {m['D_recall']:.2f} | {m['D_F1']:.2f} | "
                     f"{m['stoppage_recall']:.2f} of {m['stoppages_true']} | {m['stoppage_precision']:.2f} of {m['stoppages_called']} | "
                     f"{m['start_delay_median_s']:+.1f} | {m['dead_share']:.2f} ({r['truth']['dead_share']:.2f}) |")
    L += ["", "Stoppages found, by true stoppage length (fresh hard):", "",
          "| Length | n | Referee | GRU |", "|---|---|---|---|"]
    fr, fg = s["fresh_hard"]["referee_v3"]["recall_by_length"], s["fresh_hard"]["gru_final"]["recall_by_length"]
    for k in fr:
        L.append(f"| {k} | {fr[k]['n']} | {fr[k]['found']:.2f} | {fg[k]['found']:.2f} |")
    L += ["", "spellF1, GRU − Referee (paired episode bootstrap, 90% CI): " + "; ".join(
        f"{k} {s[k]['gru_minus_referee']['diff']:+.3f} [{s[k]['gru_minus_referee']['lo']:+.3f}, {s[k]['gru_minus_referee']['hi']:+.3f}]"
        for k in ("tune_hard", "fresh_hard", "fresh_easy"))]
    b = s["barath"]
    L += ["", f"BAR-ATH (no labels): GRU dead {b['gru_dead_min']:.1f} min, Referee dead {b['referee_dead_min']:.1f} min, "
          f"both {b['both_dead_min']:.1f} min; {100 * b['share_of_gru_dead_also_referee']:.0f}% of the GRU's dead frames are also "
          f"Referee-dead; label agreement {100 * b['label_agreement']:.0f}%. Referee-only dead minutes by cause: "
          + ", ".join(f"{k} {v:.1f}" for k, v in b["referee_only_dead_min_by_cause"].items()) + "."]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    os.makedirs(OUT, exist_ok=True)
    s = {k: run(*k.split("_")) for k in ("tune_hard", "fresh_hard", "fresh_easy")}
    s["barath"] = barath()
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    write_md(s)


if __name__ == "__main__":
    main()
