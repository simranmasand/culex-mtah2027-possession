"""A6: does the Referee Model (colleague's rules model, v3) improve the GRU on external data?

    python experiments/e21_referee_combo.py

GRU = the D19 final (A4 draws3), cached LOGO outputs from e19/cache, decoded with its tuned
decoding. Referee = experiments/referee_v3 (preset v3) run on the SAME degraded bench games
(tracking only: the Metrica games have no Smart Tagging), then exported with its own
export_dead_ball.dead_ball_columns. Combination = combine_referee.combine (variants dead /
restart / both). The Referee has no parameters fitted here, so nothing is tuned.
Scored on the standard hard bench and the fresh hard + easy benches: spellF1 per game, and the
paired episode bootstrap of (variant - GRU) pooled over games (90% CI). Keep rule (D4): helps on
hard, doesn't hurt easy. No BAR-ATH labels; public 5 min untouched.
Results -> experiments/e21/summary.json
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(HERE, "referee_v3"), HERE, ROOT, os.path.join(ROOT, "starter")]

from possession import pipeline, settings, signals  # noqa: E402  (referee_v3 copy, imported first)
import export_dead_ball  # noqa: E402
assert "referee_v3" in settings.__file__, settings.__file__
import e19_data_mix as B  # noqa: E402
import bench  # noqa: E402
from e9_confirm import fresh_bench  # noqa: E402
from combine_referee import combine, VARIANTS  # noqa: E402

A = B.A
OUT = os.path.join(HERE, "e21")
S19 = json.load(open(os.path.join(B.OUT, "summary.json")))
DEC = tuple(S19["variants"]["draws3"]["decode"])
PITCH = (105.0, 68.0)


def referee_columns(d):
    settings.use("v3")
    scale = np.array(PITCH, dtype=np.float32)
    ball = (d["ball"] / scale).astype(np.float32)
    players = (d["players"][:, :, :2] / scale).astype(np.float32)
    speeds = d["players"][:, :, 2].astype(np.float32)
    sig = signals.compute(ball, players, speeds, d["team_of_slot"].astype(np.int8), PITCH, 25)
    blocks = []
    for p in (1, 2):
        f = np.flatnonzero(d["period"] == p)
        if len(f):
            blocks.append((p, int(f[0]), int(f[-1])))
    labels, report = pipeline.label_frames(sig, blocks, [], len(ball), 25, None)
    c = export_dead_ball.dead_ball_columns(labels, report, blocks)
    return (c["dead"], c["dead_cause"].astype(str), c["restart"], c["restart_team"].astype(str)), labels


def run(kind, level):
    G = bench.bench_games(level) if kind == "tune" else fresh_bench(level)
    outs = {d["name"]: B.outputs("draws3", kind, level, G, d) for d in G}
    gru = A.Decoder(G, outs).method(DEC)
    ref = {d["name"]: referee_columns(d) for d in G}
    methods = {"gru": gru, "referee_alone": lambda d: ref[d["name"]][1]}
    for v, (dd, rr) in VARIANTS.items():
        if v != "gru":
            methods[v] = (lambda dd, rr: lambda d: combine(gru(d), *ref[d["name"]][0], use_dead=dd, use_restart=rr))(dd, rr)
    res = {}
    for name, m in methods.items():
        rows = bench.score(m, G)
        res[name] = {"spellF1_games": [round(r["spellF1"], 4) for r in rows],
                     "spellF1_mean": float(np.mean([r["spellF1"] for r in rows])),
                     "poss": int(sum(r["poss_pred"] for r in rows)),
                     "dead_share": float(np.mean(np.concatenate([m(d) == "D" for d in G])))}
        if name != "gru":
            res[name]["minus_gru"] = bench.compare(m, gru, G)
    res["truth"] = {"dead_share": float(np.mean(np.concatenate([d["labels"] == "D" for d in G])))}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {"gru_decode": DEC}
    for kind, level in (("tune", "hard"), ("fresh", "hard"), ("fresh", "easy")):
        key = f"{kind}_{level}"
        summary[key] = run(kind, level)
        print(f"== {key} (truth dead share {summary[key]['truth']['dead_share']:.2f})")
        for name, r in summary[key].items():
            if name == "truth":
                continue
            c = r.get("minus_gru")
            ctxt = f"  vs GRU {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]" if c else ""
            print(f"  {name:14s} spellF1 {r['spellF1_mean']:.3f} {r['spellF1_games']}  poss {r['poss']:4d}"
                  f"  D {r['dead_share']:.2f}{ctxt}", flush=True)
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
