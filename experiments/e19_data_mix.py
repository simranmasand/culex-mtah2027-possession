"""A4: more varied training data for the A3 model (bw5_cp1, D18).

    python experiments/e19_data_mix.py

Base = A3 bw5_cp1 (no-shape bi-GRU 3x64, loss x5 within 1 s of a label change + change-point head).
Variants change only the training data / length:
  mix     train on the bench level + the other degradation level of the same games (hard + easy copies)
  draws3  train on the bench level + 2 extra random degradations of the same level (seeds 5000+g, 6000+g;
          never the fresh-bench seed 777+g)
  ep30    30 epochs instead of 20
Training steps per epoch scale with data size (same rule as before), so mix/draws3 also train longer.
Protocol (as A3): tune on the standard hard bench (LOGO, 4 seeds, same decoding grid), the best variant(s) that
beat bw5_cp1 on tune are confirmed on the fresh bench (hard + easy) vs bw5_cp1 (cached in e18/), paired episode
bootstrap, keep rule D4. The held-out game never appears in training, in any degradation.
Results -> experiments/e19/ (results.tsv, summary.json, run.log). No BAR-ATH labels; public 5 min untouched.
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e18_boundary as A  # noqa: E402  (sets no-shape inputs, D16)
import bench  # noqa: E402
from e9_confirm import fresh_bench  # noqa: E402
from external_metrica import load_game, GAMES  # noqa: E402

OUT = os.path.join(HERE, "e19")
CACHE = os.path.join(OUT, "cache")
BASE = "bw5_cp1"
S18 = json.load(open(os.path.join(A.OUT, "summary.json")))
BASE_VAR = S18["variants"][BASE]["var"]
BASE_DEC = tuple(S18["variants"][BASE]["decode"])
VARIANTS = ("mix", "draws3", "ep30")


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


_draw_cache = {}


def draws(level, seed0):
    key = (level, seed0)
    if key not in _draw_cache:
        gaps = bench._barath_gap_lengths()
        out = []
        for g in GAMES:
            raw = load_game(g)
            d = {"players": raw["players"], "ball": raw["ball"], "team_of_slot": raw["team_of_slot"],
                 "play": raw["labels"] != "X", "labels": raw["labels"], "name": f"g{g}", "period": raw["period"]}
            out.append(bench.degrade(d, level, np.random.default_rng(seed0 + g), gaps))
        _draw_cache[key] = out
    return _draw_cache[key]


def training_set(variant, games, kind, level, held):
    """Training games for held-out game `held`; kind 'tune' (seed 2027 family) or 'fresh' (777 family)."""
    own = [g for g in games if g["name"] != held]
    if variant == "mix":
        other = "easy" if level == "hard" else "hard"
        extra = bench.bench_games(other) if kind == "tune" else fresh_bench(other)
        return own + [g for g in extra if g["name"] != held]
    if variant == "draws3":
        extra = draws(level, 5000) + draws(level, 6000)
        return own + [g for g in extra if g["name"] != held]
    return own


def outputs(variant, kind, level, games, d):
    path = os.path.join(CACHE, f"{variant}_{kind}_{level}_{d['name']}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["lp"], z["cp"]
    tr = training_set(variant, games, kind, level, d["name"])
    A.TRAIN = dict(A.CFG["train_config"], epochs=30 if variant == "ep30" else A.CFG["train_config"]["epochs"])
    Xd = torch.tensor(A.E.steps(A.E.frame_inputs(d)))[None]
    lps, cps = [], []
    for s in A.SEEDS:
        t = time.time()
        model = A.train(BASE_VAR, tr, s)
        with torch.no_grad():
            lg, cpl = model(Xd)
        lps.append(torch.log_softmax(lg, -1)[0].numpy())
        cps.append(torch.sigmoid(cpl)[0].numpy())
        log(f"  trained {variant} {kind}_{level} {d['name']} seed {s} ({time.time() - t:.0f} s, {len(tr)} training games)")
    lp = np.mean(lps, axis=0)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    cp = np.mean(cps, axis=0)
    np.savez(path, lp=lp, cp=cp)
    return lp, cp


def main():
    os.makedirs(CACHE, exist_ok=True)
    tsv = os.path.join(OUT, "results.tsv")
    with open(tsv, "w") as fh:
        fh.write("variant\tspellF1_hard_tune\tposs\talpha\tstay\tha\tbeta\n")
    hard = bench.bench_games("hard")
    base = A.Decoder(hard, {d["name"]: A.outputs(BASE, BASE_VAR, "tune_hard", hard, d) for d in hard})
    b_sc, b_poss = base.score(BASE_DEC)
    log(f"base {BASE} {BASE_DEC}: tune {b_sc:.4f} (poss {b_poss})")
    summary = {"base": {"name": BASE, "decode": BASE_DEC, "tune_spellF1": b_sc, "tune_poss": b_poss}, "variants": {}}
    with open(tsv, "a") as fh:
        fh.write(f"{BASE}\t{b_sc:.4f}\t{b_poss}\t" + "\t".join(map(str, BASE_DEC)) + "\n")
    for v in VARIANTS:
        log(f"== {v} ==")
        outs = {d["name"]: outputs(v, "tune", "hard", hard, d) for d in hard}
        p, sc, poss = A.tune(A.Decoder(hard, outs), True)
        summary["variants"][v] = {"decode": p, "tune_spellF1": sc, "tune_poss": poss}
        with open(tsv, "a") as fh:
            fh.write(f"{v}\t{sc:.4f}\t{poss}\t" + "\t".join(map(str, p)) + "\n")
        log(f"  {v}: {sc:.4f} (poss {poss}) decode {p}")
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    ranked = sorted(VARIANTS, key=lambda k: -summary["variants"][k]["tune_spellF1"])
    top = [k for k in ranked if summary["variants"][k]["tune_spellF1"] > b_sc + 0.005][:2]
    summary["confirmed"] = top
    log(f"to confirm (beat base on tune by > 0.005): {top}")
    for level in ("hard", "easy"):
        if not top:
            break
        G = fresh_bench(level)
        tag = f"fresh_{level}"
        bdec = A.Decoder(G, {d["name"]: A.outputs(BASE, BASE_VAR, tag, G, d) for d in G})
        bs, bp = bdec.score(BASE_DEC)
        summary.setdefault("confirm", {})[level] = {"base": {"spellF1": bs, "poss": bp}}
        for v in top:
            dec = A.Decoder(G, {d["name"]: outputs(v, "fresh", level, G, d) for d in G})
            p = tuple(summary["variants"][v]["decode"])
            s, poss = dec.score(p)
            c = bench.compare(dec.method(p), bdec.method(BASE_DEC), G)
            summary["confirm"][level][v] = {"spellF1": s, "poss": poss, "minus_base": c}
            log(f"  {level} {v}: {s:.4f} (poss {poss}) vs {BASE} {bs:.4f} (poss {bp}); "
                f"diff {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
            json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for v in top:
        cs = [summary["confirm"][lv][v]["minus_base"] for lv in ("hard", "easy")]
        summary["variants"][v]["keep"] = bool(any(c["lo"] > 0 for c in cs) and not any(c["hi"] < 0 for c in cs))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    log("done " + json.dumps({k: summary["variants"][k].get("keep") for k in top}))


if __name__ == "__main__":
    main()
