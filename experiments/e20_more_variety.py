"""A5: push the A4 result (degradation variety helps) further.

    python experiments/e20_more_variety.py

Base = A4 draws3 (D18 model trained on the bench level + 2 extra random degradations of the same level).
Variants (training data only; same network, A3 loss, decoding re-tuned per variant as before):
  draws5      bench level + 4 extra random degradations of the same level (seeds 5000/6000/7000/8000 + g)
  draws3_mix  draws3 + the same three draws of the other degradation level (hard and easy, 6 copies)
Extra seeds never overlap the fresh-bench seed (777 + g); the held-out game is never in training.
Protocol as A3/A4: tune on the standard hard bench (LOGO, 4 seeds); variants beating draws3 on tune by > 0.005 are
confirmed on the fresh bench (hard + easy) vs draws3 (cached in e19/), paired episode bootstrap, keep rule D4.
Results -> experiments/e20/ (results.tsv, summary.json, run.log). No BAR-ATH labels; public 5 min untouched.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e19_data_mix as B  # noqa: E402  (imports e18: no-shape inputs, A3 loss, decoder)
import bench  # noqa: E402
from e9_confirm import fresh_bench  # noqa: E402

A = B.A
OUT = os.path.join(HERE, "e20")
BASE = "draws3"
S19 = json.load(open(os.path.join(B.OUT, "summary.json")))
BASE_DEC = tuple(S19["variants"][BASE]["decode"])
VARIANTS = ("draws5", "draws3_mix")
E19_CACHE = B.CACHE


def training_set(variant, games, kind, level, held):
    own = [g for g in games if g["name"] != held]
    if variant == "draws3":
        extra = B.draws(level, 5000) + B.draws(level, 6000)
    elif variant == "draws5":
        extra = sum((B.draws(level, s) for s in (5000, 6000, 7000, 8000)), [])
    elif variant == "draws3_mix":
        other = "easy" if level == "hard" else "hard"
        base_other = bench.bench_games(other) if kind == "tune" else fresh_bench(other)
        extra = (B.draws(level, 5000) + B.draws(level, 6000)
                 + list(base_other) + B.draws(other, 5000) + B.draws(other, 6000))
    else:
        raise ValueError(variant)
    return own + [g for g in extra if g["name"] != held]


def outputs(variant, kind, level, games, d):
    B.CACHE = E19_CACHE if variant == BASE else os.path.join(OUT, "cache")
    return B.outputs(variant, kind, level, games, d)


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def main():
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    B.training_set = training_set
    B.OUT = OUT                       # e19's per-seed log lines go to e20/run.log
    tsv = os.path.join(OUT, "results.tsv")
    with open(tsv, "w") as fh:
        fh.write("variant\tspellF1_hard_tune\tposs\talpha\tstay\tha\tbeta\n")
    hard = bench.bench_games("hard")
    base = A.Decoder(hard, {d["name"]: outputs(BASE, "tune", "hard", hard, d) for d in hard})
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
    top = [k for k in ranked if summary["variants"][k]["tune_spellF1"] > b_sc + 0.005]
    summary["confirmed"] = top
    log(f"to confirm (beat {BASE} on tune by > 0.005): {top}")
    for level in ("hard", "easy"):
        if not top:
            break
        G = fresh_bench(level)
        bdec = A.Decoder(G, {d["name"]: outputs(BASE, "fresh", level, G, d) for d in G})
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
