"""A4 follow-up: paired bootstrap draws3 vs mix on the fresh bench, from e19 cached outputs (no retraining).

    python experiments/e19_head_to_head.py   # -> experiments/e19/head_to_head.json
"""
import json
import os

import e19_data_mix as B
import bench
from e9_confirm import fresh_bench

S = json.load(open(os.path.join(B.OUT, "summary.json")))
res = {}
for level in ("hard", "easy"):
    G = fresh_bench(level)
    m = {}
    for v in ("draws3", "mix"):
        dec = B.A.Decoder(G, {d["name"]: B.outputs(v, "fresh", level, G, d) for d in G})
        m[v] = dec.method(tuple(S["variants"][v]["decode"]))
    res[level] = bench.compare(m["draws3"], m["mix"], G)
    print(level, res[level], flush=True)
json.dump(res, open(os.path.join(B.OUT, "head_to_head.json"), "w"), indent=1)
