"""A3 follow-up: paired bootstrap bw5_cp1 vs bw5_r5 on the fresh bench, from e18 cached outputs (no retraining).

    python experiments/e18_head_to_head.py   # -> experiments/e18/head_to_head.json
"""
import json
import os

import e18_boundary as A
import bench
from e9_confirm import fresh_bench

S = json.load(open(os.path.join(A.OUT, "summary.json")))
res = {}
for level in ("hard", "easy"):
    G = fresh_bench(level)
    m = {}
    for name in ("bw5_cp1", "bw5_r5"):
        v = S["variants"][name]
        dec = A.Decoder(G, {d["name"]: A.outputs(name, v["var"], f"fresh_{level}", G, d) for d in G})
        m[name] = dec.method(tuple(v["decode"]))
    res[level] = bench.compare(m["bw5_cp1"], m["bw5_r5"], G)
    print(level, res[level], flush=True)
json.dump(res, open(os.path.join(A.OUT, "head_to_head.json"), "w"), indent=1)
