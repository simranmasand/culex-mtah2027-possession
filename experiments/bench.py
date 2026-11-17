"""Evaluation bench: Metrica sample games, degraded to look like BAR-ATH, scored with the official metric.

    from experiments.bench import bench_games, score, compare
    games = bench_games("hard")                 # list of dicts, same layout as atd_cache.load()
    res = score(method, games)                  # per-game spellF1 / macroF1 / possession counts
    diff = compare(method_a, method_b, games)   # paired episode bootstrap of spellF1(A) - spellF1(B)

A *method* is a function ``method(d) -> labels`` taking the array dict (players, ball,
team_of_slot, play) and returning a <U1 array of H/A/D/X over all frames. The same function runs
on BAR-ATH (``barath()``), so nothing is tuned on one data layout and applied to another.

Degradation (V1). The sample tracking is clean: 22 players always, ball blanked almost only in
dead time. BAR-ATH: ~20.5 tracked players, duplicates on ~6% of frames, ball missing 41% of graded
frames in long gaps (median ~11 s), and dead time where the ball is still visible. Two variants:

    easy : players dropped/duplicated to BAR-ATH rates; ball untouched
    hard : + extra ball gaps in live play (lengths resampled from BAR-ATH's gaps) up to 41% missing,
           + ball revealed (frozen at its restart position) in half of the dead stretches

Both are seeded and deterministic.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from external_metrica import load_game, GAMES  # noqa: E402
from possession_metrics import evaluate, label_codes, segments, valid_runs, possession_spells  # noqa: E402

FPS = 25
SEED = 2027


# -- data ---------------------------------------------------------------------------------------

def barath():
    """BAR-ATH arrays in bench layout. `play` = graded frames."""
    from atd_cache import load
    d = load()
    return {"players": d["players"], "ball": d["ball"], "team_of_slot": d["team_of_slot"],
            "play": d["graded"], "name": "BAR-ATH"}


def _barath_gap_lengths():
    d = barath()
    miss = np.isnan(d["ball"][:, 0]) & d["play"]
    e = np.diff(np.r_[0, miss.astype(int), 0])
    return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def degrade(d, level, rng, gap_lengths):
    players, ball = d["players"].copy(), d["ball"].copy()
    n, s = players.shape[:2]
    if level in ("easy", "hard"):
        # players: drop chunks so ~1.5 of 22 are missing on average (BAR-ATH 20.5 tracked)
        for slot in range(s):
            t = 0
            while t < n:
                t += int(rng.exponential(60 * FPS))
                if t >= n:
                    break
                L = int(rng.exponential(4 * FPS)) + 1
                players[t:t + L, slot] = np.nan
                t += L
        # duplicates on ~6% of frames: copy a random player (0.5 m jitter) into a spare slot of
        # the same team. Episodes start at rate 6% / mean length (2 s).
        dup = np.full((n, 2, 3), np.nan, dtype=np.float32)           # spare slot per team
        for t0 in np.flatnonzero(rng.random(n) < 0.06 / (2 * FPS)):
            L = int(rng.exponential(2 * FPS)) + 1
            src = rng.integers(s)
            seg = players[t0:t0 + L, src].copy()
            seg[:, :2] += rng.normal(0, 0.5, size=(len(seg), 2))
            dup[t0:t0 + L, d["team_of_slot"][src]] = seg
        players = np.concatenate([players, dup], axis=1)
        team_of_slot = np.r_[d["team_of_slot"], np.array([0, 1], dtype=np.int8)]
    else:
        team_of_slot = d["team_of_slot"]

    if level == "hard":
        lab = d["labels"]
        alive = np.isin(lab, ["H", "A"])
        for a, b in _runs(lab == "D"):
            if rng.random() < 0.5:
                nxt = np.flatnonzero(~np.isnan(ball[b:b + 10 * FPS, 0]))
                if len(nxt):
                    ball[a:b] = ball[b + nxt[0]]                # frozen at the restart spot
        target = 0.41 * d["play"].sum()
        missing = np.isnan(ball[:, 0]) & d["play"]
        while missing.sum() < target:
            t0 = rng.integers(n)
            if not alive[t0]:
                continue
            L = int(rng.choice(gap_lengths))
            ball[t0:t0 + L] = np.nan
            missing = np.isnan(ball[:, 0]) & d["play"]
    return {"players": players, "ball": ball, "team_of_slot": team_of_slot,
            "play": d["play"], "labels": d["labels"], "name": d["name"], "period": d["period"]}


_CACHE = {}


def bench_games(level="hard"):
    if level in _CACHE:
        return _CACHE[level]
    gaps = _barath_gap_lengths()
    out = []
    for g in GAMES:
        raw = load_game(g)
        d = {"players": raw["players"], "ball": raw["ball"], "team_of_slot": raw["team_of_slot"],
             "play": raw["labels"] != "X", "labels": raw["labels"], "name": f"g{g}",
             "period": raw["period"]}
        out.append(degrade(d, level, np.random.default_rng(SEED + g), gaps))
    _CACHE[level] = out
    return out


# -- scoring --------------------------------------------------------------------------------------

def score(method, games, verbose=False):
    rows = []
    for d in games:
        pred = method(d)
        r = evaluate(label_codes(d["labels"]), label_codes(pred))
        n_pred = len(possession_spells(label_codes(pred), valid_runs(label_codes(d["labels"]))))
        n_gt = len(possession_spells(label_codes(d["labels"]), valid_runs(label_codes(d["labels"]))))
        rows.append({"game": d["name"], "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                     "poss_pred": n_pred, "poss_true": n_gt})
    if verbose:
        for r in rows:
            print("  {game}: spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  poss {poss_pred}/{poss_true}".format(**r))
    return rows


def _episode_counts(gt, pred, ious=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Per-episode (tp per IoU, n_gt, n_pred). Episodes = stretches between GT dead balls."""
    runs = valid_runs(gt)
    gs = possession_spells(gt, runs)
    ps = possession_spells(pred, runs)
    # episode id of a frame: index of the alive stretch (between D runs) it belongs to
    alive = (gt == label_codes(np.array(["H"]))[0]) | (gt == label_codes(np.array(["A"]))[0])
    ep = np.cumsum(np.r_[0, np.diff(alive.astype(int)) == 1])
    n_ep = ep.max() + 1
    tp = np.zeros((n_ep, len(ious))); ng = np.zeros(n_ep); npred = np.zeros(n_ep)
    for a, b, _ in gs:
        ng[ep[a]] += 1
    for a, b, _ in ps:
        npred[ep[(a + b) // 2]] += 1
    pairs = []
    for i, (ga, gb, gc) in enumerate(gs):
        for j, (pa, pb, pc) in enumerate(ps):
            if pc != gc or pb <= ga or gb <= pa:
                continue
            pairs.append(((min(gb, pb) - max(ga, pa)) / (max(gb, pb) - min(ga, pa)), i, j))
    pairs.sort(key=lambda x: -x[0])
    for k, thr in enumerate(ious):
        used_g, used_p = set(), set()
        for iou, i, j in pairs:
            if iou < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i); used_p.add(j)
            tp[ep[gs[i][0]], k] += 1
    return tp, ng, npred


def _f1(tp, ng, npred):
    p = np.where(npred.sum() > 0, tp.sum(0) / max(npred.sum(), 1), 0)
    r = np.where(ng.sum() > 0, tp.sum(0) / max(ng.sum(), 1), 0)
    return float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)))


def compare(method_a, method_b, games, n_boot=1000, seed=0):
    """Paired episode bootstrap of pooled spellF1(A) - spellF1(B) across all games."""
    A, B = [], []
    for d in games:
        gt = label_codes(d["labels"])
        A.append(_episode_counts(gt, label_codes(method_a(d))))
        B.append(_episode_counts(gt, label_codes(method_b(d))))
    cat = lambda xs, k: np.concatenate([x[k] for x in xs])
    tpA, ngA, npA = cat(A, 0), cat(A, 1), cat(A, 2)
    tpB, npB = cat(B, 0), cat(B, 2)
    rng = np.random.default_rng(seed)
    n = len(ngA)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        diffs[i] = _f1(tpA[k], ngA[k], npA[k]) - _f1(tpB[k], ngA[k], npB[k])
    point = _f1(tpA, ngA, npA) - _f1(tpB, ngA, npB)
    lo, hi = np.percentile(diffs, [5, 95])
    return {"diff": point, "lo": float(lo), "hi": float(hi), "episodes": n}
