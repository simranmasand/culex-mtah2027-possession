"""Possession methods with one interface: method(d) -> <U1 labels (H/A/D, X outside d['play']).

d: players (F,S,3) m / m / m/s, ball (F,2) m, team_of_slot (S,), play (F,) bool.
Every method here runs unchanged on the bench games and on BAR-ATH.
"""

import numpy as np
import pandas as pd

FPS = 25
PITCH = (105.3, 68.0)
H, A, NOBODY = 0, 1, -1


# -- shared features ------------------------------------------------------------------------------

def features(d):
    """Per-frame features, cached on the dict."""
    if "_feat" in d:
        return d["_feat"]
    P, B, T = d["players"], d["ball"], d["team_of_slot"]
    seen = ~np.isnan(P[:, :, 0])
    ball = ~np.isnan(B[:, 0])
    diff = P[:, :, :2] - B[:, None, :]
    dist = np.sqrt((diff ** 2).sum(2))
    dist[~seen] = np.inf
    nearest = dist.argmin(1)
    near_d = dist.min(1)
    near_team = np.where(ball & np.isfinite(near_d), T[nearest], NOBODY).astype(np.int8)
    with np.errstate(all="ignore"):
        spd = np.nanmean(np.where(seen, P[:, :, 2], np.nan), 1)
    spd = pd.Series(spd).ffill().bfill().fillna(0).to_numpy()
    spd1s = pd.Series(spd).rolling(FPS, center=True, min_periods=1).mean().to_numpy()
    out_pitch = ball & ((B[:, 0] < -0.5) | (B[:, 0] > PITCH[0] + 0.5) |
                        (B[:, 1] < -0.5) | (B[:, 1] > PITCH[1] + 0.5))
    # length of the ball gap each frame sits in (0 when visible)
    gap_len = np.zeros(len(B))
    e = np.diff(np.r_[0, (~ball).astype(int), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        gap_len[a:b] = (b - a) / FPS
    f = {"ball": ball, "near_team": near_team, "near_d": np.where(ball, near_d, np.nan),
         "spd": spd1s, "out": out_pitch, "gap_len": gap_len}
    d["_feat"] = f
    return f


def _blocks(play):
    e = np.diff(np.r_[0, play.astype(int), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def _finish(labels, play):
    labels = labels.copy()
    labels[~play] = "X"
    return labels


# -- M0: published baseline -----------------------------------------------------------------------

def baseline(window=50):
    def run(d):
        f = features(d)
        codes = f["near_team"]
        hc = np.r_[0, np.cumsum(codes == H)]; ac = np.r_[0, np.cumsum(codes == A)]
        i = np.arange(len(codes))
        lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
        h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
        lab = np.full(len(codes), "D", dtype="<U1")
        seen = (h + a) > 0
        lab[seen] = np.where(h[seen] >= a[seen], "H", "A")
        return _finish(lab, d["play"])
    run.__name__ = f"baseline{window}"
    return run


# -- M1: E2 dead-ball rules + team smoothing ------------------------------------------------------

def rules(window=50, min_gap_s=3.0, slow=1.6, out_dead=True):
    """D where: ball out of pitch, or ball missing in a gap >= min_gap_s with players slow.
    Team: majority vote of nearest team over the window; through ball gaps that are not dead,
    keep the last team (possession persists)."""
    def run(d):
        f = features(d)
        dead = (f["gap_len"] >= min_gap_s) & (f["spd"] < slow)
        if out_dead:
            dead |= f["out"]
        codes = f["near_team"]
        hc = np.r_[0, np.cumsum(codes == H)]; ac = np.r_[0, np.cumsum(codes == A)]
        i = np.arange(len(codes))
        lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
        h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
        team = np.where(h + a > 0, np.where(h >= a, "H", "A"), "")
        lab = pd.Series(np.where(team == "", None, team))
        for a0, b0 in _blocks(d["play"]):
            lab.iloc[a0:b0] = lab.iloc[a0:b0].ffill().bfill()
        lab = lab.fillna("D").to_numpy().astype("<U1")
        lab[dead] = "D"
        return _finish(lab, d["play"])
    run.__name__ = f"rules_w{window}_g{min_gap_s}_s{slow}"
    return run


# -- M2: E4 HMM (H/A/D) with emissions learned on external games --------------------------------
#
# Observations per frame are discretised into a few independent channels; emissions are learned by
# counting on labelled training games (naive Bayes), made H/A-symmetric by pooling each game with
# its team-swapped copy, and tempered by `alpha` because neighbouring frames are not independent.
# Transitions: P(leave state) = stay_scale / mean duration of that state in the training labels.
# Decoding: Viterbi over each contiguous play block (whole halves on BAR-ATH).

GAP_BINS = (0, 1e-9, 1, 5, 15, np.inf)          # visible, <1 s, 1-5, 5-15, >15 s
SPD_BINS = (-np.inf, 1.0, 1.6, 2.2, np.inf)
DIST_BINS = (0, 2, 5, np.inf)


SHAPE_BINS = (-np.inf, -2, -1, -0.3, 0.3, 1, 2, np.inf)    # symmetric -> swap mirrors the bin


def observations(d, swap=False, shape_model=None):
    f = features(d)
    nt = f["near_team"].copy()
    if swap:
        nt = np.where(nt >= 0, 1 - nt, nt)
    dist_bin = np.digitize(np.nan_to_num(f["near_d"], nan=99), DIST_BINS[1:-1])      # 0..2
    near = np.where(nt < 0, 6, nt * 3 + dist_bin)                                   # 0..6
    gap = np.digitize(f["gap_len"], GAP_BINS[1:-1], right=True)                     # 0..4
    spd = np.digitize(f["spd"], SPD_BINS[1:-1])                                     # 0..3
    obs, sizes = [near, f["out"].astype(int), gap, spd], [7, 2, 5, 4]
    if shape_model is not None:
        sb = np.digitize(shape_model.score(d), SHAPE_BINS[1:-1])                    # 0..6
        obs.append(6 - sb if swap else sb); sizes.append(7)
    return obs, sizes


def _state_codes(labels, swap=False):
    s = np.select([labels == "H", labels == "A", labels == "D"], [0, 1, 2], -1)
    if swap:
        s = np.where(s == 0, 1, np.where(s == 1, 0, s))
    return s


class HMM:
    def __init__(self, alpha=0.3, stay_scale=1.0, channels=(0, 1, 2, 3), shape=False, d_bias=0.0):
        self.alpha, self.stay_scale, self.shape, self.d_bias = alpha, stay_scale, shape, d_bias
        self.channels = tuple(channels) + ((4,) if shape and 4 not in channels else ())

    def fit(self, games):
        counts, durs = None, {0: [], 1: [], 2: []}
        self.shape_model = None
        if self.shape:
            from team_shape import ShapeScore
            self.shape_model = ShapeScore().fit(games)
        for d in games:
            for swap in (False, True):
                obs, sizes = observations(d, swap, self.shape_model)
                st = _state_codes(d["labels"], swap)
                ok = st >= 0
                if counts is None:
                    counts = [np.ones((3, k)) for k in sizes]          # Laplace smoothing
                for c, o in enumerate(obs):
                    np.add.at(counts[c], (st[ok], o[ok]), 1)
            st = _state_codes(d["labels"])
            e = np.flatnonzero(np.diff(st) != 0)
            bounds = np.r_[0, e + 1, len(st)]
            for a, b in zip(bounds[:-1], bounds[1:]):
                if st[a] >= 0:
                    durs[st[a]].append(b - a)
        self.log_emit = [np.log(c / c.sum(1, keepdims=True)) for c in counts]
        hd = np.mean(durs[0] + durs[1]); dd = np.mean(durs[2])
        leave = np.array([1 / hd, 1 / hd, 1 / dd]) * self.stay_scale
        # from H: to A or D; from A: to H or D; from D: to H or A (split by training frequencies)
        n_ha = sum(1 for _ in durs[0]) + sum(1 for _ in durs[1])
        T = np.zeros((3, 3))
        T[0] = [1 - leave[0], leave[0] * 0.6, leave[0] * 0.4]
        T[1] = [leave[1] * 0.6, 1 - leave[1], leave[1] * 0.4]
        T[2] = [leave[2] * 0.5, leave[2] * 0.5, 1 - leave[2]]
        self.log_T = np.log(T)
        return self

    def emissions(self, d):
        obs, _ = observations(d, False, self.shape_model)
        le = np.zeros((len(obs[0]), 3))
        for c in self.channels:
            le += self.log_emit[c][:, obs[c]].T
        le = self.alpha * le
        le[:, 2] += self.d_bias          # per-frame log-offset for the dead-ball state
        return le

    def decode(self, d):
        le = self.emissions(d)
        out = np.full(len(le), "X", dtype="<U1")
        names = np.array(["H", "A", "D"])
        for a, b in _blocks(d["play"]):
            out[a:b] = names[viterbi(le[a:b], self.log_T)]
        return out


def viterbi(log_e, log_T):
    n = len(log_e)
    back = np.zeros((n, 3), dtype=np.int8)
    v = log_e[0] + np.log(np.array([0.2, 0.2, 0.6]))
    for t in range(1, n):
        cand = v[:, None] + log_T
        back[t] = cand.argmax(0)
        v = cand.max(0) + log_e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = v.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def hmm(train_games, **kw):
    model = HMM(**kw).fit(train_games)
    def run(d):
        return model.decode(d)
    run.__name__ = "hmm_" + "_".join(f"{k}{v}" for k, v in kw.items())
    run.model = model
    return run


def hmm_logo(**kw):
    """Leave-one-game-out wrapper for the bench: each game decoded by a model fit on the others."""
    def run(d):
        G = run.games
        train = [g for g in G if g["name"] != d["name"]]
        return HMM(**kw).fit(train).decode(d)
    run.__name__ = "hmmLOGO_" + "_".join(f"{k}{v}" for k, v in kw.items())
    return run
