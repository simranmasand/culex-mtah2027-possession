"""E5: direction-normalised team-shape features and a home-vs-away possession score.

Attacking direction is a constant per half/period (see the plan's *E5 requirement*):
estimated per contiguous play block from where each team's deepest player sits (goalkeeper end).
On BAR-ATH this is checked against the metadata (`attack_direction_first_half`).

Features are all "home minus away", each measured along that team's own attacking direction, so
swapping the teams negates every feature -> the score is antisymmetric (no intercept), which keeps
the HMM's H/A symmetry.
"""

import numpy as np
import pandas as pd

FPS = 25
MID_X = 52.65
FEATURES = ("fwd_mean", "fwd_max", "fwd_min", "width", "length", "vel_fwd")


def _blocks(play):
    e = np.diff(np.r_[0, play.astype(int), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def _segments(seg):
    """[(a, b)] contiguous runs of the same non-zero value."""
    change = np.flatnonzero(np.diff(np.r_[0, seg, 0]) != 0)
    return [(a, b) for a, b in zip(change[:-1], change[1:]) if seg[a] != 0]


def home_direction(d):
    """+1 where home attacks towards +x, -1 towards -x, 0 outside play. One value per play block."""
    P, T = d["players"], d["team_of_slot"]
    out = np.zeros(len(P), dtype=np.int8)
    with np.errstate(all="ignore"):
        hmin = np.nanmin(np.where(T == 0, P[:, :, 0], np.nan), 1)
        amin = np.nanmin(np.where(T == 1, P[:, :, 0], np.nan), 1)
    seg = d["play"].astype(int)
    if "period" in d:                                   # bench games: one block per period
        seg = np.where(d["play"], d["period"], 0)
    for a, b in _segments(seg):
        # home defends the left goal if its deepest player is usually further left than away's
        left = np.nanmean((hmin[a:b] < amin[a:b]).astype(float))
        out[a:b] = 1 if left > 0.5 else -1
    return out


def shape_features(d):
    if "_shape" in d:
        return d["_shape"]
    P, T = d["players"], d["team_of_slot"]
    dirh = home_direction(d).astype(float)
    x, y = P[:, :, 0], P[:, :, 1]
    vx = np.vstack([np.zeros((1, x.shape[1])), np.diff(x, axis=0)]) * FPS
    vx = pd.DataFrame(vx).rolling(FPS, center=True, min_periods=1).mean().to_numpy()
    feats = {}
    with np.errstate(all="ignore"):
        per = {}
        for team, sign in ((0, 1.0), (1, -1.0)):
            m = T == team
            fwd = (x[:, m] - MID_X) * (dirh[:, None] * sign)
            per[team] = {
                "fwd_mean": np.nanmean(fwd, 1), "fwd_max": np.nanmax(fwd, 1), "fwd_min": np.nanmin(fwd, 1),
                "width": np.nanstd(y[:, m], 1), "length": np.nanstd(x[:, m], 1),
                "vel_fwd": np.nanmean(vx[:, m] * (dirh[:, None] * sign), 1),
            }
    for k in FEATURES:
        feats[k] = np.nan_to_num(per[0][k] - per[1][k])
    X = np.column_stack([feats[k] for k in FEATURES])
    X[dirh == 0] = 0
    d["_shape"] = X
    return X


class ShapeScore:
    """Logistic regression (no intercept, antisymmetric) for P(home | alive) from shape features."""

    def fit(self, games, l2=1.0, iters=25):
        Xs, ys = [], []
        for d in games:
            X = shape_features(d)
            lab = d["labels"]
            ok = np.isin(lab, ["H", "A"]) & d["play"]
            Xs.append(X[ok]); ys.append((lab[ok] == "H").astype(float))
        X = np.vstack(Xs); y = np.concatenate(ys)
        self.mu = np.zeros(X.shape[1]); self.sd = X.std(0) + 1e-9   # scale only (keep antisymmetry)
        Z = X / self.sd
        w = np.zeros(Z.shape[1])
        for _ in range(iters):                       # Newton-Raphson
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + l2 * w
            Hm = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
            w -= np.linalg.solve(Hm, g)
        self.w = w
        p = 1 / (1 + np.exp(-Z @ w))
        self.train_acc = float(np.mean((p > 0.5) == (y > 0.5)))
        return self

    def score(self, d):
        return (shape_features(d) / self.sd) @ self.w     # logit of P(home)
