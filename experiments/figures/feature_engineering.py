"""Figures + numbers for the technical document's feature-engineering section.

    python experiments/figures/feature_engineering.py

Writes experiments/figures/fe_*.png and experiments/figures/fe_numbers.json.
Label-conditioned panels use the Metrica bench (hard variant = what the final model trains on).
BAR-ATH panels use no labels. The public 5-minute ground truth is not read.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path[:0] = [ROOT, EXP, os.path.join(ROOT, "starter")]

import bench  # noqa: E402
import methods as M  # noqa: E402
import e9_autoresearch as E  # noqa: E402
from team_shape import shape_features, FEATURES  # noqa: E402

HOME, AWAY, DEAD = "#2a78d6", "#eb6834", "#8a8f98"
INK, MUTED, GRID = "#1f2328", "#5b616b", "#e6e8eb"
FPS = 25
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})
NUM = {}


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), facecolor="white")
    plt.close(fig)


def gap_runs(missing, play):
    m = missing & play
    e = np.diff(np.r_[0, m.astype(int), 0])
    return (np.flatnonzero(e == -1) - np.flatnonzero(e == 1)) / FPS


def bench_frame_table(games):
    """Concatenate per-frame features + labels over bench games (graded/alive part only)."""
    cols = {k: [] for k in ("near_team", "near_d", "gap_len", "spd", "out", "lab", "fwd_mean")}
    for d in games:
        f = M.features(d)
        ok = d["play"]
        cols["near_team"].append(f["near_team"][ok]); cols["near_d"].append(f["near_d"][ok])
        cols["gap_len"].append(f["gap_len"][ok]); cols["spd"].append(f["spd"][ok])
        cols["out"].append(f["out"][ok]); cols["lab"].append(d["labels"][ok])
        cols["fwd_mean"].append(shape_features(d)[ok, 0])
    return {k: np.concatenate(v) for k, v in cols.items()}


# -- 1. what the model sees: two minutes of BAR-ATH ---------------------------------------------
def fig_timeline(B):
    a, b = 40000, 43000                                   # 2 min of first-half open play (video frames)
    f = M.features(B)
    X = E.frame_inputs(B)
    t = (np.arange(a, b) - a) / FPS
    fig, ax = plt.subplots(5, 1, figsize=(10, 7.2), sharex=True,
                           gridspec_kw={"height_ratios": [0.7, 1, 1, 1, 1]})
    nt = f["near_team"][a:b]
    for k, c, lab in ((0, HOME, "Barcelona (home) nearest"), (1, AWAY, "Athletic (away) nearest"),
                      (-1, DEAD, "ball not seen")):
        ax[0].fill_between(t, 0, 1, where=nt == k, color=c, step="mid", linewidth=0, label=lab)
    ax[0].set_yticks([]); ax[0].set_ylim(0, 1); ax[0].grid(False)
    ax[0].set_title("Nearest team to the ball", loc="left", fontsize=10, color=INK)
    ax[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.25), fontsize=9)
    ax[1].plot(t, f["near_d"][a:b], color=INK, lw=1)
    ax[1].set_ylabel("nearest\ndistance (m)")
    ax[2].plot(t, f["gap_len"][a:b], color=INK, lw=1)
    ax[2].set_ylabel("ball-gap\nlength (s)")
    ax[3].plot(t, f["spd"][a:b], color=INK, lw=1)
    ax[3].set_ylabel("mean player\nspeed (m/s)")
    ax[4].plot(t, X[a:b, 7], color=INK, lw=1)
    ax[4].axhline(0, color=MUTED, lw=0.8)
    ax[4].set_ylabel("ball x towards\nhome's attack")
    ax[4].set_ylim(-1.1, 1.1)
    ax[4].set_xlabel("seconds (BAR-ATH, first half, video frames 40,000–43,000)")
    fig.suptitle("What the model sees: per-frame features over two minutes", x=0.01, ha="left",
                 fontsize=12, color=INK, y=1.03)
    fig.align_ylabels(ax)
    save(fig, "fe_timeline.png")


# -- 2. ball gaps: BAR-ATH vs bench, and what a gap says about dead ball --------------------------
def fig_gaps(B, hard, clean, T):
    g_bar = gap_runs(np.isnan(B["ball"][:, 0]), B["play"])
    g_hard = np.concatenate([gap_runs(np.isnan(d["ball"][:, 0]), d["play"]) for d in hard])
    g_clean = np.concatenate([gap_runs(np.isnan(d["ball"][:, 0]), d["play"]) for d in clean])
    NUM["gaps"] = {
        "barath_missing_share": float(np.mean(np.isnan(B["ball"][B["play"], 0]))),
        "barath_median_gap_s": float(np.median(g_bar)), "barath_n_gaps": int(len(g_bar)),
        "barath_share_of_missing_time_in_gaps_over_5s": float(g_bar[g_bar > 5].sum() / g_bar.sum()),
        "bench_hard_missing_share": float(np.mean(np.concatenate(
            [np.isnan(d["ball"][d["play"], 0]) for d in hard]))),
        "bench_hard_median_gap_s": float(np.median(g_hard)),
        "bench_clean_missing_share": float(np.mean(np.concatenate(
            [np.isnan(d["ball"][d["play"], 0]) for d in clean]))),
    }
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    bins = np.logspace(np.log10(0.04), np.log10(600), 40)
    for g, c, lab in ((g_bar, INK, "BAR-ATH (target match)"), (g_hard, HOME, "Metrica bench, hard"),
                      (g_clean, MUTED, "Metrica bench, clean")):
        w = g / g.sum()                                    # weight by time, not by count
        ax[0].hist(g, bins=bins, weights=w, histtype="step", lw=1.8 if c != MUTED else 1.2,
                   color=c, label=lab, linestyle="-" if c != MUTED else "--")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("ball-gap length (s, log scale)")
    ax[0].set_ylabel("share of missing-ball time")
    ax[0].set_title("Missing ball comes in long gaps", loc="left", fontsize=10)
    ax[0].legend(fontsize=8.5, loc="upper left")
    # P(dead | gap-length bin), bench hard, missing-ball frames
    edges = np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64, 1e9])
    names = ["<0.5", "0.5–1", "1–2", "2–4", "4–8", "8–16", "16–32", "32–64", ">64"]
    vis = T["gap_len"] == 0
    p_vis = np.mean(T["lab"][vis] == "D")
    pd_, n_ = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (T["gap_len"] > lo) & (T["gap_len"] <= hi)
        pd_.append(np.mean(T["lab"][m] == "D") if m.any() else np.nan); n_.append(int(m.sum()))
    x = np.arange(len(names))
    ax[1].bar(x, pd_, color=DEAD, width=0.75)
    ax[1].axhline(p_vis, color=INK, lw=1, ls="--")
    ax[1].text(-0.4, p_vis + 0.02, f"dashed: ball visible, {p_vis:.0%} dead", ha="left", fontsize=8.5, color=INK)
    ax[1].set_xticks(x, names, rotation=0, fontsize=8)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("share of frames that are dead ball")
    ax[1].set_xlabel("length of the gap the frame sits in (s)")
    ax[1].set_title("A missing ball roughly doubles the chance of dead ball", loc="left", fontsize=10)
    NUM["gaps"]["bench_hard_p_dead_visible"] = float(p_vis)
    NUM["gaps"]["bench_hard_p_dead_by_gap_bin"] = dict(zip(names, [None if np.isnan(v) else round(float(v), 3) for v in pd_]))
    fig.tight_layout()
    save(fig, "fe_ball_gaps.png")


# -- 3. nearest player: how often is the nearest team the team in possession? ---------------------
def fig_nearest(T):
    alive = np.isin(T["lab"], ["H", "A"]) & (T["near_team"] >= 0)
    truth = (T["lab"][alive] == "A").astype(int)
    d = T["near_d"][alive]; nt = T["near_team"][alive]
    edges = np.array([0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 40])
    acc, share = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        acc.append(np.mean(nt[m] == truth[m])); share.append(m.mean())
    NUM["nearest"] = {"overall_accuracy_alive_visible": float(np.mean(nt == truth)),
                      "accuracy_within_2m": float(np.mean((nt == truth)[d < 2])),
                      "accuracy_over_10m": float(np.mean((nt == truth)[d >= 10])),
                      "median_distance_alive_m": float(np.median(d)),
                      "median_distance_dead_m": float(np.nanmedian(T["near_d"][(T["lab"] == "D") & (T["near_team"] >= 0)]))}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    mids = [f"{lo:g}–{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    x = np.arange(len(mids))
    ax[0].bar(x, acc, color=HOME, width=0.75)
    ax[0].axhline(0.5, color=MUTED, lw=1, ls=":")
    ax[0].set_xticks(x, mids, fontsize=8); ax[0].set_ylim(0.4, 1)
    ax[0].set_xlabel("distance from ball to nearest player (m)")
    ax[0].set_ylabel("nearest team = team in possession")
    ax[0].set_title("Nearest player is reliable only when close", loc="left", fontsize=10)
    for i, (s, v) in enumerate(zip(share, acc)):
        ax[0].text(i, v + 0.01, f"{s:.0%}", ha="center", fontsize=7.5, color=MUTED)
    ax[0].text(len(mids) - 0.5, 0.97, "labels above bars = share of frames · dotted line = coin flip", ha="right", fontsize=7.5, color=MUTED)
    ax[0].set_ylim(0.4, 1.02)
    bins = np.linspace(0, 25, 51)
    for lab, c, name in (("live", INK, "live (H or A)"), ("D", DEAD, "dead ball")):
        m = (np.isin(T["lab"], ["H", "A"]) if lab == "live" else T["lab"] == "D") & (T["near_team"] >= 0)
        ax[1].hist(T["near_d"][m], bins=bins, density=True, histtype="step", lw=1.8, color=c, label=name)
    ax[1].set_xlabel("distance from ball to nearest player (m)")
    ax[1].set_ylabel("density")
    ax[1].set_title("Dead balls sit further from players", loc="left", fontsize=10)
    ax[1].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fe_nearest_player.png")


# -- 4. player speed: live vs dead ---------------------------------------------------------------
def fig_speed(T):
    bins = np.linspace(0, 4, 61)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for lab, c, name in (("live", INK, "live (H or A)"), ("D", DEAD, "dead ball")):
        m = np.isin(T["lab"], ["H", "A"]) if lab == "live" else T["lab"] == "D"
        ax.hist(T["spd"][m], bins=bins, density=True, histtype="step", lw=1.8, color=c, label=name)
        NUM.setdefault("speed", {})[f"median_{lab}_m_s"] = float(np.median(T["spd"][m]))
    ax.set_xlabel("mean speed of tracked players, 1-s average (m/s)")
    ax.set_ylabel("density")
    ax.set_title("Players slow down when the ball is dead (bench, hard)", loc="left", fontsize=10)
    ax.legend(fontsize=8.5)
    save(fig, "fe_player_speed.png")


# -- 5. team shape (tested, dropped from the final model) -----------------------------------------
def fig_shape(T):
    bins = np.linspace(-60, 60, 61)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for lab, c, name in (("H", HOME, "home has the ball"), ("A", AWAY, "away has the ball")):
        m = T["lab"] == lab
        ax.hist(T["fwd_mean"][m], bins=bins, density=True, histtype="step", lw=1.8, color=c, label=name)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("home minus away: mean forward position along own attack (m)")
    ax.set_ylabel("density")
    ax.set_title("Team shape leans the right way, but overlaps heavily (dropped, E5c)", loc="left", fontsize=10, pad=28)
    ax.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    h = T["fwd_mean"][T["lab"] == "H"]; a = T["fwd_mean"][T["lab"] == "A"]
    NUM["shape"] = {"median_fwd_mean_home_poss_m": float(np.median(h)),
                    "median_fwd_mean_away_poss_m": float(np.median(a)),
                    "sign_accuracy": float((np.sum(h > 0) + np.sum(a < 0)) / (len(h) + len(a)))}
    save(fig, "fe_team_shape.png")


def main():
    B = bench.barath()
    hard = bench.bench_games("hard")
    clean = []
    from external_metrica import load_game, GAMES
    for g in GAMES:
        raw = load_game(g)
        clean.append({"ball": raw["ball"], "play": raw["labels"] != "X"})
    T = bench_frame_table(hard)
    fig_timeline(B)
    fig_gaps(B, hard, clean, T)
    fig_nearest(T)
    fig_speed(T)
    fig_shape(T)
    NUM["final_model_inputs"] = ["near_team==home", "near_team==away", "ball not seen", "near_d/20",
                                 "log1p(gap_len)/4", "out", "spd/4", "ball_x (attack-normalised)",
                                 "ball_y"]
    NUM["dropped_inputs"] = list(FEATURES)
    json.dump(NUM, open(os.path.join(HERE, "fe_numbers.json"), "w"), indent=1)
    print(json.dumps(NUM, indent=1))


if __name__ == "__main__":
    main()
