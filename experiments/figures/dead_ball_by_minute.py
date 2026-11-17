"""Ball coverage per minute of BAR-ATH vs where each model calls dead ball.

    python experiments/figures/dead_ball_by_minute.py                    # Referee Model (logic) vs final GRU
    python experiments/figures/dead_ball_by_minute.py nn_a4 nn_a3 nn_final

Top panel: % of frames with a tracked ball, per minute of video. One panel per model below: % of graded frames
labelled D (dead ball) in that minute. Ungraded minutes (before kick-off, halftime, after the whistle) are shaded.
No labels used: tracking + our own submission files only. Writes experiments/figures/dead_ball_by_minute.png + .json.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path[:0] = [ROOT, EXP, os.path.join(ROOT, "starter")]

from atd_cache import load  # noqa: E402
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

BALL, DEAD = "#2a78d6", "#8a8f98"
INK, MUTED, GRID, SHADE = "#1f2328", "#5b616b", "#e6e8eb", "#f3f4f6"
FPM = 25 * 60
NAMES = {"nn_a4": "GRU, learned — final, D19", "referee_v3": "Referee Model v3, logic", "nn_a3": "D18", "nn_final": "D17",
         "nn_best": "with team shape, D13", "nn_noshape": "D16"}
BENCH = json.load(open(os.path.join(EXP, "eval_table.json")))
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


def per_minute(x, mask):
    n = len(x) // FPM * FPM
    x, mask = x[:n].reshape(-1, FPM), mask[:n].reshape(-1, FPM)
    cnt = mask.sum(1)
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, (x * mask).sum(1) / np.maximum(cnt, 1), np.nan), cnt / FPM


def main(files):
    d = load()
    graded = d["graded"]
    seen = ~np.isnan(d["ball"][:, 0])
    cov, _ = per_minute(seen.astype(float), graded)
    _, graded_share = per_minute(np.ones(len(graded)), graded)
    ungraded = graded_share < 0.5
    mins = np.arange(len(cov))
    miss = 1 - per_minute(seen.astype(float), graded)[0]

    fig, axes = plt.subplots(1 + len(files), 1, figsize=(13, 2.0 + 1.6 * len(files)), sharex=True,
                             gridspec_kw={"height_ratios": [1.4] + [1] * len(files), "hspace": 0.45})
    nums = {"ball_coverage_graded": float(seen[graded].mean()), "models": {}}

    def shade(ax):
        for m in mins[ungraded]:
            ax.axvspan(m - 0.5, m + 0.5, color=SHADE, lw=0, zorder=0)
        for ko in KICKOFF_VIDEO_FRAMES.values():
            ax.axvline(ko / FPM, color=MUTED, lw=0.8, ls=":", zorder=1)

    ax = axes[0]
    shade(ax)
    ok0 = ~np.isnan(cov)
    ax.bar(mins[ok0], cov[ok0] * 100, width=0.8, color=BALL, lw=0)
    avg = 100 * seen[graded].mean()
    ax.axhline(avg, color=INK, lw=1, ls="--")
    ax.text(mins[-1] + 1, avg, f"graded avg {avg:.0f}%", va="center", ha="left", color=INK, fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("% frames\nwith a ball")
    ax.set_title("Ball coverage per minute, graded frames (tracking)", loc="left", fontsize=10, color=INK)

    for ax, f in zip(axes[1:], files):
        lab = pd.read_csv(os.path.join(ROOT, "evaluation", "results", f + ".csv"))["label"].astype(str).str.strip().to_numpy()
        dead, _ = per_minute((lab == "D").astype(float), graded)
        ok = ~np.isnan(dead)
        r = float(np.corrcoef(dead[ok], miss[ok])[0, 1])
        b = BENCH["hard"].get(f, {}).get("spellF1")
        nums["models"][f] = {"D_share_graded": float((lab[graded] == "D").mean()),
                             "corr_minuteD_vs_ball_missing": r, "bench_hard": b}
        shade(ax)
        ax.bar(mins[ok], dead[ok] * 100, width=0.8, color=DEAD, lw=0)
        mm = np.where(ok, miss * 100, np.nan)
        ax.step(mins, mm, where="mid", color=BALL, lw=1.2)
        if ax is axes[1]:
            ax.text(mins[-1] + 1, 80, "— ball missing %", color=INK, fontsize=9, va="center")
            ax.text(mins[-1] + 1, 50, "▮ dead ball %", color=INK, fontsize=9, va="center")
        ax.set_ylim(0, 100); ax.set_ylabel("% dead ball")
        bench_txt = f" · bench hard {b:.3f}" if b is not None else ""
        ax.set_title(f"{f} ({NAMES.get(f, f)}): dead ball {100 * nums['models'][f]['D_share_graded']:.0f}% of graded frames"
                     f"{bench_txt} · r(dead, ball missing) per minute = {r:.2f}",
                     loc="left", fontsize=10, color=INK)
    axes[-1].set_xlabel("minute of video (25 fps); shaded = not graded, dotted = kick-offs")
    axes[-1].set_xlim(-0.5, mins[-1] + 0.5)
    rs = {f: nums["models"][f]["corr_minuteD_vs_ball_missing"] for f in files}
    fig.suptitle("Where each model calls dead ball, minute by minute, against ball coverage ("
                 + ", ".join(f"{f}: r {r:.2f}" for f, r in rs.items()) + ")",
                 x=0.125, ha="left", fontsize=12, color=INK, y=1.0)
    fig.savefig(os.path.join(HERE, "dead_ball_by_minute.png"), facecolor="white")
    json.dump(nums, open(os.path.join(HERE, "dead_ball_by_minute.json"), "w"), indent=1)
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["referee_v3", "nn_a4"])
