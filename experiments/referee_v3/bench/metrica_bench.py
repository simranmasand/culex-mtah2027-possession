"""Score the possession models on the three public Metrica sample games (external data).

    python bench/metrica_bench.py --data path/to/sample-data/data
    python bench/metrica_bench.py --data ... --labels path/to/labels   # score on your own labels

Run from the root of the challenge repository (it uses evaluation/possession_metrics.py).
External data only: github.com/metrica-sports/sample-data, commit e706dd5 (the data folder holds
Sample_Game_1, Sample_Game_2 and Sample_Game_3). Nothing here reads the competition match.

--labels: a folder with Sample_Game_1.csv, Sample_Game_2.csv and Sample_Game_3.csv in the
submission format (frame,label with H/A/D/X, 0-based frames). Use it to score against another
set of event-derived labels, such as the ones the GRU benchmark uses, so the numbers line up.

Ground truth is built by this script from Metrica's own event data (the files carry no
possession labels), following the challenge's definitions:
  * possession changes when the other team makes a controlled action (a pass, carry, set piece,
    shot or recovery; a "ball lost" or "ball out" event also shows its team had the ball);
  * the ball is dead from a ball-out, a foul, an offside or a goal until the next restart event,
    and before each period's kick-off and after its last event.
Scores are only comparable between models scored on the same labels.

Each game is scored twice: on the clean tracking ("clean"), and with the ball hidden in random
gaps covering about 41% of frames, median gap about 10.7 s, as on the competition feed
("degraded", fixed seed). Model presets v1, v2, v3 and the organisers' smooth50 baseline.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.environ.get("METRICA_DATA", "")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))

from possession import pipeline, settings, signals  # noqa: E402
import possession_metrics as pm  # noqa: E402

PITCH = (105.0, 68.0)
FPS = 25
POSSESSION_TYPES = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT"}


# ---------------------------------------------------------------------------------------------
# Loading

def data_file(name):
    """A sample-data file, whether the data folder is flat or split into Sample_Game_N folders."""
    flat = os.path.join(DATA, name)
    if os.path.isfile(flat):
        return flat
    return os.path.join(DATA, "_".join(name.split("_")[:3]), name)   # Sample_Game_N/


def load_csv_game(n):
    def read(side):
        path = data_file("Sample_Game_{}_RawTrackingData_{}_Team.csv".format(n, side))
        return pd.read_csv(path, skiprows=2, low_memory=False)
    home, away = read("Home"), read("Away")
    frames = home["Frame"].values.astype(int)
    n_frames = int(frames.max())
    def players(df):
        cols = [c for c in df.columns if c.startswith("Player")]
        xy = []
        for c in cols:
            i = df.columns.get_loc(c)
            xy.append(df.iloc[:, [i, i + 1]].values.astype(np.float32))
        return np.stack(xy, axis=1)
    ph, pa = players(home), players(away)
    ball_cols = home.columns.get_loc("Ball")
    ball_rows = home.iloc[:, [ball_cols, ball_cols + 1]].values.astype(np.float32)
    idx = frames - 1
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball[idx] = ball_rows
    pl = np.full((n_frames, ph.shape[1] + pa.shape[1], 2), np.nan, dtype=np.float32)
    pl[idx, :ph.shape[1]] = ph
    pl[idx, ph.shape[1]:] = pa
    slot_team = np.array([0] * ph.shape[1] + [1] * pa.shape[1], dtype=np.int8)
    periods = home["Period"].values.astype(int)
    blocks = []
    for p in sorted(set(periods.tolist())):
        f = idx[periods == p]
        blocks.append((p, int(f.min()), int(f.max())))

    ev = pd.read_csv(data_file("Sample_Game_{}_RawEventsData.csv".format(n)))
    events = []
    for r in ev.itertuples(index=False):
        events.append({"team": 0 if r.Team == "Home" else 1, "type": str(r.Type),
                       "sub": "" if pd.isna(r.Subtype) else str(r.Subtype),
                       "start": int(r[4]) - 1, "end": int(r[6]) - 1, "period": int(r.Period)})
    return ball, pl, slot_team, blocks, events, n_frames


def load_epts_game(n):
    meta = ET.parse(data_file("Sample_Game_{}_metadata.xml".format(n))).getroot()
    teams = [t.get("id") for t in meta.iter("Team")]
    home_id = teams[0]
    team_of = {p.get("id"): p.get("teamId") for p in meta.iter("Player")}
    order = [c.get("playerId") for c in meta.iter("PlayerChannel") if c.get("channelId") == "x"]
    seen, slots = set(), []
    for pid in order:
        if pid not in seen:
            seen.add(pid)
            slots.append(pid)
    slot_team = np.array([0 if team_of[p] == home_id else 1 for p in slots], dtype=np.int8)
    params = {pp.findtext("Name"): pp.findtext("Value") for pp in meta.iter("ProviderParameter")}

    frames, ball_rows, player_rows = [], [], []
    with open(data_file("Sample_Game_{}_tracking.txt".format(n))) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frames.append(int(parts[0]))
            player_rows.append(parts[1].replace(";", ","))
            ball_rows.append(parts[2])
    n_frames = max(frames)
    idx = np.array(frames) - 1
    flat = np.array(",".join(player_rows).split(","), dtype=np.float32).reshape(len(frames), -1, 2)
    # The column order changes after substitutions: each DataFormatSpecification says, for its
    # frame range, which player each column holds.
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    slot_of = {p: i for i, p in enumerate(slots)}
    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        cols = [slot_of[channel_player[r.get("playerChannelId")]]
                for r in spec.iter("PlayerChannelRef") if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), cols))
    bxy = np.array(",".join(b.split(",")[0] + "," + b.split(",")[1] for b in ball_rows).split(","),
                   dtype=np.float32).reshape(-1, 2)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball[idx] = bxy
    pl = np.full((n_frames, len(slots), 2), np.nan, dtype=np.float32)
    frame_ids = np.array(frames)
    for start, end, cols in specs:
        rows = np.flatnonzero((frame_ids >= start) & (frame_ids <= end))
        pl[np.ix_(idx[rows], cols)] = flat[rows][:, :len(cols)]
    blocks = [(1, int(params["first_half_start"]) - 1, int(params["first_half_end"]) - 1),
              (2, int(params["second_half_start"]) - 1, min(n_frames, int(params["second_half_end"])) - 1)]

    data = json.load(open(data_file("Sample_Game_{}_events.json".format(n))))["data"]
    events = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        else:
            sub = ""
        events.append({"team": 0 if e["team"]["id"] == home_id else 1, "type": e["type"]["name"],
                       "sub": sub, "start": int(e["start"]["frame"]) - 1,
                       "end": int(e["end"]["frame"] or e["start"]["frame"]) - 1,
                       "period": int(e["period"])})
    return ball, pl, slot_team, blocks, events, n_frames


# ---------------------------------------------------------------------------------------------
# Ground truth from events

def dead_trigger(e):
    """Frame from which this event makes the ball dead, or None."""
    t, s = e["type"], e["sub"].upper()
    if t == "BALL OUT" or (t == "SHOT" and "OUT" in s) or "OFFSIDE" in s:
        return max(e["start"], e["end"])
    if t == "SHOT" and "GOAL" in s:
        return max(e["start"], e["end"])
    if t == "FAULT RECEIVED":
        return e["start"]
    return None


def ground_truth(events, blocks, n_frames):
    labels = np.full(n_frames, "X", dtype="<U1")
    for period, first, last in blocks:
        evs = sorted((e for e in events if e["period"] == period and first <= e["start"] <= last),
                     key=lambda e: e["start"])
        changes = [(first, "D")]       # dead until the kick-off
        owner, dead, last_end = None, True, first
        for e in evs:
            last_end = max(last_end, e["end"], e["start"])
            if e["type"] in POSSESSION_TYPES:
                team = "H" if e["team"] == 0 else "A"
                if dead:
                    if e["start"] > changes[-1][0] or changes[-1][1] != "D":
                        changes.append((e["start"], team))
                    else:
                        changes[-1] = (e["start"], team)
                    owner, dead = team, False
                elif team != owner:
                    changes.append((e["start"], team))
                    owner = team
            trigger = dead_trigger(e)
            if trigger is not None and not dead:
                changes.append((trigger, "D"))
                dead = True
        changes.append((min(last, last_end + 1), "D"))   # after the final event
        changes.sort(key=lambda c: c[0])
        for i, (frame, label) in enumerate(changes):
            stop = changes[i + 1][0] if i + 1 < len(changes) else last + 1
            labels[max(frame, first):min(stop, last + 1)] = label
    return labels


# ---------------------------------------------------------------------------------------------
# Degrading the ball track like the competition feed

def degrade(ball, blocks, seed=0, share=0.41, median_s=10.7):
    rng = np.random.RandomState(seed)
    out = ball.copy()
    graded = sum(last - first + 1 for _, first, last in blocks)
    target, hidden = share * graded, 0
    while hidden < target:
        half = blocks[rng.randint(len(blocks))]
        length = int(rng.lognormal(np.log(median_s * FPS), 0.9))
        start = rng.randint(half[1], max(half[1] + 1, half[2] - length))
        before = np.isnan(out[start:start + length, 0]).sum()
        out[start:start + length] = np.nan
        hidden += length - before
    return out


# ---------------------------------------------------------------------------------------------
# Models

def baseline(ball, players, slot_team, n_frames, window=50):
    scale = np.array(PITCH, dtype=np.float32)
    d = np.sqrt((((players * scale) - (ball * scale)[:, None, :]) ** 2).sum(axis=2))
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=1)
    codes = np.where(np.isnan(ball[:, 0]) | ~np.isfinite(d.min(axis=1)), -1,
                     slot_team[nearest]).astype(np.int8)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(n_frames)
    lo, hi = np.maximum(0, i - window), np.minimum(n_frames, i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    labels = np.full(n_frames, "D", dtype="<U1")
    seen = (h + a) > 0
    labels[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return labels


def rules_model(name, ball, players, slot_team, blocks, n_frames):
    settings.use(name)
    speeds = np.full(players.shape[:2], np.nan, dtype=np.float32)
    sig = signals.compute(ball, players, speeds, slot_team, PITCH, FPS)
    labels, _ = pipeline.label_frames(sig, blocks, [], n_frames, FPS, None)
    return labels


def score(gt, pred, blocks):
    pred = pred.copy()
    pred[gt == "X"] = "X"
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=200)
    return r


def main():
    global DATA
    parser = argparse.ArgumentParser(description="Score the models on the Metrica sample games.")
    parser.add_argument("--data", default=DATA,
                        help="the sample-data 'data' folder (or set METRICA_DATA)")
    parser.add_argument("--labels", default=None,
                        help="folder of Sample_Game_N.csv label files to score against instead")
    parser.add_argument("--out", default=os.path.join(HERE, "metrica_bench_results.csv"))
    args = parser.parse_args()
    if not args.data or not os.path.isdir(args.data):
        parser.error("point --data at the sample-data 'data' folder")
    DATA = args.data

    games = [(1, load_csv_game), (2, load_csv_game), (3, load_epts_game)]
    rows = []
    for n, loader in games:
        ball, players, slot_team, blocks, events, n_frames = loader(n)
        if args.labels:
            path = os.path.join(args.labels, "Sample_Game_{}.csv".format(n))
            given = pd.read_csv(path)["label"].astype(str).values
            gt = np.full(n_frames, "X", dtype="<U1")
            gt[:min(n_frames, len(given))] = given[:n_frames]
        else:
            gt = ground_truth(events, blocks, n_frames)
        g_codes = pm.label_codes(gt)
        stats = pm.possession_stats(g_codes, pm.valid_runs(g_codes))
        n_gt = len(pm.possession_spells(g_codes, pm.valid_runs(g_codes)))
        dead_min = np.sum(gt == "D") / FPS / 60.0
        live = (gt == "H") | (gt == "A")
        print("game {}: {} frames graded, truth {} possessions (mean {:.1f} s), {:.1f} min dead, "
              "ball missing on {:.0%} of graded frames ({:.0%} of live, {:.0%} of dead)".format(
                  n, int(np.sum(gt != "X")), n_gt, stats["mean_duration_s"], dead_min,
                  float(np.mean(np.isnan(ball[gt != "X", 0]))),
                  float(np.mean(np.isnan(ball[live, 0]))),
                  float(np.mean(np.isnan(ball[gt == "D", 0])))))
        for variant, b in (("clean", ball), ("degraded", degrade(ball, blocks, seed=n))):
            preds = {"baseline_smooth50": baseline(b, players, slot_team, n_frames)}
            for name in ("v1", "v2", "v3"):
                preds["KOone_" + name] = rules_model(name, b, players, slot_team, blocks, n_frames)
            for model, pred in preds.items():
                r = score(gt, pred, blocks)
                rows.append({"game": n, "tracking": variant, "model": model,
                             "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                             "possessions": r["n_pred_spells"], "truth": r["n_gt_spells"],
                             "mean_s": r["mean_spell_duration_s"],
                             "D_precision": r["per_class"]["D"]["precision"],
                             "D_recall": r["per_class"]["D"]["recall"],
                             "spell_low": r["spell_bootstrap_low"],
                             "spell_high": r["spell_bootstrap_high"]})
                print("  {:9s} {:18s} spellF1 {:.3f}  macroF1 {:.3f}  poss {:4d}/{:4d}  "
                      "D P/R {:.2f}/{:.2f}".format(variant, model, rows[-1]["spellF1"],
                                                   rows[-1]["macroF1"], rows[-1]["possessions"],
                                                   rows[-1]["truth"], rows[-1]["D_precision"],
                                                   rows[-1]["D_recall"]))
    settings.use(settings.DEFAULT)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\nmean over the three games:")
    print(df.groupby(["tracking", "model"])[["spellF1", "macroF1", "possessions", "truth"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
