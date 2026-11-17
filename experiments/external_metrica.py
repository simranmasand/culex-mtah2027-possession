"""Metrica public sample data (3 games) as BAR-ATH-shaped arrays + event-derived H/A/D labels.

Source: https://github.com/metrica-sports/sample-data (commit e706dd5), cloned into
data/external/metrica_sample/ (gitignored). Terms (its README): use responsibly and acknowledge
the source in anything public -- credit "Metrica Sports sample data" in the technical document.

These are OTHER games, so labels derived from their (human-produced) event data are allowed for
training and validation (RULES.md: "Hand-labelled data from other games is fine").

    from experiments.external_metrica import load_game
    g = load_game(3)
    g["players"]       float32 (F, S, 3)  x m, y m, speed m/s (speed from finite differences)
    g["ball"]          float32 (F, 2)     x m, y m
    g["team_of_slot"]  int8 (S,)          0 home, 1 away
    g["period"]        int8 (F,)          1 / 2, 0 outside play
    g["labels"]        <U1 (F,)           H / A / D on played frames, X elsewhere
    g["events"]        DataFrame          normalised raw events (frames are array indices)

Array row i == tracking frame i + 1 (both feeds count frames from 1), same as BAR-ATH.
The tracking is clean (ball nearly always present); degrading it to look like BAR-ATH is a
separate step (bench, V1).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

SRC = os.path.join(DATA_DIR, "external", "metrica_sample", "data")
CACHE = os.path.join(DATA_DIR, "cache")
PITCH = (105.0, 68.0)   # sample-data README
FPS = 25
GAMES = (1, 2, 3)

# -- label rules ---------------------------------------------------------------------------------
# Possession = team of the most recent event that implies control of the ball. BALL LOST and
# CHALLENGE do not move possession on their own: a loose / contested ball stays with the last team
# (CHALLENGE.md), and possession changes when the other side has a controlled event (usually a
# RECOVERY, then PASS/CARRY).
CONTROL_TYPES = {"PASS", "CARRY", "SET PIECE", "RECOVERY", "SHOT"}
# Events after which the ball is dead until the next SET PIECE.
def _is_stop(row):
    t, s = row["type"], row["subtype"] if isinstance(row["subtype"], str) else ""
    if t == "BALL OUT" or t == "FAULT RECEIVED":
        return True
    if t == "SHOT" and ("GOAL" in s or "OUT" in s):
        return True
    if t == "BALL LOST" and any(k in s for k in ("OFFSIDE", "HAND BALL", "END HALF")):
        return True
    if t == "PASS" and "OFFSIDE" in s:
        return True
    return False

# A stoppage is only accepted if a SET PIECE follows within this long and no live-play event
# happens in between (filters fouls where advantage was played).
MAX_STOPPAGE_S = 180
LIVE_TYPES = {"PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "CHALLENGE"}


# -- events ---------------------------------------------------------------------------------------

def _events_csv(game):
    e = pd.read_csv(os.path.join(SRC, f"Sample_Game_{game}", f"Sample_Game_{game}_RawEventsData.csv"))
    return pd.DataFrame({
        "type": e["Type"].str.upper(),
        "subtype": e["Subtype"].where(e["Subtype"].notna(), None),
        "side": e["Team"].map({"Home": 0, "Away": 1}),
        "period": e["Period"].astype(int),
        "start": e["Start Frame"].astype(int) - 1,
        "end": e["End Frame"].astype(int) - 1,
    })


def _events_json(game, home_id):
    raw = json.load(open(os.path.join(SRC, f"Sample_Game_{game}", f"Sample_Game_{game}_events.json")))["data"]
    rows = []
    for x in raw:
        sub = x["subtypes"]
        if isinstance(sub, dict):
            sub = sub["name"]
        elif sub:
            sub = "-".join(s["name"] for s in sub)
        rows.append({"type": x["type"]["name"].upper(), "subtype": sub or None,
                     "side": 0 if x["team"]["id"] == home_id else 1, "period": int(x["period"]),
                     "start": int(x["start"]["frame"]) - 1,
                     "end": int(x["end"]["frame"] or x["start"]["frame"]) - 1})
    return pd.DataFrame(rows)


# -- tracking -------------------------------------------------------------------------------------

def _tracking(game):
    from kloppy import metrica
    d = os.path.join(SRC, f"Sample_Game_{game}")
    if game in (1, 2):
        ds = metrica.load_tracking_csv(
            home_data=os.path.join(d, f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"),
            away_data=os.path.join(d, f"Sample_Game_{game}_RawTrackingData_Away_Team.csv"),
            coordinates="metrica")
    else:
        ds = metrica.load_tracking_epts(meta_data=os.path.join(d, "Sample_Game_3_metadata.xml"),
                                        raw_data=os.path.join(d, "Sample_Game_3_tracking.txt"),
                                        coordinates="metrica")
    df = ds.to_df()
    home, away = ds.metadata.teams
    slots = [(p.player_id, 0) for p in home.players] + [(p.player_id, 1) for p in away.players]
    slots = [(pid, side) for pid, side in slots if f"{pid}_x" in df.columns]
    n = int(df["frame_id"].max())
    idx = df["frame_id"].to_numpy(int) - 1

    players = np.full((n, len(slots), 3), np.nan, dtype=np.float32)
    for s, (pid, _) in enumerate(slots):
        players[idx, s, 0] = df[f"{pid}_x"].to_numpy(float) * PITCH[0]
        players[idx, s, 1] = df[f"{pid}_y"].to_numpy(float) * PITCH[1]
    step = np.linalg.norm(np.diff(players[:, :, :2], axis=0), axis=2) * FPS
    speed = pd.DataFrame(np.vstack([step[:1], step])).rolling(5, center=True, min_periods=1).mean()
    players[:, :, 2] = speed.to_numpy(np.float32)

    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[idx, 0] = df["ball_x"].to_numpy(float) * PITCH[0]
    ball[idx, 1] = df["ball_y"].to_numpy(float) * PITCH[1]
    period = np.zeros(n, dtype=np.int8)
    period[idx] = df["period_id"].to_numpy(int)
    return players, ball, np.array([side for _, side in slots], dtype=np.int8), period, home.team_id


# -- labels ---------------------------------------------------------------------------------------

def derive_labels(events, period):
    """Per-frame H/A/D from events. X outside the frames where each period is being played."""
    n = len(period)
    labels = np.full(n, "X", dtype="<U1")
    ev = events.sort_values(["start", "end"]).reset_index(drop=True)

    for p in sorted(set(ev["period"])):
        pe = ev[ev["period"] == p]
        kick = pe[(pe["type"] == "SET PIECE") & pe["subtype"].fillna("").str.contains("KICK OFF")]
        first = int(kick["start"].min()) if len(kick) else int(pe["start"].min())
        last = int(pe["end"].max())
        frames = np.flatnonzero(period == p)
        a, b = int(frames.min()), int(frames.max()) + 1
        labels[a:b] = "D"                                  # before kick-off / after last event

        # possession: forward-fill the team of the latest control event
        team = np.full(n, -1, dtype=np.int8)
        for _, r in pe[pe["type"].isin(CONTROL_TYPES)].iterrows():
            team[r["start"]] = r["side"]
        cur = -1
        for i in range(first, last + 1):
            if team[i] >= 0:
                cur = team[i]
            team[i] = cur
        alive = np.zeros(n, bool)
        alive[first:last + 1] = True

        # stoppages: stop event -> next SET PIECE, if no live play in between
        sp_starts = pe.loc[pe["type"] == "SET PIECE", "start"].to_numpy()
        live_starts = np.sort(pe.loc[pe["type"].isin(LIVE_TYPES), "start"].to_numpy())
        for _, r in pe[pe.apply(_is_stop, axis=1)].iterrows():
            stop = int(max(r["start"], r["end"]))
            nxt = sp_starts[sp_starts > stop]
            if not len(nxt) or (nxt[0] - stop) > MAX_STOPPAGE_S * FPS:
                continue
            between = live_starts[(live_starts > stop) & (live_starts < nxt[0])]
            if len(between):
                continue
            alive[stop:nxt[0]] = False

        lab = np.where(team == 0, "H", np.where(team == 1, "A", "D"))
        sel = np.arange(a, b)
        labels[sel] = np.where(alive[sel] & (team[sel] >= 0), lab[sel], "D")
    return labels


# -- public API -----------------------------------------------------------------------------------

def build(game):
    players, ball, team_of_slot, period, home_id = _tracking(game)
    events = _events_csv(game) if game in (1, 2) else _events_json(game, home_id)
    labels = derive_labels(events, period)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, f"metrica_g{game}.npz"), players=players, ball=ball,
             team_of_slot=team_of_slot, period=period, labels=labels)
    events.to_csv(os.path.join(CACHE, f"metrica_g{game}_events.csv"), index=False)


def load_game(game, rebuild=False):
    path = os.path.join(CACHE, f"metrica_g{game}.npz")
    if rebuild or not os.path.exists(path):
        build(game)
    with np.load(path) as npz:
        out = {k: npz[k] for k in npz.files}
    out["events"] = pd.read_csv(os.path.join(CACHE, f"metrica_g{game}_events.csv"))
    return out


def spells(labels):
    """[(start, stop, label)] maximal H/A runs, never across D or X -- same as the scorer."""
    out, i, n = [], 0, len(labels)
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        if labels[i] in ("H", "A"):
            out.append((i, j, labels[i]))
        i = j
    return out


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    for g in GAMES:
        d = load_game(g, rebuild)
        lab, played = d["labels"], d["labels"] != "X"
        sp = spells(lab)
        dur = np.array([b - a for a, b, _ in sp]) / FPS
        print(f"game {g}: frames {len(lab)}, played {played.sum()}, slots {len(d['team_of_slot'])}, "
              f"ball present {np.mean(~np.isnan(d['ball'][played, 0])):.1%}")
        print(f"   labels H {np.mean(lab[played] == 'H'):.1%}  A {np.mean(lab[played] == 'A'):.1%}  "
              f"D {np.mean(lab[played] == 'D'):.1%} | possessions {len(sp)}, "
              f"mean {dur.mean():.1f}s, median {np.median(dur):.1f}s, <1s {np.mean(dur < 1):.0%}")
