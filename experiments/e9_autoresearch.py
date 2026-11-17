"""E9: automatic researcher for a small sequence net (karpathy/autoresearch-style loop, but bounded).

    python experiments/e9_autoresearch.py --minutes 45

Loop: propose a config (mutate the best so far, or random) -> train leave-one-game-out on the HARD
bench -> decode with Viterbi (HMM transitions) -> mean spellF1 -> log to results.tsv -> keep if best.
Fixed evaluation (bench.score / bench.compare) is never edited by the loop. BAR-ATH labels are never
used; the public 5 min is not touched here.

At the end: the best config vs the E4 HMM on hard AND easy (paired episode bootstrap), and a
BAR-ATH submission from a model trained on all three hard games -> evaluation/results/nn_e9.csv.
Everything is seeded; results land in experiments/e9_nn/.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "starter")]

import bench  # noqa: E402
import methods as M  # noqa: E402
from team_shape import shape_features, home_direction  # noqa: E402

OUT = os.path.join(HERE, "e9_nn")
TB = None                     # tensorboard SummaryWriter, set in main()
MON = {"run": 0, "fold": "", "epoch": 0, "epochs": 0, "t_end": 0.0, "best": -1.0, "gstep": 0}


DESCRIPTIONS = {
    "loss/train": "How wrong the net is on the games it trains on (lower = better). Should fall epoch by epoch.",
    "loss/test_game": "Same loss on the held-out game it never trained on. If this rises while train loss falls, the net is overfitting.",
    "step_accuracy/train": "Share of 0.2-s steps labelled correctly (H/A/D) on the training games, before Viterbi.",
    "step_accuracy/test_game": "Same accuracy on the held-out game. The gap to train accuracy shows how well it generalises.",
    "search/minutes_left": "Time left in the search budget.",
    "search/spellF1_hard_LOGO": "The challenge's ranking metric (whole possessions matched) for each configuration tried, averaged over the 3 held-out games. Higher = better.",
    "search/best_spellF1": "Best score found so far in the search. Steps up when a configuration beats the previous best.",
    "search/hmm_reference": "The HMM's score on the same test (0.434) - the line to beat.",
    "search/possessions_pred (true 1022)": "How many possessions the configuration predicts across the 3 games (truth: 1,022). Far below = merging possessions; far above = splitting them.",
}
_DESCRIBED = set()


def tb_scalar(tag, value, step):
    """add_scalar, plus a one-line description shown in TensorBoard's tooltip (first write only)."""
    if TB is None:
        return
    if tag in _DESCRIBED or tag not in DESCRIPTIONS:
        TB.add_scalar(tag, value, step)
        return
    from torch.utils.tensorboard.summary import scalar as _scalar_summary
    summ = _scalar_summary(tag, value, new_style=True)
    summ.value[0].metadata.summary_description = DESCRIPTIONS[tag]
    TB._get_file_writer().add_summary(summ, step)
    _DESCRIBED.add(tag)


def tb_readme():
    rows = "\n".join(f"| `{k}` | {v} |" for k, v in DESCRIPTIONS.items())
    TB.add_text("README", "### What each chart shows\n\nx-axis: `search/*` = configuration number; "
                "`loss/*`, `step_accuracy/*` = running epoch count across configurations and held-out games.\n\n"
                "| chart | meaning |\n|---|---|\n" + rows, 0)

def _progress():
    left = max(0.0, MON["t_end"] - time.time())
    MON["minutes_left"] = round(left / 60, 1)
    json.dump({k: v for k, v in MON.items() if k != "t_end"}, open(os.path.join(OUT, "progress.json"), "w"))
SUB = 5                       # 25 fps -> 5 Hz steps
torch.set_num_threads(4)
DEV = "cpu"                   # deterministic + small model: CPU is fine

# -- inputs ---------------------------------------------------------------------------------------

def frame_inputs(d, swap=False):
    f = M.features(d)
    nt = f["near_team"]
    if swap:
        nt = np.where(nt >= 0, 1 - nt, nt)
    dirh = home_direction(d).astype(np.float32)
    if swap:
        dirh = -dirh
    bx = np.nan_to_num((d["ball"][:, 0] - 52.65) * dirh / 52.65)
    by = np.nan_to_num((d["ball"][:, 1] - 34.0) / 34.0)
    shape = shape_features(d) / 10.0
    if swap:
        shape = -shape
    X = np.column_stack([
        nt == 0, nt == 1, nt < 0,
        np.clip(np.nan_to_num(f["near_d"], nan=20.0), 0, 20) / 20.0,
        np.log1p(f["gap_len"]) / 4.0, f["out"], f["spd"] / 4.0,
        bx, by, np.clip(shape, -3, 3),
    ]).astype(np.float32)
    return X


def steps(X):
    n = len(X) // SUB * SUB
    return X[:n].reshape(-1, SUB, X.shape[1]).mean(1)


def step_labels(lab, swap=False):
    s = M._state_codes(lab, swap)
    n = len(s) // SUB * SUB
    s = s[:n].reshape(-1, SUB)
    out = np.full(len(s), -1, dtype=np.int64)
    for k in (0, 1, 2):
        out[(s == k).mean(1) > 0.5] = k
    return out


def prepared(games):
    data = []
    for d in games:
        for swap in (False, True):
            data.append((steps(frame_inputs(d, swap)), step_labels(d["labels"], swap)))
    return data


# -- models ---------------------------------------------------------------------------------------

class GRUNet(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(n_in, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        return self.head(self.rnn(x)[0])


class TCN(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout):
        super().__init__()
        mods, c = [], n_in
        for i in range(layers * 2):
            mods += [nn.Conv1d(c, hidden, 3, padding=2 ** i, dilation=2 ** i), nn.ReLU(), nn.Dropout(dropout)]
            c = hidden
        self.net = nn.Sequential(*mods)
        self.head = nn.Conv1d(hidden, 3, 1)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2))).transpose(1, 2)


def build(cfg, n_in):
    cls = GRUNet if cfg["arch"] == "gru" else TCN
    return cls(n_in, cfg["hidden"], cfg["layers"], cfg["dropout"]).to(DEV)


def train(cfg, train_games, seed=0, val_game=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    data = prepared(train_games)
    model = build(cfg, data[0][0].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    L, B = cfg["window"], 32
    total = sum(len(x) for x, _ in data)
    per_epoch = max(1, total // (L * B))
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    val = None
    if val_game is not None:
        val = (torch.tensor(steps(frame_inputs(val_game)))[None],
               torch.tensor(step_labels(val_game["labels"]))[None])
    MON["epochs"] = cfg["epochs"]
    for ep in range(cfg["epochs"]):
        model.train()
        losses, accs = [], []
        for _ in range(per_epoch):
            xb, yb = [], []
            for _ in range(B):
                x, y = data[np.random.randint(len(data))]
                s = np.random.randint(0, len(x) - L)
                xb.append(x[s:s + L]); yb.append(y[s:s + L])
            xb = torch.tensor(np.stack(xb)); yb = torch.tensor(np.stack(yb))
            logits = model(xb)
            loss = lossf(logits.reshape(-1, 3), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            ok = yb >= 0
            losses.append(loss.item()); accs.append((logits.argmax(-1)[ok] == yb[ok]).float().mean().item())
        MON["epoch"] = ep + 1; MON["gstep"] += 1
        if TB is not None:
            tb_scalar("loss/train", np.mean(losses), MON["gstep"])
            tb_scalar("step_accuracy/train", np.mean(accs), MON["gstep"])
            if val is not None:
                model.eval()
                with torch.no_grad():
                    lv = model(val[0]); okv = val[1] >= 0
                    tb_scalar("loss/test_game", lossf(lv.reshape(-1, 3), val[1].reshape(-1)).item(), MON["gstep"])
                    tb_scalar("step_accuracy/test_game", (lv.argmax(-1)[okv] == val[1][okv]).float().mean().item(), MON["gstep"])
            tb_scalar("search/minutes_left", max(0, MON["t_end"] - time.time()) / 60, MON["gstep"])
        _progress()
    model.eval()
    return model


def nn_method(cfg, train_games, seed=0, val_game=None):
    model = train(cfg, train_games, seed, val_game)
    hmm = M.HMM(alpha=0.3, stay_scale=cfg["stay"]).fit(train_games)

    def run(d):
        X = steps(frame_inputs(d))
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor(X)[None]), -1)[0].numpy()
        le = np.repeat(logp, SUB, axis=0)
        le = np.vstack([le, np.repeat(le[-1:], len(d["ball"]) - len(le), axis=0)]) * cfg["alpha"]
        out = np.full(len(le), "X", dtype="<U1")
        names = np.array(["H", "A", "D"])
        for a, b in M._blocks(d["play"]):
            out[a:b] = names[M.viterbi(le[a:b], hmm.log_T)]
        return out
    run.__name__ = "nn"
    return run


def logo(cfg, games, seed=0):
    preds = {}

    def run(d):
        if d["name"] not in preds:
            tr = [g for g in games if g["name"] != d["name"]]
            MON["fold"] = d["name"]
            preds[d["name"]] = nn_method(cfg, tr, seed, val_game=d)(d)
        return preds[d["name"]]
    return run


# -- search space ---------------------------------------------------------------------------------

SPACE = {"arch": ["gru", "tcn"], "hidden": [16, 32, 64, 128], "layers": [1, 2], "dropout": [0.0, 0.1, 0.3],
         "lr": [1e-3, 3e-3, 1e-2], "epochs": [2, 4, 8, 12], "window": [64, 128, 256],
         "alpha": [0.05, 0.1, 0.2, 0.4], "stay": [0.5, 1.0, 2.0]}
# best of the first (unmonitored) run, 0.4142 hard LOGO
START = {"arch": "gru", "hidden": 32, "layers": 1, "dropout": 0.1, "lr": 3e-3, "epochs": 12,
         "window": 128, "alpha": 0.1, "stay": 1.0}


def propose(best, rng):
    if best is None:
        return dict(START)
    cfg = dict(best)
    for k in rng.sample(list(SPACE), k=rng.choice([1, 1, 2, 3])):
        cfg[k] = rng.choice(SPACE[k])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tsv = os.path.join(OUT, "results.tsv")
    with open(tsv, "w") as fh:
        fh.write("run\tspellF1_hard\tmacroF1_hard\tposs\tseconds\tstatus\tconfig\n")
    global TB
    from torch.utils.tensorboard import SummaryWriter
    TB = SummaryWriter(os.path.join(OUT, "tb", time.strftime("%Y%m%d-%H%M%S")))
    hard = bench.bench_games("hard")
    rng = random.Random(2027)
    best, best_score, seen, t_end, run_id = None, -1.0, set(), time.time() + args.minutes * 60, 0
    MON["t_end"] = t_end
    tb_readme()
    while time.time() < t_end:
        cfg = propose(best, rng)
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key); run_id += 1
        MON["run"] = run_id; MON["config"] = cfg
        t = time.time()
        try:
            rows = bench.score(logo(cfg, hard), hard)
            sp = float(np.mean([r["spellF1"] for r in rows])); mf = float(np.mean([r["macroF1"] for r in rows]))
            poss = int(sum(r["poss_pred"] for r in rows))
            status = "keep" if sp > best_score else "discard"
            if status == "keep":
                best, best_score = cfg, sp
        except Exception as exc:  # noqa: BLE001
            sp = mf = 0.0; poss = 0; status = f"crash:{type(exc).__name__}"
        with open(tsv, "a") as fh:
            fh.write(f"{run_id}\t{sp:.4f}\t{mf:.4f}\t{poss}\t{time.time()-t:.0f}\t{status}\t{key}\n")
        print(f"[{run_id}] {sp:.4f} {status} {key}", flush=True)
        MON["best"] = best_score; MON["last"] = sp
        tb_scalar("search/spellF1_hard_LOGO", sp, run_id)
        tb_scalar("search/best_spellF1", best_score, run_id)
        tb_scalar("search/hmm_reference", 0.4337, run_id)
        tb_scalar("search/possessions_pred (true 1022)", poss, run_id)
        TB.flush(); _progress()

    # -- final checks for the best config ---------------------------------------------------------
    summary = {"best_config": best, "best_hard_logo": best_score, "runs": run_id}
    hmm_cmp = M.hmm_logo(alpha=0.3)
    for level in ("hard", "easy"):
        G = bench.bench_games(level)
        hmm_cmp.games = G
        m = logo(best, G)
        rows = bench.score(m, G)
        summary[f"{level}_spellF1"] = float(np.mean([r["spellF1"] for r in rows]))
        summary[f"{level}_vs_hmm"] = bench.compare(m, hmm_cmp, G)
    B = bench.barath()
    from baseline import write_submission
    lab = nn_method(best, hard)(B)
    write_submission(os.path.join(ROOT, "evaluation", "results", "nn_e9.csv"), lab)
    g = B["play"]
    summary["barath_possessions"] = int(sum(1 for i in np.flatnonzero(g)[1:]
                                            if lab[i] in "HA" and lab[i] != lab[i - 1]))
    summary["barath_D_share"] = float(np.mean(lab[g] == "D"))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
