"""Regenerate the final BAR-ATH submission from raw data with one command.

    python experiments/final/reproduce_final.py                       # -> evaluation/results/final.csv
    python experiments/final/reproduce_final.py --check evaluation/results/nn_best.csv
    python experiments/final/reproduce_final.py --rebuild-cache       # re-parse raw ATD + Metrica first

What it does (model fully defined by experiments/final/config.json):
  1. checks the raw inputs exist (BAR-ATH tactical ATD, Metrica sample data clone);
  2. builds the degraded Metrica training bench (fixed degradation seed 2027 + game);
  3. trains one bi-GRU per seed on all 3 bench games (CPU, seeded), averages per-step log-probs;
     config `train_loss` {bw, r, cp} = A3 loss (D18): steps within r of a label change weighted bw, plus a
     change-point head with weight cp; config `train_extra_draws` = extra degradation seeds of the same bench
     level added to training (A4); decode `beta` > 0 = change-point head scales the leave-state cost per step;
  4. Viterbi-decodes each graded half of BAR-ATH with transitions fitted on the bench;
  5. (config `pre_kickoff_dead`) forces D on graded frames before each half's kick-off — published rule:
     2nd half graded from 110,808, kick-off 113,742, those 2,934 frames are all D (FAQ.md / DATA.md, D17);
  6. writes frame,label CSV, runs the official validator, writes a manifest (config, versions, sha256);
  7. with --check FILE: reports how many frames differ from FILE (0 = exact reproduction).

No BAR-ATH labels are used anywhere; the public 5-minute ground truth is not read.
Training data: Metrica Sports sample data (https://github.com/metrica-sports/sample-data).
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path[:0] = [ROOT, EXP, os.path.join(ROOT, "starter")]

import torch  # noqa: E402

import bench  # noqa: E402
import e9_autoresearch as E  # noqa: E402
import e13_autoresearch as X  # noqa: E402
import e18_boundary as A  # noqa: E402  (A3 loss: boundary weighting + change-point head)
E.frame_inputs = A._full   # e18 swaps in no-shape inputs on import; undo it, set_inputs() decides from the config
from games import DATA_DIR  # noqa: E402

SHAPE_COLS = slice(9, 15)


def train_model(cfg, games, seed):
    """Plain cross-entropy GRU (E9/D13), or the A3 loss when the config has `train_loss` {bw, r, cp} (D18)."""
    loss = cfg.get("train_loss")
    if not loss:
        return E.train(cfg["train_config"], games, seed)
    A.TRAIN = cfg["train_config"]
    return A.train(loss, games, seed)


def check_inputs():
    need = {
        "BAR-ATH tactical ATD": os.path.join(DATA_DIR, "tactical"),
        "Metrica sample data": os.path.join(DATA_DIR, "external", "metrica_sample", "data"),
    }
    missing = [f"{k}: {p}" for k, p in need.items() if not os.path.isdir(p)]
    if missing:
        sys.exit("Missing inputs:\n  " + "\n  ".join(missing))


def set_inputs(kind):
    full = E.frame_inputs.__wrapped__ if hasattr(E.frame_inputs, "__wrapped__") else E.frame_inputs
    if kind == "full":
        return
    if kind == "no_shape":
        def no_shape(d, swap=False):
            return np.delete(full(d, swap), np.r_[SHAPE_COLS], axis=1)
        no_shape.__wrapped__ = full
        E.frame_inputs = no_shape
        return
    sys.exit(f"unknown inputs '{kind}'")


def force_pre_kickoff_dead(labels):
    """D on graded frames between a half's first graded frame and its kick-off. Returns frames changed."""
    from games import half_video_blocks
    from load_smart_tagging import KICKOFF_VIDEO_FRAMES
    names = {1: "1st Half", 2: "2nd Half"}
    changed = 0
    for half, first, _last in half_video_blocks("BAR-ATH"):
        ko = KICKOFF_VIDEO_FRAMES[names[half]]
        if ko > first:
            changed += int((labels[first:ko] != "D").sum())
            labels[first:ko] = "D"
    return changed


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "final.csv"))
    ap.add_argument("--check", help="existing submission to compare against frame by frame")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--save-weights", metavar="DIR", help="also save each seed's trained network to DIR/seed_<s>.pt")
    ap.add_argument("--weights", metavar="DIR",
                    help="skip training: load the 8 trained networks from DIR (e.g. experiments/final/weights)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    check_inputs()
    E.OUT = HERE                                  # progress.json stays out of the research folders
    E.TB = None
    set_inputs(cfg["inputs"])
    t0 = time.time()

    if args.rebuild_cache:
        from atd_cache import load
        from external_metrica import load_game, GAMES
        load(rebuild=True)
        for g in GAMES:
            load_game(g, rebuild=True)

    train = list(bench.bench_games(cfg["train_bench"]))
    for seed0 in ([] if args.weights else cfg.get("train_extra_draws", [])):   # A4 (e19 draws3): extra degradations
        import e19_data_mix as D
        train += D.draws(cfg["train_bench"], seed0)
    B = bench.barath()
    Xb = torch.tensor(E.steps(E.frame_inputs(B)))[None]
    lps, cps = [], []
    for s in cfg["seeds"]:
        if args.weights:                      # pre-trained networks (trained on Metrica only) instead of training
            tc, loss = cfg["train_config"], cfg.get("train_loss") or {}
            if not loss:
                raise SystemExit("--weights supports configs with train_loss (A3 network)")
            model = A.Net(Xb.shape[2], tc["hidden"], tc["layers"], tc["dropout"], loss["cp"])
            model.load_state_dict(torch.load(os.path.join(args.weights, f"seed_{s}.pt"), weights_only=True))
            model.eval()
        else:
            model = train_model(cfg, train, s)
            if args.save_weights:
                os.makedirs(args.save_weights, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_weights, f"seed_{s}.pt"))
        with torch.no_grad():
            out = model(Xb)
            logits = out[0] if isinstance(out, tuple) else out   # A3 net also returns the change-point logit
            lps.append(torch.log_softmax(logits, -1)[0].numpy())
            if isinstance(out, tuple) and out[1] is not None:
                cps.append(torch.sigmoid(out[1])[0].numpy())
        print(f"seed {s} {'loaded' if args.weights else 'trained'} ({time.time() - t0:.0f} s)", flush=True)
    lp = np.mean(lps, axis=0)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    dec = cfg["decode"]
    log_T = X.transitions(bench.bench_games(cfg["train_bench"]), dec["stay"], dec["ha"])
    if dec.get("beta", 0) > 0:      # change-point head scales the cost of leaving a state per step (e18 Decoder)
        n = len(B["ball"])
        cp = np.mean(cps, axis=0)
        le = dec["alpha"] * A.frames(lp, n)
        sc = A.frames((cp / cp.mean()) ** dec["beta"], n)
        labels = np.full(n, "X", dtype="<U1")
        names = np.array(["H", "A", "D"])
        for a, b in X.M._blocks(B["play"]):
            labels[a:b] = names[A.viterbi_tv(le[a:b], np.exp(log_T), sc[a:b])]
    else:
        labels = X.decode(lp, B, log_T, dec["alpha"])
    n_pre_ko = force_pre_kickoff_dead(labels) if cfg.get("pre_kickoff_dead") else 0

    from baseline import write_submission
    write_submission(args.out, labels)
    v = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), args.out],
                       capture_output=True, text=True)
    valid = v.returncode == 0 and "Valid." in v.stdout

    manifest = {"output": os.path.relpath(args.out, ROOT), "sha256": sha256(args.out), "valid": valid,
                "config": cfg, "seconds": round(time.time() - t0), "weights": args.weights or "trained",
                "python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "torch": torch.__version__,
                "label_counts": {k: int((labels == k).sum()) for k in "HADX"},
                "pre_kickoff_frames_set_to_D": n_pre_ko}
    if args.check:
        import pandas as pd
        ref = pd.read_csv(args.check)["label"].astype(str).str.strip().str.upper().to_numpy()
        manifest["check_against"] = os.path.relpath(os.path.abspath(args.check), ROOT)
        manifest["frames_differing"] = int((ref != labels).sum()) if len(ref) == len(labels) else "length mismatch"
    json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))
    if not valid:
        print(v.stdout, v.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
