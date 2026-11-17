"""Glue: tracking and tags in, one label per video frame out."""

import os

import numpy as np

from . import dead_ball, engine, load, settings, signals


def label_frames(sig, half_blocks, set_pieces, n_frames, fps=settings.FPS, phase_hints=None):
    """One label per video frame: H, A, D inside the halves, X everywhere else.

    half_blocks is [(half, first_frame, last_frame)], e.g. games.half_video_blocks(GAME).
    phase_hints is load.phases_from(tags), or None to ignore the phase tags.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    parts = {}
    struck = dead_ball.ball_struck(sig, fps)
    evidence = dead_ball.dead_evidence(sig, fps, parts, struck)
    starts = dead_ball.restart_starts(sig, struck, fps)
    resume_from = dead_ball.play_resumed(sig, fps)
    phase = engine.phase_masks(phase_hints, n_frames) if phase_hints else None
    report = {"dead_stretches": 0, "live_stretches": 0, "live_without_team": 0,
              "switches_from_tracking": 0, "switches_from_tags": 0}

    report["dead_reasons"] = []
    report["kickoff_candidates"] = {
        half: dead_ball.kickoff_candidates(sig, first, last, struck, fps)
        for half, first, last in half_blocks}
    report["lineup"] = {half: dead_ball.lineup_frame(sig, first, last, fps)
                        for half, first, last in half_blocks}
    for _half, first, last in half_blocks:
        for stretch in dead_ball.split_half(sig, first, last, set_pieces, fps, evidence, struck,
                                            resume_from, starts, report["dead_reasons"]):
            if stretch[0] == "dead":
                labels[stretch[1]:stretch[2] + 1] = "D"
                report["dead_stretches"] += 1
                continue
            _, start, end, team = stretch
            report["live_stretches"] += 1
            switches = engine.possession_in_stretch(sig, start, end, team, fps, phase, report)
            if not switches:
                report["live_without_team"] += 1
            engine.fill_labels(labels, start, end, switches)
    report["parts"] = parts
    return labels, report


def summary(labels, fps=settings.FPS):
    """Counts that describe the OUTPUT (safe to share): no match data in here."""
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    spells = []
    run_label, run_length = None, 0
    for label in labels.tolist() + ["X"]:
        if label == run_label:
            run_length += 1
            continue
        if run_label in ("H", "A"):
            spells.append(run_length)
        run_label, run_length = label, 1
    return {
        "counts": counts,
        "possessions": len(spells),
        "mean_possession_seconds": (sum(spells) / float(len(spells)) / fps) if spells else 0.0,
    }


def build(game, data_dir=None):
    """Run the whole model on the provided match data. Needs games.py on the import path."""
    from games import GAMES, half_video_blocks, require, source_frames, asset_paths

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    fps = GAMES[game]["fps"]
    pitch = GAMES[game]["pitch"]

    slot_team = load.team_of_each_slot(meta_path)
    print("reading tracking ({} track slots) ... this takes a minute or two".format(
        len(slot_team)))
    ball, players, speeds = load.read_tracking(raw_path, n_frames, len(slot_team))

    tag_path = asset_paths(game, data_dir)["smart_tagging"]
    set_pieces, phase_hints = [], None
    if os.path.isfile(tag_path):
        tags = load.read_tags(tag_path, fps)
        set_pieces = load.set_pieces_from(tags)
        if settings.USE_PHASE_TAGS:
            phase_hints = load.phases_from(tags)
        print("tags read: {} set pieces, {} phase hints".format(
            len(set_pieces), len(phase_hints or [])))
    else:
        print("no Smart Tagging file found; using the tracking only")

    print("working out who is on the ball ...")
    sig = signals.compute(ball, players, speeds, slot_team, pitch, fps)
    del ball, players, speeds

    print("applying the possession rules ...")
    return label_frames(sig, half_video_blocks(game), set_pieces, n_frames, fps, phase_hints)
