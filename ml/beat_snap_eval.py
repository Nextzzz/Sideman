"""Oracle experiment: how much of the error is chord-boundary TIMING?

The taxonomy says ~12% of all frames are wrong within ±0.19 s of a
chord change. Here predicted segment boundaries are snapped to the
song's annotated beat grid (HookTheory refined alignment = an oracle
for what a good beat tracker would give) and the score is recomputed.
The gap between 'viterbi' and 'snap' is the ceiling for beat-sync
decoding — the product already snaps to DSP beats, so this also tells
whether improving the beat tracker is worth it.

Run:
    .venv/Scripts/python beat_snap_eval.py [rows=250]
"""
import csv
import gzip
import json
import os
import sys

import numpy as np
import onnxruntime as ort

from btc_features import features_from_wav
from eval_guitarset import to_majmin
from hooktheory_eval import ROOT, load_audio, read_lab, STEP
from infer_tricks_eval import probs_pass, combine, T
from viterbi_eval import build_log_transitions, viterbi

HERE = os.path.dirname(os.path.abspath(__file__))


def beat_times_by_id():
    with gzip.open(os.path.join(ROOT, "Hooktheory.json.gz"), "rt", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v["alignment"]["refined"]["times"] for k, v in data.items()
            if "REFINED_ALIGNMENT" in v.get("tags", [])}


def snap_to_beats(frame_labels, spf, beats, max_shift_s):
    """Move each label-change boundary to the nearest beat (if within
    max_shift_s), then re-rasterize. Beats are extended with their mean
    period so the whole song is covered, not only the annotated span."""
    beats = np.array(beats, dtype=float)
    period = np.median(np.diff(beats)) if len(beats) > 1 else 0.5
    n = len(frame_labels)
    grid = np.concatenate([
        np.arange(beats[0], 0, -period)[::-1][:-1],
        beats,
        np.arange(beats[-1], n * spf + period, period)[1:],
    ])
    # segment boundaries in seconds
    boundaries = [i * spf for i in range(1, n) if frame_labels[i] != frame_labels[i - 1]]
    snapped = []
    for b in boundaries:
        j = np.searchsorted(grid, b)
        cands = [grid[k] for k in (j - 1, j) if 0 <= k < len(grid)]
        best = min(cands, key=lambda g: abs(g - b))
        snapped.append(best if abs(best - b) <= max_shift_s else b)
    # rebuild labels: each original segment keeps its label, boundaries move
    segments = []
    start = 0
    for i in range(1, n + 1):
        if i == n or frame_labels[i] != frame_labels[start]:
            segments.append(frame_labels[start])
            start = i
    edges = [0.0] + sorted(snapped) + [n * spf]
    out = list(frame_labels)
    for k, label in enumerate(segments):
        a = int(round(edges[k] / spf))
        b = int(round(edges[k + 1] / spf))
        for i in range(max(0, a), min(n, b)):
            out[i] = label
    return out


def score(pred_full, truth_full, truth_mm, spf):
    full_s = full_c = mm_s = mm_c = 0
    t = truth_full[0][0]
    while t < truth_full[-1][1]:
        idx = min(int(t / spf), len(pred_full) - 1)
        full = next((l for s, e, l in truth_full if s <= t < e), None)
        if full is not None and full != "X":
            full_s += 1
            full_c += pred_full[idx] == full
        mm = next((l for s, e, l in truth_mm if s <= t < e), None)
        if mm is not None and mm != "X":
            mm_s += 1
            mm_c += (to_majmin(pred_full[idx]) or "X") == mm
        t += STEP
    return full_s, full_c, mm_s, mm_c


def main(rows_limit):
    session = ort.InferenceSession(os.path.join(HERE, "models", "btc_large_voca.onnx"))
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        labels = json.load(f)["labels"]
    log_trans = build_log_transitions(labels, None, 0.9)
    beats = beat_times_by_id()

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:rows_limit]

    variants = ["viterbi", "snap<=0.15s", "snap<=0.30s", "snap<=half-beat"]
    acc = {v: [0, 0, 0, 0] for v in variants}
    done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        full_path = os.path.join(ROOT, "labs_full", row["id"] + ".lab")
        mm_path = os.path.join(ROOT, "labs", row["id"] + ".lab")
        if not (os.path.exists(audio_path) and os.path.exists(full_path)
                and row["id"] in beats):
            continue
        truth_full, truth_mm = read_lab(full_path), read_lab(mm_path)
        wav = load_audio(audio_path)
        if len(wav) < 22050:
            continue
        features, spf = features_from_wav(wav)
        probs = combine([probs_pass(session, features, 0, 0),
                         probs_pass(session, features, T // 2, 0)])
        pred = [labels[i] for i in viterbi(np.log(probs + 1e-12), log_trans)]
        song_beats = beats[row["id"]]
        half_beat = float(np.median(np.diff(song_beats))) / 2 if len(song_beats) > 1 else 0.25

        preds = {
            "viterbi": pred,
            "snap<=0.15s": snap_to_beats(pred, spf, song_beats, 0.15),
            "snap<=0.30s": snap_to_beats(pred, spf, song_beats, 0.30),
            "snap<=half-beat": snap_to_beats(pred, spf, song_beats, half_beat),
        }
        for v in variants:
            for k, val in enumerate(score(preds[v], truth_full, truth_mm, spf)):
                acc[v][k] += val
        done += 1
        if done % 25 == 0:
            print(f"{done} songs...", flush=True)

    print(f"\n=== oracle beat snapping, {done} songs ===")
    for v in variants:
        fs, fc, ms, mc = acc[v]
        print(f"{v:16} full-vocab {fc / max(fs, 1):.2%}   majmin {mc / max(ms, 1):.2%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
