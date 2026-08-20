"""Where does the product pipeline actually go wrong? Frame-level error
taxonomy on the protected benchmark rows (base+overlap, uniform Viterbi
— the shipped file route minus the ensemble/key prior).

Each scored frame (truth_full != X) lands in exactly one bucket:
    correct      exact label
    flavor       same root + mode, wrong extension (Am vs Am7)      <- repeat voting can fix
    mode         same root, maj<->min swapped                        <- key prior territory
    root         different root (sub-split by interval)              <- acoustic model territory
    nochord      predicted N/X on a chord                            <- gate / training-data issue
and is tagged boundary (within ±2 frames of a truth chord change) or
interior — boundary errors are timing, interior errors are hearing.

Run:
    .venv/Scripts/python error_analysis.py [rows=250]
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
import onnxruntime as ort

from btc_features import features_from_wav
from hooktheory_eval import ROOT, load_audio, read_lab, STEP
from infer_tricks_eval import probs_pass, combine, T
from viterbi_eval import build_log_transitions, viterbi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ_Q = {"", "maj6", "maj7", "7"}
MIN_Q = {"min", "min6", "min7", "minmaj7"}
BOUNDARY_FRAMES = 2


def split(label):
    root, _, q = label.partition(":")
    mode = "maj" if q in MAJ_Q else "min" if q in MIN_Q else q
    return ROOTS.index(root), mode, q


def classify(truth, pred):
    if pred == truth:
        return "correct"
    if pred in ("N", "X"):
        return "nochord"
    tr, tm, tq = split(truth)
    pr, pm, pq = split(pred)
    if tr == pr and tm == pm:
        return "flavor"
    if tr == pr:
        return "mode"
    interval = (pr - tr) % 12
    if tm == "maj" and pm == "min" and interval == 9:
        return "root:relative-minor"
    if tm == "min" and pm == "maj" and interval == 3:
        return "root:relative-major"
    if interval in (5, 7):
        return "root:fifth"
    if interval in (1, 11):
        return "root:semitone"
    if interval in (3, 4, 8, 9):
        return "root:third"
    return "root:other"


def main(rows_limit):
    session = ort.InferenceSession(os.path.join(HERE, "models", "btc_large_voca.onnx"))
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        labels = json.load(f)["labels"]
    log_trans = build_log_transitions(labels, None, 0.9)

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:rows_limit]

    buckets = Counter()
    position = Counter()           # (bucket, boundary/interior)
    by_quality = Counter()         # (truth quality, correct?)
    quality_pred = Counter()       # (truth quality -> pred quality) when same root
    confusions = Counter()         # (truth, pred)
    per_song = []
    done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        full_path = os.path.join(ROOT, "labs_full", row["id"] + ".lab")
        if not os.path.exists(audio_path) or not os.path.exists(full_path):
            continue
        truth = read_lab(full_path)
        wav = load_audio(audio_path)
        if len(wav) < 22050:
            continue
        features, spf = features_from_wav(wav)
        probs = combine([probs_pass(session, features, 0, 0),
                         probs_pass(session, features, T // 2, 0)])
        pred = [labels[i] for i in viterbi(np.log(probs + 1e-12), log_trans)]

        changes = [s for s, _, _ in truth[1:]]
        song_scored = song_ok = 0
        t = truth[0][0]
        while t < truth[-1][1]:
            label = next((l for s, e, l in truth if s <= t < e), None)
            if label is not None and label != "X":
                idx = min(int(t / spf), len(pred) - 1)
                bucket = classify(label, pred[idx])
                near = any(abs(t - c) <= BOUNDARY_FRAMES * spf for c in changes)
                buckets[bucket] += 1
                position[(bucket.split(":")[0], "boundary" if near else "interior")] += 1
                tq = split(label)[2] or "maj"
                by_quality[(tq, bucket == "correct")] += 1
                if bucket in ("flavor", "mode", "correct"):
                    quality_pred[(tq, split(pred[idx])[2] or "maj")] += 1
                if bucket != "correct":
                    confusions[(label, pred[idx])] += 1
                song_scored += 1
                song_ok += bucket == "correct"
            t += STEP
        per_song.append((song_ok / max(song_scored, 1), f"{row['artist']}/{row['song']}"))
        done += 1
        if done % 25 == 0:
            print(f"{done} songs...", flush=True)

    total = sum(buckets.values())
    print(f"\n=== error taxonomy, {done} songs, {total} frames (full-vocab) ===")
    for bucket, n in buckets.most_common():
        print(f"  {bucket:22} {n / total:6.2%}")

    print("\n=== error buckets: boundary (±0.19 s of a chord change) vs interior ===")
    for bucket in ("flavor", "mode", "root", "nochord"):
        b = position[(bucket, "boundary")]
        i = position[(bucket, "interior")]
        print(f"  {bucket:8} boundary {b / total:6.2%}   interior {i / total:6.2%}")

    print("\n=== accuracy by truth quality (share of frames | exact-match rate) ===")
    qual_total = Counter()
    for (q, ok), n in by_quality.items():
        qual_total[q] += n
    for q, n in qual_total.most_common():
        ok = by_quality[(q, True)]
        print(f"  {q or 'maj':8} {n / total:6.2%} | {ok / n:6.1%}")

    print("\n=== same-root quality confusions (truth -> pred), top 12 ===")
    for (tq, pq), n in quality_pred.most_common(40):
        if tq != pq:
            print(f"  {tq or 'maj':8} -> {pq or 'maj':8} {n / total:6.2%}")
    print("\n=== top label confusions ===")
    for (tr, pr), n in confusions.most_common(12):
        print(f"  {tr:10} -> {pr:10} {n / total:6.2%}")
    print("\n=== worst 10 songs ===")
    for acc, name in sorted(per_song)[:10]:
        print(f"  {acc:5.1%}  {name}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
