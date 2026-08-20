"""Song-level mode consistency: one root, one mode.

Observation (1EozE3URh-8): the model labels the same chord D in some
bars and Dm in others — not within one sustained chord (a direct D<->Dm
penalty changed nothing) but across its repeats. Musically a root keeps
its mode within a song far more often than not, so: for every root that
appears as both major and minor, sum the posterior evidence for each
mode over all frames decoded as either, and relabel the minority mode to
the majority (7ths follow: 7->min7, maj7->minmaj7, maj6->min6 and back).
Roots are never touched. Measured on the protected benchmark rows
(base+overlap, uniform Viterbi), majmin + full-vocab; --song <audio>
shows before/after for one recording.

Run:
    .venv/Scripts/python mode_consistency_eval.py [rows=250]
    .venv/Scripts/python mode_consistency_eval.py --song path.m4a
"""
import csv
import json
import os
import sys

import numpy as np
import onnxruntime as ort

from btc_features import features_from_wav
from eval_guitarset import to_majmin
from hooktheory_eval import ROOT, load_audio, read_lab, STEP
from infer_tricks_eval import probs_pass, combine, T
from viterbi_eval import build_log_transitions, viterbi, label_parts

HERE = os.path.dirname(os.path.abspath(__file__))
MAJ_TO_MIN = {"maj": "min", "7": "min7", "maj7": "minmaj7", "maj6": "min6"}
MIN_TO_MAJ = {v: k for k, v in MAJ_TO_MIN.items()}
MIN_RATIO = 1.5   # winner needs this much more evidence, else leave alone


def label_of(root, quality):
    return root if quality == "maj" else f"{root}:{quality}"


def unify_modes(path, probs, labels, index):
    """Return relabeled path; roots untouched."""
    parts = [label_parts(l) for l in labels]
    out = list(path)
    roots = {p[0] for p in parts if isinstance(p, tuple)}
    for root in roots:
        maj_states = [i for i, p in enumerate(parts) if isinstance(p, tuple) and p[0] == root and p[1] in MAJ_TO_MIN]
        min_states = [i for i, p in enumerate(parts) if isinstance(p, tuple) and p[0] == root and p[1] in MIN_TO_MAJ]
        frames = [t for t, s in enumerate(path) if s in maj_states or s in min_states]
        if not frames:
            continue
        maj_ev = probs[frames][:, maj_states].sum()
        min_ev = probs[frames][:, min_states].sum()
        if maj_ev >= MIN_RATIO * min_ev:
            target, table = "maj", MIN_TO_MAJ
        elif min_ev >= MIN_RATIO * maj_ev:
            target, table = "min", MAJ_TO_MIN
        else:
            continue
        for t in frames:
            quality = parts[path[t]][1]
            if quality in table:
                out[t] = index[label_of(root, table[quality])]
    return out


def main(rows_limit, song=None):
    session = ort.InferenceSession(os.path.join(HERE, "models", "btc_large_voca.onnx"))
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        labels = json.load(f)["labels"]
    index = {l: i for i, l in enumerate(labels)}
    log_trans = build_log_transitions(labels, None, 0.9)

    def decode(features):
        probs = combine([probs_pass(session, features, 0, 0),
                         probs_pass(session, features, T // 2, 0)])
        path = viterbi(np.log(probs + 1e-12), log_trans)
        return probs, path, unify_modes(path, probs, labels, index)

    if song:
        features, spf = features_from_wav(load_audio(song))
        probs, before, after = decode(features)
        for tag, path in (("before", before), ("after", after)):
            secs = lambda l: (np.array(path) == index[l]).sum() * spf
            print(f"{tag:6} D {secs('D'):4.0f}s  Dm {secs('D:min'):4.0f}s  A {secs('A'):4.0f}s  Am {secs('A:min'):4.0f}s")
        return

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:rows_limit]
    tally = {"viterbi": [0, 0, 0, 0], "mode-unified": [0, 0, 0, 0]}
    changed = done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        mm_path = os.path.join(ROOT, "labs", row["id"] + ".lab")
        full_path = os.path.join(ROOT, "labs_full", row["id"] + ".lab")
        if not (os.path.exists(audio_path) and os.path.exists(mm_path)):
            continue
        truth_mm = read_lab(mm_path)
        truth_full = read_lab(full_path) if os.path.exists(full_path) else []
        wav = load_audio(audio_path)
        if len(wav) < 22050:
            continue
        features, spf = features_from_wav(wav)
        probs, before, after = decode(features)
        preds = {"viterbi": [labels[i] for i in before], "mode-unified": [labels[i] for i in after]}
        changed += sum(a != b for a, b in zip(before, after))

        t = truth_mm[0][0]
        while t < truth_mm[-1][1]:
            mm = next((l for s, e, l in truth_mm if s <= t < e), None)
            full = next((l for s, e, l in truth_full if s <= t < e), None)
            for d, pred in preds.items():
                idx = min(int(t / spf), len(pred) - 1)
                if mm is not None and mm != "X":
                    tally[d][0] += 1
                    tally[d][1] += (to_majmin(pred[idx]) or "X") == mm
                if full is not None and full != "X":
                    tally[d][2] += 1
                    tally[d][3] += pred[idx] == full
            t += STEP
        done += 1
        if done % 25 == 0:
            print(f"{done} songs: " + "  ".join(
                f"{d} {v[1] / max(v[0], 1):.2%}" for d, v in tally.items()), flush=True)

    print(f"\n=== song-level mode consistency (base+ovl), {done} songs, {changed} frames relabeled ===")
    for d, (ms, mc, fs, fc) in tally.items():
        print(f"{d:13} majmin {mc / max(ms, 1):.2%}   full-vocab {fc / max(fs, 1):.2%}")


if __name__ == "__main__":
    if "--song" in sys.argv:
        main(0, song=sys.argv[sys.argv.index("--song") + 1])
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 250)
