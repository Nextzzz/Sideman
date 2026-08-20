"""Inference-time accuracy tricks, measured on the protected benchmark rows.

No training involved — only how we RUN the models we already have:
  ovl  overlapping windows (second pass at half-window offset, averaged)
  tta  pitch-shift test-time augmentation (±1 semitone = ±2 CQT bins,
       predictions rolled back to the original root, averaged)
  ens  base + guitar2 probability average
Reported as majmin WCSR, same scoring as hooktheory_eval.py.

Run:
    .venv/Scripts/python infer_tricks_eval.py [rows=250]
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

HERE = os.path.dirname(os.path.abspath(__file__))
T = 108
LOG_FLOOR = float(np.log(1e-6))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def shift_features(features, k):
    if k == 0:
        return features
    out = np.full_like(features, LOG_FLOOR)
    if k > 0:
        out[:, 2 * k:] = features[:, :-2 * k]
    else:
        out[:, :2 * k] = features[:, -2 * k:]
    return out


def unshift_probs(probs, k):
    """Model heard audio k semitones up -> its root r is true root r-k."""
    if k == 0:
        return probs
    chords = probs[:, :168].reshape(-1, 12, 14)
    chords = np.roll(chords, -k, axis=1).reshape(-1, 168)
    return np.concatenate([chords, probs[:, 168:]], axis=1)


def probs_pass(session, features, offset, k):
    """One full pass: windows starting at `offset`, audio shifted by k."""
    feats = shift_features(features, k)
    n = len(feats)
    acc = np.zeros((n, 170), dtype=np.float64)
    cover = np.zeros(n, dtype=np.float64)
    start = offset
    while start < n:
        window = feats[start:start + T]
        valid = len(window)
        if valid < T:
            window = np.pad(window, ((0, T - valid), (0, 0)))
        logits = session.run(None, {"features": window[None].astype(np.float32)})[0][0]
        acc[start:start + valid] += softmax(logits[:valid])
        cover[start:start + valid] += 1
        start += T
    acc[cover > 0] /= cover[cover > 0, None]
    return unshift_probs(acc, k), cover > 0


def combine(passes):
    """Average the passes frame-wise over the frames each pass covers."""
    total = np.zeros_like(passes[0][0])
    count = np.zeros(len(total))
    for probs, covered in passes:
        total[covered] += probs[covered]
        count[covered] += 1
    total[count > 0] /= count[count > 0, None]
    return total


def main(rows_limit):
    sessions = {}
    for name in ("btc_large_voca", "btc_guitar2"):
        sessions[name] = ort.InferenceSession(os.path.join(HERE, "models", name + ".onnx"))
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        labels = json.load(f)["labels"]

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:rows_limit]

    variants = ["base", "base+ovl", "base+tta", "base+ovl+tta",
                "guitar2", "ens", "ens+ovl+tta"]
    scored = {v: 0 for v in variants}
    correct = {v: 0 for v in variants}
    done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        lab_path = os.path.join(ROOT, "labs", row["id"] + ".lab")
        if not os.path.exists(audio_path) or not os.path.exists(lab_path):
            continue
        truth = read_lab(lab_path)
        wav = load_audio(audio_path)
        if len(wav) < 22050:
            continue
        features, spf = features_from_wav(wav)

        p = {}
        for name, short in (("btc_large_voca", "b"), ("btc_guitar2", "g")):
            s = sessions[name]
            p[short + "0"] = probs_pass(s, features, 0, 0)
            p[short + "54"] = probs_pass(s, features, T // 2, 0)
            p[short + "+1"] = probs_pass(s, features, 0, 1)
            p[short + "-1"] = probs_pass(s, features, 0, -1)

        preds = {
            "base": combine([p["b0"]]),
            "base+ovl": combine([p["b0"], p["b54"]]),
            "base+tta": combine([p["b0"], p["b+1"], p["b-1"]]),
            "base+ovl+tta": combine([p["b0"], p["b54"], p["b+1"], p["b-1"]]),
            "guitar2": combine([p["g0"]]),
            "ens": combine([p["b0"], p["g0"]]),
            "ens+ovl+tta": combine(list(p.values())),
        }
        pred_labels = {v: [to_majmin(labels[i]) or "X" for i in pr.argmax(axis=1)]
                       for v, pr in preds.items()}

        t = truth[0][0]
        while t < truth[-1][1]:
            label = next((l for s, e, l in truth if s <= t < e), None)
            if label is not None and label != "X":
                idx = min(int(t / spf), len(features) - 1)
                for v in variants:
                    scored[v] += 1
                    if pred_labels[v][idx] == label:
                        correct[v] += 1
            t += STEP
        done += 1
        if done % 20 == 0:
            print(f"{done} songs: " + "  ".join(
                f"{v} {correct[v] / max(scored[v], 1):.1%}" for v in variants),
                flush=True)

    print(f"\n=== inference tricks, {done} benchmark songs ===")
    for v in variants:
        print(f"{v:14} WCSR (majmin) {correct[v] / max(scored[v], 1):.2%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
