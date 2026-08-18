"""GuitarSet evaluation of the exported BTC ONNX models — the same
frame-level majmin WCSR protocol as the C# harness (Sideman.Cli eval),
so numbers are directly comparable with the template engine.

Run:
    .venv/Scripts/python eval_guitarset.py [modelname] [limit]
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

from btc_features import features_for, TIMESTEP

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.normpath(os.path.join(HERE, "..", "datasets", "guitarset"))
STEP = 0.1

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def to_majmin(raw):
    """Dataset/model label -> 'C', 'Cm', 'N' or None (outside majmin)."""
    if raw in ("N", "X"):
        return "N"
    root, _, quality = raw.partition(":")
    quality = quality.split("/")[0].split("(")[0]
    pc = NOTE_TO_PC.get(root[0])
    if pc is None:
        return None
    for c in root[1:]:
        pc += 1 if c == "#" else -1 if c == "b" else 0
    pc %= 12
    if quality.startswith("min"):
        return NAMES[pc] + "m"
    if quality == "" or quality.startswith("maj") or quality in ("7", "9", "11", "13", "6"):
        return NAMES[pc]
    return None


def truth_segments(jams_path):
    with open(jams_path) as f:
        jams = json.load(f)
    chosen = None
    for annotation in jams["annotations"]:
        if annotation["namespace"] != "chord":
            continue
        if not annotation["annotation_metadata"]["data_source"]:
            chosen = annotation
            break
        chosen = chosen or annotation
    duration = jams["file_metadata"]["duration"]
    segments = [(d["time"], d["time"] + d["duration"], d["value"])
                for d in chosen["data"]]
    return segments, duration


def predict_labels(session, labels, features):
    """Per-frame labels over the whole file (pad tail like BTC's test.py)."""
    frames = features.shape[0]
    pad = (-frames) % TIMESTEP
    padded = np.pad(features, ((0, pad), (0, 0)))
    out = []
    for t in range(padded.shape[0] // TIMESTEP):
        window = padded[t * TIMESTEP:(t + 1) * TIMESTEP][None, ...]
        logits = session.run(None, {"features": window})[0][0]
        out.extend(labels[int(i)] for i in logits.argmax(axis=1))
    return out[:frames]


def main(model_name="btc_large_voca", limit=10 ** 9):
    session = ort.InferenceSession(os.path.join(HERE, "models", model_name + ".onnx"))
    with open(os.path.join(HERE, "models", model_name + ".json")) as f:
        labels = json.load(f)["labels"]

    jams_dir = os.path.join(DATASET, "annotation")
    audio_dir = os.path.join(DATASET, "audio_mono-mic")
    files = sorted(f for f in os.listdir(jams_dir) if "_comp" in f)[:limit]

    scored = correct = 0
    per_file = []
    confusions = {}
    for jams_name in files:
        stem = jams_name[:-5]
        wav = os.path.join(audio_dir, stem + "_mic.wav")
        if not os.path.exists(wav):
            continue
        segments, duration = truth_segments(os.path.join(jams_dir, jams_name))
        features, spf = features_for(wav)
        pred_frames = [to_majmin(l) or "X" for l in predict_labels(session, labels, features)]

        file_scored = file_correct = 0
        t = 0.0
        while t < duration:
            truth = next((to_majmin(v) for s, e, v in segments if s <= t < e), None)
            if truth is not None:
                idx = min(int(t / spf), len(pred_frames) - 1)
                pred = pred_frames[idx]
                file_scored += 1
                if pred == truth:
                    file_correct += 1
                else:
                    key = f"{truth}->{pred}"
                    confusions[key] = confusions.get(key, 0) + 1
            t += STEP
        scored += file_scored
        correct += file_correct
        per_file.append((file_correct / max(file_scored, 1), stem))

    print(f"model={model_name} files={len(per_file)} frames={scored}")
    print(f"WCSR (majmin): {correct / scored:.2%}")
    print("worst 5:", sorted(per_file)[:5])
    print("top confusions:", sorted(confusions.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "btc_large_voca"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    main(name, limit)
