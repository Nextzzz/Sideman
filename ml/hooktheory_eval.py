"""Evaluate all exported BTC models on the HookTheory benchmark —
frame-level majmin WCSR inside each song's annotated segment.

Features are computed once per song and shared across models, so
adding a model costs only inference. Raw argmax (no Viterbi/key
prior) — a fair like-for-like model ranking.

Run:
    .venv/Scripts/python hooktheory_eval.py [limit] [audio_dir]

audio_dir 'audio' (default) scores the original mixes; 'audio_sep'
scores the demucs bass+other renders from demucs_prep.py — songs
missing there are skipped, so pass the same limit for a fair A/B.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

from btc_features import features_from_wav, TIMESTEP, SAMPLE_RATE
from eval_guitarset import to_majmin, predict_labels

HERE = os.path.dirname(os.path.abspath(__file__))
# HOOK_SUBSET=valid switches every eval/collect script to the permanent
# evaluation-only pool (datasets/hooktheory/valid); default = TEST pool.
ROOT = os.path.normpath(os.path.join(
    HERE, "..", "datasets", "hooktheory", os.environ.get("HOOK_SUBSET", "")))
TOOLS = os.path.join(os.environ["LOCALAPPDATA"], "Strunika", "tools")
FFMPEG = os.path.join(TOOLS, "ffmpeg.exe")

MODELS = ["btc_large_voca", "btc_guitar", "btc_mix"]
STEP = 0.1


def load_audio(path):
    raw = subprocess.run(
        [FFMPEG, "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, timeout=120).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def read_lab(path):
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                segments.append((float(parts[0]), float(parts[1]), parts[2]))
    return segments


def main(limit, audio_dir="audio"):
    ext = ".wav" if audio_dir == "audio_sep" else ".m4a"
    sessions = {}
    labels = {}
    for name in MODELS:
        sessions[name] = ort.InferenceSession(
            os.path.join(HERE, "models", name + ".onnx"))
        with open(os.path.join(HERE, "models", name + ".json")) as f:
            labels[name] = json.load(f)["labels"]

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:limit]

    scored = {name: 0 for name in MODELS}
    correct = {name: 0 for name in MODELS}
    full_scored = {name: 0 for name in MODELS}
    full_correct = {name: 0 for name in MODELS}
    per_song = {name: [] for name in MODELS}
    done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, audio_dir, row["id"] + ext)
        lab_path = os.path.join(ROOT, "labs", row["id"] + ".lab")
        full_path = os.path.join(ROOT, "labs_full", row["id"] + ".lab")
        if not os.path.exists(audio_path) or not os.path.exists(lab_path):
            continue
        truth = read_lab(lab_path)
        truth_full = read_lab(full_path) if os.path.exists(full_path) else []
        wav = load_audio(audio_path)
        if len(wav) < SAMPLE_RATE:
            continue
        features, spf = features_from_wav(wav)

        for name in MODELS:
            raw = predict_labels(sessions[name], labels[name], features)
            pred = [to_majmin(l) or "X" for l in raw]
            song_scored = song_correct = 0
            t = truth[0][0]
            while t < truth[-1][1]:
                label = next((l for s, e, l in truth if s <= t < e), None)
                if label is not None and label != "X":
                    idx = min(int(t / spf), len(pred) - 1)
                    song_scored += 1
                    if pred[idx] == label:
                        song_correct += 1
                # Full vocabulary: exact quality match (Am7 != Am).
                full = next((l for s, e, l in truth_full if s <= t < e), None)
                if full is not None and full != "X":
                    idx = min(int(t / spf), len(raw) - 1)
                    full_scored[name] += 1
                    if raw[idx] == full:
                        full_correct[name] += 1
                t += STEP
            scored[name] += song_scored
            correct[name] += song_correct
            if song_scored > 0:
                per_song[name].append((song_correct / song_scored,
                                       f"{row['artist']}/{row['song']}"))
        done += 1
        if done % 10 == 0:
            print(f"{done} songs: " + "  ".join(
                f"{n} {correct[n] / max(scored[n], 1):.1%}" for n in MODELS),
                flush=True)

    print(f"\n=== HookTheory benchmark ({audio_dir}), {done} songs ===")
    for name in MODELS:
        print(f"{name}: WCSR (majmin) {correct[name] / max(scored[name], 1):.2%}"
              + (f"   full-vocab {full_correct[name] / full_scored[name]:.2%}"
                 if full_scored[name] else ""))
        print("  worst 3:", [(f"{a:.0%}", b)
                             for a, b in sorted(per_song[name])[:3]])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9,
         sys.argv[2] if len(sys.argv) > 2 else "audio")
