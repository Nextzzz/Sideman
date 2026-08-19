"""ChordMini "BTC CL" checkpoint vs our base model, side by side, on the
protected HookTheory benchmark rows (first BENCH_ROWS of sample.csv).

Same features, same scoring, raw argmax for both — a fair like-for-like.
ChordMini is MIT (code + weights), so a win here means a free upgrade of
the base model.

Run:
    .venv/Scripts/python chordmini_eval.py [rows=250]
"""
import csv
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CHORDMINI = os.path.join(HERE, "vendor", "ChordMini")
sys.path.insert(0, CHORDMINI)

from src.models.btc_model import BTC_model  # noqa: E402

from btc_features import features_from_wav  # noqa: E402
from eval_guitarset import to_majmin  # noqa: E402
from hooktheory_eval import ROOT, load_audio, read_lab, STEP  # noqa: E402

BENCH_ROWS = 250


def load_chordmini():
    ck = torch.load(os.path.join(CHORDMINI, "checkpoints", "btc_model_best.pth"),
                    map_location="cpu", weights_only=False)
    model = BTC_model({"num_chords": 170, "feature_size": 144})
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    mean = float(ck["normalization"]["mean"])
    std = float(ck["normalization"]["std"])
    idx_to_chord = ck["idx_to_chord"]
    return model, mean, std, idx_to_chord


def main(rows_limit):
    model, mean, std, idx_to_chord = load_chordmini()

    session = ort.InferenceSession(os.path.join(HERE, "models",
                                                "btc_large_voca.onnx"))
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        base_labels = json.load(f)["labels"]

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:rows_limit]

    scored = {"chordmini": 0, "base": 0}
    correct = {"chordmini": 0, "base": 0}
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

        with torch.no_grad():
            x = (torch.tensor(features[None].astype(np.float32)) - mean) / std
            mini_idx = model(x)[0].argmax(dim=-1).numpy()
        mini_pred = [to_majmin(idx_to_chord[int(i)]) or "X" for i in mini_idx]

        base_pred = []
        padded = np.pad(features, ((0, (-len(features)) % 108), (0, 0)))
        for t in range(len(padded) // 108):
            logits = session.run(
                None, {"features": padded[None, t * 108:(t + 1) * 108]
                       .astype(np.float32)})[0][0]
            base_pred.extend(
                to_majmin(base_labels[int(i)]) or "X"
                for i in logits.argmax(axis=1))
        base_pred = base_pred[:len(features)]

        t = truth[0][0]
        while t < truth[-1][1]:
            label = next((l for s, e, l in truth if s <= t < e), None)
            if label is not None and label != "X":
                idx = min(int(t / spf), len(features) - 1)
                for name, pred in (("chordmini", mini_pred), ("base", base_pred)):
                    scored[name] += 1
                    if pred[idx] == label:
                        correct[name] += 1
            t += STEP
        done += 1
        if done % 20 == 0:
            print(f"{done} songs: " + "  ".join(
                f"{n} {correct[n] / max(scored[n], 1):.1%}" for n in scored),
                flush=True)

    print(f"\n=== ChordMini vs base, {done} benchmark songs ===")
    for name in scored:
        print(f"{name}: WCSR (majmin) {correct[name] / max(scored[name], 1):.2%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else BENCH_ROWS)
