"""Pseudo-labels for self-training (ChordMini-style, legally clean):
our own best teacher (base+guitar2 ensemble, overlapping windows)
labels FULL songs from the training half of the collected audio
(sample.csv rows 251+; the benchmark rows never enter). Frames below a
confidence floor are masked out; windows are ranked by mean confidence
and the best MAX_WINDOWS are kept. Features stored float16 to keep the
GPU bundle small.

Output: bundle_pseudo/data/pseudo_train.npz  (x f16, y, m)

Run:
    .venv/Scripts/python pseudo_label_prep.py [max_windows=4000]
"""
import csv
import os
import sys

import numpy as np
import onnxruntime as ort

from btc_features import features_from_wav
from hooktheory_eval import ROOT, load_audio
from infer_tricks_eval import probs_pass, combine, T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bundle_pseudo", "data")
BENCH_ROWS = 250
CONF_FLOOR = 0.6      # frames under this teacher confidence don't train
LOG_FLOOR = float(np.log(1e-6))
N_IDX = 169


def main(max_windows):
    sessions = [ort.InferenceSession(os.path.join(HERE, "models", n + ".onnx"))
                for n in ("btc_large_voca", "btc_guitar2")]
    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[BENCH_ROWS:]

    windows = []  # (mean_conf, x, y, m)
    done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        if not os.path.exists(audio_path):
            continue
        wav = load_audio(audio_path)
        if len(wav) < 22050 * 10:
            continue
        features, spf = features_from_wav(wav)
        passes = []
        for s in sessions:
            passes.append(probs_pass(s, features, 0, 0))
            passes.append(probs_pass(s, features, T // 2, 0))
        probs = combine(passes)
        labels = probs.argmax(axis=1).astype(np.int64)
        conf = probs.max(axis=1)

        for start in range(0, len(features), T):
            x = features[start:start + T]
            y = labels[start:start + T]
            c = conf[start:start + T]
            pad = T - len(x)
            mask = (c >= CONF_FLOOR).astype(np.float32)
            if pad > 0:
                x = np.pad(x, ((0, pad), (0, 0)), constant_values=LOG_FLOOR)
                y = np.pad(y, (0, pad), constant_values=N_IDX)
                mask = np.pad(mask, (0, pad))
            if mask.sum() < T * 0.5:
                continue  # mostly uncertain — skip the window entirely
            windows.append((float(c.mean()), x.astype(np.float16), y, mask))
        done += 1
        if done % 25 == 0:
            print(f"{done} songs, {len(windows)} candidate windows", flush=True)

    windows.sort(key=lambda w: -w[0])
    keep = windows[:max_windows]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "pseudo_train.npz"),
                        x=np.stack([w[1] for w in keep]),
                        y=np.stack([w[2] for w in keep]),
                        m=np.stack([w[3] for w in keep]))
    print(f"DONE: {done} songs, {len(windows)} candidates, kept {len(keep)} "
          f"(mean conf of kept: {np.mean([w[0] for w in keep]):.3f})", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
