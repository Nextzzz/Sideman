"""Golden reference files for validating the future C# port.

For each test clip, dumps:
  <name>.features.csv  raw log-CQT features (frames x 144)
  <name>.logits.csv    ONNX logits for the first 108-frame window
  <name>.labels.txt    per-frame chord labels (large voca model)

Run:
    .venv/Scripts/python make_goldens.py <wav> [<wav> ...]
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

from btc_features import features_for, TIMESTEP

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "goldens")


def main(paths):
    session = ort.InferenceSession(os.path.join(HERE, "models", "btc_large_voca.onnx"))
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        labels = json.load(f)["labels"]

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        features, _ = features_for(path)

        np.savetxt(os.path.join(GOLDEN_DIR, f"{name}.features.csv"),
                   features, delimiter=",", fmt="%.6f")

        window = features[:TIMESTEP]
        if window.shape[0] < TIMESTEP:
            window = np.pad(window, ((0, TIMESTEP - window.shape[0]), (0, 0)))
        logits = session.run(None, {"features": window[None, ...]})[0][0]
        np.savetxt(os.path.join(GOLDEN_DIR, f"{name}.logits.csv"),
                   logits, delimiter=",", fmt="%.5f")

        frame_labels = [labels[int(i)] for i in logits.argmax(axis=1)]
        with open(os.path.join(GOLDEN_DIR, f"{name}.labels.txt"), "w") as f:
            f.write("\n".join(frame_labels))

        print(f"{name}: {features.shape[0]} frames, "
              f"first-window chords: {sorted(set(frame_labels))}")


if __name__ == "__main__":
    main(sys.argv[1:])
