"""Build the training cache from aligned Billboard audio:
per song -> CQT features + frame labels (170-class vocabulary), with the
alignment offset applied so audio and annotation share one clock.

Run:
    .venv/Scripts/python billboard_cache.py
"""
import csv
import os
from collections import Counter

import numpy as np

from billboard_align import load_audio, BILLBOARD, AUDIO_DIR
from btc_features import features_from_wav, SAMPLE_RATE
from finetune import label_to_idx, X_IDX, N_IDX

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_billboard")


def frame_labels(lab_path, n_frames, spf, unmapped):
    labels = np.full(n_frames, N_IDX, dtype=np.int64)
    with open(lab_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            start, end, raw = float(parts[0]), float(parts[1]), parts[2]
            idx = label_to_idx(raw)
            if idx == X_IDX and raw not in ("X",):
                unmapped[raw.partition(":")[2].split("/")[0]] += 1
            lo = max(0, int(start / spf))
            hi = min(n_frames, int(end / spf))
            labels[lo:hi] = idx
    return labels


def main():
    os.makedirs(CACHE, exist_ok=True)
    unmapped = Counter()
    built = skipped = 0

    with open(os.path.join(BILLBOARD, "alignment.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["verdict"] == "pass"]

    for n, row in enumerate(rows):
        entry = row["id"] if len(row["id"]) == 4 else f"{int(row['id']):04d}"
        out = os.path.join(CACHE, entry + ".npz")
        if os.path.exists(out):
            skipped += 1
            continue

        wav = load_audio(os.path.join(AUDIO_DIR, entry + ".m4a"), SAMPLE_RATE)
        offset = float(row["offset_s"])
        # offset > 0: audio runs ahead of the annotation clock — drop its head.
        if offset > 0:
            wav = wav[int(offset * SAMPLE_RATE):]
        elif offset < 0:
            wav = np.concatenate(
                [np.zeros(int(-offset * SAMPLE_RATE), dtype=np.float32), wav])

        features, spf = features_from_wav(wav)
        labels = frame_labels(
            os.path.join(BILLBOARD, "McGill-Billboard", entry, "full.lab"),
            features.shape[0], spf, unmapped)
        np.savez_compressed(out, features=features, labels=labels)
        built += 1
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(rows)} cached", flush=True)

    print(f"CACHE DONE: built={built} skipped={skipped}", flush=True)
    print("top unmapped qualities:", unmapped.most_common(12), flush=True)


if __name__ == "__main__":
    main()
