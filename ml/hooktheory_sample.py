"""Build a modern-songs benchmark sample from the HookTheory dataset
(Donahue et al., ISMIR'22; CC BY-NC-SA — INTERNAL EVALUATION ONLY:
never train product weights on it, never redistribute the data).

Selection: TEST split, refined YouTube alignment, harmony present,
single key, one (longest) segment per song. Ground truth: harmony
events mapped beats->seconds via the refined alignment, reduced to
majmin ('G', 'Gm') or 'X' for out-of-vocabulary qualities.

Output:
    datasets/hooktheory/sample.csv           song list for collection
    datasets/hooktheory/labs/<id>.lab        absolute-time truth labels

Run:
    .venv/Scripts/python hooktheory_sample.py [n_songs=200]
"""
import csv
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "datasets", "hooktheory"))
LABS = os.path.join(ROOT, "labs")

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def majmin_label(event):
    intervals = event.get("root_position_intervals") or []
    root = NAMES[event["root_pitch_class"] % 12]
    if intervals[:2] == [4, 3]:
        return root
    if intervals[:2] == [3, 4]:
        return root + "m"
    return "X"  # dim/aug/sus/power — excluded from majmin scoring


def main(n_songs):
    with gzip.open(os.path.join(ROOT, "Hooktheory.json.gz"), "rt",
                   encoding="utf-8") as f:
        data = json.load(f)

    best = {}  # (artist, song) -> (num_beats, id, entry)
    for entry_id, entry in data.items():
        tags = set(entry.get("tags", []))
        annotations = entry.get("annotations") or {}
        if entry.get("split") != "TEST":
            continue
        if not {"AUDIO_AVAILABLE", "REFINED_ALIGNMENT", "HARMONY"} <= tags:
            continue
        if not annotations.get("harmony") or len(annotations.get("keys") or []) != 1:
            continue
        key = (entry["hooktheory"]["artist"], entry["hooktheory"]["song"])
        candidate = (annotations["num_beats"], entry_id, entry)
        if key not in best or candidate[0] > best[key][0]:
            best[key] = candidate

    picked = sorted(best.values(), key=lambda c: c[1])[:n_songs]
    os.makedirs(LABS, exist_ok=True)

    with open(os.path.join(ROOT, "sample.csv"), "w", encoding="utf-8",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "artist", "song", "yt_id", "yt_duration",
                         "seg_start", "seg_end"])
        for _, entry_id, entry in picked:
            refined = entry["alignment"]["refined"]
            beats = np.array(refined["beats"], dtype=float)
            times = np.array(refined["times"], dtype=float)

            lines = []
            for event in entry["annotations"]["harmony"]:
                onset, offset = event["onset"], event["offset"]
                if onset < beats[0] or offset > beats[-1]:
                    continue  # outside the aligned span
                start = float(np.interp(onset, beats, times))
                end = float(np.interp(offset, beats, times))
                lines.append(f"{start:.3f} {end:.3f} {majmin_label(event)}")
            if not lines:
                continue
            with open(os.path.join(LABS, entry_id + ".lab"), "w") as lab:
                lab.write("\n".join(lines))

            writer.writerow([entry_id, entry["hooktheory"]["artist"],
                             entry["hooktheory"]["song"],
                             entry["youtube"]["id"],
                             f"{entry['youtube']['duration']:.0f}",
                             f"{times[0]:.2f}", f"{times[-1]:.2f}"])
    print(f"sampled {len(picked)} songs -> sample.csv + labs/")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
