"""Training data from the collected HookTheory audio (LICENSE NOTE:
CC BY-NC-SA — the resulting model is a RESEARCH PROTOTYPE ONLY, used to
decide whether a commercial HookTheory license is worth requesting; it
must never ship in the product).

Song-level split against our own benchmark: the first BENCH_ROWS rows
of sample.csv stay a clean held-out benchmark (superset of every number
reported so far); training songs are the rest. For each training song,
ALL of its usable dataset segments are cached (the benchmark used only
the longest one), labels in the full 170-class vocabulary.

Output:
    bundle_hook/data/hook_train.npz    windows [N,108,144] f32 + labels + mask
    bundle_hook/data/hook_val.npz      song-level held-out slice for early stop

Run:
    .venv/Scripts/python hooktheory_train_prep.py
"""
import csv
import gzip
import json
import os
import subprocess

import numpy as np

from btc_features import features_from_wav, TIMESTEP, SAMPLE_RATE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "datasets", "hooktheory"))
OUT = os.path.join(HERE, "bundle_hook", "data")
TOOLS = os.path.join(os.environ["LOCALAPPDATA"], "Strunika", "tools")
FFMPEG = os.path.join(TOOLS, "ffmpeg.exe")

BENCH_ROWS = 250  # sample.csv rows reserved for the benchmark, forever
X_IDX, N_IDX = 168, 169
LOG_FLOOR = float(np.log(1e-6))

# root_position_intervals -> index into the BTC quality list
# [min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4]
INTERVALS_TO_QUALITY = {
    (3, 4): 0, (4, 3): 1, (3, 3): 2, (4, 4): 3,
    (3, 4, 2): 4, (4, 3, 2): 5, (3, 4, 3): 6, (3, 4, 4): 7,
    (4, 3, 4): 8, (4, 3, 3): 9, (3, 3, 3): 10, (3, 3, 4): 11,
    (2, 5): 12, (5, 2): 13,
}


def label_idx(event):
    intervals = tuple(event.get("root_position_intervals") or ())
    quality = INTERVALS_TO_QUALITY.get(intervals)
    if quality is None:
        quality = INTERVALS_TO_QUALITY.get(intervals[:3])
    if quality is None:
        quality = INTERVALS_TO_QUALITY.get(intervals[:2])
    if quality is None:
        return X_IDX
    return (event["root_pitch_class"] % 12) * 14 + quality


def load_mono(path):
    raw = subprocess.run(
        [FFMPEG, "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, timeout=300).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def segment_windows(features, spf, seg_events):
    """Cut the segment span into 108-frame windows with per-frame labels."""
    lo = max(0, int(seg_events[0][0] / spf))
    hi = min(len(features), int(seg_events[-1][1] / spf))
    if hi - lo < 20:
        return []
    labels = np.full(hi - lo, N_IDX, dtype=np.int64)
    for start, end, idx in seg_events:
        a = max(0, int(start / spf) - lo)
        b = min(hi - lo, int(end / spf) - lo)
        labels[a:b] = idx
    chunk_f = features[lo:hi]
    out = []
    for s in range(0, len(chunk_f), TIMESTEP):
        f = chunk_f[s:s + TIMESTEP]
        l = labels[s:s + TIMESTEP]
        pad = TIMESTEP - len(f)
        mask = np.ones(TIMESTEP, dtype=np.float32)
        if pad > 0:
            f = np.pad(f, ((0, pad), (0, 0)), constant_values=LOG_FLOOR)
            l = np.pad(l, (0, pad), constant_values=N_IDX)
            mask[TIMESTEP - pad:] = 0
        out.append((f.astype(np.float32), l, mask))
    return out


def main():
    with gzip.open(os.path.join(ROOT, "Hooktheory.json.gz"), "rt",
                   encoding="utf-8") as f:
        data = json.load(f)

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train_rows = rows[BENCH_ROWS:]

    # All usable segments per training song (the audio is per-song).
    by_song = {}
    for entry_id, entry in data.items():
        tags = set(entry.get("tags", []))
        annotations = entry.get("annotations") or {}
        if not {"AUDIO_AVAILABLE", "REFINED_ALIGNMENT", "HARMONY"} <= tags:
            continue
        if not annotations.get("harmony") or len(annotations.get("keys") or []) != 1:
            continue
        key = (entry["hooktheory"]["artist"], entry["hooktheory"]["song"])
        by_song.setdefault(key, []).append(entry)

    packs = {"train": ([], [], []), "val": ([], [], [])}
    used_songs = used_segments = 0
    for n_row, row in enumerate(train_rows):
        # Every 10th training song goes to the early-stop validation slice.
        xs, ys, ms = packs["val" if n_row % 10 == 0 else "train"]
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        if not os.path.exists(audio_path):
            continue
        entries = by_song.get((row["artist"], row["song"]), [])
        if not entries:
            continue
        wav = load_mono(audio_path)
        if len(wav) < SAMPLE_RATE * 10:
            continue
        features, spf = features_from_wav(wav)
        song_added = False
        for entry in entries:
            refined = entry["alignment"]["refined"]
            beats = np.array(refined["beats"], dtype=float)
            times = np.array(refined["times"], dtype=float)
            events = []
            for ev in entry["annotations"]["harmony"]:
                if ev["onset"] < beats[0] or ev["offset"] > beats[-1]:
                    continue
                events.append((float(np.interp(ev["onset"], beats, times)),
                               float(np.interp(ev["offset"], beats, times)),
                               label_idx(ev)))
            if not events:
                continue
            for f, l, m in segment_windows(features, spf, events):
                xs.append(f)
                ys.append(l)
                ms.append(m)
            used_segments += 1
            song_added = True
        used_songs += song_added
        if used_songs % 50 == 0 and song_added:
            print(f"{used_songs} songs, {used_segments} segments", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for name, (xs, ys, ms) in packs.items():
        np.savez_compressed(os.path.join(OUT, f"hook_{name}.npz"),
                            x=np.stack(xs), y=np.stack(ys), m=np.stack(ms))
        print(f"hook_{name}.npz: {len(xs)} windows", flush=True)
    print(f"DONE: {used_songs} songs, {used_segments} segments", flush=True)


if __name__ == "__main__":
    main()
