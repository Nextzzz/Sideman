"""Home benchmark: user-verified analyses become ground truth.

When an analysis in the app is judged correct by ear, `add` turns it
into a benchmark entry under datasets/hooktheory/home/ (so bench.py
--subset home and hooktheory_eval work unchanged): copies the cached
audio, writes labs_full/<id>.lab (model label format) and labs/<id>.lab
(majmin), appends to sample.csv. Our own labels, our own repertoire —
the benchmark the product is actually used on.

Run:
    .venv/Scripts/python home_bench.py add <analysis.json> [--id NAME]
    .venv/Scripts/python home_bench.py list
"""
import csv
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.normpath(os.path.join(HERE, "..", "datasets", "hooktheory", "home"))

# Display label ("Am7", "E7", "Cmaj7", "Bm7b5") -> model label ("A:min7", "E:7", ...).
SUFFIX_TO_QUALITY = {
    "": "", "m": "min", "7": "7", "maj7": "maj7", "m7": "min7", "6": "maj6", "m6": "min6",
    "mMaj7": "minmaj7", "dim": "dim", "dim7": "dim7", "m7b5": "hdim7", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4",
}


def to_model_label(pretty):
    if pretty in ("—", "", "N"):
        return "N"
    m = re.match(r"^([A-G]#?)(.*)$", pretty)
    if not m or m[2] not in SUFFIX_TO_QUALITY:
        return "X"
    root, quality = m[1], SUFFIX_TO_QUALITY[m[2]]
    return root if quality == "" else f"{root}:{quality}"


def to_majmin(model_label):
    if model_label in ("N", "X"):
        return model_label
    root, _, q = model_label.partition(":")
    if q in ("", "maj6", "maj7", "7"):
        return root
    if q in ("min", "min6", "min7", "minmaj7"):
        return root + "m"
    return "X"


def add(analysis_path, entry_id=None):
    with open(analysis_path, encoding="utf-8") as f:
        a = json.load(f)
    source = a.get("Source", "")
    yt = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", source)
    entry_id = entry_id or (yt.group(1) if yt else os.path.splitext(os.path.basename(a["AudioPath"]))[0])
    audio_src = a["AudioPath"]
    if not os.path.exists(audio_src):
        sys.exit(f"audio not found: {audio_src} (temp cache cleaned?) — re-analyze and add again")

    for sub in ("audio", "labs", "labs_full"):
        os.makedirs(os.path.join(HOME, sub), exist_ok=True)
    ext = os.path.splitext(audio_src)[1] or ".m4a"
    shutil.copy2(audio_src, os.path.join(HOME, "audio", entry_id + ".m4a" if ext == ".m4a" else entry_id + ext))

    full_lines, mm_lines = [], []
    for s in a["Segments"]:
        label = to_model_label(s["Chord"])
        full_lines.append(f"{s['Start']:.3f} {s['End']:.3f} {label}")
        mm_lines.append(f"{s['Start']:.3f} {s['End']:.3f} {to_majmin(label)}")
    with open(os.path.join(HOME, "labs_full", entry_id + ".lab"), "w") as f:
        f.write("\n".join(full_lines))
    with open(os.path.join(HOME, "labs", entry_id + ".lab"), "w") as f:
        f.write("\n".join(mm_lines))

    sample = os.path.join(HOME, "sample.csv")
    rows = []
    if os.path.exists(sample):
        with open(sample, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["id"] != entry_id]
    rows.append({"id": entry_id, "artist": "home", "song": source or entry_id,
                 "yt_id": yt.group(1) if yt else "", "yt_duration": f"{a['DurationSeconds']:.0f}",
                 "seg_start": f"{a['Segments'][0]['Start']:.2f}",
                 "seg_end": f"{a['Segments'][-1]['End']:.2f}"})
    with open(sample, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "artist", "song", "yt_id", "yt_duration", "seg_start", "seg_end"])
        w.writeheader()
        w.writerows(rows)
    print(f"added {entry_id}: {len(full_lines)} segments, {a['DurationSeconds']:.0f}s, engine '{a['Engine']}'")


def list_entries():
    sample = os.path.join(HOME, "sample.csv")
    if not os.path.exists(sample):
        print("home benchmark is empty")
        return
    with open(sample, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            print(f"{r['id']:14} {r['yt_duration']:>5}s  {r['song']}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "add":
        add(sys.argv[2], sys.argv[sys.argv.index("--id") + 1] if "--id" in sys.argv else None)
    else:
        list_entries()
