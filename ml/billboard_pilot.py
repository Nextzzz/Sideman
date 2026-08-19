"""Feasibility pilot: how many McGill Billboard songs can we match on
YouTube by duration? Checks top-1 search hit duration against the LAB
annotation duration — no audio is downloaded.

Run:
    .venv/Scripts/python billboard_pilot.py [sample_size]
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BILLBOARD = os.path.normpath(os.path.join(HERE, "..", "datasets", "billboard"))
YTDLP = os.path.join(os.environ["LOCALAPPDATA"], "Strunika", "tools", "yt-dlp.exe")


def lab_duration(entry_id):
    path = os.path.join(BILLBOARD, "McGill-Billboard", f"{entry_id:04d}", "full.lab")
    if not os.path.exists(path):
        return None
    last = 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                last = float(parts[1])
    return last if last > 60 else None


def youtube_durations(query, top=5):
    """Durations of the top-N search hits (metadata only, no download)."""
    try:
        result = subprocess.run(
            [YTDLP, f"ytsearch{top}:{query}", "--skip-download",
             "--print", "duration", "--no-warnings", "--js-runtimes", "node"],
            capture_output=True, text=True, timeout=90)
        return [float(l) for l in result.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return []


def main(sample_size=40):
    songs = []
    with open(os.path.join(BILLBOARD, "index.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["title"] and row["artist"]:
                songs.append((int(row["id"]), row["artist"], row["title"]))

    print(f"annotated songs in index: {len(songs)}", flush=True)
    step = max(1, len(songs) // sample_size)
    sample = songs[::step][:sample_size]

    tight = loose = checked = 0
    for entry_id, artist, title in sample:
        truth = lab_duration(entry_id)
        if truth is None:
            continue
        durations = youtube_durations(f"{artist} {title}")
        checked += 1
        best = min((abs(d - truth) for d in durations), default=1e9)
        tight += best <= 3.0
        loose += best <= 10.0
        tag = "TIGHT" if best <= 3 else "LOOSE" if best <= 10 else "MISS "
        print(f"{tag} {artist} - {title}: lab {truth:.0f}s, best delta {best:.0f}s "
              f"of {len(durations)} hits", flush=True)

    print(f"match rate: tight(<=3s) {tight}/{checked} = {tight / max(checked, 1):.0%}, "
          f"loose(<=10s, alignment-recoverable) {loose}/{checked} = "
          f"{loose / max(checked, 1):.0%}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
