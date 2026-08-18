"""Collect audio for McGill Billboard annotations from YouTube.

Per song: top-5 search (metadata only) -> pick the hit closest to the
LAB duration (<=10 s) -> download m4a to datasets/billboard/audio/.
Protected music retries via web_embedded + local bgutil PO-token server.
Idempotent: already-downloaded entries are skipped, safe to restart.

Run:
    .venv/Scripts/python billboard_collect.py
"""
import csv
import os
import random
import socket
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BILLBOARD = os.path.normpath(os.path.join(HERE, "..", "datasets", "billboard"))
AUDIO_DIR = os.path.join(BILLBOARD, "audio")
TOOLS = os.path.join(os.environ["LOCALAPPDATA"], "Sideman", "tools")
YTDLP = os.path.join(TOOLS, "yt-dlp.exe")
BGUTIL = os.path.join(TOOLS, "bgutil", "server", "build", "main.js")

MAX_DELTA = 10.0


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


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def ensure_bgutil():
    if port_open(4416):
        return
    print("starting bgutil PO-token server...", flush=True)
    subprocess.Popen(["node", BGUTIL],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        time.sleep(1)
        if port_open(4416):
            print("bgutil up", flush=True)
            return
    print("bgutil did not start; protected songs will fail", flush=True)


def run_ytdlp(args, timeout):
    return subprocess.run([YTDLP, *args, "--no-warnings", "--js-runtimes", "node"],
                          capture_output=True, text=True, timeout=timeout)


def search_best(query, truth):
    result = run_ytdlp([f"ytsearch5:{query}", "--skip-download",
                        "--print", "%(id)s %(duration)s"], timeout=90)
    best_id, best_delta = None, 1e9
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            delta = abs(float(parts[1]) - truth)
        except ValueError:
            continue
        if delta < best_delta:
            best_id, best_delta = parts[0], delta
    return best_id, best_delta


def download(video_id, out_path):
    url = f"https://www.youtube.com/watch?v={video_id}"
    base = ["-f", "ba[ext=m4a]", "--no-playlist", "-o", out_path, url]
    result = run_ytdlp(base, timeout=300)
    if result.returncode == 0 and os.path.exists(out_path):
        return "ok"
    ensure_bgutil()
    result = run_ytdlp(
        ["--extractor-args", "youtube:player_client=web_embedded", *base],
        timeout=300)
    if result.returncode == 0 and os.path.exists(out_path):
        return "ok-embedded"
    return "fail: " + result.stderr.strip()[-120:]


def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    songs = []
    with open(os.path.join(BILLBOARD, "index.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["title"] and row["artist"]:
                songs.append((int(row["id"]), row["artist"], row["title"]))

    report_path = os.path.join(BILLBOARD, "collection.csv")
    new_report = not os.path.exists(report_path)
    report = open(report_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(report)
    if new_report:
        writer.writerow(["id", "artist", "title", "lab_dur", "yt_id", "delta", "status"])

    ok = miss = fail = skipped = 0
    for n, (entry_id, artist, title) in enumerate(songs):
        out_path = os.path.join(AUDIO_DIR, f"{entry_id:04d}.m4a")
        if os.path.exists(out_path):
            skipped += 1
            continue

        truth = lab_duration(entry_id)
        if truth is None:
            continue

        try:
            video_id, delta = search_best(f"{artist} {title}", truth)
            if video_id is None or delta > MAX_DELTA:
                miss += 1
                writer.writerow([entry_id, artist, title, f"{truth:.0f}",
                                 video_id or "", f"{delta:.0f}", "no-duration-match"])
            else:
                status = download(video_id, out_path)
                if status.startswith("ok"):
                    ok += 1
                else:
                    fail += 1
                writer.writerow([entry_id, artist, title, f"{truth:.0f}",
                                 video_id, f"{delta:.1f}", status])
        except Exception as ex:
            fail += 1
            writer.writerow([entry_id, artist, title, f"{truth:.0f}", "", "",
                             f"error: {ex}"])
        report.flush()

        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(songs)}: ok={ok} miss={miss} fail={fail} "
                  f"skipped={skipped}", flush=True)
        time.sleep(random.uniform(2.0, 4.0))

    report.close()
    print(f"COLLECTION DONE: ok={ok} miss={miss} fail={fail} skipped={skipped}",
          flush=True)


if __name__ == "__main__":
    main()
