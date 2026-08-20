"""Download audio for the HookTheory benchmark sample.

Direct video ids from sample.csv (no search needed — the dataset
pins exact YouTube videos its alignments refer to). Idempotent.

Run:
    .venv/Scripts/python hooktheory_collect.py [limit]
"""
import csv
import os
import random
import sys
import time

from billboard_collect import download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(
    HERE, "..", "datasets", "hooktheory", os.environ.get("HOOK_SUBSET", "")))
AUDIO_DIR = os.path.join(ROOT, "audio")


def main(limit):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:limit]

    ok = fail = skipped = 0
    for n, row in enumerate(rows):
        out_path = os.path.join(AUDIO_DIR, row["id"] + ".m4a")
        if os.path.exists(out_path):
            skipped += 1
            continue
        status = download(row["yt_id"], out_path)
        if status.startswith("ok"):
            ok += 1
        else:
            fail += 1
            print(f"  {row['artist']}/{row['song']}: {status}", flush=True)
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(rows)}: ok={ok} fail={fail} skipped={skipped}",
                  flush=True)
        time.sleep(random.uniform(2.0, 4.0))
    print(f"COLLECTION DONE: ok={ok} fail={fail} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9)
