"""Analyze one song with the ChordMini checkpoint — for ear-testing
against our models (Song tab / Strunika.Cli analyze).

Accepts a local audio file or a YouTube URL (downloaded via the same
yt-dlp + bgutil chain as the collectors). Prints a chord timeline:
raw argmax + merge of equal neighbours, no Viterbi/key prior — so any
difference you hear is the model, not post-processing.

Run:
    .venv/Scripts/python chordmini_analyze.py <file-or-youtube-url>
"""
import os
import re
import sys
import tempfile

import numpy as np
import torch

from btc_features import features_from_wav
from chordmini_eval import load_chordmini
from hooktheory_eval import load_audio


FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
                 "Cb": "B", "Fb": "E"}


def pretty(label):
    """Match the app's display exactly (ChordLabels.Pretty + sharp roots)."""
    root, _, quality = label.partition(":")
    root = FLAT_TO_SHARP.get(root, root)
    return root + {
        "": "", "maj": "", "min": "m", "dim": "dim", "aug": "aug",
        "min6": "m6", "maj6": "6", "min7": "m7", "minmaj7": "mMaj7",
        "maj7": "maj7", "7": "7", "dim7": "dim7", "hdim7": "m7b5",
        "sus2": "sus2", "sus4": "sus4",
    }.get(quality, ":" + quality)


def resolve(target):
    if not target.lower().startswith("http"):
        return target
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", target)
    if not match:
        sys.exit("cannot find a video id in the URL")
    video_id = match.group(1)
    out = os.path.join(tempfile.gettempdir(), "strunika", f"cm_{video_id}.m4a")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(out):
        from billboard_collect import download
        print("downloading...", flush=True)
        status = download(video_id, out)
        if not status.startswith("ok"):
            sys.exit(f"download failed: {status}")
    return out


def main(target):
    path = resolve(target)
    print(f"Analyzing (ChordMini BTC-CL) {os.path.basename(path)}...")
    wav = load_audio(path)
    features, spf = features_from_wav(wav)

    model, mean, std, idx_to_chord = load_chordmini()
    with torch.no_grad():
        x = (torch.tensor(features[None].astype(np.float32)) - mean) / std
        idx = model(x)[0].argmax(dim=-1).numpy()

    labels = [idx_to_chord[int(i)] for i in idx]
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            begin, end = start * spf, i * spf
            if end - begin >= 0.3 and labels[start] != "N":
                print(f"  {int(begin) // 60}:{begin % 60:04.1f} - "
                      f"{int(end) // 60}:{end % 60:04.1f}  "
                      f"{pretty(labels[start])}")
            start = i


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: chordmini_analyze.py <file-or-youtube-url>")
    main(sys.argv[1])
