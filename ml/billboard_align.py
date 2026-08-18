"""Align downloaded Billboard audio against LAB chord annotations.

For each downloaded song: build the EXPECTED chroma from the annotation
(chord templates over time), compute the REAL chroma from audio, and
find the global offset (-12..+12 s) with the best agreement. A weak best
score means a wrong version (live/cover/re-record) -> rejected.

Output: datasets/billboard/alignment.csv (id, offset, score, verdict).

Run:
    .venv/Scripts/python billboard_align.py
"""
import csv
import os

import librosa
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.environ["LOCALAPPDATA"], "Sideman", "tools")
os.environ["PATH"] = TOOLS + os.pathsep + os.environ["PATH"]  # ffmpeg for m4a

BILLBOARD = os.path.normpath(os.path.join(HERE, "..", "datasets", "billboard"))
AUDIO_DIR = os.path.join(BILLBOARD, "audio")

SR = 22050
HOP = 2048
SPF = HOP / SR
MAX_OFFSET_S = 12.0
PASS_SCORE = 0.55  # calibrate after the first run if needed

ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def chord_template(label):
    """12-dim binary template for a Harte label; None for N/X/unparseable."""
    if label in ("N", "X"):
        return None
    root_part, _, quality = label.partition(":")
    quality = quality.split("/")[0].split("(")[0]
    pc = ROOTS.get(root_part[0])
    if pc is None:
        return None
    for c in root_part[1:]:
        if c == "#":
            pc += 1
        elif c == "b":
            pc -= 1
        else:
            break
    pc %= 12
    third = 3 if quality.startswith("min") or quality.startswith("dim") else 4
    template = np.zeros(12, dtype=np.float32)
    template[pc] = 1.0
    template[(pc + third) % 12] = 0.8
    template[(pc + 7) % 12] = 0.8
    return template


def expected_chroma(lab_path, n_frames):
    expected = np.zeros((n_frames, 12), dtype=np.float32)
    with open(lab_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            template = chord_template(label)
            if template is None:
                continue
            lo = max(0, int(start / SPF))
            hi = min(n_frames, int(end / SPF))
            expected[lo:hi] = template
    return expected


def align(audio_path, lab_path):
    wav, _ = librosa.load(audio_path, sr=SR, mono=True)
    real = librosa.feature.chroma_stft(y=wav, sr=SR, hop_length=HOP).T
    real /= np.linalg.norm(real, axis=1, keepdims=True) + 1e-9

    expected = expected_chroma(lab_path, real.shape[0] + int(MAX_OFFSET_S / SPF))
    norms = np.linalg.norm(expected, axis=1, keepdims=True)
    expected = expected / (norms + 1e-9)
    voiced = norms[:, 0] > 0

    max_shift = int(MAX_OFFSET_S / SPF)
    best_score, best_shift = -1.0, 0
    # shift > 0: audio starts EARLIER than the annotation clock.
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            r = real[shift:]
            e, v = expected[:len(r)], voiced[:len(r)]
        else:
            r = real[:shift]
            e, v = expected[-shift:-shift + len(r)], voiced[-shift:-shift + len(r)]
        n = min(len(r), len(e))
        if n < 200:
            continue
        sims = (r[:n] * e[:n]).sum(axis=1)
        mask = v[:n]
        if mask.sum() < 100:
            continue
        score = float(sims[mask].mean())
        if score > best_score:
            best_score, best_shift = score, shift
    return best_shift * SPF, best_score


def main():
    files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".m4a"))
    out_path = os.path.join(BILLBOARD, "alignment.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "offset_s", "score", "verdict"])
        passed = 0
        for n, name in enumerate(files):
            entry = name[:-4]
            lab = os.path.join(BILLBOARD, "McGill-Billboard", entry, "full.lab")
            if not os.path.exists(lab):
                continue
            try:
                offset, score = align(os.path.join(AUDIO_DIR, name), lab)
                verdict = "pass" if score >= PASS_SCORE else "reject"
                passed += verdict == "pass"
                writer.writerow([entry, f"{offset:.2f}", f"{score:.3f}", verdict])
            except Exception as ex:
                writer.writerow([entry, "", "", f"error: {ex}"])
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{len(files)} aligned, {passed} passed", flush=True)
                out.flush()
    print(f"ALIGNMENT DONE: {passed}/{len(files)} passed", flush=True)


if __name__ == "__main__":
    main()
