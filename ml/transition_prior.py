"""Chord-transition prior from the McGill Billboard annotations (CC0).

Transposition-invariant bigram: P(next quality, root interval | current
quality), counted over chord CHANGES (beat-level repeats collapsed) in
890 songs. Stored per quality NAME so any model with the BTC vocabulary
can use it; the decoder keeps its own stay probability and spends the
switch mass according to this prior instead of uniformly.

Output: models/transition_prior.json
    qualities       14 BTC quality names
    chord_to_chord  [q_from][interval 0..11][q_to]  P(switch to ...)  (sums to 1 with to_n)
    to_n            [q_from]                        P(switch to N)
    from_n          [q_to]                          P(N -> quality), uniform over roots

Run:
    .venv/Scripts/python transition_prior.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABS = os.path.normpath(os.path.join(HERE, "..", "datasets", "billboard", "McGill-Billboard"))
OUT = os.path.join(HERE, "models", "transition_prior.json")

ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
QUALITIES = ["min", "maj", "dim", "aug", "min6", "maj6", "min7",
             "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]
# Same leadsheet-oddity folding as finetune.py's label_to_idx.
FALLBACK = (("minmaj", "minmaj7"), ("min9", "min7"), ("min11", "min7"), ("min", "min"),
            ("maj9", "maj7"), ("maj6", "maj6"), ("maj", "maj"), ("hdim", "hdim7"),
            ("dim7", "dim7"), ("dim", "dim"), ("sus4", "sus4"), ("sus2", "sus2"),
            ("aug", "aug"), ("9", "7"), ("11", "7"), ("13", "7"), ("7", "7"),
            ("6", "maj6"), ("5", "maj"), ("1", "maj"))
SMOOTHING = 0.5


def parse(label):
    """Harte label -> (root pc, quality index) | 'N' | None (unmapped)."""
    if label in ("N", "X", ""):
        return "N"
    root_part, _, quality = label.partition(":")
    quality = quality.split("/")[0].split("(")[0]
    pc = ROOTS.get(root_part[0])
    if pc is None:
        return None
    for c in root_part[1:]:
        pc += 1 if c == "#" else -1 if c == "b" else 0
    pc %= 12
    if quality == "":
        quality = "maj"
    if quality not in QUALITIES:
        for base, mapped in FALLBACK:
            if quality.startswith(base):
                quality = mapped
                break
        else:
            return None
    return pc, QUALITIES.index(quality)


def main():
    nq = len(QUALITIES)
    c2c = np.full((nq, 12, nq), SMOOTHING)
    to_n = np.full(nq, SMOOTHING)
    from_n = np.full(nq, SMOOTHING)
    songs = changes = 0
    for entry in sorted(os.listdir(LABS)):
        path = os.path.join(LABS, entry, "full.lab")
        if not os.path.exists(path):
            continue
        sequence = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                chord = parse(parts[2])
                if chord is None:
                    continue
                if not sequence or sequence[-1] != chord:
                    sequence.append(chord)
        songs += 1
        for prev, cur in zip(sequence, sequence[1:]):
            changes += 1
            if prev == "N" and cur != "N":
                from_n[cur[1]] += 1
            elif prev != "N" and cur == "N":
                to_n[prev[1]] += 1
            elif prev != "N" and cur != "N":
                interval = (cur[0] - prev[0]) % 12
                c2c[prev[1], interval, cur[1]] += 1

    # Normalize per source quality over every destination (chords + N).
    out_c2c = np.zeros_like(c2c)
    out_to_n = np.zeros(nq)
    for q in range(nq):
        total = c2c[q].sum() + to_n[q]
        out_c2c[q] = c2c[q] / total
        out_to_n[q] = to_n[q] / total
    out_from_n = from_n / from_n.sum()

    with open(OUT, "w") as f:
        json.dump({
            "source": "McGill Billboard annotations (CC0), chord changes only",
            "songs": songs,
            "changes": changes,
            "qualities": QUALITIES,
            "chord_to_chord": out_c2c.round(6).tolist(),
            "to_n": out_to_n.round(6).tolist(),
            "from_n": out_from_n.round(6).tolist(),
        }, f)

    print(f"{songs} songs, {changes} chord changes -> {OUT}")
    maj = QUALITIES.index("maj")
    top = sorted(((out_c2c[maj, i, q], i, QUALITIES[q]) for i in range(12) for q in range(nq)),
                 reverse=True)[:6]
    print("after a major chord, most likely next:",
          ", ".join(f"+{i} {q} {p:.1%}" for p, i, q in top))


if __name__ == "__main__":
    main()
