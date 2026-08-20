"""Decoder A/B on the protected benchmark rows: raw argmax vs uniform
Viterbi vs Viterbi with the Billboard transition prior, all over the
same base+overlap probabilities. Answers whether a learned transition
prior is worth porting to the C# decoder.

Run:
    .venv/Scripts/python viterbi_eval.py [rows=250] [--self 0.9]
"""
import csv
import json
import os
import sys

import numpy as np
import onnxruntime as ort

from btc_features import features_from_wav
from eval_guitarset import to_majmin
from hooktheory_eval import ROOT, load_audio, read_lab, STEP
from infer_tricks_eval import probs_pass, combine, T

HERE = os.path.dirname(os.path.abspath(__file__))


def label_parts(label):
    if label in ("N", "X"):
        return label
    root, _, quality = label.partition(":")
    return root, (quality or "maj")


def build_log_transitions(labels, prior, stay, alpha=1.0):
    """[S,S] log P(s -> s'): stay on the diagonal, switch mass spread
    uniformly (prior=None) or by the Billboard prior raised to `alpha`
    (0.5 = softened, 1 = as counted, 2 = sharpened)."""
    n = len(labels)
    roots = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
             "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    switch = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(switch, 0)
    if prior is not None:
        qualities = prior["qualities"]
        c2c = np.array(prior["chord_to_chord"])
        to_n = np.array(prior["to_n"])
        from_n = np.array(prior["from_n"])
        parts = [label_parts(l) for l in labels]
        n_idx = labels.index("N")
        switch = np.zeros((n, n))
        for s, ps in enumerate(parts):
            if ps == "X":
                switch[s] = 1.0 / (n - 1)
                switch[s, s] = 0
                continue
            for d, pd in enumerate(parts):
                if d == s or pd == "X":
                    continue
                if ps == "N":
                    if pd != "N":
                        switch[s, d] = from_n[qualities.index(pd[1])] / 12
                elif pd == "N":
                    switch[s, d] = to_n[qualities.index(ps[1])]
                else:
                    interval = (roots[pd[0]] - roots[ps[0]]) % 12
                    switch[s, d] = c2c[qualities.index(ps[1]), interval, qualities.index(pd[1])]
            switch[s] = switch[s] ** alpha
            switch[s] /= switch[s].sum()
    trans = np.full((n, n), 0.0)
    trans += (1 - stay) * switch
    np.fill_diagonal(trans, stay)
    return np.log(trans + 1e-12)


def viterbi(log_probs, log_trans):
    frames, n = log_probs.shape
    score = log_probs[0].copy()
    back = np.zeros((frames, n), dtype=np.int32)
    for t in range(1, frames):
        cand = score[:, None] + log_trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + log_probs[t]
    path = np.zeros(frames, dtype=np.int32)
    path[-1] = score.argmax()
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def main(rows_limit, stay):
    session = ort.InferenceSession(os.path.join(HERE, "models", "btc_large_voca.onnx"))
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        labels = json.load(f)["labels"]
    with open(os.path.join(HERE, "models", "transition_prior.json")) as f:
        prior = json.load(f)

    decoders = {
        "argmax": None,
        "viterbi-uniform": build_log_transitions(labels, None, stay),
        "prior-soft": build_log_transitions(labels, prior, stay, alpha=0.5),
        "prior": build_log_transitions(labels, prior, stay, alpha=1.0),
        "prior-sharp": build_log_transitions(labels, prior, stay, alpha=2.0),
    }

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:rows_limit]

    scored = {d: 0 for d in decoders}
    correct = {d: 0 for d in decoders}
    done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        lab_path = os.path.join(ROOT, "labs", row["id"] + ".lab")
        if not os.path.exists(audio_path) or not os.path.exists(lab_path):
            continue
        truth = read_lab(lab_path)
        wav = load_audio(audio_path)
        if len(wav) < 22050:
            continue
        features, spf = features_from_wav(wav)
        probs = combine([probs_pass(session, features, 0, 0),
                         probs_pass(session, features, T // 2, 0)])
        log_probs = np.log(probs + 1e-12)

        preds = {}
        for name, log_trans in decoders.items():
            path = log_probs.argmax(axis=1) if log_trans is None else viterbi(log_probs, log_trans)
            preds[name] = [to_majmin(labels[i]) or "X" for i in path]

        t = truth[0][0]
        while t < truth[-1][1]:
            label = next((l for s, e, l in truth if s <= t < e), None)
            if label is not None and label != "X":
                idx = min(int(t / spf), len(features) - 1)
                for name in decoders:
                    scored[name] += 1
                    if preds[name][idx] == label:
                        correct[name] += 1
            t += STEP
        done += 1
        if done % 20 == 0:
            print(f"{done} songs: " + "  ".join(
                f"{d} {correct[d] / max(scored[d], 1):.1%}" for d in decoders), flush=True)

    print(f"\n=== decoders (base+ovl probs, stay={stay}), {done} benchmark songs ===")
    for d in decoders:
        print(f"{d:16} WCSR (majmin) {correct[d] / max(scored[d], 1):.2%}")


if __name__ == "__main__":
    numeric = [a for a in sys.argv[1:] if a.replace(".", "").isdigit()]
    rows = int(numeric[0]) if numeric and "." not in numeric[0] else 250
    stay = float(sys.argv[sys.argv.index("--self") + 1]) if "--self" in sys.argv else 0.9
    main(rows, stay)
