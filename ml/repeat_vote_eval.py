"""Repetition voting ("music expert" post-correction), flavor level only.

Idea (user-specified): when the same passage repeats in a song, the
model's chord FLAVOR (Am vs Am7, E vs E7, C vs Cmaj7) should agree
across repeats; the root/mode must never be overruled — a genuinely
different chord in verse 2 stays.

Mechanism: frame-level self-similarity on the model's own posteriors
(context window of 2W+1 frames, majmin-family reduction, cosine).
For each frame, near-duplicate frames elsewhere in the song that carry
the SAME root+mode vote on the extension; if another extension wins
by a clear margin, the frame takes it. Roots untouched by construction,
so majmin WCSR is unchanged and only the full-vocabulary score can move.

Run:
    .venv/Scripts/python repeat_vote_eval.py [rows=250]
"""
import csv
import json
import os
import sys

import numpy as np
import onnxruntime as ort

from btc_features import features_from_wav
from hooktheory_eval import ROOT, load_audio, read_lab, STEP
from infer_tricks_eval import probs_pass, combine, T
from viterbi_eval import build_log_transitions, viterbi

HERE = os.path.dirname(os.path.abspath(__file__))
W = 8                 # context half-window in frames (~0.75 s each side)
MIN_GAP = 32          # frames (~3 s): neighbours must come from elsewhere
MIN_VOTERS = 3
MARGIN = 0.6          # winner must hold this share of neighbour weight

MAJ_Q = {"", "maj6", "maj7", "7"}
MIN_Q = {"min", "min6", "min7", "minmaj7"}


def family_of(label):
    """(root, mode) for voting eligibility; None for N/X/other."""
    if label in ("N", "X"):
        return None
    root, _, q = label.partition(":")
    if q in MAJ_Q:
        return root, "maj"
    if q in MIN_Q:
        return root, "min"
    return root, q  # dim/aug/sus families: voting only within themselves


def family_matrix(labels):
    """[S, 25] reduction of posteriors to 12 maj + 12 min + other."""
    roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    m = np.zeros((len(labels), 25))
    for i, l in enumerate(labels):
        fam = family_of(l)
        if fam is None or fam[1] not in ("maj", "min"):
            m[i, 24] = 1
        else:
            m[i, roots.index(fam[0]) + (12 if fam[1] == "min" else 0)] = 1
    return m


def context_vectors(probs, fam_matrix):
    reduced = probs @ fam_matrix                       # [T, 25]
    n = len(reduced)
    padded = np.pad(reduced, ((W, W), (0, 0)))
    ctx = np.concatenate([padded[k:k + n] for k in range(2 * W + 1)], axis=1)
    ctx /= np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9
    return ctx


def vote(path_labels, probs, fam_matrix, threshold):
    ctx = context_vectors(probs, fam_matrix)
    sim = ctx @ ctx.T
    n = len(path_labels)
    idx = np.arange(n)
    out = list(path_labels)
    fams = [family_of(l) for l in path_labels]
    for i in range(n):
        if fams[i] is None:
            continue
        cand = np.where((sim[i] >= threshold) & (np.abs(idx - i) >= MIN_GAP))[0]
        votes = {}
        for j in cand:
            if fams[j] == fams[i]:
                votes[path_labels[j]] = votes.get(path_labels[j], 0.0) + sim[i, j]
        if len(cand) < MIN_VOTERS or not votes:
            continue
        total = sum(votes.values())
        winner, weight = max(votes.items(), key=lambda kv: kv[1])
        if winner != path_labels[i] and weight / total >= MARGIN:
            out[i] = winner
    return out


def score(pred, truth_full, spf):
    scored = correct = 0
    t = truth_full[0][0]
    while t < truth_full[-1][1]:
        full = next((l for s, e, l in truth_full if s <= t < e), None)
        if full is not None and full != "X":
            idx = min(int(t / spf), len(pred) - 1)
            scored += 1
            correct += pred[idx] == full
        t += STEP
    return scored, correct


def main(rows_limit):
    session = ort.InferenceSession(os.path.join(HERE, "models", "btc_large_voca.onnx"))
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        labels = json.load(f)["labels"]
    log_trans = build_log_transitions(labels, None, 0.9)
    fam_matrix = family_matrix(labels)

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:rows_limit]

    variants = ["viterbi", "vote@0.85", "vote@0.90", "vote@0.95"]
    scored = {v: 0 for v in variants}
    correct = {v: 0 for v in variants}
    changed = {v: 0 for v in variants}
    done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        full_path = os.path.join(ROOT, "labs_full", row["id"] + ".lab")
        if not os.path.exists(audio_path) or not os.path.exists(full_path):
            continue
        truth_full = read_lab(full_path)
        wav = load_audio(audio_path)
        if len(wav) < 22050:
            continue
        features, spf = features_from_wav(wav)
        probs = combine([probs_pass(session, features, 0, 0),
                         probs_pass(session, features, T // 2, 0)])
        path = viterbi(np.log(probs + 1e-12), log_trans)
        base_labels = [labels[i] for i in path]

        preds = {"viterbi": base_labels}
        for th in (0.85, 0.90, 0.95):
            voted = vote(base_labels, probs, fam_matrix, th)
            preds[f"vote@{th:.2f}"] = voted
            changed[f"vote@{th:.2f}"] += sum(a != b for a, b in zip(voted, base_labels))
        for v in variants:
            s, c = score(preds[v], truth_full, spf)
            scored[v] += s
            correct[v] += c
        done += 1
        if done % 20 == 0:
            print(f"{done} songs: " + "  ".join(
                f"{v} {correct[v] / max(scored[v], 1):.2%}" for v in variants), flush=True)

    print(f"\n=== repetition voting (full-vocab accuracy), {done} benchmark songs ===")
    for v in variants:
        print(f"{v:10} {correct[v] / max(scored[v], 1):.2%}   frames changed: {changed[v]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
