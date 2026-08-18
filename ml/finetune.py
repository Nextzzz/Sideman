"""Fine-tune BTC (large voca) on GuitarSet comp takes.

Honest player split: players 00-03 train, 04-05 test (never seen).
Augmentation: pitch shift in CQT space (+-k semitones = roll of 2k bins)
with matching label transposition — multiplies 2 hours of data ~12x.

Run:
    .venv/Scripts/python finetune.py            # cache features + train
    .venv/Scripts/python finetune.py --epochs 30
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

if not hasattr(np, "int"):
    np.int = int  # noqa
if not hasattr(np, "float"):
    np.float = float  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "vendor", "BTC-ISMIR19")
sys.path.insert(0, REPO)

from btc_model import BTC_model  # noqa: E402
from utils.hparams import HParams  # noqa: E402

from btc_features import features_for, TIMESTEP  # noqa: E402

DATASET = os.path.normpath(os.path.join(HERE, "..", "datasets", "guitarset"))
CACHE = os.path.join(HERE, "cache")
MODELS = os.path.join(HERE, "models")

TRAIN_PLAYERS = ("00_", "01_", "02_", "03_")
TEST_PLAYERS = ("04_", "05_")

ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
QUALITIES = ["min", "maj", "dim", "aug", "min6", "maj6", "min7",
             "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]
X_IDX, N_IDX = 168, 169
LOG_FLOOR = float(np.log(1e-6))

MAJ_FAMILY = {1, 5, 8, 9}    # maj, maj6, maj7, 7
MIN_FAMILY = {0, 4, 6, 7}    # min, min6, min7, minmaj7


def label_to_idx(raw):
    if raw in ("N",):
        return N_IDX
    if raw in ("X",):
        return X_IDX
    root_part, _, quality = raw.partition(":")
    quality = quality.split("/")[0].split("(")[0]
    pc = ROOTS.get(root_part[0])
    if pc is None:
        return X_IDX
    for c in root_part[1:]:
        pc += 1 if c == "#" else -1 if c == "b" else 0
    pc %= 12

    if quality == "":
        quality = "maj"
    if quality not in QUALITIES:
        # Leadsheet oddities -> nearest vocabulary member.
        for base, mapped in (("minmaj", "minmaj7"), ("min9", "min7"), ("min", "min"),
                             ("maj9", "maj7"), ("maj", "maj"), ("hdim", "hdim7"),
                             ("dim", "dim"), ("sus4", "sus4"), ("sus2", "sus2"),
                             ("aug", "aug"), ("9", "7"), ("11", "7"), ("13", "7"),
                             ("6", "maj6"), ("5", "maj")):
            if quality.startswith(base):
                quality = mapped
                break
        else:
            return X_IDX
    return pc * 14 + QUALITIES.index(quality)


def truth_frames(jams_path, n_frames, spf):
    with open(jams_path) as f:
        jams = json.load(f)
    chosen = None
    for annotation in jams["annotations"]:
        if annotation["namespace"] != "chord":
            continue
        if not annotation["annotation_metadata"]["data_source"]:
            chosen = annotation
            break
        chosen = chosen or annotation
    segments = [(d["time"], d["time"] + d["duration"], label_to_idx(d["value"]))
                for d in chosen["data"]]

    labels = np.full(n_frames, N_IDX, dtype=np.int64)
    for i in range(n_frames):
        t = (i + 0.5) * spf
        for start, end, idx in segments:
            if start <= t < end:
                labels[i] = idx
                break
    return labels


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    jams_dir = os.path.join(DATASET, "annotation")
    audio_dir = os.path.join(DATASET, "audio_mono-mic")
    stems = sorted(f[:-5] for f in os.listdir(jams_dir)
                   if f.endswith(".jams") and "_comp" in f)
    for n, stem in enumerate(stems):
        out = os.path.join(CACHE, stem + ".npz")
        if os.path.exists(out):
            continue
        wav = os.path.join(audio_dir, stem + "_mic.wav")
        if not os.path.exists(wav):
            continue
        features, spf = features_for(wav)
        labels = truth_frames(os.path.join(jams_dir, stem + ".jams"),
                              features.shape[0], spf)
        np.savez_compressed(out, features=features, labels=labels)
        print(f"cached {n + 1}/{len(stems)}: {stem}", flush=True)


def load_windows(players):
    """Non-overlapping 108-frame windows: (features, labels, mask) arrays."""
    xs, ys, ms = [], [], []
    for name in sorted(os.listdir(CACHE)):
        if not name.endswith(".npz") or not name.startswith(players):
            continue
        data = np.load(os.path.join(CACHE, name))
        features, labels = data["features"], data["labels"]
        for start in range(0, len(features), TIMESTEP):
            chunk = features[start:start + TIMESTEP]
            lab = labels[start:start + TIMESTEP]
            pad = TIMESTEP - len(chunk)
            mask = np.ones(TIMESTEP, dtype=np.float32)
            if pad > 0:
                chunk = np.pad(chunk, ((0, pad), (0, 0)),
                               constant_values=LOG_FLOOR)
                lab = np.pad(lab, (0, pad), constant_values=N_IDX)
                mask[TIMESTEP - pad:] = 0
            xs.append(chunk)
            ys.append(lab)
            ms.append(mask)
    return (np.stack(xs).astype(np.float32),
            np.stack(ys),
            np.stack(ms))


def pitch_shift(features, labels, k):
    """+k semitones = +2k CQT bins; labels transpose roots by k."""
    if k == 0:
        return features, labels
    shifted = np.full_like(features, LOG_FLOOR)
    if k > 0:
        shifted[:, 2 * k:] = features[:, :-2 * k]
    else:
        shifted[:, :2 * k] = features[:, -2 * k:]
    new_labels = labels.copy()
    chord = labels < X_IDX
    new_labels[chord] = ((labels[chord] // 14 + k) % 12) * 14 + labels[chord] % 14
    return shifted, new_labels


def majmin_accuracy(pred, truth, mask):
    """Frame majmin accuracy over scoreable truth frames (mirrors C# eval)."""

    def to_family(idx):
        if idx == N_IDX:
            return -1                      # N
        if idx == X_IDX:
            return -2                      # excluded
        quality = idx % 14
        root = idx // 14
        if quality in MAJ_FAMILY:
            return root * 2
        if quality in MIN_FAMILY:
            return root * 2 + 1
        return -2

    scored = correct = 0
    for p, t, m in zip(pred.ravel(), truth.ravel(), mask.ravel()):
        if m == 0:
            continue
        tf = to_family(t)
        if tf == -2:
            continue
        scored += 1
        if to_family(p) == tf:
            correct += 1
    return correct / max(scored, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    build_cache()

    train_x, train_y, train_m = load_windows(TRAIN_PLAYERS)
    test_x, test_y, test_m = load_windows(TEST_PLAYERS)
    print(f"train windows: {len(train_x)}, test windows: {len(test_x)}", flush=True)

    config = HParams.load(os.path.join(REPO, "run_config.yaml"))
    config.feature["large_voca"] = True
    config.model["num_chords"] = 170
    model = BTC_model(config=config.model)
    checkpoint = torch.load(os.path.join(REPO, "test", "btc_model_large_voca.pt"),
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    mean, std = float(checkpoint["mean"]), float(checkpoint["std"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(7)

    def forward_logits(batch_x):
        hidden, _ = model.self_attn_layers((batch_x - mean) / std)
        return model.output_layer.output_projection(hidden)

    def evaluate():
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(test_x), 16):
                logits = forward_logits(torch.tensor(test_x[i:i + 16]))
                preds.append(logits.argmax(-1).numpy())
        model.train()
        pred = np.concatenate(preds)
        full = float(((pred == test_y) * test_m).sum() / test_m.sum())
        return full, majmin_accuracy(pred, test_y, test_m)

    full0, majmin0 = evaluate()
    print(f"baseline on test players: full-vocab {full0:.4f}, majmin {majmin0:.4f}", flush=True)

    best = majmin0
    since_best = 0
    order = np.arange(len(train_x))
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        total_loss = steps = 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            batch_x, batch_y, batch_m = [], [], []
            for j in idx:
                k = int(rng.integers(-5, 7))
                fx, fy = pitch_shift(train_x[j], train_y[j], k)
                batch_x.append(fx)
                batch_y.append(fy)
                batch_m.append(train_m[j])
            logits = forward_logits(torch.tensor(np.stack(batch_x)))
            loss = F.cross_entropy(
                logits.reshape(-1, 170),
                torch.tensor(np.stack(batch_y)).reshape(-1),
                reduction="none")
            loss = (loss * torch.tensor(np.stack(batch_m)).reshape(-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss)
            steps += 1

        full, majmin = evaluate()
        marker = ""
        if majmin > best:
            best = majmin
            since_best = 0
            os.makedirs(MODELS, exist_ok=True)
            torch.save({"model": model.state_dict(), "mean": mean, "std": std},
                       os.path.join(MODELS, "btc_guitar.pt"))
            marker = "  <-- saved"
        else:
            since_best += 1
        print(f"epoch {epoch}: loss {total_loss / steps:.4f}, "
              f"test full {full:.4f}, majmin {majmin:.4f}{marker}", flush=True)
        if since_best >= args.patience:
            print("early stop", flush=True)
            break

    print(f"best test majmin: {best:.4f} (baseline {majmin0:.4f})", flush=True)


if __name__ == "__main__":
    main()
