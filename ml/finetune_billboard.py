"""Fine-tune BTC (large voca) on the McGill Billboard cache — full mixes.

Self-contained for a rented GPU box: needs only torch + numpy, the
vendor/BTC-ISMIR19 code+checkpoint and the cache_billboard/ directory
next to this script. Song-level split: ids ending in 8/9 are the test set.

Run (GPU or CPU):
    python finetune_billboard.py [--epochs 40] [--batch 32]
"""
import argparse
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

CACHE = os.path.join(HERE, "cache_billboard")
TIMESTEP = 108
X_IDX, N_IDX = 168, 169
LOG_FLOOR = float(np.log(1e-6))
MAJ_FAMILY = {1, 5, 8, 9}
MIN_FAMILY = {0, 4, 6, 7}


def load_windows(test_split):
    xs, ys, ms = [], [], []
    for name in sorted(os.listdir(CACHE)):
        if not name.endswith(".npz"):
            continue
        is_test = int(name[:-4]) % 10 >= 8
        if is_test != test_split:
            continue
        data = np.load(os.path.join(CACHE, name))
        features, labels = data["features"], data["labels"]
        for start in range(0, len(features), TIMESTEP):
            chunk = features[start:start + TIMESTEP]
            lab = labels[start:start + TIMESTEP]
            pad = TIMESTEP - len(chunk)
            mask = np.ones(TIMESTEP, dtype=np.float32)
            if pad > 0:
                chunk = np.pad(chunk, ((0, pad), (0, 0)), constant_values=LOG_FLOOR)
                lab = np.pad(lab, (0, pad), constant_values=N_IDX)
                mask[TIMESTEP - pad:] = 0
            xs.append(chunk.astype(np.float32))
            ys.append(lab)
            ms.append(mask)
    return np.stack(xs), np.stack(ys), np.stack(ms)


def pitch_shift(features, labels, k):
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


def to_family(idx):
    if idx == N_IDX:
        return -1
    if idx == X_IDX:
        return -2
    quality = idx % 14
    if quality in MAJ_FAMILY:
        return (idx // 14) * 2
    if quality in MIN_FAMILY:
        return (idx // 14) * 2 + 1
    return -2


def majmin_accuracy(pred, truth, mask):
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
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_x, train_y, train_m = load_windows(test_split=False)
    test_x, test_y, test_m = load_windows(test_split=True)
    print(f"train windows: {len(train_x)}, test windows: {len(test_x)}", flush=True)

    config = HParams.load(os.path.join(REPO, "run_config.yaml"))
    config.feature["large_voca"] = True
    config.model["num_chords"] = 170
    model = BTC_model(config=config.model).to(device)
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
            for i in range(0, len(test_x), 64):
                logits = forward_logits(
                    torch.tensor(test_x[i:i + 64]).to(device))
                preds.append(logits.argmax(-1).cpu().numpy())
        model.train()
        pred = np.concatenate(preds)
        full = float(((pred == test_y) * test_m).sum() / test_m.sum())
        return full, majmin_accuracy(pred, test_y, test_m)

    full0, majmin0 = evaluate()
    print(f"baseline on test songs: full {full0:.4f}, majmin {majmin0:.4f}", flush=True)

    best = majmin0
    since_best = 0
    order = np.arange(len(train_x))
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        total_loss = steps = 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            bx, by, bm = [], [], []
            for j in idx:
                k = int(rng.integers(-5, 7))
                fx, fy = pitch_shift(train_x[j], train_y[j], k)
                bx.append(fx)
                by.append(fy)
                bm.append(train_m[j])
            logits = forward_logits(torch.tensor(np.stack(bx)).to(device))
            loss = F.cross_entropy(
                logits.reshape(-1, 170),
                torch.tensor(np.stack(by)).reshape(-1).to(device),
                reduction="none")
            loss = (loss * torch.tensor(np.stack(bm)).reshape(-1).to(device)).mean()
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
            torch.save({"model": model.state_dict(), "mean": mean, "std": std},
                       os.path.join(HERE, "btc_mix.pt"))
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
