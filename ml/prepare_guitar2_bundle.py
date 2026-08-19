"""Pack windowed training data for the btc_guitar2 robustness fine-tune.

Turns feature caches into five compact float32 window packs so the GPU
bundle carries no raw caches:

    bundle_guitar2/data/guitar_train.npz       players 00-03, clean + aug
    bundle_guitar2/data/guitar_test_clean.npz  players 04-05, clean
    bundle_guitar2/data/guitar_test_aug.npz    players 04-05, aug
    bundle_guitar2/data/billboard_train.npz    subsample, anti-forgetting mix
    bundle_guitar2/data/billboard_test.npz     subsample, forgetting monitor

Run:
    .venv/Scripts/python prepare_guitar2_bundle.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bundle_guitar2", "data")

TIMESTEP = 108
N_IDX = 169
LOG_FLOOR = float(np.log(1e-6))

TRAIN_PLAYERS = ("00_", "01_", "02_", "03_")
TEST_PLAYERS = ("04_", "05_")


def windows_of(cache_dir, keep):
    xs, ys, ms = [], [], []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".npz") or not keep(name):
            continue
        data = np.load(os.path.join(cache_dir, name))
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
            ys.append(lab.astype(np.int64))
            ms.append(mask)
    return np.stack(xs), np.stack(ys), np.stack(ms)


def save(name, pack):
    x, y, m = pack
    np.savez_compressed(os.path.join(OUT, name), x=x, y=y, m=m)
    print(f"{name}: {len(x)} windows")


def main():
    os.makedirs(OUT, exist_ok=True)
    clean, aug, billboard = (os.path.join(HERE, d) for d in
                             ("cache", "cache_guitar_aug", "cache_billboard"))

    guitar_train = windows_of(clean, lambda n: n.startswith(TRAIN_PLAYERS))
    aug_train = windows_of(aug, lambda n: n.startswith(TRAIN_PLAYERS))
    combined = tuple(np.concatenate([a, b]) for a, b in zip(guitar_train, aug_train))
    save("guitar_train.npz", combined)
    save("guitar_test_clean.npz", windows_of(clean, lambda n: n.startswith(TEST_PLAYERS)))
    save("guitar_test_aug.npz", windows_of(aug, lambda n: n.startswith(TEST_PLAYERS)))

    # Billboard song-level split matches finetune_billboard.py: id%10>=8 test.
    bb_train = windows_of(billboard, lambda n: int(n[:-4]) % 10 < 8)
    bb_test = windows_of(billboard, lambda n: int(n[:-4]) % 10 >= 8)
    rng = np.random.default_rng(7)
    n_mix = min(len(bb_train[0]), int(1.5 * len(combined[0])))
    pick = rng.choice(len(bb_train[0]), n_mix, replace=False)
    save("billboard_train.npz", tuple(a[pick] for a in bb_train))
    pick_test = rng.choice(len(bb_test[0]), min(300, len(bb_test[0])), replace=False)
    save("billboard_test.npz", tuple(a[pick_test] for a in bb_test))


if __name__ == "__main__":
    main()
