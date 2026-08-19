"""Consumer-mic augmentation for guitar recordings + cache builder.

Simulates what separates a phone/laptop take from GuitarSet's studio
condenser: low rolloff, midrange coloration, room reverb, background
noise, AGC-ish level mangling. Length is preserved exactly so the
clean cache's frame labels stay valid for the augmented features.

Builds datasets from GuitarSet audio:
    ml/cache_guitar_aug/<stem>__v{k}.npz   (features + labels)

Run:
    .venv/Scripts/python augment.py [variants=2]
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

from btc_features import features_from_wav, SAMPLE_RATE

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.normpath(os.path.join(HERE, "..", "datasets", "guitarset"))
CLEAN_CACHE = os.path.join(HERE, "cache")
AUG_CACHE = os.path.join(HERE, "cache_guitar_aug")


def peaking(center_hz, gain_db, q, sr):
    """RBJ cookbook peaking EQ -> second-order sections."""
    amp = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * center_hz / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * amp, -2 * np.cos(w0), 1 - alpha * amp])
    a = np.array([1 + alpha / amp, -2 * np.cos(w0), 1 - alpha / amp])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def pink_noise(n, rng):
    spectrum = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    spectrum /= np.sqrt(np.arange(len(spectrum)) + 1.0)
    noise = np.fft.irfft(spectrum, n)
    return noise / (np.std(noise) + 1e-9)


def augment(wav, rng, sr=SAMPLE_RATE):
    """One random consumer-mic rendition; output length == input length."""
    out = wav.astype(np.float64)
    n = len(out)

    # Cheap-mic frequency response: low rolloff + midrange peak/dip.
    sos_hp = butter(2, rng.uniform(80, 250), "highpass", fs=sr, output="sos")
    out = sosfilt(sos_hp, out)
    out = sosfilt(peaking(rng.uniform(1000, 4000), rng.uniform(-6, 6), 1.0, sr), out)
    if rng.random() < 0.5:
        sos_lp = butter(2, rng.uniform(5000, 9000), "lowpass", fs=sr, output="sos")
        out = sosfilt(sos_lp, out)

    # Small-room reverb: exponentially decaying noise IR.
    rt60 = rng.uniform(0.15, 0.5)
    ir_len = int(rt60 * sr)
    t = np.arange(ir_len) / sr
    ir = rng.standard_normal(ir_len) * np.exp(-6.91 * t / rt60)
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    out = out + rng.uniform(0.1, 0.35) * fftconvolve(out, ir)[:n]

    # Background noise at a random SNR.
    snr_db = rng.uniform(15, 35)
    signal_rms = np.sqrt((out ** 2).mean()) + 1e-9
    out = out + pink_noise(n, rng) * signal_rms * 10 ** (-snr_db / 20)

    # AGC-ish level: random target peak, occasional soft saturation.
    out = out / (np.abs(out).max() + 1e-9) * rng.uniform(0.3, 0.95)
    if rng.random() < 0.3:
        drive = rng.uniform(1.5, 3.0)
        out = np.tanh(out * drive) / np.tanh(drive)

    return out.astype(np.float32)


def main(variants):
    import librosa

    os.makedirs(AUG_CACHE, exist_ok=True)
    stems = sorted(name[:-4] for name in os.listdir(CLEAN_CACHE)
                   if name.endswith(".npz"))
    audio_dir = os.path.join(DATASET, "audio_mono-mic")
    for i, stem in enumerate(stems):
        labels = np.load(os.path.join(CLEAN_CACHE, stem + ".npz"))["labels"]
        wav = None
        for k in range(variants):
            out = os.path.join(AUG_CACHE, f"{stem}__v{k}.npz")
            if os.path.exists(out):
                continue
            if wav is None:
                wav, _ = librosa.load(os.path.join(audio_dir, stem + "_mic.wav"),
                                      sr=SAMPLE_RATE, mono=True)
            rng = np.random.default_rng(abs(hash((stem, k))) % 2 ** 32)
            features, _ = features_from_wav(augment(wav, rng))
            assert features.shape[0] == len(labels), stem
            np.savez_compressed(out, features=features, labels=labels)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(stems)}", flush=True)
    print("AUG CACHE DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
