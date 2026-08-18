"""Measure domain features that separate 'solo guitar' from 'full mix',
plus both models' confidence, over GuitarSet vs GTZAN.

Output: domain_probe.csv -> pick the auto-routing rule from real data.

Run:
    .venv/Scripts/python domain_probe.py
"""
import os

import librosa
import numpy as np
import onnxruntime as ort
from scipy.ndimage import median_filter

from btc_features import features_for, TIMESTEP

HERE = os.path.dirname(os.path.abspath(__file__))
GUITARSET = os.path.normpath(os.path.join(HERE, "..", "datasets", "guitarset", "audio_mono-mic"))
GTZAN = os.path.normpath(os.path.join(HERE, "..", "datasets", "gtzan", "genres"))

SR = 22050
N_FFT = 2048
HOP = 1024


def audio_stats(wav, sr):
    spec = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    total = spec.sum() + 1e-9

    low = spec[freqs < 70].sum() / total
    high = spec[freqs > 5000].sum() / total

    mag = np.sqrt(spec)
    harmonic = median_filter(mag, size=(1, 17))   # smooth along time
    percussive = median_filter(mag, size=(17, 1)) # smooth along frequency
    perc = mag[percussive > harmonic].sum() / (mag.sum() + 1e-9)
    return low, high, perc


def model_confidence(session, features):
    window = features[:TIMESTEP]
    if window.shape[0] < TIMESTEP:
        window = np.pad(window, ((0, TIMESTEP - window.shape[0]), (0, 0)))
    logits = session.run(None, {"features": window[None, ...]})[0][0]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return float(probs.max(axis=1).mean())


def main():
    base = ort.InferenceSession(os.path.join(HERE, "models", "btc_large_voca.onnx"))
    guitar = ort.InferenceSession(os.path.join(HERE, "models", "btc_guitar.onnx"))

    jobs = []
    files = sorted(f for f in os.listdir(GUITARSET) if f.endswith(".wav"))
    jobs += [("guitar", os.path.join(GUITARSET, f)) for f in files[::3]]
    for genre in sorted(os.listdir(GTZAN)):
        genre_dir = os.path.join(GTZAN, genre)
        if not os.path.isdir(genre_dir):
            continue
        wavs = sorted(f for f in os.listdir(genre_dir)
                      if f.endswith(".wav") and not f.startswith("._"))
        for f in wavs[:20]:
            jobs.append((f"mix:{genre}", os.path.join(genre_dir, f)))

    out = open(os.path.join(HERE, "domain_probe.csv"), "w")
    out.write("cls,file,low,high,perc,conf_base,conf_guitar\n")
    for n, (cls, path) in enumerate(jobs):
        try:
            wav, _ = librosa.load(path, sr=SR, mono=True, duration=28.0)
            low, high, perc = audio_stats(wav, SR)
            features, _ = features_for(path)
            cb = model_confidence(base, features)
            cg = model_confidence(guitar, features)
            out.write(f"{cls},{os.path.basename(path)},{low:.4f},{high:.4f},"
                      f"{perc:.4f},{cb:.4f},{cg:.4f}\n")
        except Exception as ex:
            print(f"skip {path}: {ex}", flush=True)
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(jobs)}", flush=True)
            out.flush()
    out.close()
    print("done -> domain_probe.csv", flush=True)


if __name__ == "__main__":
    main()
