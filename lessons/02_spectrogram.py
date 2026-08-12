"""Lesson 0: waveform + spectrogram with a hand-rolled STFT.

No librosa here on purpose: the STFT below is the foundation of everything
this project will do, so we build it from numpy and understand every line.

Run:
    python lessons/02_spectrogram.py audio/take_XXXX.wav
"""
import os
import sys

import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_FFT = 2048   # frame length: 2048/44100 = 46 ms -> frequency resolution ~21.5 Hz
HOP = 512      # frame step: 12 ms -> 75% overlap between frames
FMAX = 4000    # guitar fundamentals + first harmonics live below this


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Short-Time Fourier Transform, the honest way.

    Returns complex matrix of shape (n_frames, n_fft // 2 + 1):
    one row per time frame, one column per frequency bin.
    """
    # Hann window: smooth bell curve, kills spectral leakage at frame edges.
    window = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack(
        [x[i * hop : i * hop + n_fft] * window for i in range(n_frames)]
    )
    # rfft: FFT for real input — only non-negative frequencies are returned.
    return np.fft.rfft(frames, axis=1)


def to_db(spectrum: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    """Magnitude -> decibels relative to the loudest bin, clipped at floor_db."""
    magnitude = np.abs(spectrum)
    db = 20.0 * np.log10(magnitude + 1e-10)
    db -= db.max()
    return np.maximum(db, floor_db)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python lessons/02_spectrogram.py <file.wav>")
    path = sys.argv[1]

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)  # mono
    duration = len(x) / sr
    print(f"{path}: {duration:.1f}s at {sr} Hz, {len(x)} samples")

    spec_db = to_db(stft(x)).T  # -> (freq_bins, frames) for plotting

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    keep = freqs <= FMAX

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )

    t = np.arange(len(x)) / sr
    ax1.plot(t, x, linewidth=0.4)
    ax1.set_ylabel("amplitude")
    ax1.set_title(os.path.basename(path))

    img = ax2.imshow(
        spec_db[keep],
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, FMAX],
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    ax2.set_xlabel("time, s")
    ax2.set_ylabel("frequency, Hz")
    fig.colorbar(img, ax=ax2, label="dB")

    os.makedirs("output", exist_ok=True)
    out = "output/02_" + os.path.splitext(os.path.basename(path))[0] + ".png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Wrote {out}")
    print("Look for: vertical columns = attacks, horizontal lines = ringing "
          "strings, ladders above them = harmonics.")


if __name__ == "__main__":
    main()
