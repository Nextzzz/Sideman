"""Lesson 0: waveform + spectrogram with a hand-rolled STFT.

No librosa here on purpose: the STFT (see sideman/dsp.py) is the foundation
of everything this project will do, so we build it from numpy and
understand every line.

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

from sideman.dsp import HOP, N_FFT, stft, to_db

FMAX = 4000    # guitar fundamentals + first harmonics live below this


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
