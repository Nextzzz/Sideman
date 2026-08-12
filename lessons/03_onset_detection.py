"""Lesson 1: onset detection via spectral flux.

An onset is the moment a note/chord attack begins. We detect attacks by
measuring how much NEW energy appears in the spectrum between consecutive
STFT frames (spectral flux), then peak-picking over an adaptive threshold.

Run:
    python lessons/03_onset_detection.py --demo          # synthetic plucks, known truth
    python lessons/03_onset_detection.py audio/take.wav  # your own recording
"""
import os
import sys

# Work from the repo root and find the sideman package without installation.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sideman.dsp import HOP, N_FFT, stft, to_db, spectral_flux, pick_onsets
from sideman.synth import karplus_strong, with_clicks

DELTA = 0.05      # peak must rise this far above the local median — tune me!
MIN_GAP_S = 0.05  # two attacks can't be closer than 50 ms
FMAX = 4000
TOLERANCE = 0.05  # demo scoring: detection within ±50 ms of truth counts as a hit


def make_demo(sr: int = 44100):
    """Karplus-Strong plucks at known times = free ground truth."""
    truth = [0.50, 1.10, 1.55, 2.20, 2.65, 3.40]
    freqs = [82.41, 110.0, 146.83, 196.0, 110.0, 82.41]  # E2 A2 D3 G3 A2 E2
    x = np.zeros(int(round(4.7 * sr)))
    for t0, f in zip(truth, freqs):
        pluck = karplus_strong(f, 1.2, sr, seed=int(f))
        i = int(round(t0 * sr))
        x[i : i + len(pluck)] += pluck[: len(x) - i]
    x *= 0.8 / np.max(np.abs(x))

    os.makedirs("audio", exist_ok=True)
    sf.write("audio/demo_plucks.wav", x.astype(np.float32), sr)
    return "audio/demo_plucks.wav", truth


def main() -> None:
    truth = None
    if "--demo" in sys.argv:
        path, truth = make_demo()
        print(f"Generated {path} with plucks at: {truth}")
    elif len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        sys.exit("usage: python lessons/03_onset_detection.py <file.wav> | --demo")

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    frame_rate = sr / HOP

    magnitude = np.abs(stft(x))
    novelty = spectral_flux(magnitude)
    peaks, threshold = pick_onsets(
        novelty, frame_rate, delta=DELTA, min_gap_s=MIN_GAP_S
    )
    onset_times = peaks * HOP / sr

    print(f"{len(peaks)} onsets detected: "
          + ", ".join(f"{t:.2f}" for t in onset_times))

    if truth is not None:
        hits = 0
        for t in truth:
            hit = bool(np.any(np.abs(onset_times - t) <= TOLERANCE))
            hits += hit
            print(f"  truth {t:.2f}s -> {'HIT' if hit else 'MISS'}")
        false_alarms = [
            f"{d:.2f}" for d in onset_times
            if all(abs(d - t) > TOLERANCE for t in truth)
        ]
        print(f"  score: {hits}/{len(truth)} hits, "
              f"false alarms: {false_alarms or 'none'}")

    # Audible check: original + tick at every detected onset.
    os.makedirs("output", exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    click_path = f"output/03_{stem}_clicks.wav"
    sf.write(click_path, with_clicks(x, sr, onset_times), sr)
    print(f"Wrote {click_path} — listen: ticks must land exactly on your hits.")

    # Visual check: waveform, spectrogram and novelty, onsets everywhere.
    duration = len(x) / sr
    frame_t = np.arange(len(novelty)) / frame_rate
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    keep = freqs <= FMAX

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 1]},
    )
    ax1.plot(np.arange(len(x)) / sr, x, linewidth=0.4)
    ax1.set_ylabel("amplitude")
    ax1.set_title(os.path.basename(path))

    ax2.imshow(
        to_db(magnitude).T[keep], origin="lower", aspect="auto",
        extent=[0, duration, 0, FMAX], cmap="magma", vmin=-80, vmax=0,
    )
    ax2.set_ylabel("frequency, Hz")

    ax3.plot(frame_t, novelty, label="spectral flux")
    ax3.plot(frame_t, threshold, "--", label="adaptive threshold")
    ax3.plot(onset_times, novelty[peaks], "rx", markersize=9, label="onsets")
    ax3.set_xlabel("time, s")
    ax3.set_ylabel("novelty")
    ax3.legend(loc="upper right")

    for ax in (ax1, ax2):
        for t in onset_times:
            ax.axvline(t, color="cyan", linestyle="--", linewidth=0.8, alpha=0.8)
    if truth is not None:
        for t in truth:
            ax3.axvline(t, color="green", linewidth=0.8, alpha=0.5)

    out_png = f"output/03_{stem}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
