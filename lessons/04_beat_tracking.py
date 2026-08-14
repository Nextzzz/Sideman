"""Lesson 2: tempo estimation and beat tracking.

From onsets to PULSE: autocorrelation of the novelty curve gives the tempo,
dynamic programming (Ellis, 2007) places the actual beat grid. The rendered
click track is a metronome that follows YOU — the seed of the drummer.

Run:
    python lessons/04_beat_tracking.py --demo          # synthetic groove at 100 BPM
    python lessons/04_beat_tracking.py audio/take.wav  # your own recording
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

from sideman.dsp import (
    HOP, N_FFT, stft, spectral_flux, pick_onsets, estimate_tempo, track_beats,
)
from sideman.synth import karplus_strong, with_clicks

TIGHTNESS = 100.0  # rhythm rigidity: high = steady grid, low = follows rubato
TOLERANCE = 0.07   # demo scoring: beat within ±70 ms of truth counts as a hit


def make_demo(sr: int = 44100):
    """Quarter-note plucks at 100 BPM with human jitter + two eighth-note
    pickups (attacks that are NOT beats — the tracker must not be fooled)."""
    bpm_truth = 100.0
    period = 60.0 / bpm_truth
    rng = np.random.default_rng(42)

    beats = 0.5 + np.arange(7) * period
    beats += rng.uniform(-0.01, 0.01, len(beats))  # nobody plays like a machine
    eighths = np.array([beats[2] + period / 2, beats[4] + period / 2])
    onsets = np.sort(np.concatenate([beats, eighths]))

    freqs = [82.41, 110.0, 146.83, 196.0]  # E2 A2 D3 G3
    x = np.zeros(int(round((onsets[-1] + 1.2) * sr)))
    for k, t0 in enumerate(onsets):
        pluck = karplus_strong(freqs[k % 4], 0.9, sr, seed=k)
        i = int(round(t0 * sr))
        x[i : i + len(pluck)] += pluck[: len(x) - i]
    x *= 0.8 / np.max(np.abs(x))

    os.makedirs("audio", exist_ok=True)
    sf.write("audio/demo_groove.wav", x.astype(np.float32), sr)
    return "audio/demo_groove.wav", bpm_truth, beats


def main() -> None:
    truth_bpm, truth_beats = None, None
    if "--demo" in sys.argv:
        path, truth_bpm, truth_beats = make_demo()
        print(f"Generated {path}: {truth_bpm:.0f} BPM, "
              f"beats at {np.round(truth_beats, 2)}")
    elif len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        sys.exit("usage: python lessons/04_beat_tracking.py <file.wav> | --demo")

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    frame_rate = sr / HOP

    novelty = spectral_flux(np.abs(stft(x)))
    onset_peaks, _ = pick_onsets(novelty, frame_rate)
    bpm, bpm_axis, tempo_score = estimate_tempo(novelty, frame_rate)
    beat_frames = track_beats(novelty, frame_rate, bpm, tightness=TIGHTNESS)
    # Center-of-window timestamps, same convention as lesson 1.
    beat_times = (beat_frames * HOP + N_FFT // 2) / sr

    print(f"Estimated tempo: {bpm:.1f} BPM")
    print(f"{len(beat_times)} beats: "
          + ", ".join(f"{t:.2f}" for t in beat_times))

    if truth_beats is not None:
        print(f"  tempo: {bpm:.1f} vs truth {truth_bpm:.0f} "
              f"-> {'OK' if abs(bpm - truth_bpm) <= 4 else 'FAIL'}")
        hits, devs = 0, []
        for t in truth_beats:
            dev = float(np.min(np.abs(beat_times - t)))
            hit = dev <= TOLERANCE
            hits += hit
            devs.append(dev * 1000)
            print(f"  beat {t:.2f}s -> {'HIT' if hit else 'MISS'} "
                  f"(off by {dev * 1000:.0f} ms)")
        print(f"  score: {hits}/{len(truth_beats)} beats, "
              f"mean deviation {np.mean(devs):.0f} ms")

    # The metronome that follows you.
    os.makedirs("output", exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    click_path = f"output/04_{stem}_beats.wav"
    sf.write(click_path, with_clicks(x, sr, beat_times), sr)
    print(f"Wrote {click_path} — the ticks are BEATS, not every attack.")

    # Plots: waveform + beats, novelty + beats, tempo score.
    duration = len(x) / sr
    frame_t = (np.arange(len(novelty)) * HOP + N_FFT // 2) / sr

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    ax1.plot(np.arange(len(x)) / sr, x, linewidth=0.4)
    ax1.set_ylabel("amplitude")
    ax1.set_title(f"{os.path.basename(path)} — {bpm:.1f} BPM")

    ax2.plot(frame_t, novelty, label="novelty")
    onset_t = (onset_peaks * HOP + N_FFT // 2) / sr
    ax2.plot(onset_t, novelty[onset_peaks], "c.", markersize=8, label="onsets")
    ax2.set_ylabel("novelty")
    ax2.set_xlabel("time, s")
    ax2.legend(loc="upper right")

    for ax in (ax1, ax2):
        ax.set_xlim(0, duration)
        for t in beat_times:
            ax.axvline(t, color="red", linewidth=1.0, alpha=0.7)

    ax3.plot(bpm_axis, tempo_score)
    ax3.axvline(bpm, color="red", linestyle="--", label=f"pick: {bpm:.1f} BPM")
    if truth_bpm is not None:
        ax3.axvline(truth_bpm, color="green", alpha=0.5,
                    label=f"truth: {truth_bpm:.0f} BPM")
    ax3.set_xlabel("tempo hypothesis, BPM")
    ax3.set_ylabel("weighted autocorrelation")
    ax3.legend(loc="upper right")

    out_png = f"output/04_{stem}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
