"""Core DSP building blocks, grown lesson by lesson."""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

N_FFT = 2048  # frame length: 46 ms at 44100 Hz -> ~21.5 Hz frequency resolution
HOP = 512     # frame step: 12 ms -> 75% overlap


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Short-Time Fourier Transform (lesson 0).

    Returns complex matrix of shape (n_frames, n_fft // 2 + 1):
    one row per time frame, one column per frequency bin.
    """
    # Hann window: smooth bell curve, kills spectral leakage at frame edges.
    window = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack(
        [x[i * hop : i * hop + n_fft] * window for i in range(n_frames)]
    )
    return np.fft.rfft(frames, axis=1)


def to_db(spectrum: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    """Magnitude -> decibels relative to the loudest bin, clipped at floor_db."""
    magnitude = np.abs(spectrum)
    db = 20.0 * np.log10(magnitude + 1e-10)
    db -= db.max()
    return np.maximum(db, floor_db)


def spectral_flux(magnitude: np.ndarray, gamma: float = 100.0) -> np.ndarray:
    """Onset novelty function (lesson 1).

    For each frame: how much NEW energy appeared in the spectrum compared
    to the previous frame. Positive changes only — decaying (ringing)
    strings must not count, only fresh attacks.
    """
    # Log compression: quiet harmonics matter to the ear, boost them.
    compressed = np.log1p(gamma * magnitude)
    diff = np.diff(compressed, axis=0)
    flux = np.maximum(diff, 0.0).sum(axis=1)  # half-wave rectification
    flux = np.concatenate([[0.0], flux])      # keep same length as frames
    peak = flux.max()
    return flux / peak if peak > 0 else flux


def pick_onsets(
    novelty: np.ndarray,
    frame_rate: float,
    delta: float = 0.05,
    median_window_s: float = 0.4,
    min_gap_s: float = 0.05,
):
    """Turn a novelty curve into discrete onset frames.

    A fixed threshold fails when playing gets louder/quieter, so the
    threshold adapts: local median of the curve + delta. Peaks closer
    than min_gap_s are merged (a guitarist can't strum faster anyway).

    Returns (peak_frame_indices, threshold_curve).
    """
    window = max(3, int(median_window_s * frame_rate) | 1)  # odd size
    threshold = median_filter(novelty, size=window, mode="nearest") + delta
    distance = max(1, int(min_gap_s * frame_rate))
    peaks, _ = find_peaks(novelty, height=threshold, distance=distance)
    return peaks, threshold
