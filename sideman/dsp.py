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


def estimate_tempo(
    novelty: np.ndarray,
    frame_rate: float,
    bpm_min: float = 40.0,
    bpm_max: float = 200.0,
    prior_bpm: float = 120.0,
    prior_octaves: float = 1.0,
):
    """Tempo estimation via autocorrelation of the novelty curve (lesson 2).

    If attacks repeat every T seconds, the novelty curve correlates with
    itself shifted by T (and 2T, 3T...). The strongest lag in the plausible
    BPM range is the beat period. A soft log-Gaussian prior around
    prior_bpm resolves the octave ambiguity (100 vs 50 vs 200 BPM) the
    same way humans default to a comfortable clapping speed.

    Returns (bpm, bpm_axis, score_axis) — the axes are for plotting.
    """
    n = len(novelty)
    centered = novelty - novelty.mean()
    ac = np.correlate(centered, centered, mode="full")[n - 1 :]
    if ac[0] > 0:
        ac = ac / ac[0]

    lags = np.arange(1, n)
    bpms = 60.0 * frame_rate / lags
    valid = (bpms >= bpm_min) & (bpms <= bpm_max)
    if not valid.any():
        return prior_bpm, np.array([]), np.array([])

    weight = np.exp(-0.5 * (np.log2(bpms[valid] / prior_bpm) / prior_octaves) ** 2)
    score = ac[1:][valid] * weight
    bpm = float(bpms[valid][np.argmax(score)])

    order = np.argsort(bpms[valid])
    return bpm, bpms[valid][order], score[order]


def track_beats(
    novelty: np.ndarray,
    frame_rate: float,
    bpm: float,
    tightness: float = 100.0,
) -> np.ndarray:
    """Beat tracking by dynamic programming (Ellis, 2007) — lesson 2.

    Greedy "put a beat on every strong peak" breaks on syncopation and
    missed beats. Instead we score every possible CHAIN of beats:
    reward = novelty at each beat, penalty = -tightness * ln(gap/period)^2
    for gaps deviating from the beat period, and let DP find the best
    chain globally. The log makes "twice too fast" and "twice too slow"
    equally wrong.

    Returns frame indices of beats.
    """
    period = 60.0 * frame_rate / bpm
    n = len(novelty)
    # Candidate gaps to the previous beat: half to double the period.
    gaps = np.arange(int(round(period / 2)), int(round(period * 2)) + 1)
    penalty = -tightness * np.log(gaps / period) ** 2

    cumscore = np.zeros(n)
    backlink = np.full(n, -1, dtype=int)
    for i in range(n):
        prev = i - gaps
        ok = prev >= 0
        if not ok.any():
            cumscore[i] = novelty[i]
            continue
        vals = cumscore[prev[ok]] + penalty[ok]
        j = int(np.argmax(vals))
        if vals[j] > 0:  # extending a chain beats starting fresh
            cumscore[i] = novelty[i] + vals[j]
            backlink[i] = int(prev[ok][j])
        else:
            cumscore[i] = novelty[i]

    # Backtrack from the strongest chain ending near the end of the signal.
    tail = max(1, int(round(period)))
    end = n - tail + int(np.argmax(cumscore[-tail:]))
    chain = [end]
    while backlink[chain[-1]] >= 0:
        chain.append(backlink[chain[-1]])
    beats = np.array(chain[::-1])

    # Trim "beats" the DP placed in leading/trailing near-silence.
    strong = novelty[beats] >= 0.05
    if strong.any():
        beats = beats[np.argmax(strong) : len(strong) - np.argmax(strong[::-1])]
    return beats
