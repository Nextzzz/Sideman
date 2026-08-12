"""Tiny synthesis helpers: test signals with known ground truth and audible checks."""
import numpy as np


def karplus_strong(
    freq: float, dur: float, sr: int = 44100, decay: float = 0.996, seed: int | None = None
) -> np.ndarray:
    """Plucked-string synthesis (Karplus-Strong, 1983).

    A burst of noise circulates through a delay line of one period length;
    averaging two neighbours on every pass acts as a low-pass filter, so
    the noise quickly settles into a decaying harmonic tone — remarkably
    close to a plucked guitar string.
    """
    rng = np.random.default_rng(seed)
    period = int(round(sr / freq))
    buf = rng.uniform(-1.0, 1.0, period)
    out = np.empty(int(dur * sr))
    for i in range(len(out)):
        out[i] = buf[i % period]
        buf[i % period] = decay * 0.5 * (buf[i % period] + buf[(i + 1) % period])
    # Release fade: a raw cutoff is a step discontinuity = broadband click,
    # and the onset detector would (rightly!) fire on it. Strings don't
    # stop instantly — neither should we.
    fade = min(int(0.1 * sr), len(out))
    out[-fade:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade)))
    return out * 0.7


def click(sr: int = 44100, freq: float = 1500.0, dur: float = 0.03) -> np.ndarray:
    """Short metronome-like tick for audible verification."""
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 90.0)


def with_clicks(x: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    """Mix a tick into a copy of x at each time (seconds) — trust your ears."""
    y = x.copy()
    c = click(sr)
    for t in times:
        i = int(t * sr)
        if i >= len(y):
            continue
        j = min(len(y), i + len(c))
        y[i:j] += c[: j - i]
    peak = np.max(np.abs(y))
    return y * (0.9 / peak) if peak > 0.9 else y
