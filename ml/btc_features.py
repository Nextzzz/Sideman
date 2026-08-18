"""Reference feature pipeline — an exact replica of BTC's
audio_file_to_features (10-second chunked CQT, log compression).
The future C# port must match THIS, bit for bit within tolerance.
"""
import librosa
import numpy as np

SAMPLE_RATE = 22050
N_BINS = 144
BINS_PER_OCTAVE = 24
HOP = 2048
INST_LEN = 10.0
TIMESTEP = 108


def features_for(audio_path):
    """Returns (features [frames, 144], seconds_per_frame)."""
    wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # BTC computes CQT in 10-second chunks and concatenates — replicate
    # exactly, chunk boundaries included.
    current = 0
    feature = None
    chunk = int(SAMPLE_RATE * INST_LEN)
    while len(wav) > current + chunk:
        tmp = librosa.cqt(
            wav[current:current + chunk], sr=sr,
            n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP)
        tmp = tmp[:, :TIMESTEP]  # 10 s -> exactly 108 frames
        feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
        current += chunk
    tmp = librosa.cqt(
        wav[current:], sr=sr,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP)
    feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)

    feature = np.log(np.abs(feature) + 1e-6)
    return feature.T.astype(np.float32), INST_LEN / TIMESTEP
