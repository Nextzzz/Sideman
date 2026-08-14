# Sideman

> *sideman (n.) — a professional musician hired to accompany a soloist or bandleader.*

Chord and rhythm analysis for guitarists: a tuner, chord recognition for any
audio (file / recording / YouTube link), and **live chord detection while you
play**. Desktop app now; the analysis core is written to be ported to mobile
(.NET MAUI) unchanged. Future direction: an AI band member that follows your
playing in real time.

## Features

- **Tuner** — YIN pitch detection, ±2 cents on synthetic references.
- **Song analysis** — open wav/mp3/m4a, record from the mic, or paste a
  YouTube link: chord timeline (24 major/minor triads + "no chord"),
  tempo and beat grid.
- **Live chords** — play into the mic and see the chord in ~150 ms:
  causal Viterbi filtering over the same emission model as offline analysis.

## Architecture

```
Sideman.Core   pure C#, zero dependencies — DSP + analysis (portable to mobile)
Sideman.Media  desktop I/O: NAudio decode/capture, YouTube audio download
Sideman.App    WPF desktop app (CommunityToolkit.Mvvm)
Sideman.Cli    command-line analysis & calibration harness
tests/         NUnit; synthetic Karplus-Strong fixtures with known ground truth
```

Analysis pipeline: STFT → semitone-weighted log chroma + bass-note detection
→ harmonic-aware chord templates (with anti-third discrimination) → Viterbi
smoothing → segments. Rhythm: spectral flux → autocorrelation tempo →
dynamic-programming beat tracking (Ellis 2007).

## Build & run

```
dotnet test tests/Sideman.Core.Tests          # 30 tests
dotnet run --project src/Sideman.Cli -- demo  # synthesize + analyze a progression
dotnet run --project src/Sideman.Cli -- analyze path/to/song.mp3
# desktop app:
dotnet run --project src/Sideman.App
```

## Status

Working: tuner, offline chord/tempo analysis, live detection, file/record/
YouTube input, calibrated against synthetic strummed progressions (clean and
noisy). In progress: calibration on real guitar recordings, richer chord
vocabulary (7ths, sus), chord diagrams. See [ROADMAP.md](ROADMAP.md).
