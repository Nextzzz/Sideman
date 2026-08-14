# Roadmap

## Phase 1 — Analysis core (C#) ✅
- [x] FFT/STFT, chroma with semitone-proximity weighting and bass-note detection
- [x] Chord recognition: harmonic templates + anti-third + Viterbi (24 triads + N)
- [x] Tuner: YIN pitch detection (±2 cents)
- [x] Rhythm: spectral flux onsets, autocorrelation tempo, Ellis DP beats
- [x] Streaming (causal) chord detector for live playing
- [x] 30 NUnit tests on synthetic ground truth (clean + noisy progressions)

## Phase 2 — Desktop app ✅ (first cut)
- [x] WPF: tuner tab, live chords tab, song analysis tab
- [x] Inputs: audio file (wav/mp3/m4a), mic recording, YouTube link
- [x] CLI: `analyze`, `demo`

## Phase 3 — Real-world calibration ⏳
- [ ] Record real guitar takes (several guitars/rooms), build a labeled set
- [ ] Evaluation harness: accuracy vs .lab annotations, parameter sweeps
- [ ] Tune: silence gate, no-chord floor, self-transition, bass weighting
- [ ] Tuning-offset estimation (guitars not at A440)

## Phase 4 — Feature depth
- [ ] Chord vocabulary: dominant 7 / maj7 / m7 / sus2 / sus4 / power chords
- [ ] Chord diagrams (fingerings) in the app
- [ ] Beat-synchronized chord output (chords snapped to the beat grid)
- [ ] Monophonic riff transcription (tab for single-note lines)
- [ ] Capo detection / transposition helper

## Phase 5 — Mobile (.NET MAUI)
- [ ] Extract ViewModels to a shared project
- [ ] Platform audio capture (AVAudioEngine / AudioRecord bindings)
- [ ] Allocation-free audio path in Core (mobile GC discipline)

## Phase 6 — The band member (the big goal)
- [ ] Tempo phase-locking (PLL) — accompaniment that follows live tempo
- [ ] Chord prediction (progression language model)
- [ ] Drum/bass pattern engine scheduled ahead of predicted beats
- [ ] AI generation experiments (ONNX on-device / server)
