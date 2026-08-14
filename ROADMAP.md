# Roadmap

Each phase ends with something runnable and demoable.

## Phase 0 — Digital audio fundamentals ✅
- [x] Project scaffold, environment
- [x] Sine wave, samples, sample rate (lesson 00)
- [x] Record own guitar from mic (lesson 01)
- [x] Hand-rolled STFT + spectrogram of own playing (lesson 02)

## Phase 1 — Offline analyzer (WAV in → music out) ⏳
- [x] Onset detection via spectral flux, by hand (lesson 03)
- [x] Tempo estimation + beat tracking, Ellis DP (lesson 04)
- [ ] Compare hand-rolled tracker against a library on real recordings
- [ ] Chroma features by hand
- [ ] Chord recognition: chroma → template matching → HMM/Viterbi smoothing
- [ ] CLI: `analyze take.wav` → tempo, beat grid, chord timeline

## Phase 2 — Real-time dashboard
- [ ] Audio callback + ring buffer, latency measurement
- [ ] Streaming onset/beat tracking
- [ ] Streaming chord estimation
- [ ] Live display: current tempo, phase, chord

## Phase 3 — The drummer
- [ ] MIDI out → synth (FluidSynth / sampled drums)
- [ ] Scheduler: events placed ahead of predicted beats
- [ ] Tempo phase-locking (PLL): follow gradual speed-ups/slow-downs
- [ ] Confidence rules: simplify pattern when tracking is uncertain

## Phase 4 — The bassist
- [ ] Chord progression corpus → Markov model of "what comes next"
- [ ] Bass lines from predicted harmony (root/fifth/walking patterns)
- [ ] Dynamics following (velocity tracks player's energy)

## Phase 5 — Demo & polish
- [ ] 90-second demo video: play guitar, band joins in
- [ ] README with architecture + lessons learned
- [ ] Pitch one-pager

## Later / product track
- [ ] Port real-time engine to .NET (ONNX Runtime + NAudio/ASIO) or web
- [ ] Style library (rock/blues/funk grooves)
- [ ] ML generation experiments (ReaLchords-style)
