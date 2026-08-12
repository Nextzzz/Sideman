# sideman

> *sideman (n.) — a professional musician hired to accompany a soloist or bandleader.*

An AI band member that **listens to you play guitar and follows you in real time** — matching your tempo, harmony and dynamics like a live rhythm section, instead of forcing you to follow a static backing track.

**Status: work in progress.** Currently building the analysis foundation (offline audio analysis → real-time tracking → adaptive accompaniment). This repository doubles as a learning journal: every algorithm is first implemented by hand before reaching for a library.

## Why

- Backing tracks don't listen. Every existing tool (Band-in-a-Box, iReal Pro, Moises AI Studio) either plays at a fixed tempo or generates parts *after* you record.
- A real band member doesn't react — they **predict**. The core of this project is a predictive model of the player (tempo phase-locking + chord anticipation), with generation scheduled ahead of time and gently corrected.
- 90% of new guitarists quit within a year (Fender). A band that is always ready to play with you — and never plays too fast — is a retention machine.

## Architecture (target)

```
mic → ring buffer → [analysis: onsets / beats / chroma→chords]
                  → [musician model: tempo PLL + chord prediction]
                  → [accompaniment: style patterns scheduled 1-2 beats ahead]
                  → MIDI → synth → speakers
```

## Repository layout

| Path | What |
|---|---|
| `lessons/` | Numbered, runnable scripts — one concept each, hand-rolled DSP first |
| `docs/` | Theory notes for each lesson |
| `audio/` | Local recordings (not committed) |
| `output/` | Generated plots (not committed) |

See [ROADMAP.md](ROADMAP.md) for the build plan.

## Setup

```
pip install -r requirements.txt
python lessons\00_check_setup.py
```

Lesson scripts can be run from any directory — they anchor themselves to the repo root. A virtualenv works too, if you prefer one:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python lessons\00_check_setup.py
```
