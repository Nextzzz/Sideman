"""Lesson 0: verify the audio stack and meet your first samples.

Run:
    python lessons/00_check_setup.py          # headless: saves wav + png
    python lessons/00_check_setup.py --play   # also plays the tone
"""
import os
import sys

import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 44100      # samples per second
FREQ = 440.0    # A4, concert pitch
DUR = 2.0       # seconds

os.makedirs("output", exist_ok=True)
os.makedirs("audio", exist_ok=True)

print("=== Audio devices ===")
print(sd.query_devices())
print()

# Sound is just numbers: 44100 measurements per second of a moving membrane.
t = np.arange(int(SR * DUR)) / SR
x = 0.3 * np.sin(2 * np.pi * FREQ * t)

sf.write("audio/sine_440.wav", x.astype(np.float32), SR)
print(f"Wrote audio/sine_440.wav ({len(x)} samples = {DUR}s at {SR} Hz)")

# Zoom into the first 5 ms — each dot is one sample.
n = int(0.005 * SR)
plt.figure(figsize=(10, 4))
plt.plot(t[:n] * 1000, x[:n], marker="o", markersize=3)
plt.title(f"{FREQ:.0f} Hz sine, first 5 ms — {n} samples")
plt.xlabel("time, ms")
plt.ylabel("amplitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/00_sine_samples.png", dpi=120)
print("Wrote output/00_sine_samples.png")

if "--play" in sys.argv:
    print("Playing A4...")
    sd.play(x, SR)
    sd.wait()

print("OK")
