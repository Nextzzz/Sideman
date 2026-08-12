"""Lesson 0: record your own guitar from the default microphone.

Run:
    python lessons/01_record.py [seconds]     # default 10
"""
import os
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 44100
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

os.makedirs("audio", exist_ok=True)

print(f"Recording {DUR:.0f}s from: {sd.query_devices(kind='input')['name']}")
for i in (3, 2, 1):
    print(f"  {i}...")
    time.sleep(1)
print("GO — play something!")

x = sd.rec(int(DUR * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()
x = x[:, 0]

peak = float(np.max(np.abs(x)))
path = time.strftime("audio/take_%Y%m%d_%H%M%S.wav")
sf.write(path, x, SR)

print(f"Saved {path} | peak amplitude {peak:.2f}")
if peak > 0.95:
    print("WARNING: clipping — lower the input gain and re-record.")
elif peak < 0.1:
    print("WARNING: very quiet — raise the input gain or get closer to the mic.")
