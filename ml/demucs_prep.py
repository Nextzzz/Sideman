"""Source-separation preprocessing for the HookTheory benchmark A/B.

For each benchmark song: htdemucs 4-stem separation, then keep the
harmonic content (bass + other) and drop vocals + drums. Output goes
to datasets/hooktheory/audio_sep/<id>.wav (22050 mono) so
hooktheory_eval.py can score the same songs with and without
separation.

Thread cap keeps the machine usable while it grinds (~2 min/song CPU).

Run:
    .venv/Scripts/python demucs_prep.py [limit]
"""
import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TORCH_HOME",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   ".cache_torch"))

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "datasets", "hooktheory"))
OUT_DIR = os.path.join(ROOT, "audio_sep")
TOOLS = os.path.join(os.environ["LOCALAPPDATA"], "Strunika", "tools")
FFMPEG = os.path.join(TOOLS, "ffmpeg.exe")
OUT_SR = 22050


def load_stereo_44k(path):
    import subprocess
    raw = subprocess.run(
        [FFMPEG, "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "2",
         "-ar", "44100", "-"],
        capture_output=True, timeout=300).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T
    return torch.tensor(audio.copy())


def save_mono_22k(path, wav_44k_stereo):
    import subprocess
    mono = wav_44k_stereo.mean(dim=0).numpy().astype(np.float32)
    subprocess.run(
        [FFMPEG, "-v", "quiet", "-y", "-f", "f32le", "-ac", "1",
         "-ar", "44100", "-i", "-", "-ar", str(OUT_SR), path],
        input=mono.tobytes(), timeout=300, check=True)


def main(limit):
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    model = get_model("htdemucs")
    model.eval()
    keep = [model.sources.index("bass"), model.sources.index("other")]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:limit]

    done = skipped = 0
    for row in rows:
        src = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        dst = os.path.join(OUT_DIR, row["id"] + ".wav")
        if not os.path.exists(src) or os.path.exists(dst):
            skipped += 1
            continue
        wav = load_stereo_44k(src)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.no_grad():
            stems = apply_model(model, wav[None], device="cpu",
                                progress=False)[0]
        stems = stems * (ref.std() + 1e-8) + ref.mean()
        save_mono_22k(dst, stems[keep].sum(dim=0))
        done += 1
        print(f"{done + skipped}/{len(rows)}: {row['artist']}/{row['song']}",
              flush=True)
    print(f"SEPARATION DONE: {done} new, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9)
