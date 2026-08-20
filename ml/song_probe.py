"""Probe one song through the C# product pipeline with several models and
key-prior strengths, and tally how long given chord labels are shown.
Diagnoses cases like "all those D's should be Dm".

Run:
    .venv/Scripts/python song_probe.py <audio> [labels=D,Dm,D7,Dm7,A,E,F]
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
MODELS = ["btc_large_voca", "btc_self", "btc_guitar2"]
PRIORS = ["0", "0.5", "1.5"]
LINE = re.compile(r"\s+(\d+):(\d+[,.]\d)\s+-\s+(\d+):(\d+[,.]\d)\s+(\S+)")


def analyze(audio, model, prior, ensemble=None):
    cmd = ["dotnet", "run", "--project", "src/Strunika.Cli", "--no-build", "--",
           "analyze", audio, f"--neural=ml/models/{model}.onnx",
           f"--keyprior={prior}", "--ovl"]
    if ensemble:
        cmd.append(f"--ens=ml/models/{ensemble}.onnx")
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    key, totals = "?", {}
    for line in out.splitlines():
        if line.startswith("key:"):
            key = line.split(":", 1)[1].strip()
        m = LINE.match(line)
        if m:
            start = int(m[1]) * 60 + float(m[2].replace(",", "."))
            end = int(m[3]) * 60 + float(m[4].replace(",", "."))
            totals[m[5]] = totals.get(m[5], 0.0) + end - start
    return key, totals


def main(audio, labels):
    print(f"{'model':18} {'prior':5} {'key':5} " + " ".join(f"{l:>5}" for l in labels))
    runs = [(m, p, None) for m in MODELS for p in PRIORS]
    runs.append(("btc_large_voca", "0.5", "btc_self"))
    for model, prior, ens in runs:
        key, totals = analyze(audio, model, prior, ens)
        name = model.replace("btc_", "") + (f"+{ens.replace('btc_', '')}" if ens else "")
        print(f"{name:18} {prior:5} {key:5} "
              + " ".join(f"{totals.get(l, 0):5.0f}" for l in labels), flush=True)


if __name__ == "__main__":
    labels = sys.argv[2].split(",") if len(sys.argv) > 2 else ["D", "Dm", "D7", "Dm7", "A", "E", "F"]
    main(sys.argv[1], labels)
