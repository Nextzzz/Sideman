"""One benchmark harness for model/config comparisons.

Scores any combination on a HookTheory pool (TEST rows or the permanent
VALID pool), majmin + full-vocabulary, with the product-relevant knobs:

    --models base,guitar2        single models (comma list; names map to
                                 btc_large_voca / btc_guitar2 / any ml/models/*.onnx)
    --ens base+guitar2           probability-averaged ensemble(s)
    --ovl                        overlapping windows (second half-offset pass)
    --viterbi                    uniform Viterbi (stay 0.9) instead of argmax
    --rows N                     first N rows of sample.csv (default all)
    --subset valid               use datasets/hooktheory/valid (else TEST pool)

Example — the shipped file route vs a plain base:
    .venv/Scripts/python bench.py --subset valid --models base --ens base+guitar2 --ovl --viterbi
"""
import argparse
import csv
import json
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ALIASES = {"base": "btc_large_voca", "guitar": "btc_guitar", "guitar2": "btc_guitar2",
           "mix": "btc_mix", "hook": "btc_hook", "self": "btc_self"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="base")
    parser.add_argument("--ens", default="")
    parser.add_argument("--ovl", action="store_true")
    parser.add_argument("--viterbi", action="store_true")
    parser.add_argument("--rows", type=int, default=10 ** 9)
    parser.add_argument("--subset", default="")
    args = parser.parse_args()
    if args.subset:
        os.environ["HOOK_SUBSET"] = args.subset

    # Imported after HOOK_SUBSET is set: these modules resolve ROOT on import.
    from btc_features import features_from_wav
    from eval_guitarset import to_majmin
    from hooktheory_eval import ROOT, load_audio, read_lab, STEP
    from infer_tricks_eval import probs_pass, combine, T
    from viterbi_eval import build_log_transitions, viterbi

    def resolve(name):
        return ALIASES.get(name, name)

    configs = {}  # display name -> list of model names to average
    for m in filter(None, args.models.split(",")):
        configs[m] = [resolve(m)]
    for e in filter(None, args.ens.split(",")):
        configs["ens:" + e] = [resolve(p) for p in e.split("+")]
    needed = sorted({m for ms in configs.values() for m in ms})
    sessions = {m: ort.InferenceSession(os.path.join(HERE, "models", m + ".onnx"))
                for m in needed}
    with open(os.path.join(HERE, "models", "btc_large_voca.json")) as f:
        labels = json.load(f)["labels"]
    log_trans = build_log_transitions(labels, None, 0.9) if args.viterbi else None

    with open(os.path.join(ROOT, "sample.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:args.rows]

    tally = {c: [0, 0, 0, 0] for c in configs}   # mm scored/correct, full scored/correct
    done = 0
    for row in rows:
        audio_path = os.path.join(ROOT, "audio", row["id"] + ".m4a")
        mm_path = os.path.join(ROOT, "labs", row["id"] + ".lab")
        full_path = os.path.join(ROOT, "labs_full", row["id"] + ".lab")
        if not (os.path.exists(audio_path) and os.path.exists(mm_path)):
            continue
        truth_mm = read_lab(mm_path)
        truth_full = read_lab(full_path) if os.path.exists(full_path) else []
        wav = load_audio(audio_path)
        if len(wav) < 22050:
            continue
        features, spf = features_from_wav(wav)

        passes = {}
        for m in needed:
            passes[m] = [probs_pass(sessions[m], features, 0, 0)]
            if args.ovl:
                passes[m].append(probs_pass(sessions[m], features, T // 2, 0))

        preds = {}
        for name, members in configs.items():
            probs = combine([p for m in members for p in passes[m]])
            path = (viterbi(np.log(probs + 1e-12), log_trans) if args.viterbi
                    else probs.argmax(axis=1))
            preds[name] = [labels[i] for i in path]

        t = truth_mm[0][0]
        while t < truth_mm[-1][1]:
            idx = None
            mm = next((l for s, e, l in truth_mm if s <= t < e), None)
            full = next((l for s, e, l in truth_full if s <= t < e), None)
            for name, pred in preds.items():
                idx = min(int(t / spf), len(pred) - 1)
                if mm is not None and mm != "X":
                    tally[name][0] += 1
                    tally[name][1] += (to_majmin(pred[idx]) or "X") == mm
                if full is not None and full != "X":
                    tally[name][2] += 1
                    tally[name][3] += pred[idx] == full
            t += STEP
        done += 1
        if done % 25 == 0:
            print(f"{done} songs: " + "  ".join(
                f"{n} {v[1] / max(v[0], 1):.1%}" for n, v in tally.items()), flush=True)

    knobs = ("ovl " if args.ovl else "") + ("viterbi " if args.viterbi else "argmax ")
    print(f"\n=== bench [{args.subset or 'test'}] {done} songs, {knobs}===")
    for name, (ms, mc, fs, fc) in tally.items():
        print(f"{name:18} majmin {mc / max(ms, 1):.2%}   full-vocab {fc / max(fs, 1):.2%}")


if __name__ == "__main__":
    main()
