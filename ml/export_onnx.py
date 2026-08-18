"""Export BTC chord-recognition checkpoints to ONNX.

The wrapper bakes feature normalization (checkpoint mean/std) into the
graph, so the consumer feeds RAW log-CQT features [1, 108, 144] and reads
logits [1, 108, num_chords]. Nothing model-specific leaks into C#.

Run:
    .venv/Scripts/python export_onnx.py
"""
import json
import os
import sys

import numpy as np
import torch

# Compatibility shims for the 2019-era vendor code on modern numpy —
# only for aliases numpy 2.x actually removed.
if not hasattr(np, "int"):
    np.int = int  # noqa
if not hasattr(np, "float"):
    np.float = float  # noqa

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "BTC-ISMIR19")
sys.path.insert(0, REPO)

from btc_model import BTC_model  # noqa: E402
from utils.hparams import HParams  # noqa: E402
from utils.mir_eval_modules import idx2chord, idx2voca_chord  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class BtcOnnxWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(float(std), dtype=torch.float32))

    def forward(self, features):  # [1, timestep, 144] raw log-CQT
        x = (features - self.mean) / self.std
        hidden, _ = self.model.self_attn_layers(x)
        return self.model.output_layer.output_projection(hidden)


def export(checkpoint_path, large_voca, out_name, labels):
    config = HParams.load(os.path.join(REPO, "run_config.yaml"))
    if large_voca:
        config.feature["large_voca"] = True
        config.model["num_chords"] = 170

    model = BTC_model(config=config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    wrapper = BtcOnnxWrapper(model, checkpoint["mean"], checkpoint["std"]).eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name + ".onnx")
    dummy = torch.randn(1, config.model["timestep"], config.model["feature_size"])
    torch.onnx.export(
        wrapper, (dummy,), out_path,
        input_names=["features"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )

    # Parity check: torch vs onnxruntime on random input.
    import onnxruntime as ort
    with torch.no_grad():
        torch_logits = wrapper(dummy).numpy()
    session = ort.InferenceSession(out_path)
    ort_logits = session.run(None, {"features": dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(torch_logits - ort_logits)))

    with open(os.path.join(OUT_DIR, out_name + ".json"), "w") as f:
        json.dump({
            "sample_rate": 22050,
            "n_bins": 144,
            "bins_per_octave": 24,
            "hop_length": 2048,
            "timestep": config.model["timestep"],
            "num_chords": config.model["num_chords"],
            "labels": labels,
        }, f, indent=1)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"{out_name}: exported {size_mb:.1f} MB, torch-vs-ort max diff {max_diff:.2e}")


if __name__ == "__main__":
    export(
        os.path.join(REPO, "test", "btc_model_large_voca.pt"),
        large_voca=True,
        out_name="btc_large_voca",
        labels=[idx2voca_chord()[i] for i in range(170)],
    )
    export(
        os.path.join(REPO, "test", "btc_model.pt"),
        large_voca=False,
        out_name="btc_majmin",
        labels=list(idx2chord),
    )
