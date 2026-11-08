"""Package a pretrained crop-disease model into three local HTML files.

The resulting page runs locally in the browser without a backend or network
requests. Keep crop-doctor.html, crop-model.html, and crop-runtime.html together.
The model is pretrained; this script exports and packages it, it does not train
or fine-tune it.

Setup:
    pip install torch transformers onnx onnxruntime numpy

Run:
    python crop.py
    python crop.py --full-precision
"""
import argparse
import base64
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

REPO = "aashituli/promblemo"
CROP_WORDS = {"corn": "maize", "maize": "maize", "rice": "rice", "wheat": "wheat", "potato": "potato"}
ORT_VERSION = "1.20.1"
ORT_JS_URL = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort.min.js"
ORT_MJS_URL = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.mjs"
ORT_WASM_URL = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.wasm"
MODEL_PLACEHOLDER = "/*__MODEL__*/ null"
ORT_PLACEHOLDER = "/*__ORT_RUNTIME__*/"
MAX_HTML_SIZE = 20_000_000


def parse_label(raw):
    text = re.sub(r"[_\-]+", " ", raw).strip()
    low = text.lower()
    if "invalid" in low:
        return {"raw": raw, "crop": "none", "disease": "Invalid", "invalid": True}
    for word, crop in CROP_WORDS.items():
        if re.search(rf"\b{word}\b", low):
            disease = re.sub(rf"\b{word}\b", " ", text, flags=re.I)
            disease = re.sub(r"\s+", " ", disease).strip().title() or "Healthy"
            return {"raw": raw, "crop": crop, "disease": disease, "invalid": False}
    raise SystemExit(f"Could not tell which crop the label '{raw}' belongs to. "
                     f"Edit CROP_WORDS or parse_label() in this script.")


def download(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m(pixel_values=x).logits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    project_dir = Path(__file__).parent
    ap.add_argument("--template", default=str(project_dir / "crop-doctor-template.html"))
    ap.add_argument("--out", default=str(project_dir / "crop-doctor.html"))
    ap.add_argument(
        "--full-precision",
        action="store_true",
        help="keep float32 model weights instead of the smaller float16 weights",
    )
    a = ap.parse_args()

    print("Downloading model...")
    model = AutoModelForImageClassification.from_pretrained(REPO).eval()
    proc = AutoImageProcessor.from_pretrained(REPO)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels = [id2label[i] for i in range(len(id2label))]
    cmap = [parse_label(l) for l in labels]
    print("Label map:")
    for i, c in enumerate(cmap):
        print(f"  {i:2d}  {c['raw']!r:35} -> crop={c['crop']:7} disease={c['disease']}")

    size = proc.size.get("height") or proc.size.get("shortest_edge") or 224
    mean, std = list(proc.image_mean), list(proc.image_std)
    print(f"Input size {size}, mean {mean}, std {std}")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.onnx")
        dummy = torch.randn(1, 3, size, size)
        kw = dict(input_names=["pixel_values"], output_names=["logits"], opset_version=14)
        try:
            torch.onnx.export(Wrap(model), (dummy,), path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(Wrap(model), (dummy,), path, **kw)

        if not a.full_precision:
            from onnxruntime.transformers.float16 import convert_float_to_float16
            import onnx

            q = os.path.join(d, "model_q.onnx")
            converted = convert_float_to_float16(
                onnx.load(path),
                keep_io_types=True,
            )
            onnx.save(converted, q)
            path = q

        # sanity check: ONNX output must match the original model
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = Wrap(model)(dummy).numpy()
        got = sess.run(None, {"pixel_values": dummy.numpy()})[0]
        print("Max difference vs original model:", float(np.abs(ref - got).max()),
              "| same top class:", int(ref.argmax()) == int(got.argmax()))
        if int(ref.argmax()) != int(got.argmax()):
            raise SystemExit("Exported model changed the top prediction; refusing to package it.")
        blob = Path(path).read_bytes()

    print("Downloading the pinned ONNX Runtime Web bundle...")
    ort_js = download(ORT_JS_URL).decode("utf-8")
    ort_mjs = download(ORT_MJS_URL)
    wasm_url_expr = b'new URL("ort-wasm-simd-threaded.wasm",import.meta.url)'
    if wasm_url_expr not in ort_mjs:
        raise SystemExit("Could not patch the ONNX Runtime module for single-file use.")
    ort_mjs = ort_mjs.replace(
        wasm_url_expr,
        b'new URL("ort-wasm-simd-threaded.wasm","https://localhost/")',
    )
    ort_wasm = download(ORT_WASM_URL)

    payload = {
        "modelFile": "crop-model.html",
        "runtimeFile": "crop-runtime.html",
        "mjs": base64.b64encode(ort_mjs).decode("ascii"),
        "size": size,
        "mean": mean,
        "std": std,
        "map": cmap,
    }
    html = Path(a.template).read_text(encoding="utf-8")
    if MODEL_PLACEHOLDER not in html:
        raise SystemExit(f"Model placeholder missing from {a.template}")
    if ORT_PLACEHOLDER not in html:
        raise SystemExit(f"ONNX Runtime placeholder missing from {a.template}")
    html = html.replace(MODEL_PLACEHOLDER, json.dumps(payload, separators=(",", ":")))
    html = html.replace(ORT_PLACEHOLDER, ort_js.replace("</script", "<\\/script"))

    def make_asset(name, data):
        value = base64.b64encode(data).decode("ascii")
        script_data = json.dumps(
            {"type": "crop-doctor-asset", "asset": name, "data": value},
            separators=(",", ":"),
        )
        return (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>Crop Doctor {name} data</title></head><body>"
            f"<script>parent.postMessage({script_data}, \"*\");</script>"
            "</body></html>"
        )

    out = Path(a.out)
    model_html = make_asset("model", blob)
    runtime_html = make_asset("runtime", ort_wasm)
    outputs = {
        out: html,
        out.with_name("crop-model.html"): model_html,
        out.with_name("crop-runtime.html"): runtime_html,
    }
    oversized = [
        f"{path.name}: {len(content.encode('utf-8')) / 1e6:.1f} MB"
        for path, content in outputs.items()
        if len(content.encode("utf-8")) >= MAX_HTML_SIZE
    ]
    if oversized:
        raise SystemExit(
            "Each generated HTML file must be under 20 MB. "
            "Use the default float16 model or choose a smaller model. "
            "Oversized: " + ", ".join(oversized)
        )
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB).")