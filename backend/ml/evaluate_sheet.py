"""Compare frozen ONNX models on held-out ROI crops and synthetic controls.

Real metrics are agreement with reviewed pseudo-labels, NOT physical accuracy
or automatic full-photo localization accuracy. No training/model selection here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from .dataset import sample, SIZE
from .sheet_data import load_manifest, load_pair, crop_pair, explicit_split


def mask_metrics(pred, target):
    pred, target = pred > 0, target > 0
    intersection = int(np.count_nonzero(pred & target))
    union = int(np.count_nonzero(pred | target))
    return {"iou": intersection / max(1, union),
            "dice": 2 * intersection / max(1, int(pred.sum()+target.sum()))}


def largest_mask(binary):
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(binary.shape, np.uint8)
    if contours:
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return result


def dimensions(mask, mm_per_pixel):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).astype(np.float32) * mm_per_pixel
    lengths = sorted(cv2.minAreaRect(contour)[1], reverse=True)
    simplified = cv2.approxPolyDP(contour, .15, True)
    return {"long_mm": float(lengths[0]), "short_mm": float(lengths[1]),
            "perimeter_mm": float(cv2.arcLength(simplified, True))}


def make_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def predict(session, image):
    inp = cv2.resize(image, (SIZE, SIZE)).astype(np.float32).transpose(2, 0, 1)[None] / 255
    start = time.perf_counter()
    logits = session.run(["logits"], {"image": inp})[0][0, 0]
    ms = (time.perf_counter() - start) * 1000
    # Resize probabilities to ROI before thresholding, mirroring interpolated contour scale.
    probability = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
    binary = cv2.resize(probability, (image.shape[1], image.shape[0])) > .5
    return binary, largest_mask(binary), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=Path("ml/lens_unet.onnx"))
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.data_dir / "manifest.csv")
    _, _, test = explicit_split(records, args.split_file)
    preparation = {Path(r["source"]).stem: r for r in json.loads((args.data_dir / "preparation.json").read_text())}
    results = {"scope": "Held-out capture groups; label-anchored ROI; reviewed OpenCV pseudo-label agreement; physical lens identity unconfirmed",
               "physical_accuracy_mm": None, "marker_side_mm": 37.2,
               "test_count": len(test), "models": {}}
    predictions = {}
    for model_name, model_path in (("baseline", args.baseline), ("candidate", args.candidate)):
        session = make_session(model_path)
        rows = []
        for record in test:
            filename = record["filename"]
            image, target = crop_pair(*load_pair(record, args.data_dir))
            raw, pred, ms = predict(session, image)
            predictions[(model_name, filename)] = pred
            mm_px = preparation[Path(filename).stem]["mm_per_pixel"]
            metrics = mask_metrics(pred, target)
            pred_dims, label_dims = dimensions(pred, mm_px), dimensions(target, mm_px)
            row = {"filename": filename, "raw_iou": mask_metrics(raw, target)["iou"],
                   **metrics, "inference_ms": ms, "prediction": pred_dims, "pseudo_label": label_dims}
            if pred_dims:
                row["absolute_difference_from_pseudo_label_mm"] = {k: abs(pred_dims[k]-label_dims[k]) for k in pred_dims}
            rows.append(row)
        control = []
        for i in range(40):
            image, target, _ = sample(seed=900_000 + i)
            raw, _, _ = predict(session, image)
            control.append(mask_metrics(raw, target)["iou"])
        results["models"][model_name] = {
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "mean_iou": float(np.mean([r["iou"] for r in rows])),
            "mean_raw_iou": float(np.mean([r["raw_iou"] for r in rows])),
            "mean_dice": float(np.mean([r["dice"] for r in rows])),
            "mean_inference_ms": float(np.mean([r["inference_ms"] for r in rows])),
            "synthetic_control_count": len(control), "synthetic_control_iou": float(np.mean(control)),
            "per_photo": rows}
        if model_name == "candidate" and args.candidate.with_suffix(".parity.npz").exists():
            with np.load(args.candidate.with_suffix(".parity.npz")) as parity:
                actual = session.run(["logits"], {"image": parity["image"]})[0]
                maximum = float(np.max(np.abs(actual - parity["logits"])))
                results["onnx_max_logit_difference"] = maximum
                if maximum > 1e-4:
                    raise ValueError(f"ONNX export differs from PyTorch: {maximum}")
        del session
    tiles = []
    for record in test:
        name = record["filename"]
        image, target = crop_pair(*load_pair(record, args.data_dir))
        panel = Image.new("RGB", (960, 355), "white")
        for i, (label, mask, color) in enumerate((
                ("Pseudo-label", target, (0, 110, 255)),
                ("Before", predictions[("baseline", name)], (255, 100, 0)),
                ("After", predictions[("candidate", name)], (0, 210, 60)))):
            view = image.copy()
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, contours, -1, color, 2)
            thumb = Image.fromarray(view)
            thumb.thumbnail((310, 310))
            panel.paste(thumb, (i*320+(320-thumb.width)//2, 35))
            ImageDraw.Draw(panel).text((i*320+12, 10), f"{name} - {label}", fill="black")
        tiles.append(panel)
    sheet = Image.new("RGB", (960, 355*len(tiles)), "white")
    for i, panel in enumerate(tiles):
        sheet.paste(panel, (0, i*355))
    sheet.save(args.out / "comparison.jpg")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k,v in model.items() if k != "per_photo"}
                      for name, model in results["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
