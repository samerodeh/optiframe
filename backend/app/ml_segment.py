"""ONNX lens-mask inference on the rectified preview plane.

Loads backend/ml/lens_unet.onnx lazily via onnxruntime (CPU). Returns the
largest plausible mask component as a preview-pixel contour plus a
confidence score. Returns None when the model file is missing or the mask
is unreliable, so callers fall back to the deterministic baseline.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "ml"
# Prefer the sheet fine-tune when present, else the generic base model.
MODEL_CANDIDATES = (MODEL_DIR / "lens_unet_sheet.onnx", MODEL_DIR / "lens_unet.onnx")
INPUT_SIZE = 256

_session = None
_missing = False


def _model_path():
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def available() -> bool:
    return _model_path() is not None


def _session_lazy():
    global _session, _missing
    if _session is not None:
        return _session
    path = _model_path()
    if _missing or path is None:
        return None
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        # Keep serverless-sized footprints: no arena pre-allocation growth.
        opts.add_session_config_entry("arena_extend_strategy", "kSameAsRequested")
        _session = ort.InferenceSession(str(path), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
        return _session
    except Exception:
        _missing = True
        return None


def predict_contour(rectified_rgb: np.ndarray) -> tuple[np.ndarray, float] | None:
    """rectified_rgb: HxWx3 uint8 in preview-pixel space. Returns (contour_px, confidence) or None."""
    sess = _session_lazy()
    if sess is None:
        return None
    h, w = rectified_rgb.shape[:2]
    if h < 32 or w < 32:
        return None
    small = cv2.resize(rectified_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = (small.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    try:
        logits = sess.run(["logits"], {"image": x})[0][0, 0]
    except Exception:
        return None
    prob = 1 / (1 + np.exp(-logits))
    mask = (prob > 0.5).astype(np.uint8) * 255
    # keep largest connected component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[biggest, cv2.CC_STAT_AREA])
    if area < 0.004 * INPUT_SIZE * INPUT_SIZE:
        return None
    comp = (labels == biggest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 12:
        return None
    # solidity / extent sanity in 256 space
    area_c = float(cv2.contourArea(cnt))
    hull = float(cv2.contourArea(cv2.convexHull(cnt)))
    if hull <= 0 or area_c / hull < 0.6:
        return None
    cnt = cnt.reshape(-1, 2).astype(np.float32)
    cnt[:, 0] *= w / INPUT_SIZE
    cnt[:, 1] *= h / INPUT_SIZE
    mean_prob = float(prob[labels == biggest].mean()) if (labels == biggest).any() else 0.5
    confidence = round(float(np.clip((mean_prob - 0.5) * 2, 0, 1) * 0.7 + 0.3 * min(area / (0.05 * INPUT_SIZE * INPUT_SIZE), 1)), 3)
    return cnt.reshape(-1, 1, 2), confidence
