"""ArUco capture-sheet calibration (OptiFrame sheet, DICT_4X4_50, 40 mm markers).

Sheet layout: markers ID 0..3 in the corners (TL, TR, BR, BL), 40 mm side.
A 100 mm check ruler and a dashed lens zone with a nasal arrow sit between them.

Stage 1 (works now, no sheet dimensions needed): detect any marker and derive
millimetres-per-pixel from its known 40 mm side. Photograph nearly overhead and
this scale already dimensions the lens approximately.

Stage 2 (needs MARKER_CENTERS_MM from docs/capture-sheet.pdf): full-plane
homography from all visible markers, perspective-corrected rectified preview,
then reuse of app.measurement on the rectified plane.
"""
from __future__ import annotations

import base64
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

MARKER_SIZE_MM = 40.0
MARKER_IDS = (0, 1, 2, 3)
MAX_PREVIEW_SIDE = 1600

# Stage 2 geometry: marker centers in sheet millimetres, origin at ID 0 center.
# Fill in from docs/capture-sheet.pdf (helper: ml/sheet_geometry.py).
MARKER_CENTERS_MM: dict[int, tuple[float, float]] | None = None


class SheetRetry(BaseModel):
    status: Literal["retry"] = "retry"
    reason: Literal["markers_missing", "markers_cropped", "image_blurry", "low_confidence"]
    message: str


class SheetReady(BaseModel):
    status: Literal["calibrated"] = "calibrated"
    markers: list[int]
    marker_corners_px: list[list[tuple[float, float]]]
    millimetres_per_pixel: float
    source_to_sheet_mm: list[list[float]] | None = None
    confidence: float
    preview_width_px: int
    preview_height_px: int
    preview_data_url: str


MESSAGES = {
    "markers_missing": "We could not find the sheet markers. Photograph the whole sheet so the black square ArUco markers in the corners are visible, with space around them.",
    "markers_cropped": "A marker touches the photo edge. Move back until the whole sheet and a clear margin are visible, then retake the photo.",
    "image_blurry": "The markers look blurry. Add even light, tap a marker to focus, hold still, and retake the photo.",
    "low_confidence": "The markers are not readable. Flatten the sheet, remove glare and shadows, photograph the whole sheet from nearly overhead.",
}


def retry(reason: str) -> SheetRetry:
    return SheetRetry(reason=reason, message=MESSAGES[reason])


def _detect(image: Image.Image) -> tuple[list[int], list[np.ndarray], np.ndarray, tuple[int, int]]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    return ids, corners, rgb, (rgb.shape[1], rgb.shape[0])


def calibrate_sheet(image: Image.Image) -> SheetReady | SheetRetry:
    oriented = image  # caller EXIF-transposes before intake
    ids, corners, rgb, (width, height) = _detect(oriented)
    if ids is None or len(ids) == 0:
        return retry("markers_missing")
    found = sorted(int(i) for i in ids.reshape(-1) if int(i) in MARKER_IDS)
    if not found:
        return retry("markers_missing")

    margin = max(10, min(width, height) * 0.015)
    quads: list[np.ndarray] = []
    for marker_id, quad in sorted(zip(ids.reshape(-1).tolist(), corners), key=lambda t: t[0]):
        if int(marker_id) not in MARKER_IDS:
            continue
        q = quad.reshape(4, 2).astype(np.float32)
        if np.any(q < margin) or np.any(q[:, 0] > width - 1 - margin) or np.any(q[:, 1] > height - 1 - margin):
            return retry("markers_cropped")
        quads.append(q)

    # Scale from the known 40 mm side of every visible marker.
    scales = []
    for q in quads:
        sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        if sides.min() < 30 or sides.max() / sides.min() > 1.6:
            return retry("low_confidence")
        scales.append(float(MARKER_SIZE_MM / np.mean(sides)))
    mm_per_px = float(np.mean(scales))
    if not np.isfinite(mm_per_px) or mm_per_px <= 0:
        return retry("low_confidence")

    # Sharpness check on marker interiors.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F).ravel()
    sharp = float(np.percentile(np.abs(lap), 90))
    if sharp < 4:
        return retry("image_blurry")

    preview = oriented.copy()
    preview.thumbnail((MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE), Image.Resampling.LANCZOS)
    pw, ph = preview.size
    sx, sy = pw / width, ph / height
    overlay = np.asarray(preview.convert("RGB"))
    for q in quads:
        cv2.polylines(overlay, [(q * [sx, sy]).astype(np.int32)], True, (46, 160, 67), max(2, pw // 300))
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return retry("low_confidence")

    confidence = round(float(min(1.0, 0.5 + 0.25 * len(quads) + min(sharp / 60, 0.25))), 3)
    return SheetReady(
        markers=found,
        marker_corners_px=[q.tolist() for q in quads],
        millimetres_per_pixel=round(mm_per_px, 5),
        source_to_sheet_mm=None,  # Stage 2 fills this in once MARKER_CENTERS_MM is known.
        confidence=confidence,
        preview_width_px=pw,
        preview_height_px=ph,
        preview_data_url="data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
    )
