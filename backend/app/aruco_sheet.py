"""ArUco capture-sheet calibration (OptiFrame sheet, DICT_4X4_50, 40 mm markers).

Geometry (mm, from docs/capture-sheet.pdf, origin at ID 0 center, y down):
  markers 40x40  : ID 0 (0,0), ID 1 (140,0), ID 2 (140,227), ID 3 (0,227)
  lens zone      : x 20..120, y 73.5..153.5  (100 x 80 mm, concave side down)
  check ruler    : y 180, x 20..120  (must measure 100 mm)
  sheet (A4)     : x -35..175, y -35..262

Any visible marker yields a full homography (4 coplanar corners of known
40 mm side); multiple markers are fitted jointly. The rectified preview is
the sheet plane itself, so lens contours map to millimetres directly.
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
HALF = MARKER_SIZE_MM / 2
MAX_PREVIEW_SIDE = 1600

MARKER_CENTERS_MM: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (140.0, 0.0),
    2: (140.0, 227.0),
    3: (0.0, 227.0),
}
LENS_ZONE_MM = ((20.0, 73.5), (120.0, 153.5))
RULER_MM = ((20.0, 180.0), (120.0, 180.0))
SHEET_BOUNDS_MM = ((-35.0, -35.0), (175.0, 262.0))


class SheetRetry(BaseModel):
    status: Literal["retry"] = "retry"
    reason: Literal["markers_missing", "markers_cropped", "image_blurry", "low_confidence"]
    message: str


class SheetReady(BaseModel):
    status: Literal["calibrated"] = "calibrated"
    markers: list[int]
    marker_corners_px: list[list[tuple[float, float]]]
    millimetres_per_pixel: float
    source_to_sheet_mm: list[list[float]]
    rectified_zone_corners_px: list[tuple[float, float]]
    confidence: float
    preview_width_px: int
    preview_height_px: int
    preview_data_url: str


MESSAGES = {
    "markers_missing": "We could not find the sheet markers. Photograph the whole sheet so the black square ArUco markers in the corners are visible, with space around them.",
    "markers_cropped": "A marker touches the photo edge. Move back until the whole sheet and a clear margin are visible, then retake the photo.",
    "image_blurry": "The markers look blurry. Add even light, tap a marker to focus, hold still, and retake the photo.",
    "low_confidence": "The markers are not readable. Keep the sheet flat, remove glare and shadows, photograph the whole sheet from nearly overhead.",
}


def retry(reason: str) -> SheetRetry:
    return SheetRetry(reason=reason, message=MESSAGES[reason])


def _sheet_corners(marker_id: int) -> np.ndarray:
    cx, cy = MARKER_CENTERS_MM[marker_id]
    return np.float32([
        [cx - HALF, cy - HALF], [cx + HALF, cy - HALF],
        [cx + HALF, cy + HALF], [cx - HALF, cy + HALF],
    ])


def calibrate_sheet(image: Image.Image) -> SheetReady | SheetRetry:
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return retry("markers_missing")

    margin = max(10, min(width, height) * 0.015)
    src_pts: list[np.ndarray] = []
    dst_pts: list[np.ndarray] = []
    found: list[int] = []
    quads: list[np.ndarray] = []
    for marker_id, quad in sorted(zip(ids.reshape(-1).tolist(), corners), key=lambda t: t[0]):
        marker_id = int(marker_id)
        if marker_id not in MARKER_IDS:
            continue
        q = quad.reshape(4, 2).astype(np.float32)
        if np.any(q < margin) or np.any(q[:, 0] > width - 1 - margin) or np.any(q[:, 1] > height - 1 - margin):
            return retry("markers_cropped")
        sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        if sides.min() < 30 or sides.max() / sides.min() > 1.6:
            return retry("low_confidence")
        found.append(marker_id)
        quads.append(q)
        src_pts.append(q)
        dst_pts.append(_sheet_corners(marker_id))
    if not found:
        return retry("markers_missing")

    src = np.concatenate(src_pts).astype(np.float32)
    dst = np.concatenate(dst_pts).astype(np.float32)
    to_mm, _ = cv2.findHomography(src, dst, 0 if len(found) < 2 else cv2.RANSAC, 3.0)
    if to_mm is None:
        return retry("low_confidence")

    # Reprojection check: every marker corner must land within 1.5 mm.
    reproj = cv2.perspectiveTransform(src[None], to_mm)[0]
    if float(np.abs(reproj - dst).max()) > 1.5:
        return retry("low_confidence")

    contrasts: list[float] = []
    for q in quads:
        for a, b in zip(q, np.roll(q, -1, axis=0)):
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length < 1:
                continue
            unit = edge / length
            normal = np.array([-unit[1], unit[0]])
            centers = a + np.linspace(0.2, 0.8, 16)[:, None] * edge
            inner = centers - normal * 4.0
            outer = centers + normal * 4.0
            # Marker interior is dark; sample both polarities in case the
            # marker orientation flips inside/outside.
            gi = cv2.remap(gray, inner[:, 0].astype(np.float32),
                           inner[:, 1].astype(np.float32), cv2.INTER_LINEAR)
            go = cv2.remap(gray, outer[:, 0].astype(np.float32),
                           outer[:, 1].astype(np.float32), cv2.INTER_LINEAR)
            contrasts.append(float(np.median(np.abs(go.astype(float) - gi.astype(float)))))
    if not contrasts or float(np.percentile(contrasts, 25)) < 30:
        return retry("image_blurry")

    (x0, y0), (x1, y1) = SHEET_BOUNDS_MM
    corners_sheet = np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
    span = np.array([x1 - x0, y1 - y0])
    ppm = min((MAX_PREVIEW_SIDE - 1) / span.max(),
              float(np.linalg.norm(src[1] - src[0]) / MARKER_SIZE_MM))
    if not np.isfinite(ppm) or ppm < 1:
        return retry("low_confidence")
    canvas = np.array([[ppm, 0, -x0 * ppm], [0, ppm, -y0 * ppm], [0, 0, 1]])
    to_px = canvas @ to_mm
    to_px /= to_px[2, 2]
    size = tuple(np.ceil(span * ppm).astype(int) + 1)
    preview = cv2.warpPerspective(rgb, to_px, size, flags=cv2.INTER_LINEAR,
                                  borderValue=(235, 239, 235))
    zone = cv2.perspectiveTransform(
        np.float32([[[LENS_ZONE_MM[0][0], LENS_ZONE_MM[0][1]],
                      [LENS_ZONE_MM[1][0], LENS_ZONE_MM[0][1]],
                      [LENS_ZONE_MM[1][0], LENS_ZONE_MM[1][1]],
                      [LENS_ZONE_MM[0][0], LENS_ZONE_MM[1][1]]]]), to_px)[0]
    cv2.polylines(preview, [zone.astype(np.int32)], True, (46, 160, 67), max(2, size[0] // 400))
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return retry("low_confidence")

    return SheetReady(
        markers=found,
        marker_corners_px=[q.tolist() for q in quads],
        millimetres_per_pixel=round(1 / ppm, 5),
        source_to_sheet_mm=(to_mm / to_mm[2, 2]).tolist(),
        rectified_zone_corners_px=zone.tolist(),
        confidence=round(float(min(1.0, 0.45 + 0.18 * len(found))), 3),
        preview_width_px=int(size[0]),
        preview_height_px=int(size[1]),
        preview_data_url="data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
    )
