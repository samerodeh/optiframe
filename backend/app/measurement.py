"""Lens/rim contour measurement on a calibrated image plane.

Primary path is the trained ML mask (app/ml_segment.py, tiny UNet ONNX);
the deterministic Canny/Otsu baseline remains as automatic fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import pyclipper
from PIL import Image
from pydantic import BaseModel

from .calibration import CalibrationReady
from . import ml_segment

FRAME_INSET_MM = 1.5


class MeasurementRetry(BaseModel):
    status: Literal["retry"] = "retry"
    reason: Literal["lens_missing", "lens_cropped", "lens_blurry", "ambiguous_contour", "low_confidence"]
    message: str


class LensMeasurement(BaseModel):
    status: Literal["measured"] = "measured"
    method: Literal["direct_lens_contour", "outer_rim_inset", "ml_lens_mask", "ml_rim_inset"]
    approximate: bool
    inset_mm: float
    width_a_mm: float
    height_b_mm: float
    perimeter_mm: float
    confidence: float
    contour_px: list[tuple[float, float]]
    contour_mm: list[tuple[float, float]]
    detected_outer_contour_px: list[tuple[float, float]] | None = None


MESSAGES = {
    "lens_missing": "We calibrated the card but could not find a complete lens edge. Put one target lens or rim beside the card on a plain contrasting surface, avoid glare, and retake the photo.",
    "lens_cropped": "The target lens or rim touches the photo edge. Move back until its complete outline and a clear margin are visible, then retake the photo.",
    "lens_blurry": "The lens or rim edge is too soft to measure reliably. Add even light, tap the edge to focus, hold still, and retake the photo.",
    "ambiguous_contour": "More than one lens-sized outline was found. Photograph one target lens or rim beside the card and remove other rounded objects.",
    "low_confidence": "We found an outline but cannot measure it reliably. Use a plain contrasting surface, separate the lens from the card, remove glare and shadows, and retake from overhead.",
}


def retry(reason: str) -> MeasurementRetry:
    return MeasurementRetry(reason=reason, message=MESSAGES[reason])


@dataclass
class Candidate:
    contour: np.ndarray
    score: float
    sharpness: float
    area: float
    center: np.ndarray
    size: np.ndarray


def _rectify(image: Image.Image, calibration: CalibrationReady) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.warpPerspective(
        rgb,
        np.asarray(calibration.source_to_rectified, dtype=np.float64),
        (calibration.preview_width_px, calibration.preview_height_px),
        flags=cv2.INTER_LINEAR,
        borderValue=(235, 239, 235),
    )


def _card_mask(shape: tuple[int, int], calibration: CalibrationReady, padding_px: int) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    card = np.asarray(calibration.rectified_card_corners_px, np.float32)
    center = card.mean(axis=0)
    distances = np.linalg.norm(card - center, axis=1)
    expanded = center + (card - center) * (1 + padding_px / max(float(distances.min()), 1))
    cv2.fillConvexPoly(mask, np.rint(expanded).astype(np.int32), 255)
    return mask


def _edge_sharpness(gray: np.ndarray, contour: np.ndarray) -> float:
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    points = contour.reshape(-1, 2)
    if len(points) > 300:
        points = points[:: max(1, len(points) // 300)]
    values = gradient[
        np.clip(points[:, 1], 0, gray.shape[0] - 1),
        np.clip(points[:, 0], 0, gray.shape[1] - 1),
    ]
    return float(np.percentile(values, 30)) if len(values) else 0.0


def _candidate_contours(rectified: np.ndarray, calibration: CalibrationReady) -> tuple[list[Candidate], bool]:
    gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    median = float(np.median(blurred))
    lower = int(max(12, .45 * median))
    upper = int(min(240, max(lower + 30, 1.35 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    close_radius = max(1, round(.45 / calibration.millimetres_per_pixel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    card_mask = _card_mask(gray.shape, calibration, round(4 / calibration.millimetres_per_pixel))
    edges[card_mask > 0] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    height, width = gray.shape
    margin = max(8, round(2 / calibration.millimetres_per_pixel))
    candidates: list[Candidate] = []
    cropped = False
    mm = calibration.millimetres_per_pixel
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
        if len(contour) < 40:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        size_mm = np.array([w * mm, h * mm])
        if not (18 <= size_mm[0] <= 90 and 12 <= size_mm[1] <= 75):
            continue
        if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
            cropped = True
            continue
        area = float(abs(cv2.contourArea(contour)))
        area_mm2 = area * mm * mm
        if not 180 <= area_mm2 <= 5200:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        extent = area / float(w * h)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        if solidity < .72 or not .42 <= extent <= .92 or circularity < .30:
            continue
        # Reject anything intersecting the expanded reference-card region.
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, [contour], -1, 255, 2)
        if np.any((contour_mask > 0) & (card_mask > 0)):
            continue
        sharpness = _edge_sharpness(gray, contour)
        geometry = min(solidity / .95, 1) * .45 + min(circularity / .75, 1) * .35 + min(area_mm2 / 800, 1) * .20
        score = float(geometry * min(sharpness / 100, 1))
        moments = cv2.moments(contour)
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        candidate = Candidate(contour, score, sharpness, area, center, size_mm)
        duplicate_index = next((i for i, existing in enumerate(candidates)
                                if np.linalg.norm(existing.center - center) < max(w, h) * .08
                                and np.max(np.abs(existing.size - size_mm)) < 10), None)
        if duplicate_index is None:
            candidates.append(candidate)
        elif candidate.area > candidates[duplicate_index].area:
            candidates[duplicate_index] = candidate
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True), cropped


def _simplify_metric(contour_px: np.ndarray, calibration: CalibrationReady) -> tuple[np.ndarray, np.ndarray]:
    # A quarter-millimetre simplification suppresses pixel-staircase perimeter
    # inflation while retaining substantially finer detail than the 1 mm target.
    epsilon_px = max(.5, .25 / calibration.millimetres_per_pixel)
    simplified_px = cv2.approxPolyDP(contour_px, epsilon_px, True).reshape(-1, 2).astype(np.float32)
    rectified_to_mm = np.asarray(calibration.source_to_card_mm) @ np.linalg.inv(
        np.asarray(calibration.source_to_rectified)
    )
    contour_mm = cv2.perspectiveTransform(simplified_px[None], rectified_to_mm)[0]
    return simplified_px, contour_mm


def _inset_polygon(contour_mm: np.ndarray) -> np.ndarray | None:
    scale = 10_000
    path = np.rint(contour_mm * scale).astype(np.int64).tolist()
    offsetter = pyclipper.PyclipperOffset(miter_limit=2, arc_tolerance=.02 * scale)
    offsetter.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solutions = offsetter.Execute(-FRAME_INSET_MM * scale)
    if not solutions:
        return None
    solution = max(solutions, key=lambda points: abs(pyclipper.Area(points)))
    if len(solution) < 3:
        return None
    return np.asarray(solution, np.float32) / scale


def _finalize(
    contour_preview_px: np.ndarray,
    calibration: CalibrationReady,
    capture_mode: Literal["loose", "framed"],
    confidence: float,
    ml: bool,
) -> LensMeasurement | MeasurementRetry:
    contour_preview_px = np.asarray(contour_preview_px, dtype=np.float32).reshape(-1, 1, 2)
    # Margin / size sanity in preview pixels (mirrors deterministic filters).
    h, w = calibration.preview_height_px, calibration.preview_width_px
    mm = calibration.millimetres_per_pixel
    x, y, bw, bh = cv2.boundingRect(contour_preview_px.astype(np.int32))
    margin = max(8, round(2 / mm))
    if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
        return retry("lens_cropped")
    size_mm = np.array([bw * mm, bh * mm])
    if not (18 <= size_mm[0] <= 90 and 12 <= size_mm[1] <= 75):
        return retry("lens_missing")

    outer_px, outer_mm = _simplify_metric(contour_preview_px, calibration)
    measured_mm = outer_mm
    measured_px = outer_px
    if capture_mode == "framed":
        inset = _inset_polygon(outer_mm)
        if inset is None:
            return retry("low_confidence")
        measured_mm = inset
        mm_to_rectified = np.linalg.inv(
            np.asarray(calibration.source_to_card_mm) @ np.linalg.inv(
                np.asarray(calibration.source_to_rectified)
            )
        )
        measured_px = cv2.perspectiveTransform(inset.astype(np.float32)[None], mm_to_rectified)[0]

    minimum, maximum = measured_mm.min(axis=0), measured_mm.max(axis=0)
    width_a, height_b = maximum - minimum
    perimeter = cv2.arcLength(measured_mm.astype(np.float32).reshape(-1, 1, 2), True)
    if width_a <= 0 or height_b <= 0 or perimeter <= 0:
        return retry("low_confidence")
    if capture_mode == "loose":
        method = "ml_lens_mask" if ml else "direct_lens_contour"
    else:
        method = "ml_rim_inset" if ml else "outer_rim_inset"
    return LensMeasurement(
        method=method,
        approximate=capture_mode == "framed",
        inset_mm=FRAME_INSET_MM if capture_mode == "framed" else 0,
        width_a_mm=round(float(width_a), 2),
        height_b_mm=round(float(height_b), 2),
        perimeter_mm=round(float(perimeter), 2),
        confidence=round(float(confidence), 3),
        contour_px=measured_px.tolist(),
        contour_mm=measured_mm.tolist(),
        detected_outer_contour_px=outer_px.tolist() if capture_mode == "framed" else None,
    )


def measure_lens(
    image: Image.Image,
    calibration: CalibrationReady,
    capture_mode: Literal["loose", "framed"],
) -> LensMeasurement | MeasurementRetry:
    rectified = _rectify(image, calibration)
    # Primary: trained ML mask. Falls through to deterministic on any failure.
    if ml_segment.available():
        ml_result = ml_segment.predict_contour(rectified)
        if ml_result is not None:
            contour_px, ml_conf = ml_result
            finalized = _finalize(contour_px, calibration, capture_mode, ml_conf, ml=True)
            if isinstance(finalized, LensMeasurement):
                return finalized
    candidates, cropped = _candidate_contours(rectified, calibration)
    if not candidates:
        return retry("lens_cropped" if cropped else "lens_missing")
    best = candidates[0]
    if best.sharpness < 35:
        return retry("lens_blurry")
    if best.score < .48:
        return retry("low_confidence")
    if len(candidates) > 1 and candidates[1].score >= best.score * .82:
        return retry("ambiguous_contour")

    return _finalize(best.contour, calibration, capture_mode, best.score, ml=False)
