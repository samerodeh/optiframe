"""Conservative, deterministic calibration for one blank ID-1 card on a plain surface.

The score measures edge/geometry quality, not the probability of card identity.
A homography cannot establish the physical size or identity of an unmarked rectangle.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
MAX_WORKING_SIDE = 1600
MAX_PREVIEW_SIDE = 1600
CARD_MM = np.float32([[0, 0], [CARD_WIDTH_MM, 0],
                      [CARD_WIDTH_MM, CARD_HEIGHT_MM], [0, CARD_HEIGHT_MM]])


class CalibrationRetry(BaseModel):
    status: Literal["retry"] = "retry"
    reason: Literal["card_missing", "card_cropped", "image_blurry", "low_confidence"]
    message: str


class CalibrationReady(BaseModel):
    status: Literal["calibrated"] = "calibrated"
    card_size_mm: tuple[float, float] = (CARD_WIDTH_MM, CARD_HEIGHT_MM)
    corners_px: list[tuple[float, float]]
    rectified_card_corners_px: list[tuple[float, float]]
    source_to_rectified: list[list[float]]
    source_to_card_mm: list[list[float]]
    millimetres_per_pixel: float
    confidence: float
    preview_width_px: int
    preview_height_px: int
    preview_data_url: str


MESSAGES = {
    "card_missing": "We could not find the reference card. Place one blank 85.60 × 53.98 mm card beside the lens on a plain contrasting surface, keep all four corners visible, and retake the photo.",
    "card_cropped": "The card may be cropped or too close to the photo edge. Move back until the complete card and a clear margin around all four corners are visible, then retake the photo.",
    "image_blurry": "The card edges look blurry. Clean the camera, add even light, tap the card to focus, hold still, and retake the photo.",
    "low_confidence": "We cannot calibrate this photo reliably. Use only one blank standard-size card, remove other rectangular objects and glare, keep card and lens on the same flat surface, and retake from nearly overhead with clear, contrasting edges.",
}


def retry(reason: str) -> CalibrationRetry:
    return CalibrationRetry(reason=reason, message=MESSAGES[reason])


def _ordered(points: np.ndarray) -> np.ndarray:
    """Clockwise in image coordinates, long edge first; supports portrait/rotation.

    Start at the upper of the two possible long-edge origins. A blank card has
    unavoidable 180-degree orientation ambiguity; this does not determine lens pose.
    """
    center = points.mean(axis=0)
    q = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    lengths = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    parity = 0 if lengths[0] + lengths[2] >= lengths[1] + lengths[3] else 1
    start = min((parity, parity + 2), key=lambda i: (q[i, 1], q[i, 0]))
    return np.roll(q, -start, axis=0).astype(np.float32)


def _fit_corners(contour: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Intersect fitted straight edges rather than the rounded card's inset tips."""
    points = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        direction = b - a
        length = np.linalg.norm(direction)
        unit = direction / length
        along = (points - a) @ unit / length
        distance = np.abs((points - a) @ np.array([-unit[1], unit[0]]))
        samples = points[(along > .15) & (along < .85) & (distance < max(3, length * .025))]
        if len(samples) < 12:
            return None
        vx, vy, x, y = cv2.fitLine(samples, cv2.DIST_L2, 0, .01, .01).reshape(4)
        lines.append(np.array([-vy, vx, vy * x - vx * y]))
    corners = []
    for i in range(4):
        intersection = np.cross(lines[i - 1], lines[i])
        if abs(intersection[2]) < .15:
            return None
        corners.append(intersection[:2] / intersection[2])
    fitted = np.float32(corners)
    if np.max(np.linalg.norm(fitted - quad, axis=1)) > 15:
        return None
    return fitted


def _edge_quality(gray: np.ndarray, quad: np.ndarray) -> tuple[float, float]:
    """Weakest edge contrast/support and sharpness, sampled away from round corners.

    Local edge spread avoids global Laplacian scores that penalize a blank card or
    let a sharp textured background hide a blurred reference object.
    """
    contrasts, sharpness = [], []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        unit = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-unit[1], unit[0]])
        centers = a + np.linspace(.16, .84, 40)[:, None] * (b - a)
        coords = centers[:, None, :] + np.arange(-10, 11)[None, :, None] * normal
        profiles = cv2.remap(gray, coords[:, :, 0].astype(np.float32),
                             coords[:, :, 1].astype(np.float32), cv2.INTER_LINEAR).astype(float)
        contrast = np.abs(profiles[:, -3:].mean(axis=1) - profiles[:, :3].mean(axis=1))
        gradient = np.max(np.abs(np.diff(profiles, axis=1)), axis=1)
        # Lower quartile requires most of each edge to be supported.
        contrasts.append(float(np.percentile(contrast, 25)))
        sharpness.append(float(np.percentile(gradient / np.maximum(contrast, 1), 25)))
    return min(contrasts), min(sharpness)


@dataclass
class Candidate:
    corners: np.ndarray
    score: float
    blurry: bool


def calibrate(image: Image.Image) -> CalibrationReady | CalibrationRetry:
    width, height = image.size  # Already EXIF-transposed by intake.
    working = image.copy()
    working.thumbnail((MAX_WORKING_SIDE, MAX_WORKING_SIDE), Image.Resampling.LANCZOS)
    rgb = np.asarray(working.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(smooth, 30, 90)
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates: list[Candidate] = []
    cropped = False
    plausible = False
    margin = max(10, min(w, h) * .015)
    for mask in (edges, binary, cv2.bitwise_not(binary)):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Bound work on textured or adversarial images.
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            area = cv2.contourArea(contour)
            if not .025 * w * h < area < .85 * w * h:
                continue
            perimeter = cv2.arcLength(contour, True)
            quad = cv2.approxPolyDP(contour, .025 * perimeter, True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            quad = _ordered(quad.reshape(4, 2))
            lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            ratio = (lengths[0] + lengths[2]) / (lengths[1] + lengths[3])
            # Conservative near-overhead envelope; strong tilt makes long-edge
            # assignment unreliable on an unmarked rectangle.
            if not 1.25 <= ratio <= 2.05:
                continue
            plausible = True
            if (np.any(quad < margin) or np.any(quad[:, 0] > w - 1 - margin)
                    or np.any(quad[:, 1] > h - 1 - margin)):
                cropped = True
                continue
            if lengths.min() < 90 or max(lengths[0] / lengths[2], lengths[2] / lengths[0],
                                         lengths[1] / lengths[3], lengths[3] / lengths[1]) > 1.5:
                continue
            vectors = (np.roll(quad, -1, axis=0) - quad) / lengths[:, None]
            if np.max(np.abs(np.sum(vectors * np.roll(vectors, 1, axis=0), axis=1))) > .5:
                continue
            if area / cv2.contourArea(quad) < .94:
                continue
            fitted = _fit_corners(contour, quad)
            if fitted is None:
                continue
            if (np.any(fitted < margin) or np.any(fitted[:, 0] > w - 1 - margin)
                    or np.any(fitted[:, 1] > h - 1 - margin)):
                cropped = True
                continue
            contrast, sharpness = _edge_quality(gray, fitted)
            if contrast < 25:
                continue
            # Blank interior helps reject rims and patterned rectangular clutter.
            interior = np.zeros_like(gray)
            inner_quad = fitted.mean(axis=0) + .8 * (fitted - fitted.mean(axis=0))
            cv2.fillConvexPoly(interior, inner_quad.astype(np.int32), 255)
            pixels = gray[interior > 0]
            if np.percentile(pixels, 90) - np.percentile(pixels, 10) > 40:
                continue
            score = float(.5 * min(contrast / 80, 1) + .3 * min(sharpness / .35, 1)
                          + .2 * max(0, 1 - abs(ratio - CARD_WIDTH_MM / CARD_HEIGHT_MM) / .7))
            candidate = Candidate(fitted, score, sharpness < .18)
            duplicate = next((i for i, c in enumerate(candidates) if np.linalg.norm(
                c.corners.mean(axis=0) - fitted.mean(axis=0)) < 10
                and abs(cv2.contourArea(c.corners) / cv2.contourArea(fitted) - 1) < .1), None)
            if duplicate is None:
                candidates.append(candidate)
            elif candidate.score > candidates[duplicate].score:
                candidates[duplicate] = candidate
    if not candidates:
        return retry("card_cropped" if cropped else "low_confidence" if plausible else "card_missing")
    # Do not silently choose one of multiple plausible cards, even if one is blurry.
    if len(candidates) > 1:
        return retry("low_confidence")
    candidate = candidates[0]
    if candidate.blurry:
        return retry("image_blurry")
    if candidate.score < .78:
        return retry("low_confidence")
    # PIL resizing uses pixel-center coordinates, not only a corner-based scale.
    source_corners = (candidate.corners + .5) * [width / w, height / h] - .5
    to_mm = cv2.getPerspectiveTransform(source_corners.astype(np.float32), CARD_MM)
    bounds = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    denominators = np.column_stack((bounds, np.ones(4))) @ to_mm[2]
    # A projective horizon inside/near the photo creates an unbounded preview.
    if np.any(denominators <= 0) or denominators.min() / denominators.max() < .2:
        return retry("low_confidence")
    metric_bounds = cv2.perspectiveTransform(bounds[None], to_mm)[0]
    minimum, maximum = metric_bounds.min(axis=0), metric_bounds.max(axis=0)
    spans = maximum - minimum
    native_ppm = min(np.linalg.norm(candidate.corners[1] - candidate.corners[0]) / CARD_WIDTH_MM,
                     np.linalg.norm(candidate.corners[2] - candidate.corners[1]) / CARD_HEIGHT_MM)
    ppm = min(native_ppm, (MAX_PREVIEW_SIDE - 1) / float(spans.max()))
    if not np.isfinite(ppm) or ppm < 1:
        return retry("low_confidence")
    canvas = np.array([[ppm, 0, -minimum[0] * ppm], [0, ppm, -minimum[1] * ppm], [0, 0, 1]])
    transform = canvas @ to_mm
    transform /= transform[2, 2]
    output_size = np.ceil(spans * ppm).astype(int) + 1
    from_working = np.array([[width / w, 0, .5 * width / w - .5],
                             [0, height / h, .5 * height / h - .5], [0, 0, 1]])
    preview = cv2.warpPerspective(rgb, transform @ from_working, tuple(output_size),
                                  flags=cv2.INTER_LINEAR, borderValue=(235, 239, 235))
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return retry("low_confidence")
    return CalibrationReady(
        corners_px=source_corners.tolist(),
        rectified_card_corners_px=cv2.perspectiveTransform(
            source_corners.astype(np.float32)[None], transform)[0].tolist(),
        source_to_rectified=transform.tolist(), source_to_card_mm=to_mm.tolist(),
        millimetres_per_pixel=1 / ppm, confidence=round(candidate.score, 3),
        preview_width_px=int(output_size[0]), preview_height_px=int(output_size[1]),
        preview_data_url="data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
    )
