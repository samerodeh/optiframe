"""Offline pseudo-label preparation; never used as runtime lens detection.

Requires all four sheet markers. Actual marker side is supplied by the user;
the nominal PDF layout is assumed uniformly scaled and checked by corner
reprojection. This cannot independently verify print scale or lens dimensions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from app.aruco_sheet import _sheet_corners, calibrate_sheet


def rectify(rgb: np.ndarray, marker_mm: float, ppm: float = 5):
    if not np.isfinite(marker_mm) or marker_mm <= 0:
        raise ValueError("Marker size must be positive and finite")
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    lookup = {} if ids is None else dict(zip(ids.ravel().tolist(), corners))
    if not all(i in lookup for i in range(4)):
        raise ValueError("Offline preparation requires all four complete markers")
    src = np.concatenate([lookup[i].reshape(4, 2) for i in range(4)])
    ratio = marker_mm / 40.0
    dst = np.concatenate([_sheet_corners(i) * ratio for i in range(4)])
    homography, _ = cv2.findHomography(src, dst, 0)
    if homography is None:
        raise ValueError("Homography failed")
    errors = np.linalg.norm(cv2.perspectiveTransform(src[None], homography)[0] - dst, axis=1)
    if errors.max() > 1.5:
        raise ValueError(f"Sheet layout differs from uniform PDF scaling: {errors.max():.2f} mm residual")
    canvas = np.array([[ppm, 0, 35 * ratio * ppm], [0, ppm, 35 * ratio * ppm], [0, 0, 1]])
    size = (round(210 * ratio * ppm), round(297 * ratio * ppm))
    warped = cv2.warpPerspective(rgb, canvas @ homography, size)
    return warped, {"processed_photo_to_mm": homography.tolist(), "marker_mm": marker_mm,
                    "mm_per_pixel": 1 / ppm, "uniform_layout_scale": ratio,
                    "corner_rms_mm": float(np.sqrt(np.mean(errors ** 2))),
                    "corner_max_mm": float(errors.max())}


def pseudo_mask(rgb: np.ndarray, marker_mm: float, ppm: float = 5):
    """Edge-derived draft label, independent of the UNet. Must inspect overlays."""
    ratio = marker_mm / 40
    # Broad interior between the top and bottom markers, not the printed lens zone.
    x0, x1 = [round((x + 35) * ratio * ppm) for x in (-8, 148)]
    y0, y1 = [round((y + 35) * ratio * ppm) for y in (22, 205)]
    roi = rgb[y0:y1, x0:x1]
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    choices = []
    for low, high in ((5, 15), (10, 30), (20, 60), (35, 100)):
        edges = cv2.Canny(gray, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if not 250 * ppm**2 < area < 2500 * ppm**2:
                continue
            if x <= 2 or y <= 2 or x+w >= roi.shape[1]-2 or y+h >= roi.shape[0]-2:
                continue
            solidity = area / max(1, cv2.contourArea(cv2.convexHull(contour)))
            if solidity < .94 or max(w, h) / min(w, h) > 2.5:
                continue
            choices.append((area, contour + np.array([[[x0, y0]]])))
    if not choices:
        raise ValueError("No reliable closed contour for draft label")
    contour = max(choices, key=lambda c: c[0])[1]
    mask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    # Remove one-pixel branches from paper texture without forcing a lens shape.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--marker-mm", type=float, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    for folder in ("photos", "masks", "review"):
        (args.data_dir / folder).mkdir(parents=True, exist_ok=True)
    records, tiles = [], []
    for path in sorted((args.data_dir / "originals").glob("*.jpg")):
        record = {"source": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        record["original_oriented_size_px"] = list(image.size)
        # Keep sufficient resolution for marker corner localization; strip metadata.
        image.thumbnail((2200, 2200))
        record["processed_photo_size_px"] = list(image.size)
        rgb = np.array(image)
        runtime = calibrate_sheet(image)
        record["runtime_calibration_status"] = runtime.status
        record["runtime_calibration_reason"] = getattr(runtime, "reason", None)
        try:
            warped, calibration = rectify(rgb, args.marker_mm)
            record.update(calibration)
            Image.fromarray(warped).save(args.data_dir / "photos" / f"{path.stem}.png")
            mask = pseudo_mask(warped, args.marker_mm)
            Image.fromarray(mask).save(args.data_dir / "masks" / f"{path.stem}.png")
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            record["draft_bounds_mm"] = [w / 5, h / 5]
            record["label_status"] = "draft_pseudo_label"
            view = warped.copy()
            cv2.drawContours(view, [contour], -1, (0, 220, 0), 2)
            pad = 35
            view = view[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]
            tile = Image.fromarray(view)
            tile.thumbnail((270, 310))
            panel = Image.new("RGB", (290, 350), "white")
            panel.paste(tile, ((290-tile.width)//2, 30))
            ImageDraw.Draw(panel).text((8, 8), f"{path.stem} draft {w/5:.1f} x {h/5:.1f} mm", fill="black")
            panel.save(args.data_dir / "review" / f"{path.stem}.jpg")
            tiles.append(panel)
        except ValueError as exc:
            record["error"] = str(exc)
        records.append(record)
        print(path.name, record.get("draft_bounds_mm", record.get("error")), flush=True)
    (args.data_dir / "preparation.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if tiles:
        sheet = Image.new("RGB", (290 * 5, 350 * ((len(tiles)+4)//5)), "#cccccc")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i%5)*290, (i//5)*350))
        sheet.save(args.data_dir / "review" / "labels.jpg")


if __name__ == "__main__":
    main()
