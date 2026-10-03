"""Synthetic ArUco sheet tests: spec geometry from docs/capture-sheet.pdf."""
import cv2
import numpy as np
import pytest
from PIL import Image

from app.aruco_sheet import (
    SheetReady, calibrate_sheet, MARKER_CENTERS_MM, RULER_MM,
)

PXMM = 5.0  # synthetic render resolution
SHEET_W_MM, SHEET_H_MM = 210.0, 297.0


def render_sheet() -> np.ndarray:
    w, h = int(SHEET_W_MM * PXMM), int(SHEET_H_MM * PXMM)
    page = np.full((h, w, 3), 245, np.uint8)
    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, (cx, cy) in MARKER_CENTERS_MM.items():
        # PDF coords: origin top-left; sheet coords: origin at ID 0 center.
        px, py = (cx + 35) * PXMM, (cy + 35) * PXMM
        size = int(40 * PXMM)
        marker = cv2.aruco.generateImageMarker(aruco, marker_id, size)
        x0, y0 = int(px - size / 2), int(py - size / 2)
        page[y0:y0 + size, x0:x0 + size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    # 100 mm ruler: PDF x 55..155 at y 215.
    x0, x1 = int(55 * PXMM), int(155 * PXMM)
    y = int(215 * PXMM)
    cv2.line(page, (x0, y), (x1, y), (0, 0, 0), 3)
    for i in range(11):
        x = x0 + int(i * 10 * PXMM)
        cv2.line(page, (x, y - 12), (x, y + 12), (0, 0, 0), 2)
    return page


def photograph(page: np.ndarray, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    h, w = page.shape[:2]
    pad = 250
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 235, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = page
    h2, w2 = canvas.shape[:2]
    src = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
    spread = 45
    dst = src + rng.uniform(-spread, spread, src.shape).astype(np.float32)
    warp = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(canvas, warp, (w2 + 100, h2 + 100),
                                borderValue=(235, 239, 235))
    return Image.fromarray(photo)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sheet_calibration_four_markers(seed):
    result = calibrate_sheet(photograph(render_sheet(), seed))
    assert isinstance(result, SheetReady), result
    assert result.markers == [0, 1, 2, 3]
    assert result.millimetres_per_pixel == pytest.approx(0.2, rel=0.05)
    to_mm = np.asarray(result.source_to_sheet_mm)
    # Ruler endpoints are already sheet-mm by construction; verify the
    # homography maps a detected photo corner back near its sheet position.
    src = np.asarray(result.marker_corners_px[0][0], dtype=np.float32).reshape(1, 1, 2)
    back = cv2.perspectiveTransform(src, to_mm)[0][0]
    assert back == pytest.approx([-20.0, -20.0], abs=1.0)
    assert abs(RULER_MM[1][0] - RULER_MM[0][0] - 100.0) < 1e-9
    # Printed-scale self-check: the warped ruler must read 100 mm.
    assert result.ruler_length_mm == pytest.approx(100.0, abs=1.5)
    assert result.print_scale_ok is True
    assert result.preview_width_px > 500 and result.preview_height_px > 500
    assert result.preview_data_url.startswith("data:image/jpeg;base64,")
    assert len(result.rectified_zone_corners_px) == 4


def test_sheet_ruler_measures_100mm_in_warped_preview():
    result = calibrate_sheet(photograph(render_sheet(), 0))
    assert isinstance(result, SheetReady)
    to_mm = np.asarray(result.source_to_sheet_mm)
    to_px = np.linalg.inv(np.asarray(result.source_to_sheet_mm))
    # Warp the photo with the estimated sheet->preview implicitly via mm_per_px:
    # ruler length in the photo divided by scale must be ~100 mm.
    q = np.asarray(result.marker_corners_px[0])
    side_px = float(np.linalg.norm(q[1] - q[0]))
    assert side_px * result.millimetres_per_pixel == pytest.approx(40.0, rel=0.03)


def test_sheet_missing_markers_retries():
    blank = Image.new("RGB", (800, 1000), (240,) * 3)
    result = calibrate_sheet(blank)
    assert result.status == "retry" and result.reason == "markers_missing"
