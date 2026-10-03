import math

import cv2
import numpy as np
import pytest
from PIL import Image

from app.calibration import CalibrationReady, calibrate
from app.measurement import LensMeasurement, measure_lens
from .synthetic import scene


@pytest.mark.parametrize("angle", [0, 25, 90, 155, 270, 335])
@pytest.mark.parametrize("perspective", [0, 1, 2])
def test_loose_lens_measurements_across_pose(angle, perspective):
    image, _, _ = scene(angle, perspective)
    calibration = calibrate(image)
    assert isinstance(calibration, CalibrationReady)
    result = measure_lens(image, calibration, "loose")
    assert isinstance(result, LensMeasurement), result
    assert result.method == "direct_lens_contour"
    assert result.approximate is False
    assert result.inset_mm == 0
    assert result.width_a_mm == pytest.approx(40, abs=.75)
    assert result.height_b_mm == pytest.approx(28, abs=.75)
    expected_perimeter = math.pi * (3 * (20 + 14) - math.sqrt((3 * 20 + 14) * (20 + 3 * 14)))
    assert result.perimeter_mm == pytest.approx(expected_perimeter, abs=2.5)
    contour = np.asarray(result.contour_mm)
    assert len(contour) >= 12
    assert np.isfinite(contour).all()


@pytest.mark.parametrize("angle,perspective", [(0, 0), (25, 1), (90, 2), (335, 2)])
def test_framed_mode_offsets_outer_contour_inward(angle, perspective):
    image, _, _ = scene(angle, perspective, lens_width=12)
    calibration = calibrate(image)
    result = measure_lens(image, calibration, "framed")
    assert isinstance(result, LensMeasurement), result
    assert result.method == "outer_rim_inset"
    assert result.approximate is True
    assert result.inset_mm == 1.5
    assert result.width_a_mm == pytest.approx(37, abs=.9)
    assert result.height_b_mm == pytest.approx(25, abs=.9)
    assert result.detected_outer_contour_px is not None
    outer = np.asarray(result.detected_outer_contour_px)
    measured = np.asarray(result.contour_px)
    assert cv2.contourArea(measured.astype(np.float32)) < cv2.contourArea(outer.astype(np.float32))


def test_missing_lens_after_successful_calibration():
    image, _, _ = scene(lens=False)
    calibration = calibrate(image)
    result = measure_lens(image, calibration, "loose")
    assert result.status == "retry"
    assert result.reason == "lens_missing"


def test_ambiguous_multiple_lenses():
    image, _, _ = scene()
    pixels = np.asarray(image).copy()
    cv2.ellipse(pixels, (1120, 700), (100, 70), 0, 0, 360, (110, 140, 150), 3)
    image = Image.fromarray(pixels)
    calibration = calibrate(image)
    result = measure_lens(image, calibration, "loose")
    assert result.status == "retry"
    assert result.reason == "ambiguous_contour"


def test_cropped_lens():
    image, _, _ = scene(lens=False)
    pixels = np.asarray(image).copy()
    cv2.ellipse(pixels, (1380, 500), (100, 70), 0, 0, 360, (110, 140, 150), 3)
    image = Image.fromarray(pixels)
    calibration = calibrate(image)
    result = measure_lens(image, calibration, "loose")
    assert result.status == "retry"
    assert result.reason == "lens_cropped"
