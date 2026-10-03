import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

from app.calibration import CalibrationReady, calibrate
from .synthetic import scene


@pytest.mark.parametrize("angle", [0, 25, 45, 90, 135, 180, 270, 335])
@pytest.mark.parametrize("perspective", [0, 1, 2])
def test_corners_and_independent_metric_landmarks(angle, perspective):
    image, true_corners, landmarks = scene(angle, perspective)
    result = calibrate(image)
    assert isinstance(result, CalibrationReady), result
    detected = np.array(result.corners_px)
    # Match independently of the arbitrary 180-degree orientation of a blank card.
    error = np.linalg.norm(detected[:, None] - true_corners[None], axis=2).min(axis=1)
    assert error.max() < 2.5
    transform = np.array(result.source_to_rectified)
    mapped = cv2.perspectiveTransform(landmarks[None], transform)[0]
    # Verifies metric recovery away from the reference, not just the four fitted points.
    measured = np.linalg.norm(mapped[1:] - mapped[0], axis=1) * result.millimetres_per_pixel
    np.testing.assert_allclose(measured, [20, 10], atol=.5)
    card = np.array(result.rectified_card_corners_px)
    np.testing.assert_allclose(np.linalg.norm(np.roll(card, -1, axis=0) - card, axis=1)
                               * result.millimetres_per_pixel, [85.6, 53.98, 85.6, 53.98], atol=.001)
    assert np.isfinite(transform).all()
    preview = Image.open(BytesIO(base64.b64decode(result.preview_data_url.split(",")[1])))
    assert preview.size == (result.preview_width_px, result.preview_height_px)
    assert max(preview.size) <= 1600
    assert len(result.preview_data_url) < 4_000_000
    # Confirm the surrounding lens/landmarks were not cropped to a card-only image.
    assert np.all(mapped >= 0)
    assert np.all(mapped < preview.size)
    metric = cv2.perspectiveTransform(landmarks[None], np.array(result.source_to_card_mm))[0]
    np.testing.assert_allclose(np.linalg.norm(metric[1:] - metric[0], axis=1), [20, 10], atol=.5)


@pytest.mark.parametrize("dark,rounded,noise", [(True, True, 0), (False, False, 0), (False, True, 3)])
def test_card_appearance(dark, rounded, noise):
    image, _, _ = scene(25, 1, rounded=rounded, dark=dark, noise=noise)
    assert calibrate(image).status == "calibrated"


def test_full_resolution_coordinate_contract():
    image, corners, _ = scene()
    result = calibrate(image.resize((2800, 2200)))
    assert result.status == "calibrated"
    expected = (corners + .5) * 2 - .5
    error = np.linalg.norm(np.array(result.corners_px)[:, None] - expected[None], axis=2).min(axis=1)
    assert error.max() < 4


@pytest.mark.parametrize("blur", [3, 5, 9])
def test_blurred_card_needs_retry(blur):
    image, _, _ = scene(blur=blur)
    result = calibrate(image)
    assert result.status == "retry"
    assert result.reason == "image_blurry"


def test_card_blur_is_not_hidden_by_sharp_background():
    image, _, _ = scene(blur=5)
    draw = ImageDraw.Draw(image)
    for y in range(20, 200, 8):
        draw.line((20, y, 1300, y), fill="white", width=2)
    result = calibrate(image)
    assert result.status == "retry" and result.reason == "image_blurry"


def test_missing_card():
    image = Image.new("RGB", (1000, 800), "gray")
    ImageDraw.Draw(image).ellipse((200, 200, 500, 400), outline="white", width=4)
    assert calibrate(image).reason == "card_missing"


@pytest.mark.parametrize("x", [-20, 4, 580])
def test_cropped_or_border_card(x):
    image = Image.new("RGB", (1000, 800), (40, 40, 40))
    ImageDraw.Draw(image).rounded_rectangle((x, 200, x + 428, 470), radius=15, fill="white")
    result = calibrate(image)
    assert result.status == "retry" and result.reason == "card_cropped"


def test_ambiguous_cards():
    image = Image.new("RGB", (1400, 1000), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    for x in [100, 800]:
        draw.rounded_rectangle((x, 350, x + 428, 620), radius=15, fill="white")
    assert calibrate(image).reason == "low_confidence"


def test_low_contrast():
    image = Image.new("RGB", (1000, 800), (100, 100, 100))
    ImageDraw.Draw(image).rectangle((250, 250, 678, 520), fill=(112, 112, 112))
    assert calibrate(image).status == "retry"


def test_occluded_card():
    image, _, _ = scene()
    ImageDraw.Draw(image).rectangle((460, 350, 540, 490), fill=(45, 45, 45))
    assert calibrate(image).status == "retry"


def test_extreme_perspective():
    image = Image.new("RGB", (1200, 900), (40, 40, 40))
    ImageDraw.Draw(image).polygon([(400, 350), (690, 380), (900, 650), (150, 600)], fill="white")
    assert calibrate(image).status == "retry"
