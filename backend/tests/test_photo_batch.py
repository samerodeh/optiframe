import cv2
import numpy as np
import pytest

from app.aruco_sheet import MARKER_CENTERS_MM
from ml.prepare_photo_batch import rectify
from ml.evaluate_sheet import mask_metrics


def rendered_sheet():
    # Independently draw a sheet at actual physical dimensions: 37.2 mm markers.
    # Using 5 px/mm makes the marker side exactly 186 px.
    ppm, ratio = 5, .93
    img = np.full((1381, 977, 3), 245, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, (cx, cy) in MARKER_CENTERS_MM.items():
        x, y = [round((value + 15) * ratio * ppm) for value in (cx, cy)]
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 186)
        img[y:y+186, x:x+186] = marker[:, :, None]
    # A 50 x 35 mm red object on the same plane, not used in homography fitting.
    img[600:775, 350:600] = [255, 0, 0]
    return img


@pytest.mark.parametrize("destination", [
    [[30, 30], [1020, 30], [1020, 1490], [30, 1490]],
    [[170, 40], [960, 200], [1020, 1440], [20, 1250]],
    [[70, 210], [900, 40], [1050, 1380], [260, 1480]],
])
def test_measured_marker_side_recovers_dimensions_under_perspective(destination):
    image = rendered_sheet()
    h, w = image.shape[:2]
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]),
                                            np.float32(destination))
    photo = cv2.warpPerspective(image, transform, (1100, 1550), borderValue=(210, 210, 210))
    rectified, meta = rectify(photo, 37.2)
    red = (rectified[:, :, 0] > 230) & (rectified[:, :, 1] < 25)
    ys, xs = np.nonzero(red)
    assert (xs.max()-xs.min()+1) * meta["mm_per_pixel"] == pytest.approx(50, abs=.5)
    assert (ys.max()-ys.min()+1) * meta["mm_per_pixel"] == pytest.approx(35, abs=.5)
    assert meta["uniform_layout_scale"] == pytest.approx(.93)


def test_unmeasured_40mm_assumption_inflates_dimensions():
    image = rendered_sheet()
    widths = []
    for marker_mm in (37.2, 40):
        warped, meta = rectify(image, marker_mm)
        ys, xs = np.nonzero((warped[:, :, 0] > 230) & (warped[:, :, 1] < 25))
        widths.append((xs.max()-xs.min()+1) * meta["mm_per_pixel"])
    assert widths[1]/widths[0] == pytest.approx(40/37.2, abs=.01)


def test_wrong_marker_layout_is_rejected():
    image = rendered_sheet()
    x, y, side = 721, 70, 186
    marker = image[y:y+side, x:x+side].copy()
    image[y:y+side, x:x+side] = 245
    image[y+40:y+40+side, x:x+side] = marker
    with pytest.raises(ValueError, match="layout"):
        rectify(image, 37.2)


def test_foreground_metrics_penalize_background_flooding():
    target = np.zeros((100, 100), np.uint8)
    target[40:60, 40:60] = 255
    assert mask_metrics(target, target) == {"iou": 1, "dice": 1}
    assert mask_metrics(np.ones_like(target), target)["iou"] == pytest.approx(.04)
