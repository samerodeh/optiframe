import numpy as np
import pytest

from app import ml_segment
from app.calibration import calibrate, CalibrationReady
from app.measurement import LensMeasurement, _candidate_contours, _ml_rescue, _rectify
from ml.dataset import sample


def test_dataset_sample_shapes():
    img, mask, meta = sample(seed=0)
    assert img.shape == (256, 256, 3)
    assert mask.shape == (256, 256)
    assert mask.max() == 255 and (mask > 0).sum() > 1000
    assert meta["shape"] in {"oval", "round", "wayfarer", "aviator", "cateye"}


def test_ml_unavailable_without_model_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_segment, "MODEL_CANDIDATES", (tmp_path / "missing.onnx",))
    monkeypatch.setattr(ml_segment, "_session", None)
    monkeypatch.setattr(ml_segment, "_missing", False)
    img, _, _ = sample(seed=1)
    assert ml_segment.predict_contour(img) is None


def test_ml_rescue_never_contradicts_anchor():
    # Safety property: ML refines the deterministic ROI, never reinvents it.
    # Either rescue agrees with the anchor size or it declines (None).
    if not ml_segment.available():
        pytest.skip("no trained model present")
    from .synthetic import scene
    image, _, _ = scene(25, 1)
    calibration = calibrate(image)
    assert isinstance(calibration, CalibrationReady)
    rectified = _rectify(image, calibration)
    candidates, _ = _candidate_contours(rectified, calibration)
    assert candidates, "need a deterministic ROI seed"
    rescued = _ml_rescue(rectified, calibration, "loose", candidates)
    if rescued is None:
        return
    assert isinstance(rescued, LensMeasurement)
    anchor = np.asarray(candidates[0].size, dtype=float)
    got = np.array([rescued.width_a_mm, rescued.height_b_mm])
    assert np.all(np.abs(got - anchor) / anchor <= 0.25 + 1e-9)
