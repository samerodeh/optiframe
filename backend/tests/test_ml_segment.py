import numpy as np

from app import ml_segment
from ml.dataset import sample


def test_dataset_sample_shapes():
    img, mask, meta = sample(seed=0)
    assert img.shape == (256, 256, 3)
    assert mask.shape == (256, 256)
    assert mask.max() == 255 and (mask > 0).sum() > 1000
    assert meta["shape"] in {"oval", "round", "wayfarer", "aviator", "cateye"}


def test_ml_unavailable_without_model_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_segment, "MODEL_PATH", tmp_path / "missing.onnx")
    monkeypatch.setattr(ml_segment, "_session", None)
    monkeypatch.setattr(ml_segment, "_missing", False)
    img, _, _ = sample(seed=1)
    assert ml_segment.predict_contour(img) is None
