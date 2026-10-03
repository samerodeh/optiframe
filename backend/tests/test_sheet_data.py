import json

import numpy as np
import pytest

from ml.sheet_data import crop_pair, split_by_lens, explicit_split, load_manifest


def test_validation_holds_out_all_photos_of_a_lens():
    records = [{"lens_id": lens, "filename": f"{lens}_{i}.jpg"}
               for lens in ["A", "B", "C", "D"] for i in range(5)]
    training, validation = split_by_lens(records)
    assert len(training) == 15 and len(validation) == 5
    assert not ({r["lens_id"] for r in training} & {r["lens_id"] for r in validation})
    assert split_by_lens(records) == (training, validation)


def test_single_lens_cannot_produce_independent_lens_validation():
    with pytest.raises(ValueError, match="at least two"):
        split_by_lens([{"lens_id": "A", "filename": "one.jpg"}])


def test_crop_retains_whole_label_and_matching_image_pixels():
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    mask = np.zeros((600, 800), dtype=np.uint8)
    image[250:350, 300:500] = [200, 100, 50]
    mask[250:350, 300:500] = 255
    crop, target = crop_pair(image, mask)
    assert crop.shape == (260, 360, 3)
    assert target.shape == crop.shape[:2]
    assert np.count_nonzero(target) == 20000
    assert np.all(crop[target > 0] == [200, 100, 50])


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        crop_pair(np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100), np.uint8))


def test_missing_physical_measurements_are_unknown_not_zero(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filename,lens_id,width_a_mm,height_b_mm,nasal,notes\none.png,unknown,,,,pilot\n")
    record = load_manifest(manifest)[0]
    assert record["width_a_mm"] is None
    assert record["height_b_mm"] is None


def test_explicit_split_keeps_test_out_and_rejects_leakage(tmp_path):
    path = tmp_path / "split.json"
    records = [{"filename": name, "lens_id": "unknown"} for name in ("a.png", "b.png", "c.png")]
    spec = {"unit": "capture_group", "limitation": "Same physical lens may occur in all splits",
            "train": ["a.png"], "validation": ["b.png"], "test": ["c.png"],
            "groups": {"a.png": "one", "b.png": "two", "c.png": "three"}}
    path.write_text(json.dumps(spec))
    train, val, test = explicit_split(records, path)
    assert train == records[:1] and val == records[1:2] and test == records[2:]
    spec["test"] = ["a.png"]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Duplicate"):
        explicit_split(records, path)
    spec["test"] = ["c.png"]
    spec["groups"]["c.png"] = "one"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Capture group"):
        explicit_split(records, path)
