# ML workspace

Tiny UNet lens segmentation on the rectified preview plane.

- `ml/dataset.py`: synthetic 256x256 rectified-plane generator (oval, round,
  wayfarer, aviator, cat-eye + transparency/glare/shadow/blur). No personal
  data, CC0.
- `ml/unet.py`: TinyUNet (~0.5M params).
- `ml/train.py`: CPU training + ONNX export to `ml/lens_unet.onnx`.
- `app/ml_segment.py`: onnxruntime CPU inference. Missing model file means
  automatic fallback to the deterministic baseline.
- `app/measurement.py`: deterministic contours remain primary. ML runs only
  on padded candidate ROIs to rescue weak edges; ambiguity and the 25% size
  agreement gate still apply.

Train (from `backend/`):

```powershell
.venv\Scripts\python.exe -m ml.train --samples 800 --epochs 18 --out ml/lens_unet.onnx
```

Real-photo fine-tuning uses `ml/train_sheet.py` and `ml/sheet_data.py`.
Images and same-size binary masks live in `photos/` and `masks/`, with a
`manifest.csv`. Missing physical measurements stay blank, never invented or
replaced with image estimates. By default validation holds out whole lens IDs.
An explicit capture-group split is supported for a small pilot of unknown lens
identity, with train/validation/test disjoint and the limitation recorded.
Augmentation is training-only; model selection never uses the test split.

The 2026-10-03 batch is local at `ml/data/sheet/batch_20261003/`. All 21 originals
were copied byte-for-byte and hashed. Derived PNGs are EXIF-oriented and contain
no copied EXIF metadata. Original photos and crops include reflections and have
not been certified free of personal information. This batch is Git-ignored and
has no public redistribution license; the user's authorization covers this
local experiment. It is not part of the CC0 synthetic dataset.

The user measured these markers at **37.2 x 37.2 mm**. Preparation scales the
entire nominal PDF layout by **0.93**, including center spacings (130.2 and
211.11 mm), and checks all 16 corners for consistency. These spacings are
inferred from uniform printing, not independent physical measurements. Only
this batch uses that scale; nominal PDF geometry remains 40 mm. A printed ruler
rectified using markers on the same print cannot detect uniform print shrinkage.

```powershell
# From backend, using an environment with requirements-dev.txt installed:
python -m ml.prepare_photo_batch --data-dir ml/data/sheet/batch_20261003 --marker-mm 37.2
# Inspect review/labels.jpg. Exclude bad masks/calibrations, record reviewed
# status and write manifest.csv plus a fixed split.json BEFORE training.
python -m ml.train_sheet --data-dir ml/data/sheet/batch_20261003 --split-file ml/data/sheet/batch_20261003/split.json --epochs 8 --batch 1 --samples 100 --lr 0.0003 --out ml/runs/sheet_20261003/candidate.onnx
python -m ml.evaluate_sheet --data-dir ml/data/sheet/batch_20261003 --split-file ml/data/sheet/batch_20261003/split.json --candidate ml/runs/sheet_20261003/candidate.onnx --out ml/runs/sheet_20261003/evaluation
```

The preparation script creates draft OpenCV pseudo-labels, independently of the
UNet. Rerunning it resets review status; review new overlays before reuse. It is
offline data preparation, not a replacement for production candidate discovery.
Evaluation compares frozen ONNX models on label-anchored crops, includes all
pixels in raw IoU, also reports largest-contour IoU/Dice, and uses 40 unseen
procedural samples to check for forgetting. The label-based ROI makes this a
segmentation experiment, not a full-photo detection benchmark. Differences in
millimetres from pseudo-labels are not caliper error. Long/short sides use a
minimum-area rectangle, not nasal-oriented boxing A/B.

Candidates and atomic best checkpoints go into ignored `ml/runs/`. Existing
production ONNX files are not overwritten. CPU batch 1, a single Torch thread,
and `dynamo=False` export keep this Windows run within memory limits. The exported
ONNX logits are compared with a saved PyTorch probe before reporting success.

Validate against independently measured physical dimensions and additional
lenses before claiming 1 mm accuracy or promoting a candidate to production.
