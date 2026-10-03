# ML workspace

Tiny UNet lens segmentation on the rectified preview plane.

- `ml/dataset.py`: synthetic 256x256 rectified-plane generator (oval, round,
  wayfarer, aviator, cat-eye + transparency/glare/shadow/blur). No personal
  data, CC0.
- `ml/unet.py`: TinyUNet (~0.5M params).
- `ml/train.py`: CPU training + ONNX export to `ml/lens_unet.onnx`.
- `app/ml_segment.py`: onnxruntime CPU inference. Missing model file means
  automatic fallback to the deterministic baseline.
- `app/measurement.py`: tries `ml_lens_mask` / `ml_rim_inset` first, then
  `direct_lens_contour` / `outer_rim_inset`.

Train (from `backend/`):

```powershell
.venv\Scripts\python.exe -m ml.train --samples 800 --epochs 18 --out ml/lens_unet.onnx
```

Fine-tuning on real photos: add annotated rectified masks under
`ml/data/real/` (source photo + PNG mask + mm scale) and extend
`ml/train.py` with a mixed sampler. Validate against known physical
dimensions of the two test lenses before claiming 1 mm accuracy.
