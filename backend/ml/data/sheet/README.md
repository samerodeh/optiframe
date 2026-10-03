# Sheet photo protocol (your part — ~45 min at a table)

Do this AFTER printing the sheet. The model can only learn your lenses from
real photos of them — synthetic blobs alone will not reach 1 mm on glass.

## 1. Print (5 min)
- Print `docs/capture-sheet.pdf` at **100 %, actual size** (no fit-to-page).
- Mount on flat cardboard. Ruler-check: the check line must measure 100 mm.

## 2. Caliper ground truth (10 min)
- For each of the 2 lenses, concave side down, measure with calipers:
  - **A** = horizontal width in the sheet plane (along the 100 mm ruler).
  - **B** = vertical height. Note which side is nasal (toward the NEZ arrow).
- Write them down; you will put them in `manifest.csv` below.

## 3. Photograph (20 min, 30–50 photos)
- Lens in the dashed zone, concave side down, nasal side toward NEZ.
- Whole sheet visible + margin around all 4 markers, phone nearly overhead.
- Vary per photo: lens rotation (0°, ~45°, 90°), lighting (daylight, lamp,
  slight side-light for edge glare), phone height, both lenses.
- Sharp, no flash hotspots covering the lens edge.

## 4. Manifest (10 min)
Copy this table into `manifest.csv` (one row per photo):

```csv
filename,lens_id,width_a_mm,height_b_mm,nasal,notes
photo_001.jpg,lens_A,52.30,31.10,nasal_right,overhead daylight
photo_002.jpg,lens_A,52.30,31.10,nasal_right,lamp side light
```

- `lens_id`: e.g. `lens_A`, `lens_B` (your 2 test lenses).
- `width_a_mm` / `height_b_mm`: YOUR caliper numbers, same for every photo
  of that lens. This is the ground truth the model is graded against.
- Put the JPGs in `photos/` next to this file.

## 5. What happens next (my part, automatic)
1. I auto-generate a mask PNG per photo into `masks/` using the current
   pipeline on the sheet-rectified image.
2. You open 5–10 overlays I show you and say keep/fix — I correct the bad ones.
3. I fine-tune (`python -m ml.train_sheet`), grade A/B error against YOUR
   caliper numbers, push `lens_unet_sheet.onnx`, redeploy, re-test live.
4. Target: mean |predicted − caliper| under ~1 mm on both lenses. Until that
   number is measured on your photos, no accuracy is claimed.
