# OptiFrame

Mobile-first React and FastAPI app for measuring recycled eyeglass lenses and eventually generating a printable frame.

## Current milestone: calibration and lens measurement

The app now:

- captures or uploads original JPG, PNG, WebP, HEIC and HEIF images;
- preserves explicit loose/framed mode and left/right eye choices;
- detects the four corners of one blank 85.60 × 53.98 mm card;
- rectifies the complete scene to a top-down view and calculates millimetres per pixel;
- returns the corners, homographies, preview dimensions, scale, quality score and an in-memory JPEG preview;
- detects one lens-sized closed contour and reports boxing width A, height B and perimeter in millimetres;
- measures a loose lens edge directly, or offsets a detected outer rim inward by 1.5 mm in clearly labeled approximation mode;
- overlays the measured contour on the rectified image and returns it in both preview-pixel and card-plane millimetre coordinates;
- asks for a new photo when the card is missing, cropped, blurry, ambiguous or insufficiently reliable;
- processes images in memory, with a 4 MB upload limit and 24-megapixel decode limit.

The current contour detector tries the trained ML lens mask first and falls back
to a deterministic controlled-capture baseline. 1:1 SVG export, 3D preview
and STL generation are not implemented yet.

## Capture guide

1. Place one blank standard-size card and the lens beside each other on the same flat, plain, contrasting surface.
2. Keep the complete card, lens and a clear margin around all four card corners in the photo. Remove other rectangular objects. Align the top of the lens with the card's long edge so A and B use the intended boxing orientation.
3. Hold the phone nearly overhead, use even lighting, avoid glare and tap the card to focus.
4. Select the eye from the wearer's point of view and the capture mode, then calibrate. Check that the overlay outlines the intended card.

No printer or marker sheet is required. The camera input prefers the rear camera and file upload remains available as a fallback. Some browsers cannot show a local HEIC/HEIF preview before upload; the calibrated result is returned as JPEG.

## Detection approach and limits

A deterministic OpenCV baseline is appropriate for controlled capture, but rectangular shape alone is not robust card identification in arbitrary scenes. A four-point homography can map any quadrilateral to the known rectangle, so a successful fit is not independent proof of card identity, size or 1 mm accuracy.

The detector uses Canny edges and both polarities of Otsu thresholding. It requires a convex card-like quadrilateral, sufficient area and resolution, margin around every corner, moderate perspective, clear and sharp edge support, and a fairly uniform blank interior. It fits the straight middle portion of each edge so rounded card corners do not shrink the calibration rectangle. Multiple plausible rectangles are rejected instead of silently choosing one.

The four fitted corners define a planar homography. The full photo is mapped to a bounded rectified canvas so the lens remains visible beside the card. The output has one isotropic scale for both axes and is limited to 1600 pixels per side.

After calibration, `backend/app/measurement.py` masks the card, finds closed edge contours in the rectified scene and filters them by physical size, completeness, convexity, solidity, smoothness and edge sharpness. It rejects multiple similarly credible lens-sized outlines. The accepted contour is simplified at 0.25 mm to reduce pixel-staircase inflation before calculating the ISO-style boxing dimensions A and B and the closed-contour perimeter. Framed mode uses a Clipper polygon offset of −1.5 mm and returns both the detected outer rim and approximate inset.

Known limits:

- A blank card can be confused with another unmarked rectangle of similar proportions. The user must verify the overlay.
- Strong perspective can make the long and short sides ambiguous. A blank card also has an unavoidable 180-degree orientation ambiguity.
- Similar-colored backgrounds, shadows, glare, texture, occlusion and other rectangles can cause rejection or a false candidate.
- Card and lens must be on the same plane. Lens curvature, raised rims, card-size error and camera lens distortion can introduce measurement error.
- Transparent lenses with weak edges, glare, shadows, connected bridges/temples and patterned backgrounds can defeat this contour baseline. For framed mode, photograph one target rim clearly; a complete connected pair can be rejected as ambiguous.
- A and B are axis-aligned in the rectified card plane. The photo should keep the lens in its intended boxing orientation rather than rotated relative to the card.
- Thresholds are covered by synthetic tests but have not yet been validated on the two physical lenses. Approximately 1 mm lens accuracy remains a target.
- Local desktop checks do not establish Android Chrome, iOS Safari, Vercel cold-start or the final under-30-seconds-per-pair requirement. Those require real-device and deployed-app testing.

## Run locally

Backend:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8765
```

Frontend, in a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Vite forwards `/api` to FastAPI on port 8765. Production Python dependencies are in `backend/requirements.txt`; test dependencies are in `backend/requirements-dev.txt`.

## Verification

```powershell
cd backend
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q app
cd ../frontend
npm run build
```

Synthetic tests render cards, independent metric landmarks and a known 40 × 28 mm lens through rotations and three perspective settings. They check card-corner error, 20 mm and 10 mm distances beside the card, preview bounds, transforms, scale, A/B/perimeter recovery, the framed 1.5 mm inset and contour coordinates. Negative cases cover missing, blurred, cropped, occluded, low-contrast and ambiguous cards or lenses. API tests cover mode/eye retention, formats, EXIF rotation, invalid and spoofed uploads, transparency, multipart limits and image dimensions. The fixtures are generated procedurally and contain no personal data.

These synthetic tolerances do not claim real-world accuracy. Real-card repeatability and comparison against known physical dimensions remain required.

## API

- `GET /api/health`
- `POST /api/captures` with multipart fields `image`, `capture_mode` (`loose` or `framed`) and `eye` (`left` or `right`)

Successful measurement returns HTTP 200 with `status: "measured"`, `next_step: "capture_other_eye"`, calibration data and measurement data.

Calibration fields:

| Field | Meaning |
| --- | --- |
| `corners_px` | Four corners in the original EXIF-oriented photo |
| `source_to_card_mm` | Row-major 3×3 homography from photo coordinates to card-plane millimetres |
| `source_to_rectified` | Row-major 3×3 homography from photo coordinates to preview pixels |
| `rectified_card_corners_px` | Card overlay coordinates in the preview |
| `millimetres_per_pixel` | Uniform scale of the returned preview |
| `preview_width_px`, `preview_height_px` | JPEG preview dimensions |
| `preview_data_url` | In-memory JPEG data URL |
| `confidence` | Heuristic edge/geometry quality, not a probability or accuracy guarantee |

Corners correspond to `(0,0)`, `(85.60,0)`, `(85.60,53.98)`, `(0,53.98)` in the card plane. They are clockwise in image coordinates with a long edge first, but do not necessarily start at the image's top-left corner. Apply a homography to `[x,y,1]` and divide the first two outputs by the third.

Measurement fields:

| Field | Meaning |
| --- | --- |
| `method` | `direct_lens_contour`, `outer_rim_inset`, `ml_lens_mask` or `ml_rim_inset` |
| `approximate`, `inset_mm` | Whether framed approximation was used and its 1.5 mm inset |
| `width_a_mm`, `height_b_mm` | Axis-aligned boxing dimensions in the rectified card plane |
| `perimeter_mm` | Closed measured-contour length |
| `contour_px` | Measured overlay contour in preview pixels |
| `contour_mm` | The same contour in card-plane millimetres |
| `detected_outer_contour_px` | Framed mode's pre-offset rim contour, otherwise `null` |
| `confidence` | Heuristic contour quality, not a calibrated probability |

An ordinary retry returns HTTP 200 with `status: "retry"` and `next_step: "retry_capture"`. Calibration failures use `card_missing`, `card_cropped`, `image_blurry` or `low_confidence`. Measurement failures use `lens_missing`, `lens_cropped`, `lens_blurry`, `ambiguous_contour` or `low_confidence`. When card calibration succeeds but lens measurement fails, the response still includes the rectified calibration preview but reports no dimensions.

Invalid input uses HTTP 400, 413, 415 or 422. Upload bodies are bounded before multipart parsing, and the parser is configured to keep accepted uploads in memory rather than spooling them to disk.

## Vercel

Root `vercel.json` defines two public services:

- `frontend`: Vite at `/`
- `backend`: FastAPI at `/api/*`

The browser calls `/api/captures` on the same origin. Keep the API rewrite before the frontend catch-all rewrite. `opencv-python-headless` avoids GUI dependencies. Import the repository root using Vercel's Services preset and verify a real image upload after deployment.

## Sources and next milestones

- Challenge reference: the supplied `consignes.pdf`.
- Geometry: [OpenCV homography tutorial](https://docs.opencv.org/4.5.1/d9/dab/tutorial_homography.html) and [geometric transforms](https://docs.opencv.org/4.12.0/da/d54/group__imgproc__transform.html).
- OpenCV is Apache-2.0; NumPy is BSD-3-Clause; Pillow is HPND; Pillow-HEIF is BSD-3-Clause and bundles codec libraries documented by its project. [Pyclipper](https://github.com/fonttools/pyclipper) is MIT and its bundled Clipper core uses the Boost Software License.
- No external dataset or pretrained model is used. Synthetic fixture images are generated by `backend/tests/synthetic.py`, contain no personal data, and are dedicated to CC0-1.0. OpenAI Codex assisted development; no generative model runs in the app.

The next robustness milestone is ML-assisted lens segmentation for transparent, reflective and connected-frame cases, with real-photo validation against known physical dimensions. Product milestones after that add paired left/right capture state, 1:1 SVG export, a default 18 mm bridge, 3D preview, downloadable closed STL and digital mesh validation.
