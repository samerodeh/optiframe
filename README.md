# OptiFrame

Mobile-first React and FastAPI app for measuring recycled eyeglass lenses and eventually generating a printable frame.

## Current milestone: reference-card calibration

The app now:

- captures or uploads original JPG, PNG, WebP, HEIC and HEIF images;
- preserves explicit loose/framed mode and left/right eye choices;
- detects the four corners of one blank 85.60 × 53.98 mm card;
- rectifies the complete scene to a top-down view and calculates millimetres per pixel;
- returns the corners, homographies, preview dimensions, scale, quality score and an in-memory JPEG preview;
- asks for a new photo when the card is missing, cropped, blurry, ambiguous or insufficiently reliable;
- processes images in memory, with a 4 MB upload limit and 24-megapixel decode limit.

Calibration is not lens measurement. Lens segmentation, A/B/perimeter measurements, SVG export, 3D preview and STL generation are not implemented yet. No ML model is trained or invoked.

## Capture guide

1. Place one blank standard-size card and the lens beside each other on the same flat, plain, contrasting surface.
2. Keep the complete card, lens and a clear margin around all four card corners in the photo. Remove other rectangular objects.
3. Hold the phone nearly overhead, use even lighting, avoid glare and tap the card to focus.
4. Select the eye from the wearer's point of view and the capture mode, then calibrate. Check that the overlay outlines the intended card.

No printer or marker sheet is required. The camera input prefers the rear camera and file upload remains available as a fallback. Some browsers cannot show a local HEIC/HEIF preview before upload; the calibrated result is returned as JPEG.

## Detection approach and limits

A deterministic OpenCV baseline is appropriate for controlled capture, but rectangular shape alone is not robust card identification in arbitrary scenes. A four-point homography can map any quadrilateral to the known rectangle, so a successful fit is not independent proof of card identity, size or 1 mm accuracy.

The detector uses Canny edges and both polarities of Otsu thresholding. It requires a convex card-like quadrilateral, sufficient area and resolution, margin around every corner, moderate perspective, clear and sharp edge support, and a fairly uniform blank interior. It fits the straight middle portion of each edge so rounded card corners do not shrink the calibration rectangle. Multiple plausible rectangles are rejected instead of silently choosing one.

The four fitted corners define a planar homography. The full photo is mapped to a bounded rectified canvas so the lens remains visible beside the card. The output has one isotropic scale for both axes and is limited to 1600 pixels per side.

Known limits:

- A blank card can be confused with another unmarked rectangle of similar proportions. The user must verify the overlay.
- Strong perspective can make the long and short sides ambiguous. A blank card also has an unavoidable 180-degree orientation ambiguity.
- Similar-colored backgrounds, shadows, glare, texture, occlusion and other rectangles can cause rejection or a false candidate.
- Card and lens must be on the same plane. Lens curvature, raised rims, card-size error and camera lens distortion can introduce measurement error.
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

Synthetic tests render cards and independent metric landmarks through eight rotations and three perspective settings. They check card-corner error, 20 mm and 10 mm distances beside the card, preview bounds, transforms and scale. Negative cases cover missing, blurred, cropped, occluded, low-contrast and ambiguous cards. API tests cover mode/eye retention, formats, EXIF rotation, invalid and spoofed uploads, transparency, multipart limits and image dimensions. The fixtures are generated procedurally and contain no personal data.

These synthetic tolerances do not claim real-world accuracy. Real-card repeatability and comparison against known physical dimensions remain required.

## API

- `GET /api/health`
- `POST /api/captures` with multipart fields `image`, `capture_mode` (`loose` or `framed`) and `eye` (`left` or `right`)

Successful calibration returns HTTP 200 with `status: "calibrated"`, `next_step: "lens_segmentation"` and:

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

An ordinary retry also returns HTTP 200, with `status: "retry"`, `next_step: "retry_capture"`, a reason (`card_missing`, `card_cropped`, `image_blurry` or `low_confidence`) and a user-facing message. It has no scale, transform, or preview. Neither response contains a lens contour.

Invalid input uses HTTP 400, 413, 415 or 422. Upload bodies are bounded before multipart parsing, and the parser is configured to keep accepted uploads in memory rather than spooling them to disk.

## Vercel

Root `vercel.json` defines two public services:

- `frontend`: Vite at `/`
- `backend`: FastAPI at `/api/*`

The browser calls `/api/captures` on the same origin. Keep the API rewrite before the frontend catch-all rewrite. `opencv-python-headless` avoids GUI dependencies. Import the repository root using Vercel's Services preset and verify a real image upload after deployment.

## Sources and next milestones

- Challenge reference: the supplied `consignes.pdf`.
- Geometry: [OpenCV homography tutorial](https://docs.opencv.org/4.5.1/d9/dab/tutorial_homography.html) and [geometric transforms](https://docs.opencv.org/4.12.0/da/d54/group__imgproc__transform.html).
- OpenCV is Apache-2.0; NumPy is BSD-3-Clause; Pillow is HPND; Pillow-HEIF is BSD-3-Clause and bundles codec libraries documented by its project.
- No external dataset or pretrained model is used. Synthetic fixture images are generated by `backend/tests/synthetic.py`, contain no personal data, and are dedicated to CC0-1.0. OpenAI Codex assisted development; no generative model runs in the app.

The next milestone is lens segmentation: detect a loose lens contour directly, or detect a framed rim and apply a clearly labeled approximate 1.5 mm inward polygon offset. It must preserve different left/right shapes and compute A, B and perimeter in millimetres. Later milestones add a 1:1 SVG, default 18 mm bridge, 3D preview, downloadable closed STL and digital mesh validation.
