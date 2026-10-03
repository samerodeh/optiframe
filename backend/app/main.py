from __future__ import annotations

from io import BytesIO
from typing import Literal
from uuid import uuid4
import warnings

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .calibration import CalibrationReady, CalibrationRetry, calibrate
from .upload_limit import MAX_IMAGE_BYTES, UploadLimitMiddleware

register_heif_opener()

MAX_IMAGE_PIXELS = 24_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "HEIF", "HEIC"}
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}

CaptureMode = Literal["loose", "framed"]
Eye = Literal["left", "right"]


class HealthResponse(BaseModel):
    status: Literal["ok"]


class CaptureReceipt(BaseModel):
    capture_id: str
    capture_mode: CaptureMode
    eye: Eye
    filename: str
    width_px: int
    height_px: int
    size_bytes: int
    status: Literal["calibrated", "retry"]
    next_step: Literal["lens_segmentation", "retry_capture"]
    calibration: CalibrationReady | CalibrationRetry


app = FastAPI(
    title="OptiFrame API",
    version="0.2.0",
    description="Image intake and reference-card calibration for OptiFrame.",
)

app.add_middleware(UploadLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/captures", response_model=CaptureReceipt)
async def create_capture(
    image: UploadFile = File(...),
    capture_mode: CaptureMode = Form(...),
    eye: Eye = Form(...),
) -> CaptureReceipt:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        await image.close()
        raise HTTPException(
            status_code=415,
            detail="Choose an original JPG, PNG, HEIC, HEIF, or WebP image.",
        )

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    await image.close()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The image must be 4 MB or smaller.")

    # Decode and CPU-bound OpenCV work off the event loop. Images remain in memory.
    width_px, height_px, calibration = await run_in_threadpool(process_image, image_bytes)

    return CaptureReceipt(
        capture_id=str(uuid4()),
        capture_mode=capture_mode,
        eye=eye,
        filename=image.filename or "camera-capture",
        width_px=width_px,
        height_px=height_px,
        size_bytes=len(image_bytes),
        status=calibration.status,
        next_step="lens_segmentation" if calibration.status == "calibrated" else "retry_capture",
        calibration=calibration,
    )


def process_image(image_bytes: bytes) -> tuple[int, int, CalibrationReady | CalibrationRetry]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as opened:
                if opened.format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(status_code=415, detail="Choose an original JPG, PNG, HEIC, HEIF, or WebP image.")
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=422, detail="This image has too many pixels. Use a photo of 24 megapixels or fewer.")
                opened.verify()
            with Image.open(BytesIO(image_bytes)) as opened:
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                if min(width, height) < 480 or max(width, height) < 640:
                    raise HTTPException(status_code=422, detail="The image is too small for calibration. Use at least 640 × 480 pixels (or portrait equivalent).")
                # Composite transparency consistently; never treat invisible RGB as a card.
                if oriented.mode in ("RGBA", "LA", "P"):
                    rgba = oriented.convert("RGBA")
                    rgb = Image.new("RGB", oriented.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    rgb = oriented.convert("RGB")
                return width, height, calibrate(rgb)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=422, detail="This image has too many pixels. Use a photo of 24 megapixels or fewer.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image. Retake it or choose another original.") from exc
