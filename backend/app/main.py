from __future__ import annotations

from io import BytesIO
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pydantic import BaseModel

register_heif_opener()

MAX_IMAGE_BYTES = 15 * 1024 * 1024
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
    status: Literal["received"]
    next_step: Literal["reference_detection"]


app = FastAPI(
    title="OptiFrame API",
    version="0.1.0",
    description="Image intake API for the OptiFrame mobile capture flow.",
)

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
        raise HTTPException(
            status_code=415,
            detail="Choose an original JPG, PNG, HEIC, or WebP image.",
        )

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    await image.close()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The image must be 15 MB or smaller.")

    try:
        with Image.open(BytesIO(image_bytes)) as opened_image:
            width_px, height_px = opened_image.size
            opened_image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    if width_px < 640 or height_px < 480:
        raise HTTPException(
            status_code=422,
            detail="The image is too small for measurement. Use a photo of at least 640 × 480 pixels.",
        )

    return CaptureReceipt(
        capture_id=str(uuid4()),
        capture_mode=capture_mode,
        eye=eye,
        filename=image.filename or "camera-capture",
        width_px=width_px,
        height_px=height_px,
        size_bytes=len(image_bytes),
        status="received",
        next_step="reference_detection",
    )
