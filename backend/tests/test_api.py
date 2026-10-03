from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import app, MAX_IMAGE_BYTES
from app.upload_limit import MAX_BODY_BYTES
from .synthetic import scene

client = TestClient(app)


def upload(image, mode="loose", eye="left", format="PNG", mime="image/png", **save_options):
    buffer = BytesIO()
    image.save(buffer, format=format, **save_options)
    return client.post("/api/captures", data={"capture_mode": mode, "eye": eye},
                       files={"image": ("capture", buffer.getvalue(), mime)})


@pytest.mark.parametrize("mode,eye", [("loose", "left"), ("loose", "right"), ("framed", "left"), ("framed", "right")])
def test_success_preserves_selections(mode, eye):
    response = upload(scene()[0], mode, eye)
    assert response.status_code == 200
    result = response.json()
    assert (result["capture_mode"], result["eye"]) == (mode, eye)
    assert result["status"] == "calibrated"
    assert result["next_step"] == "lens_segmentation"
    assert len(result["calibration"]["corners_px"]) == 4
    assert "lens_contour" not in result
    assert len(response.content) < 4_000_000


def test_retry_is_structured_and_preserves_selections():
    response = upload(Image.new("RGB", (1000, 800)), "framed", "right")
    assert response.status_code == 200
    result = response.json()
    assert (result["capture_mode"], result["eye"]) == ("framed", "right")
    assert result["status"] == "retry"
    assert result["next_step"] == "retry_capture"
    assert result["calibration"]["reason"] == "card_missing"
    assert "preview_data_url" not in result["calibration"]
    assert "retake" in result["calibration"]["message"]


@pytest.mark.parametrize("format,mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("HEIF", "image/heif"), ("HEIF", "image/heic")])
def test_supported_formats(format, mime):
    response = upload(scene()[0], format=format, mime=mime)
    assert response.status_code == 200
    assert response.json()["status"] == "calibrated"


def test_exif_orientation_uses_display_coordinates():
    image = scene()[0]
    exif = Image.Exif()
    exif[274] = 6  # rotate clockwise for display
    response = upload(image, format="JPEG", mime="image/jpeg", exif=exif)
    assert response.status_code == 200
    result = response.json()
    assert (result["width_px"], result["height_px"]) == (1100, 1400)
    assert result["status"] == "calibrated"
    corners = result["calibration"]["corners_px"]
    assert all(0 <= x < 1100 and 0 <= y < 1400 for x, y in corners)


@pytest.mark.parametrize("payload,mime,code", [(b"garbage", "image/png", 400),
    (b"x", "text/plain", 415), (b"x" * (MAX_IMAGE_BYTES + 1), "image/jpeg", 413)],
    ids=["corrupt", "unsupported", "oversized"])
def test_invalid_uploads(payload, mime, code):
    response = client.post("/api/captures", data={"capture_mode": "loose", "eye": "left"},
                           files={"image": ("bad", payload, mime)})
    assert response.status_code == code
    assert isinstance(response.json()["detail"], str)


def test_spoofed_image_type():
    assert upload(Image.new("RGB", (1000, 800)), format="GIF").status_code == 415


def test_minimum_dimensions():
    assert upload(Image.new("RGB", (639, 480))).status_code == 422
    assert upload(Image.new("RGB", (480, 640))).status_code == 200


def test_pixel_limit():
    assert upload(Image.new("RGB", (6000, 4100))).status_code == 422


def test_invalid_selections():
    assert upload(scene()[0], mode="automatic").status_code == 422


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_streamed_body_limit_before_multipart_parsing():
    response = client.post("/api/captures", content=iter([b"x" * MAX_BODY_BYTES, b"x"]),
                           headers={"content-type": "multipart/form-data; boundary=invalid"})
    assert response.status_code == 413


def test_transparent_rgb_is_not_detected_as_card():
    image = scene()[0].convert("RGBA")
    image.putalpha(0)
    response = upload(image)
    assert response.status_code == 200
    assert response.json()["status"] == "retry"
