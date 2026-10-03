"""Procedural fixtures: no real photos, personal data, external dataset, or model.

Ground truth comes from the rendering transform, independently of the detector.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

CARD = np.float32([[300, 360], [728, 360], [728, 629.9], [300, 629.9]])
LANDMARKS = np.float32([[870, 510], [970, 510], [870, 560]])  # 20 mm and 10 mm apart


def scene(angle=0, perspective=0, rounded=True, dark=False, noise=0, blur=0,
          lens=True, lens_width=3):
    background, foreground = (225, 40) if dark else (45, 230)
    plane = Image.new("RGB", (1400, 1100), (background,) * 3)
    draw = ImageDraw.Draw(plane)
    draw.rounded_rectangle((300, 360, 728, 630), radius=16 if rounded else 0,
                           fill=(foreground,) * 3)
    if lens:
        draw.ellipse((820, 450, 1020, 590), outline=(110, 140, 150), width=lens_width)
    for x, y in LANDMARKS:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(180, 30, 30))
    bounds = np.float32([[0, 0], [1399, 0], [1399, 1099], [0, 1099]])
    destinations = [bounds, np.float32([[40, 60], [1350, 0], [1390, 1070], [80, 1090]]),
                    np.float32([[140, 30], [1290, 150], [1390, 1050], [30, 1090]])]
    perspective_transform = cv2.getPerspectiveTransform(bounds, destinations[perspective])
    rotation = np.vstack([cv2.getRotationMatrix2D((700, 550), angle, .82), [0, 0, 1]])
    transform = rotation @ perspective_transform
    rgb = cv2.warpPerspective(np.asarray(plane), transform, (1400, 1100),
                              borderValue=(background,) * 3)
    if noise:
        rng = np.random.default_rng(42)
        rgb = np.clip(rgb.astype(float) + rng.normal(0, noise, rgb.shape), 0, 255).astype(np.uint8)
    if blur:
        rgb = cv2.GaussianBlur(rgb, (0, 0), blur)
    return Image.fromarray(rgb), cv2.perspectiveTransform(CARD[None], transform)[0], cv2.perspectiveTransform(LANDMARKS[None], transform)[0]
