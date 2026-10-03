"""Synthetic rectified-plane lens dataset for training the tiny UNet.

Generates 256x256 RGB images + binary masks directly in the rectified
(top-down, mm-uniform) plane, which is what inference sees. Shapes cover
common lens families: oval, round, wayfarer, aviator, cat-eye-ish, rim.
Augmentations mimic the failure modes of the deterministic baseline:
weak transparent edges, glare streaks, shadows, blur, noise, background
texture. No personal data, no external dataset. CC0.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

SIZE = 256


def _lens_polygon(shape: str, rng: random.Random, cx: float, cy: float,
                  w: float, h: float, rot: float) -> np.ndarray:
    if shape in ("oval", "round"):
        axes = (w / 2, h / 2 if shape == "oval" else w / 2)
        pts = cv2.ellipse2Poly((int(cx), int(cy)),
                               (int(axes[0]), int(axes[1])),
                               int(rot), 0, 360, 8)
        return pts.astype(np.float32)
    n = 48
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    if shape == "wayfarer":  # wider top, tapered bottom
        rx, ry = w / 2, h / 2
        x = rx * np.cos(t)
        y = ry * np.sin(t) - 0.12 * ry * np.cos(t) ** 2
        x *= 1 + 0.08 * np.sin(2 * t)
    elif shape == "aviator":  # teardrop
        rx, ry = w / 2, h / 2
        x = rx * np.cos(t)
        y = ry * np.sin(t) + 0.18 * ry * (np.cos(t) ** 3)
    else:  # cat-eye-ish: lifted outer corners
        rx, ry = w / 2, h / 2
        x = rx * np.cos(t)
        y = ry * np.sin(t) - 0.15 * ry * np.abs(np.cos(t)) * np.sign(np.cos(t))
    rot_r = math.radians(rot)
    xr = x * math.cos(rot_r) - y * math.sin(rot_r) + cx
    yr = x * math.sin(rot_r) + y * math.cos(rot_r) + cy
    return np.stack([xr, yr], axis=1).astype(np.float32)


def sample(seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (rgb 256x256 uint8, mask 256x256 uint8 0/255, meta)."""
    rng = random.Random(seed)
    npr = np.random.default_rng(seed if seed is not None else rng.randint(0, 10**9))
    bg = rng.randint(200, 245)
    img = np.full((SIZE, SIZE, 3), bg, np.uint8)
    # subtle background texture
    tex = npr.normal(0, rng.uniform(2, 9), (SIZE, SIZE, 1))
    img = np.clip(img.astype(float) + tex, 0, 255).astype(np.uint8)

    shape = rng.choice(["oval", "round", "wayfarer", "aviator", "cateye"])
    w = rng.uniform(90, 170)
    h = w * rng.uniform(0.62, 0.95) if shape != "round" else w
    cx = rng.uniform(70, SIZE - 70)
    cy = rng.uniform(70, SIZE - 70)
    # keep fully inside with margin so boxing dims are well-defined
    cx = min(max(cx, w / 2 + 8), SIZE - w / 2 - 8)
    cy = min(max(cy, h / 2 + 8), SIZE - h / 2 - 8)
    rot = rng.uniform(-25, 25)
    poly = _lens_polygon(shape, rng, cx, cy, w, h, rot)

    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

    # lens interior: slight tint / transparency variation
    tint = rng.randint(-18, 18)
    tinted = np.clip(img.astype(int) + tint, 0, 255).astype(np.uint8)
    img[mask > 0] = (0.55 * img[mask > 0] + 0.45 * tinted[mask > 0]).astype(np.uint8)

    # edge: sometimes strong, sometimes faint (transparent lens)
    faint = rng.random() < 0.45
    edge_color = rng.randint(40, 120) if not faint else rng.randint(150, 200)
    thickness = rng.choice([1, 2, 2, 3])
    cv2.polylines(img, [poly.astype(np.int32)], True,
                  (edge_color,) * 3, thickness, cv2.LINE_AA)

    # glare streak
    if rng.random() < 0.4:
        x0, y0 = rng.uniform(0, SIZE), rng.uniform(0, SIZE)
        ang = rng.uniform(0, math.pi)
        length = rng.uniform(60, 200)
        x1, y1 = x0 + length * math.cos(ang), y0 + length * math.sin(ang)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 (255, 255, 255), rng.choice([2, 3, 5]), cv2.LINE_AA)

    # shadow blob
    if rng.random() < 0.3:
        sx, sy = rng.uniform(0, SIZE), rng.uniform(0, SIZE)
        cv2.ellipse(img, (int(sx), int(sy)),
                    (rng.randint(20, 60), rng.randint(10, 30)),
                    rng.uniform(0, 180), 0, 360, (bg - 30,) * 3, -1)

    # blur + noise
    if rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
    noise = rng.uniform(0, 6)
    if noise > 0.5:
        img = np.clip(img.astype(float) + npr.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    meta = {"shape": shape, "w_px": w, "h_px": h, "faint": faint}
    return img, mask, meta
