"""Train the tiny lens UNet on synthetic rectified-plane data and export ONNX.

Usage (from backend/):
    .venv/Scripts/python.exe -m ml.train --samples 800 --epochs 18 --out ml/lens_unet.onnx

CPU-only, no external data. Saves best by validation IoU.
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .dataset import sample, SIZE
from .unet import TinyUNet, count_params

# Avoid native thread-pool conflicts between torch (OpenMP) and OpenCV.
torch.set_num_threads(1)
try:
    cv2.setNumThreads(0)
except Exception:
    pass


class LensSynthetic(Dataset):
    def __init__(self, n: int, offset: int = 0):
        self.n = n
        self.offset = offset

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        img, mask, _ = sample(seed=self.offset + i)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(mask[None].astype(np.float32) / 255.0)
        return x, y


def iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_b = (pred > 0.5).float()
    inter = (pred_b * target).sum().item()
    union = ((pred_b + target) >= 1).sum().item()
    return inter / max(union, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--val", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", type=str, default="ml/lens_unet.onnx")
    args = ap.parse_args()

    device = torch.device("cpu")
    model = TinyUNet().to(device)
    print(f"params: {count_params(model)}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = LensSynthetic(args.samples, offset=0)
    val_ds = LensSynthetic(args.val, offset=100_000)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=0)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()
        model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for x, y in val_dl:
                logits = model(x.to(device))
                probs = torch.sigmoid(logits).cpu()
                for p, t in zip(probs, y):
                    scores.append(iou(p, t))
        miou = float(np.mean(scores))
        print(f"epoch {epoch+1}/{args.epochs} loss={total/len(train_ds):.4f} val_iou={miou:.3f}")
        if miou >= best:
            best = miou
            torch.save(model.state_dict(), "ml/best_unet.pt")
    print(f"best val IoU: {best:.3f}")

    model.load_state_dict(torch.load("ml/best_unet.pt", map_location=device))
    model.eval()
    dummy = torch.randn(1, 3, SIZE, SIZE)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
        dynamo=False,  # legacy TorchScript exporter: stable on Windows CPU
    )
    print(f"exported {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
