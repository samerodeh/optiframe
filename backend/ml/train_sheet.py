"""Fine-tune the lens UNet on real sheet photos (Stage 2 of the ArUco track).

Starts from the generic checkpoint (ml/best_unet.pt) when present, otherwise
from scratch, and trains on mixed batches: real sheet pairs from
ml/data/sheet/ (manifest.csv + photos/ + masks/) plus synthetic samples to
avoid forgetting generic shapes. Exports ml/lens_unet_sheet.onnx, which the
app prefers over ml/lens_unet.onnx when both exist.

Usage (from backend/):
    .venv/Scripts/python.exe -m ml.train_sheet --epochs 30
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .dataset import sample as synthetic_sample, SIZE
from .sheet_data import load_manifest, load_pair, DATA_DIR
from .train import iou
from .unet import TinyUNet, count_params


class SheetReal(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i % len(self.records)]
        img, mask = load_pair(rec)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy((mask[None] > 127).astype(np.float32))
        return x, y


class MixedSheet(Dataset):
    """Oversampled real pairs interleaved with synthetic samples."""

    def __init__(self, records: list[dict], n: int, real_ratio: float = 0.7):
        self.records = records
        self.n = n
        self.real_ratio = real_ratio
        self.real = SheetReal(records) if records else None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if self.real is not None and (i % 10) < round(self.real_ratio * 10):
            return self.real[i]
        img, mask, _ = synthetic_sample(seed=500_000 + i)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(mask[None].astype(np.float32) / 255.0)
        return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--init", type=str, default="ml/best_unet.pt")
    ap.add_argument("--out", type=str, default="ml/lens_unet_sheet.onnx")
    args = ap.parse_args()

    records = load_manifest()
    print(f"real pairs: {len(records)}")
    if not records:
        print("Add photos + manifest.csv under ml/data/sheet/ first (see its README).")
        return

    torch.set_num_threads(1)
    model = TinyUNet()
    print(f"params: {count_params(model)}")
    init = pathlib.Path(args.init)
    if init.exists():
        model.load_state_dict(torch.load(init, map_location="cpu"))
        print(f"fine-tuning from {init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    train_dl = DataLoader(MixedSheet(records, args.samples), batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(SheetReal(records), batch_size=args.batch)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        model.eval()
        scores = []
        with torch.no_grad():
            for x, y in val_dl:
                for p, t in zip(torch.sigmoid(model(x)), y):
                    scores.append(iou(p, t))
        miou = float(np.mean(scores))
        print(f"epoch {epoch+1}/{args.epochs} loss={total/args.samples:.4f} real_iou={miou:.3f}", flush=True)
        if miou >= best:
            best = miou
            torch.save(model.state_dict(), "ml/best_unet_sheet.pt")
    print(f"best real IoU: {best:.3f}")
    model.load_state_dict(torch.load("ml/best_unet_sheet.pt", map_location="cpu"))
    model.eval()
    out = pathlib.Path(args.out)
    torch.onnx.export(model, torch.randn(1, 3, SIZE, SIZE), str(out),
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=14, dynamo=False)
    print(f"exported {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
