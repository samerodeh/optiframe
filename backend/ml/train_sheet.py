"""Fine-tune the lens UNet on real sheet photos (Stage 2 of the ArUco track).

Starts from the required generic checkpoint (ml/best_unet.pt), and trains on mixed batches: real sheet pairs from
ml/data/sheet/ (manifest.csv + photos/ + masks/) plus synthetic samples to
avoid forgetting generic shapes. Holds out complete lens IDs for validation.
Exports a candidate under ml/runs/sheet/ without replacing the deployed model.
Label-anchored crops approximate the inference ROI; end-to-end candidate
discovery and rectification still require separate evaluation on real photos.

Usage (from backend/):
    .venv/Scripts/python.exe -m ml.train_sheet --data-dir PATH --dry-run
    .venv/Scripts/python.exe -m ml.train_sheet --data-dir PATH --epochs 30 --batch 2
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .dataset import sample as synthetic_sample, SIZE
from .sheet_data import load_manifest, load_pair, crop_pair, split_by_lens, explicit_split, DATA_DIR
from .train import iou
from .unet import TinyUNet, count_params


class SheetReal(Dataset):
    def __init__(self, records: list[dict], data_dir: pathlib.Path = DATA_DIR, augment=False):
        self.records = records
        self.data_dir = data_dir
        self.augment = augment
        self.pairs = [crop_pair(*load_pair(r, data_dir)) for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        img, mask = self.pairs[i % len(self.records)]
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            # Same transform on image/label; augmentation never touches held-out photos.
            k = int(np.random.randint(4))
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if np.random.random() < .5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = np.clip(img.astype(np.float32) * np.random.uniform(.8, 1.15)
                          + np.random.uniform(-10, 10), 0, 255).astype(np.uint8)
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy((mask[None] > 127).astype(np.float32))
        return x, y


class MixedSheet(Dataset):
    """Oversampled real pairs interleaved with synthetic samples."""

    def __init__(self, records: list[dict], n: int, real_ratio: float = 0.7,
                 data_dir: pathlib.Path = DATA_DIR):
        self.records = records
        self.n = n
        self.real_ratio = real_ratio
        self.real = SheetReal(records, data_dir, augment=True) if records else None
        # Independent counters prevent fixed synthetic slots from permanently
        # excluding real records whose indexes have the same modulo-10 residue.
        real_count = 0
        self.real_indices = {}
        for i in range(n):
            if (i % 10) < round(real_ratio * 10):
                self.real_indices[i] = real_count
                real_count += 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if self.real is not None and (i % 10) < round(self.real_ratio * 10):
            return self.real[self.real_indices[i]]
        img, mask, _ = synthetic_sample(seed=500_000 + i)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(mask[None].astype(np.float32) / 255.0)
        return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--init", type=str, default="ml/best_unet.pt")
    ap.add_argument("--out", type=str, default="ml/runs/sheet/candidate.onnx")
    ap.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-file", type=pathlib.Path,
                    help="Explicit capture-group train/validation/test split for limited pilot datasets")
    ap.add_argument("--dry-run", action="store_true", help="Validate pairs and lens split without training")
    args = ap.parse_args()

    if min(args.epochs, args.batch, args.samples) < 1:
        ap.error("epochs, batch and samples must be positive")
    records = load_manifest(args.data_dir / "manifest.csv")
    print(f"real pairs: {len(records)}")
    if not records:
        ap.error(f"No dataset records found in {args.data_dir / 'manifest.csv'}")
    if args.split_file:
        train_records, val_records, test_records = explicit_split(records, args.split_file)
        print("Capture-group pilot: does not measure generalization to unseen lenses.")
    else:
        train_records, val_records = split_by_lens(records, args.seed)
        test_records = []
    for record in records:
        crop_pair(*load_pair(record, args.data_dir))
    print(f"train: {len(train_records)} photos; validation: {len(val_records)} photos; "
          f"test: {len(test_records)} photos (not used by trainer)", flush=True)
    if args.dry_run:
        return

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cv2.setNumThreads(0)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out.with_suffix(".best.pt")
    model = TinyUNet()
    print(f"params: {count_params(model)}")
    init = pathlib.Path(args.init)
    if init.exists():
        model.load_state_dict(torch.load(init, map_location="cpu"))
        print(f"fine-tuning from {init}")
    else:
        ap.error(f"Initial checkpoint is missing: {init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    train_dl = DataLoader(MixedSheet(train_records, args.samples, data_dir=args.data_dir),
                          batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(SheetReal(val_records, args.data_dir), batch_size=args.batch, num_workers=0)

    def validation_iou():
        model.eval()
        scores = []
        with torch.no_grad():
            for x, y in val_dl:
                for p, t in zip(torch.sigmoid(model(x)), y):
                    scores.append(iou(p, t))
        return float(np.mean(scores))

    best = validation_iou()
    torch.save(model.state_dict(), checkpoint)
    history = [{"epoch": 0, "validation_iou": best}]
    print(f"baseline validation IoU: {best:.4f}", flush=True)
    metadata = {"seed": args.seed, "epochs": args.epochs, "batch": args.batch,
                "samples_per_epoch": args.samples, "lr": args.lr,
                "train": [r["filename"] for r in train_records],
                "validation": [r["filename"] for r in val_records],
                "test": [r["filename"] for r in test_records], "history": history}
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        miou = validation_iou()
        history.append({"epoch": epoch+1, "validation_iou": miou, "loss": total/args.samples})
        print(f"epoch {epoch+1}/{args.epochs} loss={total/args.samples:.4f} real_iou={miou:.3f}", flush=True)
        if miou >= best:
            best = miou
            pending = checkpoint.with_suffix(".tmp")
            torch.save(model.state_dict(), pending)
            pending.replace(checkpoint)
        out.with_suffix(".training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"best real IoU: {best:.3f}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    probe = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        np.savez(out.with_suffix(".parity.npz"), image=probe.numpy(), logits=model(probe).numpy())
    torch.onnx.export(model, probe, str(out),
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=14, dynamo=False)
    print(f"exported {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
