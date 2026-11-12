"""Fine-tune YOLOv8 for plate detection, log to Weights & Biases, export ONNX.

Usage:
    python scripts/train.py --data datasets/plates/data.yaml --model yolov8n.pt --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 plate detector")
    p.add_argument("--data", type=Path, default=Path("datasets/plates/data.yaml"))
    p.add_argument("--model", default="yolov8n.pt", help="Pretrained checkpoint to fine-tune")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="'cpu', '0', '0,1'...")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="plate_yolov8n")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--export-onnx", action="store_true")
    p.add_argument("--weights-out", type=Path, default=Path("weights"))
    return p.parse_args()


def setup_wandb(enabled: bool, project: str = "vn-plate-ocr") -> None:
    """Toggle Ultralytics' built-in W&B integration (``yolo settings wandb=...``)."""
    # TODO: from ultralytics import settings; settings.update({"wandb": enabled})
    raise NotImplementedError


def train(args: argparse.Namespace) -> Path:
    """Run training; return path to ``best.pt``."""
    # TODO: YOLO(args.model).train(data=..., epochs=..., imgsz=..., batch=..., ...)
    raise NotImplementedError


def validate(weights: Path, data: Path) -> dict[str, float]:
    """Validate on the test split; return mAP50, mAP50-95, precision, recall."""
    # TODO: model.val(data=..., split="test")
    raise NotImplementedError


def export_onnx(weights: Path, imgsz: int = 640) -> Path:
    """Export weights to ONNX (dynamic=False, simplify=True)."""
    # TODO
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    # TODO: setup_wandb → train → validate → copy best.pt to weights/ → (export_onnx)
    raise NotImplementedError


if __name__ == "__main__":
    main()
