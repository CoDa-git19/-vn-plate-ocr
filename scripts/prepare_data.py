"""Prepare datasets for YOLOv8 training and OCR evaluation.

Converts raw annotations (e.g. Roboflow / VOC / custom CSV) into Ultralytics
YOLO format and writes ``datasets/plates/data.yaml``:

    datasets/plates/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt   # "0 cx cy w h" normalized
        data.yaml

Usage:
    python scripts/prepare_data.py --src datasets/raw --out datasets/plates --format voc
"""

from __future__ import annotations

import argparse
from pathlib import Path

SPLITS = ("train", "val", "test")
CLASS_NAMES = ["plate"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Raw dataset directory")
    p.add_argument("--out", type=Path, default=Path("datasets/plates"))
    p.add_argument("--format", choices=["voc", "yolo", "csv"], default="yolo")
    p.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-crops", action="store_true", help="Also save plate crops for OCR labeling")
    return p.parse_args()


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert one Pascal VOC XML file to YOLO label lines."""
    # TODO
    raise NotImplementedError


def split_dataset(items: list[Path], ratios: tuple[float, float, float], seed: int) -> dict[str, list[Path]]:
    """Deterministic random split into train/val/test."""
    # TODO
    raise NotImplementedError


def write_data_yaml(out_dir: Path) -> Path:
    """Write the Ultralytics ``data.yaml`` (path, train, val, test, names)."""
    # TODO
    raise NotImplementedError


def export_crops(out_dir: Path, crops_dir: Path) -> int:
    """Save GT plate crops (for label_tool.py / OCR eval). Returns count."""
    # TODO
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    # TODO: collect items → convert → split → copy → write_data_yaml → (export_crops)
    raise NotImplementedError


if __name__ == "__main__":
    main()
