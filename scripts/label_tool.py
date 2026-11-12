"""Minimal OpenCV tool to label plate text for crops (OCR ground truth).

Shows each crop with the current OCR prediction pre-filled; type to correct.

Keys:
    Enter   accept / save        Backspace  delete char
    Tab     skip                 Esc        quit (progress is saved)
    1 / 2   set layout one_line / two_line

Usage:
    python scripts/label_tool.py --crops datasets/crops --out datasets/ocr_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

CSV_FIELDS = ("crop_path", "text", "layout")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label plate text for crops")
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("datasets/ocr_labels.csv"))
    p.add_argument("--prefill", action="store_true", help="Pre-fill with EasyOCR prediction")
    p.add_argument("--scale", type=float, default=3.0, help="Display upscale factor")
    return p.parse_args()


def load_existing(out_csv: Path) -> dict[str, dict[str, str]]:
    """Load already-labeled rows so the tool can resume."""
    # TODO
    raise NotImplementedError


def save_row(out_csv: Path, row: dict[str, str]) -> None:
    """Append one labeled row (write header if file is new)."""
    # TODO
    raise NotImplementedError


def label_one(crop_path: Path, initial_text: str, scale: float) -> tuple[str, str] | None:
    """Interactive loop for one crop.

    Returns:
        ``(text, layout)``, or ``None`` if skipped. Raises ``KeyboardInterrupt`` on Esc.
    """
    # TODO: cv2.imshow + cv2.waitKey loop
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
