"""Evaluation: end-to-end metrics, ablation study, and per-slice breakdown.

Ground truth format (CSV, one row per plate):
    image_path, x1, y1, x2, y2, text, layout, lighting, angle
where ``layout ∈ {one_line, two_line}``, ``lighting ∈ {day, night}``,
``angle ∈ {frontal, oblique}``. Missing slice columns are allowed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src import Box
from src.pipeline import PipelineConfig, PlateResult

# Ablation variants: name → overrides applied on top of the base PipelineConfig.
ABLATIONS: dict[str, dict] = {
    "baseline": {"use_rectify": False, "use_postprocess": False},
    "+postprocess": {"use_rectify": False, "use_postprocess": True},
    "+rectify": {"use_rectify": True, "use_postprocess": False},
    "full": {"use_rectify": True, "use_postprocess": True},
}

SLICE_KEYS = ("layout", "lighting", "angle", "size")


@dataclass
class GTPlate:
    image_path: Path
    box: Box
    text: str
    attrs: dict[str, str]


@dataclass
class Metrics:
    n: int
    det_recall: float      # GT plates matched by a detection (IoU >= thr)
    det_precision: float
    exact_match: float     # full plate string correct (end-to-end)
    cer: float             # mean character error rate on matched plates
    fps: float


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``xyxy`` boxes."""
    # TODO
    raise NotImplementedError


def cer(pred: str, gt: str) -> float:
    """Character error rate = Levenshtein(pred, gt) / len(gt)."""
    # TODO: Levenshtein.distance
    raise NotImplementedError


def exact_match(pred: str, gt: str) -> bool:
    """Exact match after normalization (see ``ocr.normalize_plate_text``)."""
    # TODO
    raise NotImplementedError


def load_ground_truth(csv_path: str | Path) -> list[GTPlate]:
    """Load GT CSV; derive ``size`` slice (small/medium/large) from box height."""
    # TODO
    raise NotImplementedError


def match_predictions(
    preds: list[PlateResult], gts: list[GTPlate], iou_thr: float = 0.5
) -> list[tuple[GTPlate, PlateResult | None]]:
    """Greedy IoU matching of predictions to GT plates of the same image."""
    # TODO
    raise NotImplementedError


def evaluate(config: PipelineConfig, gt: list[GTPlate], iou_thr: float = 0.5) -> tuple[Metrics, list[dict]]:
    """Run the pipeline over the GT images.

    Returns:
        ``(overall_metrics, per_plate_rows)``; rows feed :func:`metrics_by_slice`.
    """
    # TODO
    raise NotImplementedError


def metrics_by_slice(rows: list[dict], keys: tuple[str, ...] = SLICE_KEYS) -> dict[str, dict[str, Metrics]]:
    """Group per-plate rows by each slice key and compute metrics per group."""
    # TODO
    raise NotImplementedError


def run_ablation(base: PipelineConfig, gt: list[GTPlate]) -> dict[str, Metrics]:
    """Evaluate every variant in :data:`ABLATIONS`."""
    # TODO: dataclasses.replace(base, **overrides) for each variant
    raise NotImplementedError


def to_markdown_table(results: dict[str, Metrics]) -> str:
    """Render results as a Markdown table (pasted into README)."""
    # TODO
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate vn-plate-ocr")
    p.add_argument("--gt", type=Path, required=True, help="Ground-truth CSV")
    p.add_argument("--weights", type=Path, default=Path("weights/plate_yolov8n.pt"))
    p.add_argument("--ablation", action="store_true", help="Run all ABLATIONS variants")
    p.add_argument("--slices", action="store_true", help="Report metrics per slice")
    p.add_argument("--out", type=Path, default=Path("runs/eval"))
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
