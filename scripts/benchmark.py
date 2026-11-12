"""Latency / throughput benchmark: PyTorch vs ONNX Runtime, per pipeline stage.

Usage:
    python scripts/benchmark.py --images datasets/plates/images/test --weights weights/plate_yolov8n.pt weights/plate_yolov8n.onnx
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchResult:
    backend: str
    stage: str          # "detect" | "rectify" | "ocr" | "end2end"
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark vn-plate-ocr")
    p.add_argument("--images", type=Path, required=True, help="Folder of test images")
    p.add_argument("--weights", type=Path, nargs="+", required=True, help=".pt and/or .onnx")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("runs/benchmark/results.csv"))
    return p.parse_args()


def time_fn(fn, *args, warmup: int = 10, runs: int = 100) -> list[float]:
    """Call ``fn(*args)`` ``warmup + runs`` times; return per-run latencies in ms."""
    # TODO: time.perf_counter
    raise NotImplementedError


def summarize(backend: str, stage: str, latencies_ms: list[float]) -> BenchResult:
    """Compute mean / p50 / p95 / FPS."""
    # TODO: numpy.percentile
    raise NotImplementedError


def benchmark_weights(weights: Path, images: list[Path], args: argparse.Namespace) -> list[BenchResult]:
    """Benchmark each pipeline stage for one weights file."""
    # TODO
    raise NotImplementedError


def print_table(results: list[BenchResult]) -> None:
    """Print a Markdown table for the README."""
    # TODO
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
