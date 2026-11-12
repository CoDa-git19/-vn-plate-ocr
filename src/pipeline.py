"""Single-image pipeline: detect → crop → (rectify) → OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import Box
from src.detect import PlateDetector
from src.ocr import PlateReader


@dataclass
class PipelineConfig:
    """Pipeline switches — also used as ablation knobs in ``evaluate.py``."""

    weights: str | Path = "weights/plate_yolov8n.pt"
    det_conf: float = 0.25
    imgsz: int = 640
    crop_pad: float = 0.05
    use_rectify: bool = True
    use_postprocess: bool = True  # regex/confusion fixes in ocr.py
    ocr_min_conf: float = 0.0
    device: str | None = None
    gpu_ocr: bool = False


@dataclass
class PlateResult:
    """One recognized plate."""

    box: Box
    det_conf: float
    text: str
    ocr_conf: float
    rectified: bool = False
    is_valid: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


class PlatePipeline:
    """End-to-end plate recognition for still images."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Build detector + reader from ``config``."""
        self.config = config or PipelineConfig()
        self.detector: PlateDetector | None = None  # TODO
        self.reader: PlateReader | None = None  # TODO

    def recognize_crop(self, crop: np.ndarray) -> tuple[str, float, bool]:
        """Rectify (if enabled) and OCR a single crop → ``(text, conf, rectified)``."""
        # TODO
        raise NotImplementedError

    def process_image(self, image: np.ndarray) -> list[PlateResult]:
        """Run the full pipeline on a BGR image.

        Returns:
            One :class:`PlateResult` per detected plate, with per-stage timings.
        """
        # TODO: detect → crop_box → recognize_crop → PlateResult
        raise NotImplementedError

    def __call__(self, image: np.ndarray) -> list[PlateResult]:
        return self.process_image(image)


def draw_results(image: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    """Return a copy of ``image`` with boxes and plate text drawn on it."""
    # TODO: cv2.rectangle + cv2.putText (or PIL for nicer fonts)
    raise NotImplementedError


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, supporting non-ASCII (Vietnamese) paths."""
    # TODO: cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    raise NotImplementedError
