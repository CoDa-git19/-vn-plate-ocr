"""License plate detection with a YOLOv8 model (Ultralytics).

Supports both PyTorch (``.pt``) and ONNX (``.onnx``) weights through the
Ultralytics ``YOLO`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import Box

DEFAULT_WEIGHTS = Path("weights/plate_yolov8n.pt")


@dataclass(frozen=True)
class Detection:
    """A single detected plate."""

    box: Box  # (x1, y1, x2, y2), absolute pixels
    conf: float


class PlateDetector:
    """Thin wrapper around ``ultralytics.YOLO`` for plate detection."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        """Load the YOLO model.

        Args:
            weights: Path to ``.pt`` or ``.onnx`` weights.
            conf_thres: Minimum confidence to keep a detection.
            iou_thres: IoU threshold for NMS.
            imgsz: Inference image size.
            device: ``"cpu"``, ``"cuda:0"``... ``None`` lets Ultralytics choose.
        """
        self.weights = Path(weights)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = device
        self.model = None  # TODO: self.model = YOLO(str(self.weights))

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect plates in a BGR image.

        Args:
            image: HxWx3 BGR ``uint8`` image (OpenCV convention).

        Returns:
            Detections sorted by confidence (descending), boxes clipped to image.
        """
        # TODO: run self.model.predict(...), convert results[0].boxes to Detection
        raise NotImplementedError

    def track(self, frame: np.ndarray, persist: bool = True) -> list[tuple[int, Detection]]:
        """Detect + track plates in a video frame (ByteTrack via Ultralytics).

        Args:
            frame: BGR video frame.
            persist: Keep tracker state between calls (must be True for video).

        Returns:
            List of ``(track_id, detection)``. Untracked boxes are dropped.
        """
        # TODO: self.model.track(frame, persist=persist, tracker="bytetrack.yaml", ...)
        raise NotImplementedError


_default_detector: PlateDetector | None = None


def get_detector(weights: str | Path = DEFAULT_WEIGHTS) -> PlateDetector:
    """Return a lazily-created module-level detector (avoids reloading weights)."""
    # TODO: create once, cache in _default_detector
    raise NotImplementedError


def detect_plates(
    image: np.ndarray,
    detector: PlateDetector | None = None,
) -> list[tuple[Box, float]]:
    """Functional API: detect plates and return ``[(box, conf), ...]``.

    Args:
        image: BGR image.
        detector: Optional detector instance; defaults to :func:`get_detector`.
    """
    # TODO: detector = detector or get_detector(); return [(d.box, d.conf) for d in ...]
    raise NotImplementedError


def crop_box(image: np.ndarray, box: Box, pad: float = 0.05) -> np.ndarray:
    """Crop ``box`` from ``image`` with relative padding, clipped to bounds."""
    # TODO
    raise NotImplementedError
