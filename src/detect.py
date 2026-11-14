"""License plate detection with a YOLOv8 model (Ultralytics).

Supports both PyTorch (``.pt``) and ONNX (``.onnx``) weights through the
Ultralytics ``YOLO`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import Box

DEFAULT_WEIGHTS = Path("models/best.pt")


@dataclass(frozen=True)
class Detection:
    """A single detected plate."""

    box: Box  # (x1, y1, x2, y2), absolute pixels
    conf: float
    cls: int = 0  # YOLO class id (the plate model has 2 classes, '0' and '1': both are plates)


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
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = device

        from ultralytics import YOLO  # lazy: importing ultralytics/torch is slow

        self.model = YOLO(str(self.weights))

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect plates in a BGR image.

        Args:
            image: HxWx3 BGR ``uint8`` image (OpenCV convention).

        Returns:
            Detections sorted by confidence (descending), boxes clipped to image.
            Every class is kept (all classes of the plate model are plates).
        """
        if image is None or image.size == 0:
            return []
        results = self.model.predict(
            image, conf=self.conf_thres, iou=self.iou_thres, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return []

        h, w = image.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        dets = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes, strict=True):
            box = _clip_box((round(x1), round(y1), round(x2), round(y2)), w, h)
            if box is not None:
                dets.append(Detection(box=box, conf=float(conf), cls=int(cls)))
        return sorted(dets, key=lambda d: d.conf, reverse=True)

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
    global _default_detector
    if _default_detector is None or _default_detector.weights != Path(weights):
        _default_detector = PlateDetector(weights)
    return _default_detector


def detect_plates(
    image: np.ndarray,
    detector: PlateDetector | None = None,
) -> list[tuple[Box, float]]:
    """Functional API: detect plates and return ``[(box, conf), ...]``.

    Args:
        image: BGR image.
        detector: Optional detector instance; defaults to :func:`get_detector`.
    """
    detector = detector or get_detector()
    return [(d.box, d.conf) for d in detector.detect(image)]


def crop_box(image: np.ndarray, box: Box, pad: float = 0.05) -> np.ndarray:
    """Crop ``box`` from ``image`` with relative padding, clipped to bounds.

    ``pad=0.05`` grows the box by 5 % of its width on the left and right and 5 %
    of its height on the top and bottom. Returns an empty array if the box lies
    outside the image.
    """
    padded = pad_box(box, image.shape[1], image.shape[0], pad)
    if padded is None:
        return image[0:0, 0:0]
    x1, y1, x2, y2 = padded
    return image[y1:y2, x1:x2]


def pad_box(box: Box, width: int, height: int, pad: float = 0.05) -> Box | None:
    """Grow ``box`` by ``pad`` of its size on every side, then clip to the image."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return _clip_box((int(x1 - dx), int(y1 - dy), int(round(x2 + dx)), int(round(y2 + dy))), width, height)


def _clip_box(box: Box, width: int, height: int) -> Box | None:
    """Clip to ``[0, width] x [0, height]``; ``None`` if nothing is left."""
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
