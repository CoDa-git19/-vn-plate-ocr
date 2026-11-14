"""Single-image pipeline: detect -> crop (+5 % padding) -> (rectify) -> OCR -> normalize.

Why two stages instead of OCR on the whole image?
  * OCR models read text *anywhere*: on a full street scene they also read shop
    signs, banners, stickers... and there is no reliable way to tell which string
    is the plate. YOLO localises the plate first, so OCR only ever sees plate pixels.
  * A plate is often < 2 % of the frame. Cropping then upscaling the crop
    (``ocr.preprocess``) gives the characters enough pixels; a full-image OCR pass
    would see them at a few pixels high and miss or garble them.
  * Each crop is one plate, so multiple plates in one image are separated for free
    and every reading comes with its own box.
  * It is faster: OCR (the slow part) runs on a few small crops, not on a 1080p frame,
    and it is skipped entirely when no plate is detected.
  * Each stage can be measured, tuned and swapped on its own (detector mAP vs OCR CER),
    and a rectification step can be inserted between them.

CLI:
    python -m src.pipeline --image x.jpg --output y.jpg --weights models/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from src.detect import Detection, PlateDetector, pad_box
from src.ocr import get_reader, normalize_vn_plate, read_plate

DEFAULT_WEIGHTS = Path("models/best.pt")
CROP_PAD = 0.05  # 5 % of the box size on every side

COLOR_VALID = (0, 200, 0)      # BGR green
COLOR_INVALID = (0, 140, 255)  # BGR orange
COLOR_TEXT = (255, 255, 255)

# One result per plate, see PlateOCRPipeline.process()
PlateResult = dict[str, Any]


class _Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


@dataclass
class PipelineConfig:
    """Pipeline switches, also used as ablation knobs in ``evaluate.py``."""

    weights: str | Path = DEFAULT_WEIGHTS
    det_conf: float = 0.25
    imgsz: int = 640
    crop_pad: float = CROP_PAD
    use_rectify: bool = False
    use_postprocess: bool = True  # positional fixes + format check in ocr.normalize_vn_plate
    ocr_min_conf: float = 0.0
    device: str | None = None
    gpu_ocr: bool = False


class PlateOCRPipeline:
    """End-to-end plate recognition for still images (YOLOv8 + EasyOCR)."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        conf: float = 0.25,
        use_rectify: bool = False,
        *,
        pad: float = CROP_PAD,
        imgsz: int = 640,
        device: str | None = None,
        detector: _Detector | None = None,
    ) -> None:
        """Load the detector (the EasyOCR reader is a lazy module-level singleton).

        Args:
            weights: YOLO ``.pt`` / ``.onnx`` weights.
            conf: Minimum detection confidence.
            use_rectify: Perspective rectification before OCR. Reserved: ``src/rectify.py``
                is not implemented yet, so crops are currently passed through unchanged.
            pad: Relative padding around each box before cropping.
            imgsz: YOLO inference size.
            device: ``"cpu"``, ``"cuda:0"``... ``None`` lets Ultralytics choose.
            detector: Inject a ready detector (tests, sharing one model); ``weights`` is then ignored.
        """
        self.conf = conf
        self.use_rectify = use_rectify
        self.pad = pad
        self.detector = detector or PlateDetector(weights, conf_thres=conf, imgsz=imgsz, device=device)
        self.last_latency_ms: dict[str, float] = {}

    @classmethod
    def from_config(cls, config: PipelineConfig) -> PlateOCRPipeline:
        return cls(config.weights, conf=config.det_conf, use_rectify=config.use_rectify,
                   pad=config.crop_pad, imgsz=config.imgsz, device=config.device)

    def warmup(self) -> None:
        """Load EasyOCR and run one dummy detection so the first real call is not slowed down."""
        get_reader()
        self.detector.detect(np.zeros((64, 64, 3), np.uint8))

    # ------------------------------------------------------------------ #
    def rectify(self, crop: np.ndarray) -> tuple[np.ndarray, bool]:
        """Placeholder for ``src.rectify.rectify_plate`` -> ``(crop, rectified)``."""
        # TODO: once src/rectify.py is implemented:
        #   if self.use_rectify: return rectify_plate(crop)
        return crop, False

    def recognize_crop(self, crop: np.ndarray) -> dict[str, Any]:
        """Rectify (if enabled) and OCR one crop -> text fields + per-stage latency."""
        t0 = time.perf_counter()
        crop, rectified = self.rectify(crop)
        t1 = time.perf_counter()
        raw_text, ocr_conf = read_plate(crop)
        plate_text, is_valid = normalize_vn_plate(raw_text)
        t2 = time.perf_counter()
        return {
            "raw_text": raw_text,
            "plate_text": plate_text,
            "ocr_conf": float(ocr_conf),
            "is_valid_format": bool(is_valid),
            "rectified": rectified,
            "latency_ms": {"rectify": (t1 - t0) * 1000, "ocr": (t2 - t1) * 1000},
        }

    def process(self, image: np.ndarray) -> list[PlateResult]:
        """Run the full pipeline on a BGR image.

        Returns one dict per detected plate, highest detection confidence first::

            bbox             (x1, y1, x2, y2) detector box, absolute pixels (without padding)
            plate_class      YOLO class id (0 or 1; both classes are plates)
            det_conf         detection confidence
            raw_text         cleaned OCR string (lines joined with '-')
            plate_text       normalized plate, e.g. '51F-12345' (see ocr.normalize_vn_plate)
            ocr_conf         OCR confidence
            is_valid_format  plate_text matches a VN plate format
            rectified        perspective rectification was applied
            latency_ms       {'detect', 'rectify', 'ocr'}; 'detect' is shared by the whole image

        Image-level totals are kept in :attr:`last_latency_ms`. No plate (or an empty
        image) -> ``[]``.
        """
        self.last_latency_ms = {"detect": 0.0, "rectify": 0.0, "ocr": 0.0, "total": 0.0}
        if image is None or image.size == 0:
            return []
        image = _to_bgr(image)
        h, w = image.shape[:2]

        t0 = time.perf_counter()
        detections = self.detector.detect(image)
        detect_ms = (time.perf_counter() - t0) * 1000
        self.last_latency_ms["detect"] = detect_ms

        results: list[PlateResult] = []
        for det in detections:
            padded = pad_box(det.box, w, h, self.pad)
            if padded is None:
                continue
            x1, y1, x2, y2 = padded
            rec = self.recognize_crop(image[y1:y2, x1:x2])
            rec["latency_ms"] = {"detect": detect_ms, **rec["latency_ms"]}
            self.last_latency_ms["rectify"] += rec["latency_ms"]["rectify"]
            self.last_latency_ms["ocr"] += rec["latency_ms"]["ocr"]
            results.append({"bbox": tuple(det.box), "plate_class": int(det.cls),
                            "det_conf": float(det.conf), **rec})

        self.last_latency_ms["total"] = (time.perf_counter() - t0) * 1000
        return results

    def __call__(self, image: np.ndarray) -> list[PlateResult]:
        return self.process(image)


# Backwards-compatible name used by app.py / video_pipeline.py
PlatePipeline = PlateOCRPipeline


# --------------------------------------------------------------------------- #
# Drawing / IO
# --------------------------------------------------------------------------- #
def draw_results(image: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    """Return a copy of ``image`` with boxes and plate text drawn on it.

    Green box = ``is_valid_format``, orange = not. The label (``plate_text``, or
    ``raw_text`` / ``?`` if empty) sits above the box on a filled background,
    or inside the box top when there is no room above.
    """
    out = _to_bgr(image).copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 900)
    thick = max(1, round(2 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for r in results:
        x1, y1, x2, y2 = (int(v) for v in r["bbox"])
        color = COLOR_VALID if r.get("is_valid_format") else COLOR_INVALID
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + 1)

        label = r.get("plate_text") or r.get("raw_text") or "?"
        (tw, th), base = cv2.getTextSize(label, font, scale, thick)
        pad = max(2, thick * 2)
        box_h = th + base + 2 * pad
        top = y1 - box_h if y1 - box_h >= 0 else y1  # no room above -> inside the box
        left = min(max(0, x1), max(0, w - tw - 2 * pad))
        cv2.rectangle(out, (left, top), (left + tw + 2 * pad, top + box_h), color, cv2.FILLED)
        cv2.putText(out, label, (left + pad, top + pad + th), font, scale, COLOR_TEXT, thick, cv2.LINE_AA)
    return out


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, supporting non-ASCII (Vietnamese) paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image, supporting non-ASCII paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Cannot encode image as {path.suffix!r}")
    buf.tofile(str(path))


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect and read Vietnamese license plates in an image")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="Write the annotated image here")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--rectify", action="store_true", help="Perspective rectification (not implemented yet)")
    p.add_argument("--device", default=None, help="cpu, cuda:0, ... (default: auto)")
    args = p.parse_args(argv)

    if not args.image.is_file():
        sys.exit(f"Image not found: {args.image}")
    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")
    image = load_image(args.image)

    t0 = time.perf_counter()
    pipe = PlateOCRPipeline(args.weights, conf=args.conf, use_rectify=args.rectify, device=args.device)
    pipe.warmup()
    load_s = time.perf_counter() - t0

    results = pipe.process(image)
    lat = pipe.last_latency_ms

    h, w = image.shape[:2]
    print(f"image   : {args.image} ({w}x{h})")
    print(f"plates  : {len(results)}")
    for i, r in enumerate(results, 1):
        mark = "OK " if r["is_valid_format"] else "BAD"
        print(f"  [{i}] {mark} {r['plate_text']!r:<14} raw={r['raw_text']!r:<16} "
              f"ocr={r['ocr_conf']:.2f} det={r['det_conf']:.2f} cls={r['plate_class']} "
              f"bbox={r['bbox']} rectified={r['rectified']}")
    print(f"latency : detect {lat['detect']:.0f} ms | rectify {lat['rectify']:.0f} ms | "
          f"ocr {lat['ocr']:.0f} ms | total {lat['total']:.0f} ms  (model load {load_s:.1f} s)")

    if args.output:
        save_image(args.output, draw_results(image, results))
        print(f"output  : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
