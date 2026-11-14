"""Perspective rectification of plate crops.

Finds the plate border inside a (slightly padded) crop and warps it to an
axis-aligned rectangle, which usually helps OCR on tilted plates.

Algorithm:
    1. grayscale -> GaussianBlur 5x5 -> Canny
    2. findContours, keep the contour with the largest area
    3. cv2.minAreaRect -> cv2.boxPoints: 4 corners
    4. order_points: top-left, top-right, bottom-right, bottom-left (coordinate sum / difference)
    5. target width / height = Euclidean distances between the ordered corners
    6. getPerspectiveTransform + warpPerspective

Safety: the original crop is returned with ``False`` when ANY of these holds
    * no contour
    * minAreaRect rotation < 5 degrees (already straight)
    * contour area < 30 % of the crop area (not the plate border)
    * warped aspect ratio w/h outside [1.0, 6.0] (1-line plate) or [0.8, 2.0] (2-line plate)

CLI:
    python -m src.rectify --input crop.jpg --output docs/rectify_debug.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_ANGLE_DEG = 5.0
MIN_AREA_RATIO = 0.3
ASPECT_RANGES = {"one_line": (1.0, 6.0), "two_line": (0.8, 2.0)}
# Layout guess from the crop's w/h: VN 1-line plates are ~4.7:1 (520x110 mm),
# 2-line plates ~2:1 (330x165 mm, cars) or ~1.4:1 (190x140 mm, motorbikes).
ONE_LINE_MIN_CROP_RATIO = 2.0
CANNY_LOW, CANNY_HIGH = 50, 150


@dataclass
class RectifyInfo:
    """Why rectification succeeded or fell back (used by :func:`debug_rectify`)."""

    ok: bool
    reason: str
    layout: str = ""
    angle: float | None = None       # rotation of minAreaRect from horizontal/vertical, degrees
    area_ratio: float | None = None  # contour area / crop area
    aspect: float | None = None      # warped w / h


def rectify_plate(
    crop: np.ndarray,
    min_area_ratio: float = MIN_AREA_RATIO,
    layout: str | None = None,
) -> tuple[np.ndarray, bool]:
    """Rectify a plate crop.

    Args:
        crop: BGR (or gray / BGRA) plate crop.
        min_area_ratio: Reject contours covering less than this fraction of the crop.
        layout: ``"one_line"`` or ``"two_line"``; ``None`` guesses it from the crop's
            aspect ratio (``w/h >= 2.0`` -> one line). Selects the allowed warped aspect range.

    Returns:
        ``(warped, True)`` on success, otherwise ``(crop, False)`` (the very same array)
        so the caller can fall back to the raw crop.
    """
    warped, info = _rectify(crop, min_area_ratio, layout)
    return warped, info.ok


def _rectify(
    crop: np.ndarray,
    min_area_ratio: float = MIN_AREA_RATIO,
    layout: str | None = None,
) -> tuple[np.ndarray, RectifyInfo]:
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 2:
        return crop, RectifyInfo(False, "empty crop")
    h, w = crop.shape[:2]
    layout = layout or ("one_line" if w / h >= ONE_LINE_MIN_CROP_RATIO else "two_line")
    if layout not in ASPECT_RANGES:
        raise ValueError(f"layout must be one of {sorted(ASPECT_RANGES)}, got {layout!r}")

    # 1. grayscale -> blur -> edges
    gray = _to_gray(crop)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), CANNY_LOW, CANNY_HIGH)

    # 2. largest contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop, RectifyInfo(False, "no contour", layout)
    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / (w * h)

    # 3. rotated bounding rectangle -> 4 corners
    rect = cv2.minAreaRect(contour)
    angle = _rotation_deg(rect[2])
    info = RectifyInfo(False, "", layout, angle=angle, area_ratio=area_ratio)
    if angle < MIN_ANGLE_DEG:
        info.reason = f"already straight ({angle:.1f} deg < {MIN_ANGLE_DEG:g})"
        return crop, info
    if area_ratio < min_area_ratio:
        info.reason = f"contour too small ({area_ratio:.0%} < {min_area_ratio:.0%} of crop)"
        return crop, info

    # 4-6. order corners, warp to a w x h rectangle
    warped = four_point_transform(crop, cv2.boxPoints(rect))
    info.aspect = warped.shape[1] / warped.shape[0]
    lo, hi = ASPECT_RANGES[layout]
    if not lo <= info.aspect <= hi:
        info.reason = f"aspect {info.aspect:.2f} outside [{lo:g}, {hi:g}] for {layout}"
        return crop, info

    info.ok, info.reason = True, "rectified"
    return warped, info


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    top-left has the smallest ``x + y``, bottom-right the largest; top-right has
    the smallest ``y - x``, bottom-left the largest.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral ``pts`` to a rectangle.

    The target width is the longer of the top / bottom edges and the target height
    the longer of the left / right edges (Euclidean distances).
    """
    tl, tr, br, bl = order_points(pts)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    w, h = max(1, round(width)), max(1, round(height))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _rotation_deg(angle: float) -> float:
    """Deviation of a minAreaRect angle from the nearest axis, in [0, 45] degrees.

    OpenCV >= 4.5.1 reports angles in (0, 90], older versions in [-90, 0):
    both 0 and 90 mean "axis-aligned".
    """
    a = abs(angle) % 90.0
    return min(a, 90.0 - a)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img


# --------------------------------------------------------------------------- #
# Debug
# --------------------------------------------------------------------------- #
def debug_rectify(crop: np.ndarray, out_path: str | Path, height: int = 200) -> RectifyInfo:
    """Save ``[before | after]`` side by side, both scaled to ``height`` px, with a caption.

    "after" is the rectified crop, or the original again when rectification fell
    back; the caption under it gives the reason.
    """
    warped, info = _rectify(crop)
    before, after = _to_bgr(crop), _to_bgr(warped)

    def fit(img: np.ndarray) -> np.ndarray:
        scale = height / img.shape[0]
        return cv2.resize(img, (max(1, round(img.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC)

    def caption(img: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
        bar = np.full((34, max(img.shape[1], 8 * len(text) + 16), 3), 255, np.uint8)
        cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        body = np.full((height, bar.shape[1], 3), 255, np.uint8)
        body[:, : img.shape[1]] = img
        return np.vstack([body, bar])

    angle = f"{info.angle:.1f} deg" if info.angle is not None else "-"
    area = f"{info.area_ratio:.0%}" if info.area_ratio is not None else "-"
    left = caption(fit(before), f"before  angle {angle}, area {area}", (0, 0, 0))
    right = caption(fit(after), f"after: {info.reason}", (0, 150, 0) if info.ok else (0, 100, 230))
    gap = np.full((left.shape[0], 12, 3), 255, np.uint8)
    sheet = np.hstack([left, gap, right])  # same height: `height` + caption bar

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(out_path.suffix or ".png", sheet)
    if not ok:
        raise ValueError(f"Cannot encode image as {out_path.suffix!r}")
    buf.tofile(str(out_path))
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rectify a plate crop and save a before/after image")
    p.add_argument("--input", type=Path, required=True, help="Plate crop")
    p.add_argument("--output", type=Path, default=Path("docs/rectify_debug.png"))
    args = p.parse_args(argv)

    if not args.input.is_file():
        sys.exit(f"Image not found: {args.input}")
    data = np.fromfile(str(args.input), dtype=np.uint8)
    crop = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if crop is None:
        sys.exit(f"Cannot decode image: {args.input}")

    info = debug_rectify(crop, args.output)
    h, w = crop.shape[:2]
    print(f"input   : {args.input} ({w}x{h}), layout {info.layout or '-'}")
    if info.angle is not None:
        aspect = f"{info.aspect:.2f}" if info.aspect is not None else "-"
        print(f"rect    : angle {info.angle:.1f} deg, contour area {info.area_ratio:.0%}, warped aspect {aspect}")
    print(f"result  : {'RECTIFIED' if info.ok else 'fallback'} - {info.reason}")
    print(f"output  : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
