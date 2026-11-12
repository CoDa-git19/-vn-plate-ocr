"""Perspective rectification of plate crops.

Finds the plate's 4 corners inside a (slightly padded) crop and warps it to a
fronto-parallel rectangle, which usually improves OCR on angled plates.
"""

from __future__ import annotations

import numpy as np

# Output sizes (w, h) by plate layout.
ONE_LINE_SIZE = (240, 52)   # car plate ~ 520x110 mm
TWO_LINE_SIZE = (190, 140)  # motorbike plate ~ 190x140 mm


def rectify_plate(crop: np.ndarray, min_area_ratio: float = 0.3) -> tuple[np.ndarray, bool]:
    """Rectify a plate crop.

    Args:
        crop: BGR plate crop.
        min_area_ratio: Reject quads covering less than this fraction of the crop.

    Returns:
        ``(warped, success)``. If no reliable quadrilateral is found, returns
        ``(crop, False)`` so the caller can fall back to the raw crop.
    """
    # TODO: quad = find_plate_quad(crop); if None → (crop, False)
    #       else four_point_transform(crop, quad, size chosen by aspect ratio)
    raise NotImplementedError


def find_plate_quad(crop: np.ndarray, min_area_ratio: float = 0.3) -> np.ndarray | None:
    """Locate the plate border as a 4-point polygon.

    Approach: gray → bilateral filter → Canny → contours → ``approxPolyDP``,
    keep the largest convex 4-gon. Fallback: ``minAreaRect`` of largest contour.

    Returns:
        ``(4, 2)`` float32 array of corners, or ``None``.
    """
    # TODO
    raise NotImplementedError


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    # TODO: sum → tl/br, diff → tr/bl
    raise NotImplementedError


def four_point_transform(image: np.ndarray, pts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp the quadrilateral ``pts`` to a ``size=(w, h)`` rectangle."""
    # TODO: cv2.getPerspectiveTransform + cv2.warpPerspective
    raise NotImplementedError


def deskew(crop: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """Cheaper alternative: rotation-only deskew via Hough lines / minAreaRect."""
    # TODO
    raise NotImplementedError
