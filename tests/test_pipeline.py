"""Unit tests for pure-logic parts (no model weights required).

Tests marked ``TODO`` are expected to fail until the skeleton is implemented;
remove the marker once the function is done.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import cer, iou
from src.ocr import format_plate, is_valid_vn_plate, normalize_plate_text
from src.rectify import order_points, rectify_plate
from src.video_pipeline import PlateVoter

TODO = pytest.mark.xfail(reason="skeleton not implemented yet", raises=NotImplementedError, strict=False)


# --- ocr -------------------------------------------------------------------
@TODO
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("51A-123.45", "51A12345"), ("59-x1 123.45", "59X112345"), (" 30e 999.99 ", "30E99999")],
)
def test_normalize_plate_text(raw: str, expected: str) -> None:
    assert normalize_plate_text(raw) == expected


@TODO
@pytest.mark.parametrize(
    ("text", "valid"),
    [("51A12345", True), ("59X112345", True), ("30E9999", True), ("ABC123", False), ("", False)],
)
def test_is_valid_vn_plate(text: str, valid: bool) -> None:
    assert is_valid_vn_plate(text) is valid


@TODO
def test_format_plate() -> None:
    assert format_plate("51A12345") == "51A-123.45"


# --- rectify ---------------------------------------------------------------
@TODO
def test_order_points() -> None:
    pts = np.array([[10, 10], [0, 10], [0, 0], [10, 0]], dtype=np.float32)
    expected = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    np.testing.assert_array_equal(order_points(pts), expected)


@TODO
def test_rectify_fallback_on_blank_crop() -> None:
    crop = np.zeros((40, 120, 3), dtype=np.uint8)
    warped, ok = rectify_plate(crop)
    assert ok is False
    assert warped.shape == crop.shape


# --- evaluate --------------------------------------------------------------
@TODO
def test_cer() -> None:
    assert cer("51A12345", "51A12345") == 0.0
    assert cer("51A12346", "51A12345") == pytest.approx(1 / 8)


@TODO
def test_iou() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# --- video voting ----------------------------------------------------------
@TODO
def test_voter_majority() -> None:
    voter = PlateVoter(min_readings=1, min_conf=0.0)
    for text, conf in [("51A12345", 0.9), ("51A12845", 0.5), ("51A12345", 0.8)]:
        voter.add(1, text, conf)
    text, _ = voter.vote(1)
    assert text == "51A12345"


# --- integration (needs weights) -------------------------------------------
@pytest.mark.skip(reason="requires trained weights in weights/")
def test_pipeline_end_to_end() -> None:
    from src.pipeline import PlatePipeline

    results = PlatePipeline().process_image(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(results, list)
