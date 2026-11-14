"""Unit tests for pure-logic parts (no model weights required).

Tests marked ``TODO`` are expected to fail until the skeleton is implemented;
remove the marker once the function is done.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import src.ocr as ocr
from src.evaluate import cer, exact_match, iou
from src.ocr import clean_text, merge_lines, normalize_vn_plate, preprocess, read_plate
from src.rectify import order_points, rectify_plate
from src.video_pipeline import PlateVoter

TODO = pytest.mark.xfail(reason="skeleton not implemented yet", raises=NotImplementedError, strict=False)


# --- ocr -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("51f-123.45 ", "51F-123.45"), ("51F 123,45", "51F12345"), ("59-H1\n234.56", "59-H1234.56"), ("", "")],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # already correct, various separators
        ("51F-123.45", ("51F-12345", True)),
        ("51F12345", ("51F-12345", True)),
        ("30E-9999", ("30E-9999", True)),
        # 2-line motorbike joined by merge_lines, or read as one line
        ("59-H1-234.56", ("59H1-23456", True)),
        ("59-H1234.56", ("59H1-23456", True)),
        ("59H1-2345", ("59H1-2345", True)),
        # province code (positions 1-2) must be digits
        ("S1F-12345", ("51F-12345", True)),
        ("5IF-12345", ("51F-12345", True)),
        ("B8B-1234", ("88B-1234", True)),
        ("OQA-12345", ("00A-12345", True)),
        ("DLF-12345", ("01F-12345", True)),
        # series (position 3) must be a valid letter: digit -> letter
        ("510-12345", ("51D-12345", True)),
        ("518-12345", ("51B-12345", True)),
        ("515-12345", ("51S-12345", True)),
        ("512-12345", ("51Z-12345", True)),
        ("516-12345", ("51G-12345", True)),
        # ... and non-existent letters O/Q -> D
        ("51O-12345", ("51D-12345", True)),
        ("51Q-12345", ("51D-12345", True)),
        # serial digits: O/Q/D->0, I/J/L->1, Z->2, A->4, S->5, G->6, T->7, B->8
        ("51F-1234S", ("51F-12345", True)),
        ("59H1-2345O", ("59H1-23450", True)),
        ("51F-IZAST", ("51F-12457", True)),
        ("51F-GBOQD", ("51F-68000", True)),
        ("51F-JL123", ("51F-11123", True)),
        # motorbike with a 2-letter series
        ("29AA-12345", ("29AA-12345", True)),
        ("29-AA-123.45", ("29AA-12345", True)),
        ("29AA12345", ("29AA-12345", True)),
        ("29AC1234", ("29AC-1234", True)),  # 8 chars, 'C' can't be a digit -> 2-letter series
    ],
)
def test_normalize_vn_plate_fixes_by_position(raw: str, expected: tuple[str, bool]) -> None:
    assert normalize_vn_plate(raw) == expected


@pytest.mark.parametrize("letter", ["I", "J", "W", "R"])
def test_normalize_does_not_guess_unused_letters(letter: str) -> None:
    assert normalize_vn_plate(f"51{letter}-12345") == (f"51{letter}-12345", False)


def test_normalize_no_longer_maps_1_to_I() -> None:
    assert normalize_vn_plate("511-12345") == ("511-12345", False)


def test_normalize_leaves_ambiguous_4th_series_char() -> None:
    # 4th char of a 4-char series may be a digit (59H1) or a letter (29AA): 'O' is not guessed
    assert normalize_vn_plate("29AO-12345") == ("29AO-12345", False)
    # valid letters that are also in DIGIT_FIX stay letters there
    assert normalize_vn_plate("29AS-12345") == ("29AS-12345", True)
    assert normalize_vn_plate("59H1-23456") == ("59H1-23456", True)


def test_normalize_does_not_fix_blindly() -> None:
    # 'B' / 'S' at the series position are valid letters and must stay letters
    assert normalize_vn_plate("51B-12345") == ("51B-12345", True)
    assert normalize_vn_plate("29S-12345") == ("29S-12345", True)
    # implausible length: no positional fixes at all
    assert normalize_vn_plate("SO1") == ("SO1", False)
    assert normalize_vn_plate("51F-1234567") == ("51F1234567", False)


@pytest.mark.parametrize("raw", ["51FF-1234X", "12-34567", "51W-1234", "5-12345", ""])
def test_normalize_invalid(raw: str) -> None:
    assert normalize_vn_plate(raw)[1] is False


def test_merge_lines_orders_by_y_then_x() -> None:
    def box(x1, y1, x2, y2):
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    fragments = [
        (box(10, 60, 120, 100), "234.56", 0.8),  # bottom line first
        (box(60, 10, 110, 50), "H1", 0.9),       # top line, right part
        (box(10, 12, 55, 48), "59-", 0.7),       # top line, left part
    ]
    text, conf, n_lines = merge_lines(fragments)
    assert text == "59-H1-234.56"
    assert n_lines == 2
    assert conf == pytest.approx((0.8 * 6 + 0.9 * 2 + 0.7 * 3) / 11)
    assert merge_lines([]) == ("", 0.0, 0)


@pytest.mark.parametrize("shape", [(20, 80, 3), (40, 160), (120, 400, 3), (30, 100, 4)])
def test_preprocess_output(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    out = preprocess(rng.integers(0, 255, shape, dtype=np.uint8))
    assert out.ndim == 2 and out.dtype == np.uint8
    assert out.shape[0] == max(64, shape[0])
    assert set(np.unique(out)) <= {0, 255}


@pytest.mark.parametrize(
    ("bg", "fg"),
    [((255, 255, 255), (0, 0, 0)),     # white plate, black text
     ((0, 200, 255), (0, 0, 0)),       # yellow plate, black text
     ((160, 60, 20), (255, 255, 255))],  # blue plate, white text -> must be inverted
)
def test_preprocess_makes_text_dark_on_light(bg, fg) -> None:
    plate = np.full((80, 320, 3), bg, np.uint8)  # >= 64 px high: no resize, masks align
    cv2.putText(plate, "80A123", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, fg, 6)
    text_mask = np.all(plate == fg, axis=2)
    out = preprocess(plate)
    assert out[text_mask].mean() < 64      # text pixels -> black
    assert out[~text_mask].mean() > 191    # background -> white


class _FakeReader:
    """Returns different readings for the original (3-channel) and pre-processed (2-D) image."""

    def __init__(self, original: list, preprocessed: list) -> None:
        self.original, self.preprocessed = original, preprocessed

    def readtext(self, img, **kwargs):
        return self.preprocessed if img.ndim == 2 else self.original


_BOX = [[0, 0], [100, 0], [100, 30], [0, 30]]


@pytest.mark.parametrize(
    ("orig_conf", "pre_conf", "expected"),
    [(0.9, 0.6, ("51F-123.45", 0.9)), (0.4, 0.8, ("51F12345", 0.8))],
)
def test_read_plate_picks_higher_confidence(monkeypatch, orig_conf, pre_conf, expected) -> None:
    fake = _FakeReader([(_BOX, "51f-123.45", orig_conf)], [(_BOX, "51F 12345", pre_conf)])
    monkeypatch.setattr(ocr, "_reader", fake)
    text, conf = read_plate(np.zeros((30, 100, 3), np.uint8))
    assert (text, pytest.approx(conf)) == (expected[0], expected[1])


def test_read_plate_empty(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "_reader", _FakeReader([], []))
    assert read_plate(np.zeros((30, 100, 3), np.uint8)) == ("", 0.0)
    assert read_plate(np.zeros((0, 0, 3), np.uint8)) == ("", 0.0)


# --- rectify ---------------------------------------------------------------
def test_order_points() -> None:
    pts = np.array([[10, 10], [0, 10], [0, 0], [10, 0]], dtype=np.float32)
    expected = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    np.testing.assert_array_equal(order_points(pts), expected)


def test_order_points_tilted() -> None:
    # rectangle rotated ~10 deg clockwise, corners shuffled
    tl, tr, br, bl = [20, 10], [120, 28], [113, 68], [13, 50]
    np.testing.assert_array_equal(order_points(np.array([br, tl, bl, tr])), np.array([tl, tr, br, bl], np.float32))


def test_rectify_fallback_on_blank_crop() -> None:
    crop = np.zeros((40, 120, 3), dtype=np.uint8)
    warped, ok = rectify_plate(crop)
    assert ok is False
    assert warped is crop


def _synthetic_plate(angle: float, size: tuple[int, int] = (240, 52), canvas: tuple[int, int] = (270, 115),
                     ) -> np.ndarray:
    """White plate with a black border and text, rotated by ``angle`` on a dark canvas (BGR)."""
    pw, ph = size
    cw, ch = canvas
    img = np.full((ch, cw, 3), 40, np.uint8)
    x0, y0 = (cw - pw) // 2, (ch - ph) // 2
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), 3)
    cv2.putText(img, "51F12345", (x0 + 12, y0 + ph - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    m = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (cw, ch), borderValue=(40, 40, 40))


def test_rectify_straightens_tilted_plate() -> None:
    crop = _synthetic_plate(12)
    warped, ok = rectify_plate(crop)
    assert ok is True
    h, w = warped.shape[:2]
    # back to ~plate size: the contour follows the inner or outer edge of the 3 px border
    assert 230 <= w <= 250 and 45 <= h <= 60
    assert 4.0 <= w / h <= 5.5


def test_rectify_skips_straight_plate() -> None:
    crop = _synthetic_plate(2)
    warped, ok = rectify_plate(crop)
    assert ok is False and warped is crop


def test_rectify_rejects_small_contour() -> None:
    crop = _synthetic_plate(12, size=(100, 22), canvas=(300, 140))  # plate ~5 % of crop
    warped, ok = rectify_plate(crop)
    assert ok is False and warped is crop


def test_rectify_rejects_bad_aspect_for_layout() -> None:
    crop = _synthetic_plate(12)  # ~4.6:1, fine as one line ...
    assert rectify_plate(crop, layout="one_line")[1] is True
    warped, ok = rectify_plate(crop, layout="two_line")  # ... but outside [0.8, 2.0]
    assert ok is False and warped is crop


def test_pipeline_uses_rectify_only_when_enabled(monkeypatch) -> None:
    import src.pipeline as pipeline_mod

    calls = []

    def fake_rectify(crop):
        calls.append(crop.shape)
        return np.zeros((52, 240, 3), np.uint8), True

    monkeypatch.setattr(pipeline_mod, "rectify_plate", fake_rectify)
    reader = _RecordingReader("51F12345")
    monkeypatch.setattr(ocr, "_reader", reader)
    det = _FakeDetector([Detection((100, 50, 200, 90), 0.9)])
    image = np.zeros((480, 640, 3), np.uint8)

    off = PlateOCRPipeline(detector=det).process(image)
    assert calls == [] and off[0]["rectified"] is False

    on = PlateOCRPipeline(detector=det, use_rectify=True).process(image)
    assert calls == [(44, 110, 3)]          # rectify got the padded crop
    assert on[0]["rectified"] is True
    assert reader.shapes[-2] == (52, 240, 3)  # OCR (original pass) got the rectified crop


# --- evaluate --------------------------------------------------------------
def test_cer() -> None:
    assert cer("51A12345", "51A12345") == 0.0
    assert cer("51A12346", "51A12345") == pytest.approx(1 / 8)
    assert cer("51f-123.45", "51F-12345") == 0.0  # separators and case ignored
    assert cer("", "51F-12345") == 1.0             # nothing detected


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


# --- pipeline (fake detector + fake OCR reader) ----------------------------
from src.detect import Detection, crop_box, pad_box  # noqa: E402
from src.pipeline import COLOR_INVALID, COLOR_VALID, PlateOCRPipeline, draw_results, load_image  # noqa: E402


class _FakeDetector:
    def __init__(self, dets: list[Detection]) -> None:
        self.dets = dets
        self.seen_shapes: list[tuple[int, ...]] = []

    def detect(self, image):
        self.seen_shapes.append(image.shape)
        return self.dets


class _RecordingReader:
    """Records the crop shapes it is given and always reads ``text``."""

    def __init__(self, text: str, conf: float = 0.9) -> None:
        self.text, self.conf, self.shapes = text, conf, []

    def readtext(self, img, **kwargs):
        self.shapes.append(img.shape)
        return [(_BOX, self.text, self.conf)]


def test_pad_box_and_crop() -> None:
    assert pad_box((100, 50, 200, 90), 640, 480) == (95, 48, 205, 92)  # 5 % of 100x40
    assert pad_box((0, 0, 100, 40), 640, 480) == (0, 0, 105, 42)       # clipped at the border
    assert pad_box((700, 500, 800, 600), 640, 480) is None             # outside the image
    img = np.zeros((480, 640, 3), np.uint8)
    assert crop_box(img, (100, 50, 200, 90)).shape == (44, 110, 3)
    assert crop_box(img, (700, 500, 800, 600)).size == 0


def test_pipeline_no_detection_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "_reader", _RecordingReader("51F12345"))
    pipe = PlateOCRPipeline(detector=_FakeDetector([]))
    assert pipe.process(np.zeros((480, 640, 3), np.uint8)) == []
    assert pipe.process(np.zeros((0, 0, 3), np.uint8)) == []
    assert pipe.process(None) == []


def test_pipeline_result_fields(monkeypatch) -> None:
    reader = _RecordingReader("51f-123.45")
    monkeypatch.setattr(ocr, "_reader", reader)
    dets = [Detection((100, 50, 200, 90), 0.8, cls=1), Detection((300, 300, 340, 320), 0.6, cls=0)]
    pipe = PlateOCRPipeline(detector=_FakeDetector(dets))

    results = pipe.process(np.zeros((480, 640, 3), np.uint8))

    assert len(results) == 2
    r = results[0]
    assert r["bbox"] == (100, 50, 200, 90)  # original box, not the padded one
    assert r["plate_class"] == 1 and results[1]["plate_class"] == 0
    assert r["det_conf"] == pytest.approx(0.8)
    assert r["raw_text"] == "51F-123.45"
    assert r["plate_text"] == "51F-12345" and r["is_valid_format"] is True
    assert r["ocr_conf"] == pytest.approx(0.9)
    assert r["rectified"] is False
    assert set(r["latency_ms"]) == {"detect", "rectify", "ocr"}
    assert set(pipe.last_latency_ms) == {"detect", "rectify", "ocr", "total"}
    # OCR got the 5 %-padded crop (110x44), original pass is 3-channel
    assert reader.shapes[0] == (44, 110, 3)


def test_pipeline_accepts_gray_image(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "_reader", _RecordingReader("51F12345"))
    det = _FakeDetector([Detection((10, 10, 110, 50), 0.9)])
    PlateOCRPipeline(detector=det).process(np.zeros((120, 160), np.uint8))
    assert det.seen_shapes == [(120, 160, 3)]


def test_draw_results_colors() -> None:
    img = np.zeros((300, 400, 3), np.uint8)
    results = [
        {"bbox": (50, 100, 150, 140), "plate_text": "51F-12345", "is_valid_format": True},
        {"bbox": (250, 0, 350, 40), "plate_text": "", "raw_text": "", "is_valid_format": False},
    ]
    out = draw_results(img, results)
    assert out is not img and not img.any()                 # input untouched
    assert tuple(out[120, 50]) == COLOR_VALID               # left edge of the valid box
    assert tuple(out[20, 350]) == COLOR_INVALID             # right edge of the invalid box
    assert tuple(out[95, 52]) == COLOR_VALID                # label background above the box
    assert draw_results(img, []).shape == img.shape


# --- integration (needs weights) -------------------------------------------
@pytest.mark.skipif(not Path("models/best.pt").is_file(), reason="requires trained weights in models/best.pt")
def test_pipeline_end_to_end() -> None:
    results = PlateOCRPipeline("models/best.pt").process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert results == []


# --- evaluate harness (fake detector + fake OCR) ----------------------------
import src.evaluate as ev  # noqa: E402


def test_exact_match_ignores_separators() -> None:
    assert exact_match("51f-123.45", "51F-12345")
    assert exact_match("59-H1 234.56", "59H1-23456")
    assert not exact_match("51F-12346", "51F-12345")
    assert not exact_match("", "")


def test_mask_last() -> None:
    assert ev.mask_last("51F-12345") == "51F-12***"
    assert ev.mask_last("59H1-234.56") == "59H1-23*.**"
    assert ev.mask_last("AB") == "**"


class _BySizeReader:
    """Reads ``by_width[w]`` on the original (3-channel) crop, ``pre`` on the pre-processed one."""

    def __init__(self, by_width: dict[int, tuple[str, float]], pre: tuple[str, float] | None = None) -> None:
        self.by_width, self.pre = by_width, pre

    def readtext(self, img, **kwargs):
        if img.ndim == 2:
            return [(_BOX, *self.pre)] if self.pre else []
        hit = self.by_width.get(img.shape[1])
        return [(_BOX, *hit)] if hit else []


def _gt_image(tmp_path, tags=()) -> ev.GTItem:
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), np.zeros((480, 640, 3), np.uint8))
    return ev.GTItem(path, path.as_posix(), "51F-12345", list(tags))


def test_evaluate_picks_closest_plate_not_first(monkeypatch, tmp_path) -> None:
    # det 1 (more confident, first) reads garbage; det 2 reads the GT
    monkeypatch.setattr(ocr, "_reader", _BySizeReader({110: ("99Z-99999", 0.95), 44: ("51F-123.45", 0.6)}))
    dets = [Detection((100, 50, 200, 90), 0.9), Detection((300, 300, 340, 320), 0.5)]
    pipe = PlateOCRPipeline(detector=_FakeDetector(dets), use_rectify=True)

    row = ev.evaluate_image(pipe, _gt_image(tmp_path))

    assert row["n_detections"] == 2
    for cfg in ev.CONFIGS:
        assert row["predictions"][cfg]["exact"], cfg
        assert row["predictions"][cfg]["bbox"] == [300, 300, 340, 320]
    assert row["predictions"]["baseline"]["text"] == "51F-123.45"  # no rules: raw cleaned text
    assert row["predictions"][ev.FULL]["text"] == "51F-12345"


def test_evaluate_configs_are_cumulative_and_full_matches_pipeline(monkeypatch, tmp_path) -> None:
    # raw crop -> "51F-1234S" (conf .5); preprocessed -> "51F 1234S" (conf .9)
    monkeypatch.setattr(ocr, "_reader", _BySizeReader({110: ("51F-I234S", 0.5)}, pre=("51F 1234S", 0.9)))
    det = _FakeDetector([Detection((100, 50, 200, 90), 0.9)])
    pipe = PlateOCRPipeline(detector=det, use_rectify=True)
    item = _gt_image(tmp_path)

    p = ev.evaluate_image(pipe, item)["predictions"]
    assert p["baseline"]["text"] == "51F-I234S"
    assert p["+ preprocess"]["text"] == "51F1234S"   # more confident pre-processed reading
    assert p["+ rectify"]["text"] == "51F1234S"      # blank crop: rectify falls back
    assert p["+ rules"]["text"] == "51F-12345" and p["+ rules"]["exact"]
    # config 4 == what the real pipeline returns
    assert pipe.process(load_image(item.image_path))[0]["plate_text"] == p["+ rules"]["text"]


def test_evaluate_no_detection(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ocr, "_reader", _BySizeReader({}))
    pipe = PlateOCRPipeline(detector=_FakeDetector([]))
    row = ev.evaluate_image(pipe, _gt_image(tmp_path))
    assert row["n_detections"] == 0
    m = ev.summarize([row], ev.FULL)
    assert m == {"n": 1, "exact_match": 0.0, "cer": 1.0, "det_recall": 0.0}


def _row(path: str, exact: bool, cer_: float = 0.0, n_det: int = 1) -> dict:
    return {"image_path": path, "n_detections": n_det,
            "predictions": {c: {"exact": exact, "cer": cer_} for c in ev.CONFIGS}}


def test_slice_metrics_clean_and_multi_tag() -> None:
    items = [ev.GTItem(Path(p), p, "51F12345", t) for p, t in
             [("a", []), ("b", ["two_line", "small"]), ("c", ["small"]), ("d", ["custom"])]]
    rows = [_row("a", True), _row("b", False, 0.5), _row("c", True), _row("d", True)]
    s = ev.slice_metrics(rows, items)
    assert s["clean"]["n"] == 1 and s["clean"]["exact_match"] == 1.0
    assert s["small"]["n"] == 2 and s["small"]["exact_match"] == 0.5     # b counts in both slices
    assert s["two_line"]["n"] == 1 and s["angled"]["n"] == 0
    assert s["custom"]["n"] == 1                                         # unknown tags still reported
    table = ev.slice_table(s)
    assert "WARNING" in table and "angled (n=0)" in table


def test_ablation_table_delta() -> None:
    metrics = {c: {"n": 10, "exact_match": em, "cer": 0.1, "det_recall": 0.9}
               for c, em in zip(ev.CONFIGS, (0.2, 0.3, 0.3, 0.55), strict=True)}
    table = ev.ablation_table(metrics)
    assert "| 1. baseline | 20.0% | 0.100 | 90.0% | - |" in table
    assert "+10.0 pts" in table and "+0.0 pts" in table and "+25.0 pts" in table


def test_load_ground_truth_empty_tags(tmp_path) -> None:
    img = tmp_path / "x.jpg"
    cv2.imwrite(str(img), np.zeros((10, 10, 3), np.uint8))
    csv_path = tmp_path / "gt.csv"
    lines = ["image_path,ground_truth_plate,tags",
             f"{img.as_posix()},51F-12345,",
             f"{img.as_posix()},29A-12345,two_line;small",
             "missing.jpg,30A-11111,"]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    items = ev.load_ground_truth(csv_path)
    assert [it.slices for it in items] == [["clean"], ["two_line", "small"]]


def test_save_failures(tmp_path) -> None:
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), np.full((200, 300, 3), 128, np.uint8))
    rows = [{"image_path": path.as_posix(), "gt": "51F-12345", "tags": [],
             "predictions": {ev.FULL: {"exact": False, "cer": c, "text": t, "bbox": b}}}
            for c, t, b in [(1.0, "", None), (0.25, "51F-12399", [100, 80, 220, 110]),
                            (0.1, "51F-1234", [10, 10, 60, 50])]]
    out = tmp_path / "fail.png"
    assert ev.save_failures(rows, out, blur=True) == 3
    assert out.stat().st_size > 0
