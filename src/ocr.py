"""Read characters on a cropped Vietnamese license plate with EasyOCR.

Vietnamese plates come in two layouts:
  * 1 line  (most cars):             ``51F-123.45``
  * 2 lines (motorbikes, some cars): ``59-H1`` / ``234.56``

:func:`read_plate` returns the cleaned OCR string (lines joined top-to-bottom
with ``-``); :func:`normalize_vn_plate` fixes OCR confusions *by position* and
returns the canonical form ``51F-12345`` / ``59H1-23456`` / ``29AA-12345``
plus a validity flag.

CLI:
    python -m src.ocr --image path/to/crop.jpg
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MIN_HEIGHT = 64  # px, crops are upscaled to at least this height before OCR
ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-"  # keep O/I/Q... so positional fixes can act

# Series letters used on Vietnamese civilian plates: 20 letters. I, J, O, Q, R, W are
# never used as civilian series letters (I/J/O/Q are easily confused with 1/0), so OCR
# output containing them at a letter position is either a confusion or an invalid read.
VALID_SERIES_LETTERS = "ABCDEFGHKLMNPSTUVXYZ"
_L = f"[{VALID_SERIES_LETTERS}]"

# Positional confusion tables
DIGIT_FIX = {  # where a digit is required (province code, serial number)
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "J": "1", "L": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8",
}
LETTER_FIX = {  # series letter (3rd character)
    "0": "D", "8": "B", "5": "S", "2": "Z", "6": "G",  # digit -> letter
    "O": "D", "Q": "D",                                # non-existent letter -> letter
}  # I, J, W, R are deliberately NOT mapped: no guessing, the plate becomes invalid

# Valid series letters that can never be a misread digit (not a key of DIGIT_FIX).
_LETTER_ONLY = frozenset(VALID_SERIES_LETTERS) - DIGIT_FIX.keys()

CAR_RE = re.compile(rf"\d{{2}}{_L}-?\d{{4,5}}")                # 51F-12345, 51F-1234
MOTO_RE = re.compile(rf"\d{{2}}{_L}(?:\d|{_L})-?\d{{4,5}}")    # 59H1-23456, 29AA-12345

_CLEAN_RE = re.compile(r"[^A-Z0-9.\-]")

# EasyOCR fragment: (4 corner points, text, confidence)
Fragment = tuple[Sequence[Sequence[float]], str, float]

# --------------------------------------------------------------------------- #
# Reader (lazy singleton)
# --------------------------------------------------------------------------- #
_reader: Any = None
_reader_lock = threading.Lock()


def get_reader() -> Any:
    """Create the EasyOCR reader once (thread-safe) and reuse it.

    easyocr/torch are imported lazily so that the text utilities in this module
    (and their tests) do not pay the import cost.
    """
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                import torch

                _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _reader


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def preprocess(crop: np.ndarray) -> np.ndarray:
    """Upscale to >= 64 px high (INTER_CUBIC) -> grayscale -> CLAHE -> adaptive threshold.

    Returns a binary ``uint8`` image with dark text on a light background
    (inverted if needed, e.g. white-on-blue plates).
    """
    img = crop
    h, w = img.shape[:2]
    if h < MIN_HEIGHT:
        scale = MIN_HEIGHT / h
        img = cv2.resize(img, (max(1, round(w * scale)), MIN_HEIGHT), interpolation=cv2.INTER_CUBIC)

    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(img, code)
    else:
        gray = img

    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Polarity must be fixed BEFORE adaptive thresholding (its output is mostly white
    # either way). The background covers most of the plate, so if Otsu says the
    # majority is dark we have light text on a dark plate (e.g. blue) -> invert.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu.mean() < 127:
        gray = cv2.bitwise_not(gray)

    block = max(11, int(gray.shape[0] * 0.4) | 1)  # odd, ~ character height
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Uppercase and drop every character outside ``[A-Z0-9.-]``."""
    return _CLEAN_RE.sub("", text.upper())


def merge_lines(fragments: Sequence[Fragment]) -> tuple[str, float, int]:
    """Group EasyOCR fragments into lines by Y, order lines top-to-bottom, words left-to-right.

    Returns ``(text, confidence, n_lines)``: lines are joined with ``-`` and the
    confidence is the character-weighted mean of fragment confidences.
    """
    items = []
    for box, text, conf in fragments:
        t = clean_text(text)
        if not t:
            continue
        pts = np.asarray(box, dtype=float)
        items.append({"yc": pts[:, 1].mean(), "x": pts[:, 0].min(),
                      "h": pts[:, 1].max() - pts[:, 1].min(), "text": t, "conf": float(conf)})
    if not items:
        return "", 0.0, 0

    lines: list[list[dict]] = []
    for it in sorted(items, key=lambda d: d["yc"]):
        if lines:
            cur = lines[-1]
            cur_yc = float(np.mean([d["yc"] for d in cur]))
            cur_h = float(np.median([d["h"] for d in cur]))
            if abs(it["yc"] - cur_yc) <= 0.5 * max(cur_h, it["h"]):
                cur.append(it)
                continue
        lines.append([it])

    text = "-".join("".join(d["text"] for d in sorted(line, key=lambda d: d["x"])) for line in lines)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    n_chars = sum(len(d["text"]) for d in items)
    conf = sum(d["conf"] * len(d["text"]) for d in items) / n_chars
    return text, conf, len(lines)


def ocr_once(img: np.ndarray) -> tuple[str, float]:
    """Run EasyOCR on one image and merge its lines -> ``(text, conf)``."""
    fragments = get_reader().readtext(img, allowlist=ALLOWLIST, detail=1, paragraph=False,
                                      min_size=10)  # default 20 px drops chars on small crops
    text, conf, _ = merge_lines(fragments)
    return text, conf


def read_plate_candidates(crop: np.ndarray) -> dict[str, tuple[str, float]]:
    """OCR results on the original crop and on its pre-processed version."""
    return {"original": ocr_once(crop), "preprocessed": ocr_once(preprocess(crop))}


def read_plate(crop: np.ndarray) -> tuple[str, float]:
    """Read a plate crop (BGR) -> ``(cleaned_text, confidence)``.

    OCR runs on both the original and the pre-processed crop; the reading with
    the higher confidence wins. Returns ``("", 0.0)`` if nothing is read.
    """
    if crop is None or crop.size == 0:
        return "", 0.0
    return max(read_plate_candidates(crop).values(), key=lambda r: r[1])


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _series_length(cleaned: str, core: str) -> int:
    """Length of the series before the dash: 3 (``51F``) or 4 (``59H1`` / ``29AA``)."""
    n = len(core)
    if "-" in cleaned:  # a dash read by OCR, or the line break added by merge_lines
        k = len(cleaned[: cleaned.rfind("-")].replace("-", "").replace(".", ""))
        if k in (3, 4) and n - k in (4, 5):
            return k
    if n == 7:  # 51F-1234
        return 3
    if n == 9:  # 59H1-23456 / 29AA-12345
        return 4
    # n == 8 without a usable dash: car 51F-12345 or motorbike 59H1-2345 / 29AA-1234.
    # Only a letter that can never be a misread digit at position 4 points to 29AA-1234;
    # otherwise the (far more common single-line) car layout is assumed.
    return 4 if core[3] in _LETTER_ONLY else 3


def normalize_vn_plate(text: str) -> tuple[str, bool]:
    """Fix OCR confusions by position and validate against VN plate formats.

    Formats (separators removed, ``L`` = one of :data:`VALID_SERIES_LETTERS`):
        car         DD L   + 4-5 digits    51F-12345
        motorbike   DD L D + 4-5 digits    59H1-23456
        motorbike   DD L L + 4-5 digits    29AA-12345

    Positions (1-based):
        1-2  province code -> digit  (O/Q/D->0, I/J/L->1, Z->2, A->4, S->5, G->6, T->7, B->8)
        3    series letter -> letter (0->D, 8->B, 5->S, 2->Z, 6->G, O/Q->D);
             I, J, W, R are not guessed and make the plate invalid
        4    in a 4-character series: digit (59H1) or letter (29AA) -> left unchanged,
             since e.g. ``O`` could be ``0`` or ``D``
        rest serial number -> digit table
    Corrections are only applied when the length is plausible (7-9 characters),
    otherwise the text is returned unchanged with ``is_valid=False``.

    Returns:
        ``(canonical_text, is_valid)``, e.g. ``("51F-12345", True)``.
    """
    cleaned = clean_text(text)
    core = cleaned.replace("-", "").replace(".", "")
    n = len(core)
    if not 7 <= n <= 9:
        return core, False

    k = _series_length(cleaned, core)
    fixed = []
    for i, c in enumerate(core):
        if i == 2:
            fixed.append(LETTER_FIX.get(c, c))
        elif i == 3 and k == 4:
            fixed.append(c)  # ambiguous digit/letter position: never guessed
        else:
            fixed.append(DIGIT_FIX.get(c, c))
    fixed_str = "".join(fixed)

    plate = f"{fixed_str[:k]}-{fixed_str[k:]}"
    pattern = MOTO_RE if k == 4 else CAR_RE
    return plate, pattern.fullmatch(plate) is not None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _imread(path: Path) -> np.ndarray | None:
    """cv2.imread that also handles non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OCR a cropped Vietnamese license plate")
    p.add_argument("--image", type=Path, required=True, help="Path to a plate crop")
    p.add_argument("--save-preprocessed", type=Path, default=None, help="Also write the pre-processed image")
    args = p.parse_args(argv)

    if not args.image.is_file():
        sys.exit(f"Image not found: {args.image}")
    crop = _imread(args.image)
    if crop is None:
        sys.exit(f"Cannot decode image: {args.image}")

    t0 = time.perf_counter()
    get_reader()
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    cands = read_plate_candidates(crop)
    t_ocr = time.perf_counter() - t0
    text, conf = max(cands.values(), key=lambda r: r[1])
    plate, valid = normalize_vn_plate(text)

    h, w = crop.shape[:2]
    print(f"image        : {args.image} ({w}x{h})")
    for name, (t, c) in cands.items():
        print(f"{name:<13}: {t!r:<16} conf={c:.3f}")
    print(f"read_plate   : {text!r:<16} conf={conf:.3f}")
    print(f"normalized   : {plate!r:<16} valid={valid}")
    print(f"time         : reader load {t_load:.1f}s, OCR (2 passes) {t_ocr * 1000:.0f} ms")

    if args.save_preprocessed:
        args.save_preprocessed.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(args.save_preprocessed.suffix or ".png", preprocess(crop))
        if ok:
            buf.tofile(str(args.save_preprocessed))
            print(f"preprocessed -> {args.save_preprocessed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
