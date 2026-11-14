"""Ablation harness: end-to-end metrics, cumulative ablation, per-tag slices, model comparison.

Ground truth: ``data/manual_test.csv`` written by ``scripts/label_tool.py``::

    image_path,ground_truth_plate,tags
    data/manual_test/images/3.jpg,76D-18759,two_line;small
    data/manual_test/images/1.jpg,60A-69696,            <- no tag = "clean"

``image_path`` is relative to the repo root. One plate per image.

A. Ablation: 4 CUMULATIVE configurations, built from the pipeline's own stages
   (``src.ocr``, ``src.rectify``, ``src.pipeline``):
     1. baseline      raw padded crop -> EasyOCR, no preprocessing, no rules
     2. + preprocess  ocr.read_plate: OCR on the raw crop AND on the upscaled + CLAHE +
                      thresholded crop, the more confident reading wins
     3. + rectify     rectify.rectify_plate before read_plate (falls back to the raw crop)
     4. + rules       ocr.normalize_vn_plate (positional fixes + VN regex)
   Configuration 4 is exactly what ``PlateOCRPipeline(use_rectify=True)`` returns as
   ``plate_text``. Detection runs once per image; configurations reuse the OCR passes
   they share, so the whole ablation costs ~1 pipeline run (+ OCR on rectified crops).
B. Slices: configuration 4 grouped by tag (an image with 2 tags counts in both).
C. ``--compare``: the same for several weights + mAP50 (``model.val`` on the dataset's
   test split) and median latencies.

Metrics (per image):
    Exact Match       prediction == GT after removing '-', '.', spaces and case
    CER               Levenshtein(pred, gt) / len(gt) on the same separator-free strings
                      (no plate detected -> pred = "" -> CER = 1)
    Detection Recall  share of images with >= 1 detected plate
    When several plates are detected, the GT is compared with the CLOSEST prediction
    (lowest CER), separately for each configuration.

CLI:
    python -m src.evaluate --weights models/best.pt --csv data/manual_test.csv
    python -m src.evaluate --compare models/best_yolov8n.pt models/best_yolov8s.pt
    python -m src.evaluate --blur-failures        # mask the last 3 plate characters in the figure

Outputs: markdown tables on stdout, raw results in ``docs/results.json``,
the 6 worst predictions in ``docs/failure_cases.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from src import Box
from src.detect import pad_box
from src.ocr import clean_text, normalize_vn_plate, read_plate, read_plate_candidates
from src.pipeline import PlateOCRPipeline, load_image
from src.rectify import rectify_plate

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = ("baseline", "+ preprocess", "+ rectify", "+ rules")
FULL = CONFIGS[-1]
SLICE_TAGS = ("two_line", "angled", "lowlight", "small", "blurry", "clean")
MIN_SLICE = 5
N_FAILURES = 6


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compact(text: str) -> str:
    """Uppercase, drop everything outside [A-Z0-9] ('-', '.', spaces...)."""
    return re.sub(r"[^A-Z0-9]", "", clean_text(text))


def levenshtein(a: str, b: str) -> int:
    try:
        import Levenshtein

        return Levenshtein.distance(a, b)
    except ImportError:  # pure-Python fallback
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def cer(pred: str, gt: str) -> float:
    """Character error rate = Levenshtein(pred, gt) / len(gt), separators ignored."""
    p, g = compact(pred), compact(gt)
    if not g:
        return 0.0 if not p else 1.0
    return levenshtein(p, g) / len(g)


def exact_match(pred: str, gt: str) -> bool:
    """Equal after removing '-', '.', spaces and case."""
    g = compact(gt)
    return bool(g) and compact(pred) == g


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``xyxy`` boxes."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
@dataclass
class GTItem:
    image_path: Path  # absolute
    rel_path: str     # as written in the CSV
    plate: str
    tags: list[str] = field(default_factory=list)

    @property
    def slices(self) -> list[str]:
        return self.tags or ["clean"]


def load_ground_truth(csv_path: str | Path) -> list[GTItem]:
    """Read the label_tool CSV; image paths are resolved against the repo root."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        sys.exit(f"Ground-truth CSV not found: {csv_path}")
    items, missing = [], []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rel = (row.get("image_path") or "").strip()
            plate = (row.get("ground_truth_plate") or "").strip()
            if not rel or not compact(plate):
                continue
            path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
            if not path.is_file():
                missing.append(rel)
                continue
            tags = [t for t in re.split(r"[;,\s]+", (row.get("tags") or "").strip().lower()) if t]
            items.append(GTItem(path, rel, plate, list(dict.fromkeys(tags))))
    if missing:
        print(f"WARNING: {len(missing)} image(s) in {csv_path} not found, skipped: {missing[:3]}...",
              file=sys.stderr)
    if not items:
        sys.exit(f"No usable rows in {csv_path}")
    return items


# --------------------------------------------------------------------------- #
# Running the staged pipeline
# --------------------------------------------------------------------------- #
def read_all_configs(crop: np.ndarray) -> dict[str, Any]:
    """OCR one padded crop under every configuration, sharing the passes they have in common.

    Returns ``{"texts": {config: (text, conf)}, "rectified": bool, "full_ms": float}``;
    ``full_ms`` is the time the full pipeline (config 4) spends on this crop.
    """
    t0 = time.perf_counter()
    cands = read_plate_candidates(crop)  # {"original": ..., "preprocessed": ...}
    t1 = time.perf_counter()
    baseline = cands["original"]
    plus_pre = max(cands.values(), key=lambda r: r[1])  # == read_plate(crop)

    warped, rectified = rectify_plate(crop)
    t2 = time.perf_counter()
    plus_rect = read_plate(warped) if rectified else plus_pre
    t3 = time.perf_counter()

    rules = (normalize_vn_plate(plus_rect[0])[0], plus_rect[1])
    ocr_ms = (t3 - t2) if rectified else (t1 - t0)
    return {
        "texts": {CONFIGS[0]: baseline, CONFIGS[1]: plus_pre, CONFIGS[2]: plus_rect, CONFIGS[3]: rules},
        "rectified": rectified,
        "full_ms": ((t2 - t1) + ocr_ms) * 1000,
    }


def evaluate_image(pipe: PlateOCRPipeline, item: GTItem) -> dict[str, Any]:
    """Detect once, OCR every plate under every config, pick the closest plate per config."""
    image = load_image(item.image_path)
    h, w = image.shape[:2]
    t0 = time.perf_counter()
    detections = pipe.detector.detect(image)
    detect_ms = (time.perf_counter() - t0) * 1000

    plates = []
    for det in detections:
        padded = pad_box(det.box, w, h, pipe.pad)
        if padded is None:
            continue
        x1, y1, x2, y2 = padded
        plates.append({"bbox": list(det.box), "det_conf": float(det.conf), "plate_class": int(det.cls),
                       **read_all_configs(image[y1:y2, x1:x2])})

    predictions = {}
    for cfg in CONFIGS:
        best = min(plates, key=lambda p: cer(p["texts"][cfg][0], item.plate), default=None)
        text = best["texts"][cfg][0] if best else ""
        predictions[cfg] = {
            "text": text,
            "ocr_conf": float(best["texts"][cfg][1]) if best else 0.0,
            "cer": cer(text, item.plate),
            "exact": exact_match(text, item.plate),
            "bbox": best["bbox"] if best else None,
            "plate_class": best["plate_class"] if best else None,
            "rectified": best["rectified"] if best else False,
        }

    return {
        "image_path": item.rel_path,
        "gt": item.plate,
        "tags": item.tags,
        "n_detections": len(plates),
        "detect_ms": detect_ms,
        "full_ms": detect_ms + sum(p["full_ms"] for p in plates),
        "predictions": predictions,
        "plates": [{**{k: v for k, v in p.items() if k != "texts"},
                    "texts": {c: list(t) for c, t in p["texts"].items()}} for p in plates],
    }


def summarize(rows: list[dict[str, Any]], cfg: str) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "exact_match": float("nan"), "cer": float("nan"), "det_recall": float("nan")}
    return {
        "n": n,
        "exact_match": sum(r["predictions"][cfg]["exact"] for r in rows) / n,
        "cer": sum(r["predictions"][cfg]["cer"] for r in rows) / n,
        "det_recall": sum(r["n_detections"] > 0 for r in rows) / n,
    }


def slice_metrics(rows: list[dict[str, Any]], items: list[GTItem], cfg: str = FULL) -> dict[str, dict]:
    """Metrics of ``cfg`` per tag (``clean`` = no tag); unknown tags get their own slice."""
    by_path = {it.rel_path: it for it in items}
    names = list(SLICE_TAGS) + sorted({t for it in items for t in it.tags} - set(SLICE_TAGS))
    return {name: summarize([r for r in rows if name in by_path[r["image_path"]].slices], cfg)
            for name in names}


def run_weights(weights: Path, items: list[GTItem], conf: float, device: str | None) -> dict[str, Any]:
    pipe = PlateOCRPipeline(weights, conf=conf, use_rectify=True, device=device)
    pipe.warmup()  # load EasyOCR + first YOLO call outside the timings
    rows = [evaluate_image(pipe, it) for it in tqdm(items, desc=weights.name, unit="img")]
    return {
        "weights": weights.as_posix(),
        "metrics": {cfg: summarize(rows, cfg) for cfg in CONFIGS},
        "slices": slice_metrics(rows, items),
        "latency_ms": {
            "detect_p50": statistics.median(r["detect_ms"] for r in rows),
            "end_to_end_p50": statistics.median(r["full_ms"] for r in rows),
        },
        "rows": rows,
    }


def map50_on_test(weights: Path, data_yaml: Path, device: str | None) -> float | None:
    """mAP50 of ``weights`` on the dataset's test split (Ultralytics ``model.val``)."""
    try:
        from ultralytics import YOLO

        metrics = YOLO(str(weights)).val(
            data=str(data_yaml), split="test", imgsz=640, batch=8, device=device or "cpu",
            workers=0, plots=False, verbose=False, project=str(ROOT / "runs/eval"),
            name=f"val_{weights.stem}", exist_ok=True,
        )
        return float(metrics.box.map50)
    except Exception as e:  # missing dataset, bad yaml...: report, keep the rest of the comparison
        print(f"WARNING: mAP50 failed for {weights}: {e}", file=sys.stderr)
        return None


def find_data_yaml() -> Path | None:
    found = sorted((ROOT / "datasets").glob("*/data.yaml"))
    return found[0].resolve() if found else None


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return "-" if x != x else f"{x * 100:.1f}%"  # x != x: NaN


def ablation_table(metrics: dict[str, dict]) -> str:
    lines = ["| Configuration | Exact Match | CER | Detection Recall | Delta EM vs previous |",
             "|---|---:|---:|---:|---:|"]
    prev = None
    for i, cfg in enumerate(CONFIGS, 1):
        m = metrics[cfg]
        delta = "-" if prev is None else f"{(m['exact_match'] - prev) * 100:+.1f} pts"
        lines.append(f"| {i}. {cfg} | {_pct(m['exact_match'])} | {m['cer']:.3f} | "
                     f"{_pct(m['det_recall'])} | {delta} |")
        prev = m["exact_match"]
    return "\n".join(lines)


def slice_table(slices: dict[str, dict]) -> str:
    lines = ["| Slice | n | Exact Match |", "|---|---:|---:|"]
    warn = []
    for name, m in slices.items():
        flag = " (!)" if m["n"] < MIN_SLICE else ""
        lines.append(f"| {name} | {m['n']}{flag} | {_pct(m['exact_match'])} |")
        if m["n"] < MIN_SLICE:
            warn.append(f"{name} (n={m['n']})")
    if warn:
        lines.append(f"\n> WARNING: fewer than {MIN_SLICE} samples, not reliable: {', '.join(warn)}")
    return "\n".join(lines)


def compare_table(runs: list[dict]) -> str:
    lines = ["| Weights | mAP50 (test) | Exact Match | Detect p50 (ms) | End-to-end p50 (ms) |",
             "|---|---:|---:|---:|---:|"]
    for r in runs:
        m50 = "n/a" if r.get("map50") is None else f"{r['map50']:.3f}"
        lat = r["latency_ms"]
        lines.append(f"| {Path(r['weights']).name} | {m50} | {_pct(r['metrics'][FULL]['exact_match'])} | "
                     f"{lat['detect_p50']:.0f} | {lat['end_to_end_p50']:.0f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Failure figure
# --------------------------------------------------------------------------- #
def mask_last(text: str, n: int = 3, char: str = "*") -> str:
    """Replace the last ``n`` letters/digits by ``char`` (separators kept): 51F-12345 -> 51F-12***."""
    out, left = list(text), n
    for i in range(len(out) - 1, -1, -1):
        if left == 0:
            break
        if out[i].isalnum():
            out[i], left = char, left - 1
    return "".join(out)


def _pixelate(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def failure_view(image: np.ndarray, bbox: list[int] | None, blur: bool) -> np.ndarray:
    """Zoom around the predicted plate (red box); with ``blur`` hide its last characters.

    No detection: the whole image is shown (pixelated entirely when ``blur``, since the
    plate cannot be located).
    """
    img = image.copy()
    if bbox is None:
        if blur:
            _pixelate(img, 0, 0, img.shape[1], img.shape[0])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if blur:  # last 3 characters: right part of a 1-line plate, bottom-right of a 2-line plate
        if bw >= 2 * bh:
            _pixelate(img, x1 + int(bw * 0.62), y1, x2, y2)
        else:
            _pixelate(img, x1 + int(bw * 0.4), y1 + bh // 2, x2, y2)
    t = max(2, round(max(img.shape[:2]) / 400))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), t)
    m = max(bw, bh)
    h, w = img.shape[:2]
    view = img[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def save_failures(rows: list[dict], out_path: Path, blur: bool, n: int = N_FAILURES) -> int:
    """Plot the ``n`` worst predictions of the full configuration (highest CER)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    failures = sorted((r for r in rows if not r["predictions"][FULL]["exact"]),
                      key=lambda r: r["predictions"][FULL]["cer"], reverse=True)[:n]
    if not failures:
        return 0

    cols = 3
    nrows = (len(failures) + cols - 1) // cols
    fig = Figure(figsize=(cols * 4.6, nrows * 3.6), dpi=110)
    FigureCanvasAgg(fig)
    for i, r in enumerate(failures):
        pred = r["predictions"][FULL]
        gt_txt, pred_txt = r["gt"], pred["text"] or "(no plate)"
        if blur:
            gt_txt = mask_last(gt_txt)
            pred_txt = mask_last(pred_txt) if pred["text"] else pred_txt
        ax = fig.add_subplot(nrows, cols, i + 1)
        ax.imshow(failure_view(load_image(ROOT / r["image_path"]), pred["bbox"], blur))
        ax.set_axis_off()
        tags = ",".join(r["tags"]) or "clean"
        ax.set_title(f"GT {gt_txt} | Pred {pred_txt}\nCER {pred['cer']:.2f}  [{tags}]", fontsize=10)
    fig.suptitle(f"{len(failures)} worst predictions ({FULL})" + ("  - last 3 characters masked" if blur else ""),
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    return len(failures)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _json_default(o: Any) -> Any:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(type(o))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation / slice / model-comparison evaluation")
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--compare", type=Path, nargs="+", default=None, metavar="WEIGHTS",
                   help="Evaluate several weights (+ mAP50 on the dataset test split)")
    p.add_argument("--csv", type=Path, default=Path("data/manual_test.csv"), help="Ground truth (label_tool.py)")
    p.add_argument("--data", type=Path, default=None,
                   help="data.yaml for mAP50 in --compare (default: first datasets/*/data.yaml)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--device", default=None, help="cpu, cuda:0, ... (default: auto)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N images (quick check)")
    p.add_argument("--json", type=Path, default=Path("docs/results.json"))
    p.add_argument("--failures", type=Path, default=Path("docs/failure_cases.png"))
    p.add_argument("--blur-failures", action="store_true", help="Mask the last 3 plate characters in the figure")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    weights_list = args.compare or [args.weights]
    for w in weights_list:
        if not w.is_file():
            sys.exit(f"Weights not found: {w}")
    items = load_ground_truth(args.csv)[: args.limit]

    data_yaml = None
    if args.compare:
        data_yaml = args.data.resolve() if args.data else find_data_yaml()
        if data_yaml is None or not data_yaml.is_file():
            print("WARNING: no data.yaml found, mAP50 will be n/a (pass --data)", file=sys.stderr)
            data_yaml = None

    print(f"Ground truth: {args.csv} ({len(items)} images), weights: {', '.join(w.as_posix() for w in weights_list)}")
    runs = []
    for w in weights_list:
        run = run_weights(w, items, args.conf, args.device)
        if data_yaml is not None:
            run["map50"] = map50_on_test(w, data_yaml, args.device)
        runs.append(run)

    for run in runs:
        print(f"\n## Ablation - {run['weights']} (n={len(items)})\n")
        print(ablation_table(run["metrics"]))
        print(f"\n## Slices - {FULL} - {run['weights']}\n")
        print(slice_table(run["slices"]))
    if args.compare:
        print("\n## Model comparison\n")
        print(compare_table(runs))
        print("\nmAP50: Ultralytics val on the dataset test split; Exact Match / latency: manual test set, "
              "full pipeline. Latencies exclude model loading.")

    n_fail = save_failures(runs[0]["rows"], args.failures, args.blur_failures)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "csv": args.csv.as_posix(),
        "n_images": len(items),
        "configs": list(CONFIGS),
        "data_yaml": data_yaml.as_posix() if data_yaml else None,
        "runs": runs,
    }
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(f"\nraw results -> {args.json}")
    print(f"failures    -> {args.failures} ({n_fail} images, {Path(runs[0]['weights']).name})"
          if n_fail else "failures    -> none (every prediction is an exact match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
