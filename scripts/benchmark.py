"""Latency benchmark: PyTorch vs ONNX Runtime detector + per-stage breakdown of the pipeline.

    python scripts/benchmark.py --weights models/best.pt --images data/manual_test/images

1. Export ``models/best.onnx`` (opset 12, onnxslim, static 640x640 input).
2. Detector latency on 30 images: 5 warm-up calls, 30 timed calls, MEDIAN and p95 for
   PyTorch CPU, ONNX Runtime CPU and PyTorch GPU (if CUDA is available). Each call is the
   full ``PlateDetector.detect`` (letterbox + inference + NMS + box conversion).
3. Per-stage latency of the full pipeline (detect / rectify / OCR) on the same images, to
   show where the time goes.
4. Markdown tables.
5. Correctness: PyTorch and ONNX on the same 10 images must give the same number of boxes
   and coordinates within 1 px, otherwise a WARNING is printed. The check feeds both backends
   the SAME 640x640 letterboxed input: as deployed, PyTorch letterboxes to the smallest
   multiple of 32 (e.g. 640x480) while the static ONNX model always gets 640x640, so their
   boxes can also differ for that reason alone (reported separately, for information).

Why ONNX Runtime is usually faster on CPU: it loads a static graph once and optimizes it
ahead of time (constant folding, fusing Conv+BN+activation into single kernels, dropping
no-op nodes), so there is no per-op Python/autograd dispatch at run time. Its CPU kernels
(MLAS) are tuned for inference with pre-packed weights and a memory plan reused across
calls, whereas PyTorch eager executes one generic operator at a time and allocates
intermediate tensors on the fly.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import shutil
import statistics
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detect import PlateDetector, pad_box  # noqa: E402
from src.ocr import get_reader, normalize_vn_plate, read_plate  # noqa: E402
from src.pipeline import load_image  # noqa: E402
from src.rectify import rectify_plate  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
TOL_PX = 1.0


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def export_onnx(weights: Path, imgsz: int, out: Path, force: bool) -> Path:
    """``YOLO(weights).export(format="onnx", opset=12, simplify=True, dynamic=False)`` -> ``out``."""
    if out.is_file() and not force and out.stat().st_mtime >= weights.stat().st_mtime:
        print(f"[export] reusing {out} (newer than {weights.name}; --force-export to redo)")
        return out
    import torch
    from ultralytics import YOLO

    t0 = time.perf_counter()
    with _legacy_onnx_exporter(torch):
        produced = Path(YOLO(str(weights)).export(format="onnx", opset=12, simplify=True, dynamic=False,
                                                   imgsz=imgsz, device="cpu"))
    if produced.resolve() != out.resolve():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), out)
    print(f"[export] {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f} s")
    return out


@contextmanager
def _legacy_onnx_exporter(torch_mod):
    """Make ``torch.onnx.export`` default to the TorchScript exporter (``dynamo=False``) while exporting.

    torch >= 2.9 defaults to the dynamo exporter, which needs ``onnxscript`` and only targets
    opset >= 18; Ultralytics 8.3.40 does not pass ``dynamo``, so opset 12 would fail.
    """
    original = torch_mod.onnx.export
    if "dynamo" not in inspect.signature(original).parameters:  # older torch: nothing to do
        yield
        return

    @functools.wraps(original)
    def export(*a, **kw):
        kw.setdefault("dynamo", False)
        return original(*a, **kw)

    torch_mod.onnx.export = export
    try:
        yield
    finally:
        torch_mod.onnx.export = original


def load_images(folder: Path, n: int) -> list[tuple[str, np.ndarray]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)[:n]
    if not paths:
        sys.exit(f"No images in {folder}")
    return [(p.name, load_image(p)) for p in paths]


def stats(ms: list[float]) -> dict[str, float]:
    return {"median": statistics.median(ms), "p95": float(np.percentile(ms, 95)), "n": len(ms)}


# --------------------------------------------------------------------------- #
# 2. Detector latency
# --------------------------------------------------------------------------- #
def time_calls(fn: Callable[[np.ndarray], object], images: list[np.ndarray], warmup: int, runs: int) -> list[float]:
    """``warmup`` untimed calls, then ``runs`` timed calls cycling through ``images``; ms per call."""
    for i in range(warmup):
        fn(images[i % len(images)])
    out = []
    for i in range(runs):
        img = images[i % len(images)]
        t0 = time.perf_counter()
        fn(img)
        out.append((time.perf_counter() - t0) * 1000)
    return out


# --------------------------------------------------------------------------- #
# 3. Per-stage latency
# --------------------------------------------------------------------------- #
def stage_latencies(detector: PlateDetector, images: list[np.ndarray], warmup: int) -> dict[str, list[float]]:
    """Full pipeline per image (detect -> pad crop -> rectify -> read_plate + normalize)."""

    def run(img: np.ndarray) -> tuple[float, float, float, int]:
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        dets = detector.detect(img)
        t_det = time.perf_counter() - t0
        t_rect = t_ocr = 0.0
        for d in dets:
            padded = pad_box(d.box, w, h)
            if padded is None:
                continue
            x1, y1, x2, y2 = padded
            crop = img[y1:y2, x1:x2]
            t0 = time.perf_counter()
            crop, _ = rectify_plate(crop)
            t1 = time.perf_counter()
            normalize_vn_plate(read_plate(crop)[0])
            t2 = time.perf_counter()
            t_rect += t1 - t0
            t_ocr += t2 - t1
        return t_det * 1000, t_rect * 1000, t_ocr * 1000, len(dets)

    get_reader()
    for img in images[:warmup]:
        run(img)
    out: dict[str, list[float]] = {"detect": [], "rectify": [], "ocr": [], "total": [], "plates": []}
    for img in images:
        d, r, o, n = run(img)
        for k, v in (("detect", d), ("rectify", r), ("ocr", o), ("total", d + r + o), ("plates", n)):
            out[k].append(v)
    return out


# --------------------------------------------------------------------------- #
# 5. Correctness
# --------------------------------------------------------------------------- #
def raw_boxes(detector: PlateDetector, img: np.ndarray) -> np.ndarray:
    """Unrounded ``(N, 6)`` = x1, y1, x2, y2, conf, cls, sorted by x1 (order-independent comparison)."""
    r = detector.model.predict(img, conf=detector.conf_thres, iou=detector.iou_thres, imgsz=detector.imgsz,
                               device=detector.device, verbose=False)[0].boxes
    if r is None or len(r) == 0:
        return np.zeros((0, 6))
    a = np.concatenate([r.xyxy.cpu().numpy(), r.conf.cpu().numpy()[:, None], r.cls.cpu().numpy()[:, None]], axis=1)
    return a[np.argsort(a[:, 0])]


def compare_boxes(a: np.ndarray, b: np.ndarray) -> tuple[bool, float, float]:
    """``(same_count, max_coord_diff_px, max_conf_diff)``; boxes matched greedily by nearest coordinates."""
    if len(a) != len(b):
        return False, float("inf"), float("inf")
    if len(a) == 0:
        return True, 0.0, 0.0
    unused, max_px, max_conf = list(range(len(b))), 0.0, 0.0
    for row in a:
        j = min(unused, key=lambda k: np.abs(b[k, :4] - row[:4]).max())
        unused.remove(j)
        max_px = max(max_px, float(np.abs(b[j, :4] - row[:4]).max()))
        max_conf = max(max_conf, abs(float(b[j, 4] - row[4])))
    return True, max_px, max_conf


def _small(x: float) -> str:
    """Show tiny float32 differences instead of rounding them to 0."""
    return "0" if x == 0 else (f"{x:.1e}" if x < 1e-3 else f"{x:.3f}")


def letterbox_square(img: np.ndarray, size: int) -> np.ndarray:
    """Same square letterbox the static ONNX model uses (gray 114 padding)."""
    from ultralytics.data.augment import LetterBox

    return LetterBox(new_shape=(size, size), auto=False)(image=img)


def check_correctness(pt: PlateDetector, ox: PlateDetector, named: list[tuple[str, np.ndarray]],
                      imgsz: int) -> bool:
    print(f"\n## Correctness: PyTorch vs ONNX Runtime ({len(named)} images, tolerance {TOL_PX:g} px)\n")
    print("| Image | Boxes PT / ONNX | Max coord diff, same input (px) | Max conf diff | "
          "As deployed: boxes PT / ONNX, max diff (px) |")
    print("|---|---:|---:|---:|---:|")
    ok_all = True
    for name, img in named:
        sq = letterbox_square(img, imgsz)
        a, b = raw_boxes(pt, sq), raw_boxes(ox, sq)
        same, px, dconf = compare_boxes(a, b)
        ok = same and px < TOL_PX
        ok_all &= ok
        da, db = raw_boxes(pt, img), raw_boxes(ox, img)
        dsame, dpx, _ = compare_boxes(da, db)
        dep = f"{len(da)} / {len(db)}, " + (_small(dpx) if dsame else "count differs")
        flag = "" if ok else " **(!)**"
        px_s = _small(px) if same else "-"
        conf_s = _small(dconf) if same else "-"
        print(f"| {name} | {len(a)} / {len(b)} | {px_s}{flag} | {conf_s} | {dep} |")
        if not ok:
            why = "different number of boxes" if not same else f"coordinates differ by {px:.2f} px"
            print(f"\nWARNING: {name}: PyTorch and ONNX disagree on the same input ({why}).\n", file=sys.stderr)
    if ok_all:
        print(f"\nOK: same box count and < {TOL_PX:g} px on identical input for all {len(named)} images.")
    else:
        print("\nWARNING: ONNX output does NOT match PyTorch within tolerance - check the export (opset, simplify).")
    return ok_all


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime + pipeline stages (CPU)")
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--images", type=Path, default=Path("data/manual_test/images"))
    p.add_argument("--onnx", type=Path, default=None, help="Output .onnx (default: next to --weights)")
    p.add_argument("--force-export", action="store_true", help="Re-export even if the .onnx is up to date")
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--check-images", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")
    if not args.images.is_dir():
        sys.exit(f"Image folder not found: {args.images}")

    import onnxruntime as ort
    import torch

    onnx_path = export_onnx(args.weights, args.imgsz, args.onnx or args.weights.with_suffix(".onnx"),
                            args.force_export)
    named = load_images(args.images, args.n_images)
    images = [img for _, img in named]
    print(f"[setup] {len(images)} images from {args.images}, warm-up {args.warmup}, runs {args.runs}, "
          f"torch {torch.__version__} ({torch.get_num_threads()} threads), onnxruntime {ort.__version__}")

    pt_cpu = PlateDetector(args.weights, imgsz=args.imgsz, device="cpu")
    onnx_cpu = PlateDetector(onnx_path, imgsz=args.imgsz, device="cpu")
    configs: list[tuple[str, PlateDetector | None]] = [("PyTorch CPU", pt_cpu), ("ONNX Runtime CPU", onnx_cpu)]
    configs.append(("PyTorch GPU", PlateDetector(args.weights, imgsz=args.imgsz, device="cuda:0")
                    if torch.cuda.is_available() else None))

    det_stats = {}
    for label, det in configs:
        if det is None:
            continue
        print(f"[bench] {label} ...", flush=True)
        det_stats[label] = stats(time_calls(det.detect, images, args.warmup, args.runs))

    base = det_stats["PyTorch CPU"]["median"]
    print(f"\n## Detector latency ({args.runs} timed calls over {len(images)} images, after {args.warmup} warm-up)\n")
    print("| Configuration | median ms | p95 ms | speedup vs PyTorch CPU |")
    print("|---|---:|---:|---:|")
    for label, det in configs:
        if det is None:
            print(f"| {label} | n/a | n/a | no CUDA device |")
            continue
        s = det_stats[label]
        print(f"| {label} | {s['median']:.1f} | {s['p95']:.1f} | {base / s['median']:.2f}x |")
    print("\nSpeedup = PyTorch CPU median / median. Static ONNX always runs a 640x640 input; PyTorch letterboxes "
          "to the smallest multiple of 32 (e.g. 640x480), i.e. fewer pixels.")

    print("\n[bench] pipeline stages (PyTorch CPU detector, rectify on, EasyOCR) ...", flush=True)
    st = stage_latencies(pt_cpu, images, args.warmup)
    total_sum = sum(st["total"])
    print(f"\n## Pipeline stages per image ({len(images)} images, {sum(st['plates'])} plates, "
          f"after {args.warmup} warm-up images)\n")
    print("| Stage | median ms | p95 ms | share of total time |")
    print("|---|---:|---:|---:|")
    for k in ("detect", "rectify", "ocr", "total"):
        s = stats(st[k])
        share = "100%" if k == "total" else f"{sum(st[k]) / total_sum:.0%}"
        print(f"| {k} | {s['median']:.1f} | {s['p95']:.1f} | {share} |")
    worst = max(("detect", "rectify", "ocr"), key=lambda k: sum(st[k]))
    print(f"\nBottleneck: **{worst}** ({sum(st[worst]) / total_sum:.0%} of the time; rectify + OCR scale with "
          "the number of plates per image).")

    ok = check_correctness(pt_cpu, onnx_cpu, named[: args.check_images], args.imgsz)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
