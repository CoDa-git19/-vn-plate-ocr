"""Download, merge, validate and visualize YOLO-format plate datasets.

Steps:
  1. (optional) Download one or more Roboflow datasets in YOLOv8 format.
  2. If several datasets are given, merge them into one: unified class names,
     collision-free file names, MD5 de-duplication of image content.
  3. Print a summary from data.yaml: number of classes, class names, images per split.
  4. Save a 3x3 grid of random train images with GT boxes   -> docs/sample_data.png
  5. Save a histogram of box area / image area (%)           -> docs/box_size_dist.png
  6. Warn about images without labels, invalid class ids and out-of-bounds boxes.

The Roboflow API key is read from the ROBOFLOW_API_KEY environment variable
(or a Colab secret with the same name). Never hard-code it in the repo.

Examples:
    # Download the default Roboflow dataset, then inspect it
    python scripts/prepare_data.py --roboflow

    # Inspect an existing local dataset
    python scripts/prepare_data.py datasets/vietnam-license-plate-hjswj-v2

    # Merge several datasets (local and/or Roboflow) into datasets/merged
    python scripts/prepare_data.py datasets/a datasets/b --roboflow ws/project/3 --out datasets/merged

Google Colab:
    !pip install -q roboflow
    import os; from google.colab import userdata
    os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
    !python scripts/prepare_data.py --roboflow
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_DIR_ALIASES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
DEFAULT_ROBOFLOW = "vietnam-license/vietnam-license-plate-hjswj/2"
BOUNDS_TOL = 1e-3  # tolerance on normalized coordinates

ISSUE_TITLES = {
    "missing_label": "Images without a label file",
    "empty_label": "Images with an empty label file (background)",
    "bad_class": "Label lines with an invalid class id",
    "out_of_bounds": "Boxes outside the image / non-positive size",
    "malformed": "Malformed label lines",
    "orphan_label": "Label files without a matching image",
}

BoxN = tuple[float, float, float, float]  # normalized (cx, cy, w, h)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def get_roboflow_api_key() -> str:
    """Read the API key from $ROBOFLOW_API_KEY, falling back to a Colab secret."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        try:  # only works inside a notebook kernel (e.g. %run), not in `!python`
            from google.colab import userdata  # type: ignore[import-not-found]

            key = userdata.get("ROBOFLOW_API_KEY")
        except Exception:
            key = None
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set. Export it (or add it as a Colab secret) before using --roboflow.")
    return key


def download_roboflow(spec: str, datasets_dir: Path, fmt: str = "yolov8") -> Path:
    """Download ``workspace/project/version`` from Roboflow and return the dataset dir."""
    try:
        workspace, project, version = spec.strip("/").split("/")
        version_num = int(version)
    except ValueError:
        sys.exit(f"Invalid --roboflow value {spec!r}; expected WORKSPACE/PROJECT/VERSION")

    location = datasets_dir / f"{project}-v{version_num}"
    if (location / "data.yaml").is_file():
        print(f"[download] {spec} already at {location}, skipping")
        return location

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("roboflow is not installed: pip install roboflow")

    datasets_dir.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=get_roboflow_api_key())
    ds = rf.workspace(workspace).project(project).version(version_num).download(
        fmt, location=str(location), overwrite=True
    )
    return Path(ds.location)


# --------------------------------------------------------------------------- #
# Dataset loading helpers
# --------------------------------------------------------------------------- #
@dataclass
class YoloDataset:
    yaml_path: Path
    names: list[str]
    splits: dict[str, Path]  # split -> images directory

    @property
    def root(self) -> Path:
        return self.yaml_path.parent


def _names_from_cfg(cfg: dict) -> list[str]:
    names = cfg.get("names")
    if names is None:
        return [str(i) for i in range(int(cfg.get("nc", 0)))]
    if isinstance(names, dict):  # {0: "plate", 1: ...}
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def _resolve_split_dir(root: Path, base: Path, entry: object, split: str) -> Path | None:
    """Find the images dir of a split, tolerating Roboflow's odd relative paths."""
    if isinstance(entry, (list, tuple)):  # several dirs per split: rare, take the first
        entry = entry[0] if entry else None
    candidates: list[Path] = []
    if isinstance(entry, str) and entry:
        e = Path(entry)
        candidates += [base / e, root / e]
        # Roboflow writes e.g. "../train/images", which is wrong relative to data.yaml
        candidates.append(root / Path(*[p for p in e.parts if p not in ("..", ".")]))
    for alias in SPLIT_DIR_ALIASES[split]:
        candidates += [root / alias / "images", root / "images" / alias]
    return next((c.resolve() for c in candidates if c.is_dir()), None)


def load_dataset(path: Path) -> YoloDataset:
    """Load a YOLO dataset from a directory containing data.yaml (or the yaml itself)."""
    yaml_path = path if path.suffix in (".yaml", ".yml") else path / "data.yaml"
    if not yaml_path.is_file():
        sys.exit(f"data.yaml not found: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names = _names_from_cfg(cfg)
    if cfg.get("nc") is not None and int(cfg["nc"]) != len(names):
        warn(f"{yaml_path}: nc={cfg['nc']} but {len(names)} names; using the names list")

    root = yaml_path.parent.resolve()
    base = root
    if cfg.get("path"):
        p = Path(cfg["path"])
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            base = p

    splits: dict[str, Path] = {}
    for split in SPLITS:
        d = _resolve_split_dir(root, base, cfg.get(split), split)
        if d is not None:
            splits[split] = d
        elif cfg.get(split):
            warn(f"{yaml_path}: split '{split}' -> {cfg[split]!r} not found on disk")
    if not splits:
        sys.exit(f"No split directories found for {yaml_path}")
    return YoloDataset(yaml_path.resolve(), names, splits)


def list_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(img_path: Path) -> Path:
    """Ultralytics convention: replace the last ``images`` dir with ``labels``, suffix ``.txt``."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def parse_label_line(line: str) -> tuple[int, BoxN]:
    """Parse ``cls cx cy w h`` (or a ``cls x1 y1 x2 y2 ...`` polygon). Raises ValueError."""
    vals = line.split()
    cls = int(vals[0])
    coords = [float(v) for v in vals[1:]]
    if len(coords) == 4:
        return cls, (coords[0], coords[1], coords[2], coords[3])
    if len(coords) >= 6 and len(coords) % 2 == 0:  # segmentation polygon -> bbox
        xs, ys = coords[0::2], coords[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return cls, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    raise ValueError(f"expected 5 values or a polygon, got {len(vals)}")


def imread_any(path: Path) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII paths (Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


# --------------------------------------------------------------------------- #
# 2. Merge
# --------------------------------------------------------------------------- #
def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _remap_label_lines(path: Path, remap: dict[int, int]) -> tuple[list[str], list[str]]:
    """Rewrite class ids to the merged class list.

    Returns ``(remapped_lines, bad_lines)``; bad lines have a non-integer or unknown class id.
    """
    out, bad = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            cls = None
        if cls not in remap:
            bad.append(line.strip())
            continue
        out.append(" ".join([str(remap[cls]), *parts[1:]]))
    return out, bad


def merge_datasets(paths: list[Path], out_dir: Path, overwrite: bool = False, max_show: int = 10) -> Path:
    """Merge several YOLO datasets into ``out_dir`` (images/{split}, labels/{split})."""
    datasets = [load_dataset(p) for p in paths]
    out_dir = out_dir.resolve()

    for ds in datasets:
        if out_dir == ds.root or out_dir in ds.root.parents or ds.root in out_dir.parents:
            sys.exit(f"--out {out_dir} overlaps input dataset {ds.root}")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            sys.exit(f"{out_dir} is not empty; pass --overwrite to replace it")
        if not (out_dir / "data.yaml").is_file():
            sys.exit(f"Refusing to delete {out_dir}: it does not look like a dataset (no data.yaml)")
        shutil.rmtree(out_dir)

    # Unified class list, matched by (stripped) name in order of first appearance.
    names: list[str] = []
    remaps: list[dict[int, int]] = []
    for ds in datasets:
        remap = {}
        for i, n in enumerate(ds.names):
            n = n.strip()
            if n not in names:
                names.append(n)
            remap[i] = names.index(n)
        remaps.append(remap)

    seen: dict[str, tuple[int, str, Path]] = {}  # md5 -> (dataset idx, split, path)
    dup_msgs: list[str] = []
    bad_label_msgs: list[str] = []
    cross_split = 0
    rows: list[dict[str, str]] = []
    per_ds: list[tuple[int, Counter]] = []

    for di, (ds, remap) in enumerate(zip(datasets, remaps)):
        kept: Counter = Counter()
        n_in = 0
        for split in SPLITS:
            if split not in ds.splits:
                continue
            imgs = list_images(ds.splits[split])
            n_in += len(imgs)
            for img in tqdm(imgs, desc=f"merge ds{di}/{split}", unit="img", leave=False):
                digest = md5_file(img)
                if digest in seen:
                    k_di, k_split, k_path = seen[digest]
                    cross_split += k_split != split
                    dup_msgs.append(f"ds{di}/{split}/{img.name}  ==  ds{k_di}/{k_split}/{k_path.name}")
                    continue

                src_lbl = label_path_for(img)
                lines: list[str] | None = None
                if src_lbl.is_file():  # missing labels stay missing so the checks report them
                    lines, bad = _remap_label_lines(src_lbl, remap)
                    if bad:
                        # Dropping only the bad lines could turn a plate image into a false
                        # "background" image, so skip the whole image instead.
                        bad_label_msgs.append(f"ds{di}/{_rel(src_lbl, ds.root)}  {bad[0]!r}")
                        continue
                seen[digest] = (di, split, img)

                stem = f"ds{di}_{digest[:16]}"  # unique: md5 is unique after de-dup
                dst_img = out_dir / "images" / split / f"{stem}{img.suffix.lower()}"
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dst_img)

                if lines is not None:
                    dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

                kept[split] += 1
                rows.append({"image": dst_img.name, "split": split, "source_dataset": str(ds.root),
                             "source_image": _rel(img, ds.root), "md5": digest})
        per_ds.append((n_in, kept))

    cfg = {
        "path": out_dir.as_posix(),
        **{s: f"images/{s}" for s in SPLITS if (out_dir / "images" / s).is_dir()},
        "nc": len(names),
        "names": names,
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    with open(out_dir / "sources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "source_dataset", "source_image", "md5"])
        writer.writeheader()
        writer.writerows(rows)

    # ---- report ----
    print("\n=== Merge ===")
    for di, (ds, (n_in, kept)) in enumerate(zip(datasets, per_ds)):
        kept_str = " ".join(f"{s}={kept[s]}" for s in SPLITS if kept[s])
        print(f"  ds{di}: {ds.root}")
        print(f"        names={ds.names}  images in={n_in}  kept: {kept_str or '0'}")
    print(f"  merged names: {names}")
    if len(names) > max(len(ds.names) for ds in datasets):
        warn("class names differ between datasets - check whether some of them are the same class")
    print(f"  duplicate images removed (MD5): {len(dup_msgs)}  (across splits: {cross_split})")
    for m in dup_msgs[:max_show]:
        print(f"        {m}")
    if len(dup_msgs) > max_show:
        print(f"        ... {len(dup_msgs) - max_show} more")
    if bad_label_msgs:
        warn(f"skipped {len(bad_label_msgs)} images whose labels have an unknown/invalid class id")
        for m in bad_label_msgs[:max_show]:
            print(f"        {m}")
        if len(bad_label_msgs) > max_show:
            print(f"        ... {len(bad_label_msgs) - max_show} more")
    print(f"  total kept: {len(rows)} images -> {out_dir / 'data.yaml'}")
    return out_dir


# --------------------------------------------------------------------------- #
# 3 + 6. Scan, summary and checks
# --------------------------------------------------------------------------- #
@dataclass
class SplitScan:
    images: list[Path]
    boxes: dict[Path, list[tuple[int, BoxN]]] = field(default_factory=dict)
    n_labeled: int = 0
    n_background: int = 0
    n_missing: int = 0
    class_counts: Counter = field(default_factory=Counter)
    issues: dict[str, list[str]] = field(default_factory=lambda: {k: [] for k in ISSUE_TITLES})

    @property
    def n_boxes(self) -> int:
        return sum(len(b) for b in self.boxes.values())

    def area_pct(self) -> np.ndarray:
        """Box area as % of image area (normalized w*h*100)."""
        return np.array([w * h * 100 for bs in self.boxes.values() for _, (_, _, w, h) in bs], dtype=float)


def scan_split(img_dir: Path, nc: int, split: str, root: Path) -> SplitScan:
    scan = SplitScan(images=list_images(img_dir))
    expected_labels: set[Path] = set()

    for img in tqdm(scan.images, desc=f"scan {split}", unit="img", leave=False):
        lbl = label_path_for(img)
        expected_labels.add(lbl)
        if not lbl.is_file():
            scan.n_missing += 1
            scan.issues["missing_label"].append(_rel(img, root))
            continue

        raw = lbl.read_text(encoding="utf-8", errors="replace").splitlines()
        if not any(line.strip() for line in raw):
            scan.n_background += 1
            scan.issues["empty_label"].append(_rel(img, root))
            continue

        scan.n_labeled += 1
        boxes: list[tuple[int, BoxN]] = []
        for ln_no, line in enumerate(raw, 1):
            if not line.strip():
                continue
            where = f"{_rel(lbl, root)}:{ln_no}"
            try:
                cls, (cx, cy, w, h) = parse_label_line(line)
            except (ValueError, IndexError) as e:
                scan.issues["malformed"].append(f"{where}  {line.strip()!r} ({e})")
                continue
            if not 0 <= cls < nc:
                scan.issues["bad_class"].append(f"{where}  class {cls} (valid: 0..{nc - 1})")
                continue
            if (w <= 0 or h <= 0 or cx - w / 2 < -BOUNDS_TOL or cy - h / 2 < -BOUNDS_TOL
                    or cx + w / 2 > 1 + BOUNDS_TOL or cy + h / 2 > 1 + BOUNDS_TOL):
                scan.issues["out_of_bounds"].append(f"{where}  cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")
            boxes.append((cls, (cx, cy, w, h)))
            scan.class_counts[cls] += 1
        scan.boxes[img] = boxes

    label_dir = label_path_for(img_dir / "_.jpg").parent
    if label_dir.is_dir() and label_dir != img_dir:
        for lbl in sorted(label_dir.rglob("*.txt")):
            if lbl not in expected_labels:
                scan.issues["orphan_label"].append(_rel(lbl, root))
    return scan


def scan_dataset(ds: YoloDataset) -> dict[str, SplitScan]:
    return {split: scan_split(d, len(ds.names), split, ds.root) for split, d in ds.splits.items()}


def print_summary(ds: YoloDataset, scans: dict[str, SplitScan]) -> None:
    print(f"\n=== Dataset: {ds.yaml_path} ===")
    print(f"  classes: {len(ds.names)}")
    print(f"  names  : {ds.names}")
    header = f"  {'split':<6}{'images':>9}{'labeled':>9}{'empty':>9}{'no-lbl':>9}{'boxes':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    tot = Counter()
    for split, s in scans.items():
        vals = (len(s.images), s.n_labeled, s.n_background, s.n_missing, s.n_boxes)
        tot.update(dict(zip(("i", "l", "e", "m", "b"), vals)))
        print(f"  {split:<6}" + "".join(f"{v:>9}" for v in vals))
    print(f"  {'total':<6}" + "".join(f"{tot[k]:>9}" for k in ("i", "l", "e", "m", "b")))

    counts = sum((s.class_counts for s in scans.values()), Counter())
    print("  boxes per class: " + ", ".join(f"{ds.names[c]}={counts[c]}" for c in range(len(ds.names))))


def report_issues(ds: YoloDataset, scans: dict[str, SplitScan], max_show: int = 10) -> int:
    """Print issues (first ``max_show`` of each kind) and write the full list to data_issues.txt."""
    print("\n=== Data checks ===")
    report: list[str] = []
    total = 0
    for kind, title in ISSUE_TITLES.items():
        msgs = [f"[{split}] {m}" for split, s in scans.items() for m in s.issues[kind]]
        if not msgs:
            continue
        total += len(msgs)
        warn(f"{title}: {len(msgs)}")
        for m in msgs[:max_show]:
            print(f"         {m}")
        if len(msgs) > max_show:
            print(f"         ... {len(msgs) - max_show} more")
        report += [f"## {title} ({len(msgs)})", *msgs, ""]

    issues_file = ds.root / "data_issues.txt"
    if total:
        issues_file.write_text("\n".join(report), encoding="utf-8")
        print(f"  full list: {issues_file}")
    else:
        issues_file.unlink(missing_ok=True)
        print("  [OK] no issues found")
    return total


# --------------------------------------------------------------------------- #
# 4. Sample grid
# --------------------------------------------------------------------------- #
def plot_sample_grid(scan: SplitScan, names: list[str], out_path: Path, n: int = 9, seed: int = 0) -> Path:
    picks = random.Random(seed).sample(scan.images, min(n, len(scan.images)))
    rows = cols = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")

    for ax, img_path in zip(np.ravel(axes), picks):
        img = imread_any(img_path)
        if img is None:
            ax.set_title(f"unreadable: {img_path.name[:28]}", fontsize=8, color="red")
            continue
        h, w = img.shape[:2]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = scan.boxes.get(img_path, [])
        for cls, (cx, cy, bw, bh) in boxes:
            color = plt.cm.tab10(cls % 10)
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            ax.add_patch(mpatches.Rectangle((x1, y1), bw * w, bh * h, fill=False, edgecolor=color, linewidth=2))
            ax.text(x1, y1 - 2, names[cls], color="white", fontsize=7, va="bottom",
                    bbox={"facecolor": color, "edgecolor": "none", "pad": 1})
        ax.set_title(f"{img_path.name[:28]}  ({len(boxes)} box)", fontsize=8)

    fig.suptitle(f"Random train samples with GT boxes (seed={seed})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 5. Box size distribution
# --------------------------------------------------------------------------- #
def plot_box_size_dist(area_by_split: dict[str, np.ndarray], out_path: Path) -> float:
    """Histogram (log x-axis) of box area / image area in %. Returns the overall median."""
    all_area = np.concatenate([a for a in area_by_split.values() if a.size])
    all_area = all_area[all_area > 0]
    median = float(np.median(all_area))
    lo, hi = np.log10(all_area.min()), np.log10(all_area.max())
    if hi - lo < 1e-6:  # all boxes the same size -> widen so bin edges increase
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.logspace(lo, hi, 50)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist([a[a > 0] for a in area_by_split.values()], bins=bins, stacked=True,
            label=[f"{s} (n={a.size})" for s, a in area_by_split.items()], edgecolor="white", linewidth=0.3)
    ax.axvline(median, color="black", linestyle="--", linewidth=1.5, label=f"median = {median:.2f}%")
    ax.set_xscale("log")
    ax.set_xlabel("Box area / image area (%)  [log scale]")
    ax.set_ylabel("Number of boxes")
    ax.set_title(f"Plate box size distribution (n={all_area.size})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return median


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", type=Path, help="Local dataset dirs (or data.yaml files)")
    p.add_argument("--roboflow", action="append", nargs="?", const=DEFAULT_ROBOFLOW, default=[],
                   metavar="WS/PROJECT/VERSION",
                   help=f"Download from Roboflow (repeatable). No value = {DEFAULT_ROBOFLOW}")
    p.add_argument("--format", default="yolov8", help="Roboflow export format")
    p.add_argument("--datasets-dir", type=Path, default=REPO_ROOT / "datasets")
    p.add_argument("--out", type=Path, default=None, help="Merged output dir (default: <datasets-dir>/merged)")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing merged dataset")
    p.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs")
    p.add_argument("--seed", type=int, default=0, help="Seed for the sample grid")
    p.add_argument("--max-show", type=int, default=10, help="Examples printed per warning type")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plt.switch_backend("Agg")  # headless-safe (Colab `!python`, servers)

    paths = list(args.datasets) + [download_roboflow(s, args.datasets_dir, args.format) for s in args.roboflow]
    if not paths:
        sys.exit("Give at least one dataset path or --roboflow")

    if len(paths) > 1:
        final = merge_datasets(paths, args.out or args.datasets_dir / "merged", args.overwrite, args.max_show)
    else:
        final = paths[0]

    ds = load_dataset(final)
    scans = scan_dataset(ds)
    print_summary(ds, scans)
    report_issues(ds, scans, args.max_show)

    print("\n=== Plots ===")
    train = scans.get("train")
    if train and train.images:
        out = plot_sample_grid(train, ds.names, args.docs_dir / "sample_data.png", seed=args.seed)
        print(f"  sample grid      -> {out}")
    else:
        warn("no train images, skipping sample grid")

    area_by_split = {s: sc.area_pct() for s, sc in scans.items()}
    if any((a > 0).any() for a in area_by_split.values()):
        out = args.docs_dir / "box_size_dist.png"
        median = plot_box_size_dist(area_by_split, out)
        print(f"  box size hist    -> {out}")
        print(f"  median box area  : {median:.3f}% of image")
        for s, a in area_by_split.items():
            a = a[a > 0]
            if a.size:
                p5, p25, p50, p75, p95 = np.percentile(a, [5, 25, 50, 75, 95])
                print(f"    {s:<5}  p5={p5:.3f}%  p25={p25:.3f}%  median={p50:.3f}%  p75={p75:.3f}%  p95={p95:.3f}%")
    else:
        warn("no valid boxes, skipping size histogram")

    print(f"\nNext: python scripts/train.py --data {ds.yaml_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
