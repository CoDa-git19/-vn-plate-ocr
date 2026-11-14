"""Fast ground-truth labelling for the manual test set (Windows-friendly).

Shows each image of ``data/manual_test/images/`` in a matplotlib window; you type
the correct plate in the terminal, then its tags. Every answer is appended to
``data/manual_test.csv`` immediately, so you can stop at any time and resume later
(already-labelled images are skipped).

Usage:
    python scripts/label_tool.py
    python scripts/label_tool.py --images data/manual_test/images --out data/manual_test.csv

At the plate prompt:
    51f12345        plate, typed any way: case, '-', '.', spaces are normalized -> 51F-12345
    51f12345/23     plate + tags in one go (skips the tag prompt)
    skip  (s)       leave this image for later
    undo  (u)       delete the last saved row and label that image again
    quit  (q)       stop; everything entered so far is already saved
At the tag prompt: digits separated by spaces (or not): "2 3" == "23"; Enter = no tag.
    1=two_line 2=angled 3=lowlight 4=small 5=blurry

CSV (UTF-8): image_path, ground_truth_plate, tags
    image_path is relative to the repo root with '/' separators, tags are ';'-separated.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.rcParams["toolbar"] = "None"           # more room for the image
matplotlib.rcParams["figure.raise_window"] = False  # don't steal focus from the terminal on update

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ocr import clean_text, normalize_vn_plate  # noqa: E402

FIELDS = ("image_path", "ground_truth_plate", "tags")
TAGS = {"1": "two_line", "2": "angled", "3": "lowlight", "4": "small", "5": "blurry"}
TAG_HELP = "  ".join(f"{k}={v}" for k, v in TAGS.items())
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
COMMANDS = {"skip": "skip", "s": "skip", "undo": "undo", "u": "undo", "quit": "quit", "q": "quit"}


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def rel_path(path: Path) -> str:
    """Path relative to the repo root with '/' (portable between Windows and Linux)."""
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:  # outside the repo: relative to cwd, or absolute if on another drive (Windows)
        try:
            return Path(os.path.relpath(path)).as_posix()
        except ValueError:
            return path.as_posix()


def natural_key(path: Path) -> list:
    """Sort 2.jpg before 10.jpg."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=natural_key)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            sys.exit(f"{csv_path}: unexpected header {reader.fieldnames}, expected {list(FIELDS)}")
        return list(reader)


def append_row(csv_path: Path, row: dict[str, str]) -> None:
    """Append one row and flush it to disk right away (safe to Ctrl+C / close at any time)."""
    new = not csv_path.is_file() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def drop_last_row(csv_path: Path) -> dict[str, str] | None:
    """Remove the last row (atomic rewrite) and return it."""
    rows = read_rows(csv_path)
    if not rows:
        return None
    last = rows.pop()
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)
    return last


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def parse_plate(text: str) -> tuple[str, bool]:
    """Typed plate -> ``(stored_value, matches_vn_format)``.

    Valid VN plates are stored in the pipeline's canonical form (``51F-12345``),
    so they can be compared directly with ``plate_text``. Anything else is kept
    as typed (uppercased, without characters outside ``[A-Z0-9.-]``).
    """
    plate, valid = normalize_vn_plate(text)
    return (plate, True) if valid else (clean_text(text), False)


def parse_tags(text: str) -> list[str] | None:
    """``"2 3"`` / ``"23"`` -> ``["angled", "lowlight"]``; ``None`` if a digit is unknown."""
    digits = re.sub(r"[\s,;]+", "", text)
    if any(d not in TAGS for d in digits):
        return None
    return [TAGS[d] for d in dict.fromkeys(digits)]  # keep order, drop duplicates


def ask(prompt: str) -> str:
    """input() that turns Ctrl+C / Ctrl+Z into 'quit'."""
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return "quit"


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
class Viewer:
    """One reusable matplotlib window; images are swapped with set_data (fast, no flicker)."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.fig = None
        self.ax = None
        self.im = None

    def show(self, path: Path, title: str) -> None:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # phone photos: honour EXIF rotation
        img.thumbnail((self.max_size, self.max_size))  # faster redraw, still sharp on screen

        if self.fig is None or not plt.fignum_exists(self.fig.number):
            terminal = _foreground_window()
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 7))
            self.fig.canvas.manager.set_window_title("label_tool")
            self.ax.set_axis_off()
            self.fig.subplots_adjust(0, 0, 1, 0.95)
            self.im = self.ax.imshow(img)
            plt.pause(0.1)
            _focus_window(terminal)  # give the keyboard back to the terminal once
        else:
            self.im.set_data(img)
            self.im.set_extent((-0.5, img.width - 0.5, img.height - 0.5, -0.5))
            self.ax.set_xlim(-0.5, img.width - 0.5)
            self.ax.set_ylim(img.height - 0.5, -0.5)
        self.ax.set_title(title, fontsize=11)
        plt.pause(0.1)

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)


def _foreground_window() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes

    return ctypes.windll.user32.GetForegroundWindow()


def _focus_window(hwnd: int | None) -> None:
    """Best effort: bring the terminal (cmd, PowerShell, VS Code...) back to the front."""
    if hwnd:
        import ctypes

        ctypes.windll.user32.SetForegroundWindow(hwnd)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def label_one(viewer: Viewer, path: Path, progress: str) -> tuple[str, dict[str, str] | None]:
    """Ask for one image -> ``(action, row)`` with action in {save, skip, undo, quit}."""
    rel = rel_path(path)
    viewer.show(path, f"{progress}   {path.name}")
    print(f"\n[{progress}] {rel}")

    retyped: str | None = None
    while True:
        answer = retyped if retyped is not None else ask("  plate> ")
        retyped = None
        cmd = COMMANDS.get(answer.lower())
        if cmd:
            return cmd, None
        plate_part, _, tag_part = answer.partition("/")
        if not clean_text(plate_part):
            print("  (empty - type the plate, or skip / undo / quit)")
            continue

        plate, valid = parse_plate(plate_part)
        if not valid:
            again = ask(f"  '{plate}' is not a standard VN format. Enter = keep, or retype: ")
            if again:  # a new plate (checked again from the top) or a command
                retyped = again
                continue

        tags = parse_tags(tag_part) if tag_part else None
        while tags is None:
            if tag_part:
                print(f"  unknown tag in {tag_part!r}")
            tag_part = ask(f"  tags ({TAG_HELP}, Enter=none)> ")
            if tag_part.lower() in COMMANDS:
                return COMMANDS[tag_part.lower()], None
            tags = parse_tags(tag_part)

        row = {"image_path": rel, "ground_truth_plate": plate, "tags": ";".join(tags)}
        print(f"  saved: {plate}" + (f"  [{', '.join(tags)}]" if tags else ""))
        return "save", row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label ground-truth plates for the manual test set")
    p.add_argument("--images", type=Path, default=ROOT / "data/manual_test/images")
    p.add_argument("--out", type=Path, default=ROOT / "data/manual_test.csv")
    p.add_argument("--max-size", type=int, default=1600, help="Longest side shown, in px")
    args = p.parse_args(argv)

    if not args.images.is_dir():
        sys.exit(f"Image folder not found: {args.images}")
    images = list_images(args.images)
    if not images:
        sys.exit(f"No .jpg/.jpeg/.png in {args.images}")

    by_rel = {rel_path(img): img for img in images}
    done = {r["image_path"] for r in read_rows(args.out)} & by_rel.keys()
    queue = deque(img for img in images if rel_path(img) not in done)
    total = len(images)

    print(f"{len(done)}/{total} already labelled in {rel_path(args.out)}, {len(queue)} to go.")
    print(f"Tags: {TAG_HELP}   |   plate/23 = plate + tags   |   skip  undo  quit")
    if not queue:
        return 0

    viewer = Viewer(args.max_size)
    skipped = 0
    try:
        while queue:
            path = queue.popleft()
            action, row = label_one(viewer, path, f"{len(done) + 1}/{total}")
            if action == "save":
                append_row(args.out, row)
                done.add(row["image_path"])
            elif action == "skip":
                skipped += 1
            elif action == "undo":
                queue.appendleft(path)
                last = drop_last_row(args.out)
                if last is None:
                    print("  nothing to undo")
                    continue
                done.discard(last["image_path"])
                print(f"  undone: {last['image_path']} = {last['ground_truth_plate']}")
                if last["image_path"] in by_rel and last["image_path"] != rel_path(path):
                    queue.appendleft(by_rel[last["image_path"]])
            elif action == "quit":
                break
    finally:
        viewer.close()

    left = total - len(done)
    print(f"\nLabelled {len(done)}/{total}" + (f", {skipped} skipped this run" if skipped else "")
          + (f". Run again to label the remaining {left}." if left else ". All done."))
    print(f"CSV: {rel_path(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
