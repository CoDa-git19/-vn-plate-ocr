"""Video pipeline: detect + track (ByteTrack) + per-track OCR voting.

A single frame is often blurred or caught mid-motion, so one OCR reading is noisy.
Each tracked plate instead collects a few readings and the displayed text is a
confidence-weighted vote over them; once a track has ``min_votes`` readings, OCR
stops for it (it is the slow part: ~0.5-2 s per crop on CPU).

OCR runs on a tracked box only when ALL three hold:
    * frame_idx % ocr_every_n == 0
    * box area > min_box_area           (tiny, far-away plates are unreadable)
    * the track has fewer than min_votes readings

Crop / preprocess / OCR / normalization are the image pipeline's own functions
(``detect.pad_box`` + ``PlateOCRPipeline.recognize_crop`` -> ``ocr.read_plate``,
``ocr.normalize_vn_plate``).

CLI:
    python -m src.video_pipeline --input x.mp4 --output y.mp4 --weights models/best.pt [--gt 59H1-23456]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from tqdm import tqdm

from src.detect import Detection, pad_box
from src.evaluate import compact, exact_match
from src.ocr import normalize_vn_plate
from src.pipeline import DEFAULT_WEIGHTS, PlateOCRPipeline

COLOR_CONFIRMED = (0, 200, 0)    # BGR green: track has min_votes readings
COLOR_TENTATIVE = (150, 150, 150)  # gray: provisional text (or nothing read yet)
COLOR_TEXT = (255, 255, 255)

Reading = tuple[str, float]  # (plate_text, ocr_conf)


class _Tracker(Protocol):
    def track(self, frame: np.ndarray, persist: bool = True) -> list[tuple[int, Detection]]: ...

    def reset_tracker(self) -> None: ...


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VoteResult:
    text: str
    weight: float      # sum of confidences of the winning readings
    share: float       # weight / total weight of all readings
    n_votes: int       # readings for the winning text
    mean_conf: float
    valid: bool        # winning text matches a VN plate format


def weighted_vote(readings: list[Reading]) -> VoteResult | None:
    """Confidence-weighted majority over whole plate strings.

    1. readings are grouped by their separator-free form (``51F-12345`` == ``51F12345``)
    2. if any group matches a VN plate format, only those groups compete
    3. the group with the highest sum of confidences wins
    4. ties -> highest mean confidence (then more readings, then alphabetical, for determinism)

    Empty readings are ignored. Returns ``None`` when there is nothing to vote on.
    """
    groups: dict[str, list[Reading]] = defaultdict(list)
    for text, conf in readings:
        key = compact(text)
        if key:
            groups[key].append((text, float(conf)))
    if not groups:
        return None

    stats = []
    for key, items in groups.items():
        texts = [t for t, _ in items]
        display = max(set(texts), key=lambda t: (texts.count(t), t))  # most frequent spelling
        weight = sum(c for _, c in items)
        stats.append({"key": key, "text": display, "weight": weight, "n": len(items),
                      "mean": weight / len(items), "valid": normalize_vn_plate(display)[1]})

    pool = [s for s in stats if s["valid"]] or stats
    best = max(pool, key=lambda s: (round(s["weight"], 9), round(s["mean"], 9), s["n"], s["key"]))
    total = sum(s["weight"] for s in stats)
    return VoteResult(best["text"], best["weight"], best["weight"] / total if total else 0.0,
                      best["n"], best["mean"], best["valid"])


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
class VideoPlateOCR:
    """Plate recognition on video: ByteTrack tracking + confidence-weighted voting per track."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        ocr_every_n: int = 3,
        min_votes: int = 3,
        min_box_area: int = 1500,
        *,
        conf: float = 0.25,
        use_rectify: bool = False,
        device: str | None = None,
        detector: _Tracker | None = None,
    ) -> None:
        """
        Args:
            weights: YOLO weights (both classes of the plate model are plates).
            ocr_every_n: OCR only on frames where ``frame_idx % ocr_every_n == 0``.
            min_votes: Readings per track after which OCR stops and the text is final.
            min_box_area: Minimum box area (px^2) for OCR.
            conf: Detection confidence threshold.
            use_rectify: Perspective rectification before OCR (see ``src/rectify.py``).
            device: ``"cpu"``, ``"cuda:0"``... ``None`` lets Ultralytics choose.
            detector: Inject an object with ``track(frame, persist)`` and ``reset_tracker()`` (tests);
                ``weights`` is then ignored.
        """
        if ocr_every_n < 1 or min_votes < 1:
            raise ValueError("ocr_every_n and min_votes must be >= 1")
        self.ocr_every_n = ocr_every_n
        self.min_votes = min_votes
        self.min_box_area = min_box_area
        # Reuse the image pipeline (crop padding, rectify, read_plate, normalize_vn_plate).
        self.pipeline = PlateOCRPipeline(weights, conf=conf, use_rectify=use_rectify, device=device,
                                         detector=detector)
        self.detector: _Tracker = self.pipeline.detector
        self.votes: dict[int, list[Reading]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    def should_ocr(self, frame_idx: int, track_id: int, box: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = box
        return (frame_idx % self.ocr_every_n == 0
                and (x2 - x1) * (y2 - y1) > self.min_box_area
                and len(self.votes[track_id]) < self.min_votes)

    def vote(self, track_id: int) -> VoteResult | None:
        """Current decision for a track (see :func:`weighted_vote`)."""
        return weighted_vote(self.votes.get(track_id, []))

    def is_confirmed(self, track_id: int) -> bool:
        return len(self.votes.get(track_id, [])) >= self.min_votes

    def read_track(self, frame: np.ndarray, track_id: int, det: Detection) -> Reading:
        """Crop (with the pipeline's padding) -> recognize_crop -> add the reading to ``votes``."""
        h, w = frame.shape[:2]
        padded = pad_box(det.box, w, h, self.pipeline.pad)
        if padded is None:
            return "", 0.0
        x1, y1, x2, y2 = padded
        rec = self.pipeline.recognize_crop(frame[y1:y2, x1:x2])
        reading = (rec["plate_text"], rec["ocr_conf"])
        self.votes[track_id].append(reading)
        return reading

    # ------------------------------------------------------------------ #
    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None,
        max_frames: int | None = None,
        gt: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Track, OCR and vote over a video; write the annotated video to ``output_path``.

        Returns:
            ``total_frames, processing_fps, unique_tracks, plates_read, ocr_calls_made,
            ocr_calls_saved`` (+ ``single_frame_accuracy, voted_accuracy`` with ``gt``)
            and per-track details under ``tracks``.

            * plates_read      tracks that reached ``min_votes`` readings AND have a voted text
            * ocr_calls_saved  tracked boxes NOT sent to OCR, vs. OCR on every box of every frame
            * with ``gt`` (one plate or a list), over all tracks that got >= 1 reading:
              single_frame_accuracy = correct individual readings / readings,
              voted_accuracy        = tracks whose voted text is correct / tracks.
              A text is correct if it equals one of the GT plates, separators ignored.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if max_frames is not None:
            n_total = min(n_total, max_frames) if n_total else max_frames

        self.votes = defaultdict(list)
        self.detector.reset_tracker()  # fresh track IDs for every video
        tracks: dict[int, dict[str, Any]] = {}
        writer = None
        frame_idx = ocr_calls = tracked_boxes = 0
        t0 = time.perf_counter()
        try:
            with tqdm(total=n_total, desc=Path(str(video_path)).name, unit="frame") as bar:
                while max_frames is None or frame_idx < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if writer is None and output_path is not None:
                        writer = _open_writer(output_path, fps, frame.shape[1], frame.shape[0])

                    # always persist=True (see PlateDetector.track); reset happened above
                    tracked = self.detector.track(frame, persist=True)
                    for tid, det in tracked:
                        tracked_boxes += 1
                        info = tracks.setdefault(tid, {"first_frame": frame_idx, "max_area": 0})
                        info["last_frame"] = frame_idx
                        x1, y1, x2, y2 = det.box
                        info["max_area"] = max(info["max_area"], (x2 - x1) * (y2 - y1))
                        if self.should_ocr(frame_idx, tid, det.box):
                            self.read_track(frame, tid, det)
                            ocr_calls += 1

                    if writer is not None:
                        writer.write(self.draw(frame, tracked))
                    frame_idx += 1
                    bar.update(1)
                    bar.set_postfix(tracks=len(tracks), ocr=ocr_calls, refresh=False)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        elapsed = time.perf_counter() - t0

        track_rows = {}
        for tid, info in sorted(tracks.items()):
            v = self.vote(tid)
            track_rows[tid] = {
                **info,
                "text": v.text if v else "",
                "valid_format": v.valid if v else False,
                "vote_share": round(v.share, 3) if v else 0.0,
                "mean_conf": round(v.mean_conf, 3) if v else 0.0,
                "n_readings": len(self.votes.get(tid, [])),
                "confirmed": self.is_confirmed(tid),
                "readings": [[t, round(c, 3)] for t, c in self.votes.get(tid, [])],
            }

        result: dict[str, Any] = {
            "total_frames": frame_idx,
            "processing_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
            "unique_tracks": len(tracks),
            "plates_read": sum(r["confirmed"] and bool(r["text"]) for r in track_rows.values()),
            "ocr_calls_made": ocr_calls,
            "ocr_calls_saved": tracked_boxes - ocr_calls,
        }
        if gt:
            result.update(self.accuracy(gt))
        result["plates"] = sorted({r["text"] for r in track_rows.values() if r["confirmed"] and r["text"]})
        result["tracks"] = track_rows
        return result

    def accuracy(self, gt: str | list[str]) -> dict[str, float | None]:
        """single-frame vs voted accuracy over every track with >= 1 reading (see process_video)."""
        gts = [gt] if isinstance(gt, str) else list(gt)

        def correct(text: str) -> bool:
            return any(exact_match(text, g) for g in gts)

        read = {tid: r for tid, r in self.votes.items() if r}
        readings = [t for r in read.values() for t, _ in r]
        voted = [self.vote(tid) for tid in read]
        return {
            "single_frame_accuracy": (round(sum(map(correct, readings)) / len(readings), 4)
                                      if readings else None),
            "voted_accuracy": (round(sum(v is not None and correct(v.text) for v in voted) / len(voted), 4)
                               if voted else None),
        }

    # ------------------------------------------------------------------ #
    def draw(self, frame: np.ndarray, tracked: list[tuple[int, Detection]]) -> np.ndarray:
        """Box + ``#id text``: green once the track is confirmed, gray while provisional."""
        out = frame.copy()
        h, w = out.shape[:2]
        scale = max(0.5, min(w, h) / 900)
        thick = max(1, round(2 * scale))
        font = cv2.FONT_HERSHEY_SIMPLEX
        for tid, det in tracked:
            x1, y1, x2, y2 = det.box
            v = self.vote(tid)
            color = COLOR_CONFIRMED if self.is_confirmed(tid) else COLOR_TENTATIVE
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + 1)

            label = f"#{tid} {v.text if v else '...'}"
            (tw, th), base = cv2.getTextSize(label, font, scale, thick)
            pad = max(2, thick * 2)
            box_h = th + base + 2 * pad
            top = y1 - box_h if y1 - box_h >= 0 else y1
            left = min(max(0, x1), max(0, w - tw - 2 * pad))
            cv2.rectangle(out, (left, top), (left + tw + 2 * pad, top + box_h), color, cv2.FILLED)
            cv2.putText(out, label, (left + pad, top + pad + th), font, scale, COLOR_TEXT, thick, cv2.LINE_AA)
        return out


def _open_writer(path: str | Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Cannot open VideoWriter for {path} (mp4v). On Windows, avoid non-ASCII characters "
                      "in the output path.")
    return writer


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plate recognition on video with ByteTrack + OCR voting")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="Annotated .mp4 (mp4v)")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--gt", nargs="+", default=None, metavar="PLATE",
                   help='Ground-truth plate(s) in the video, e.g. --gt "59H1-23456"')
    p.add_argument("--ocr-every-n", type=int, default=3)
    p.add_argument("--min-votes", type=int, default=3)
    p.add_argument("--min-box-area", type=int, default=1500)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--rectify", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu, cuda:0, ... (default: auto)")
    p.add_argument("--json", type=Path, default=None, help="Also save the full result dict (with tracks)")
    args = p.parse_args(argv)

    if not args.input.is_file():
        sys.exit(f"Video not found: {args.input}")
    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")

    t0 = time.perf_counter()
    vp = VideoPlateOCR(args.weights, args.ocr_every_n, args.min_votes, args.min_box_area,
                       conf=args.conf, use_rectify=args.rectify, device=args.device)
    vp.pipeline.warmup()
    print(f"models loaded in {time.perf_counter() - t0:.1f} s")

    res = vp.process_video(args.input, args.output, max_frames=args.max_frames, gt=args.gt)

    for key in ("total_frames", "processing_fps", "unique_tracks", "plates_read",
                "ocr_calls_made", "ocr_calls_saved", "single_frame_accuracy", "voted_accuracy"):
        if key in res:
            print(f"{key:<22}: {res[key]}")
    print("tracks:")
    for tid, t in res["tracks"].items():
        if t["n_readings"]:
            state = "final" if t["confirmed"] else "tentative"
            print(f"  #{tid:<4} {t['text'] or '-':<12} {state:<9} share={t['vote_share']:.2f} "
                  f"readings={[r[0] for r in t['readings']]}")
    skipped = sum(1 for t in res["tracks"].values() if not t["n_readings"])
    if skipped:
        print(f"  ({skipped} track(s) never OCR'd: box <= {args.min_box_area} px^2 or too short)")
    if args.output:
        print(f"output: {args.output}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"json  : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
