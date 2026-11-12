"""Video pipeline: detect + track (ByteTrack) + per-track OCR voting.

Running OCR on every frame is noisy; instead each track accumulates readings
and the final plate text is decided by confidence-weighted voting.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import Box
from src.pipeline import PipelineConfig, PlatePipeline


@dataclass
class TrackState:
    """Accumulated information for one tracked plate."""

    track_id: int
    first_frame: int
    last_frame: int
    last_box: Box
    readings: list[tuple[str, float]] = field(default_factory=list)
    final_text: str = ""
    final_conf: float = 0.0


class PlateVoter:
    """Aggregates OCR readings per track."""

    def __init__(self, min_readings: int = 3, max_readings: int = 30, min_conf: float = 0.3) -> None:
        """
        Args:
            min_readings: Readings required before a track is considered stable.
            max_readings: Stop OCR on a track after this many readings (saves compute).
            min_conf: Ignore readings below this OCR confidence.
        """
        self.min_readings = min_readings
        self.max_readings = max_readings
        self.min_conf = min_conf
        self.readings: dict[int, list[tuple[str, float]]] = defaultdict(list)

    def add(self, track_id: int, text: str, conf: float) -> None:
        """Record a reading for ``track_id``."""
        # TODO
        raise NotImplementedError

    def needs_ocr(self, track_id: int) -> bool:
        """Whether this track still benefits from more OCR readings."""
        # TODO
        raise NotImplementedError

    def vote(self, track_id: int, method: str = "char") -> tuple[str, float]:
        """Decide the plate text for a track.

        Args:
            method: ``"string"`` — weighted majority over whole strings;
                ``"char"`` — per-position weighted vote among readings of the
                modal length.

        Returns:
            ``(text, agreement_score)``.
        """
        # TODO
        raise NotImplementedError


def iter_frames(source: str | Path | int, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_idx, frame)`` from a video file or webcam index."""
    # TODO: cv2.VideoCapture, skip by stride, release on exit
    raise NotImplementedError


def process_video(
    source: str | Path | int,
    output_path: str | Path | None = None,
    config: PipelineConfig | None = None,
    stride: int = 1,
    vote_method: str = "char",
    max_frames: int | None = None,
) -> dict[int, TrackState]:
    """Process a video end to end.

    Args:
        source: Video path or webcam index.
        output_path: If given, write an annotated ``.mp4``.
        config: Pipeline configuration.
        stride: Process every ``stride``-th frame.
        vote_method: See :meth:`PlateVoter.vote`.
        max_frames: Stop early (useful for demos/tests).

    Returns:
        Mapping ``track_id → TrackState`` with ``final_text`` filled in.
    """
    # TODO:
    #   pipeline = PlatePipeline(config); voter = PlateVoter()
    #   for idx, frame in iter_frames(...):
    #       for tid, det in pipeline.detector.track(frame):
    #           if voter.needs_ocr(tid): crop → pipeline.recognize_crop → voter.add
    #       draw current voted text, write frame
    #   finalize tracks with voter.vote
    raise NotImplementedError


def draw_tracks(frame: np.ndarray, tracks: dict[int, TrackState]) -> np.ndarray:
    """Draw track boxes, IDs and current voted text on a frame."""
    # TODO
    raise NotImplementedError


def export_tracks_csv(tracks: dict[int, TrackState], path: str | Path) -> None:
    """Save per-track results (id, text, conf, first/last frame) to CSV."""
    # TODO
    raise NotImplementedError


__all__ = ["PlatePipeline", "PlateVoter", "TrackState", "process_video"]
