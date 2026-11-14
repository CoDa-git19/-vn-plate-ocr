"""Gradio demo: Vietnamese license plate recognition on images and videos (CPU-friendly).

Usage:
    python app.py                                  # http://127.0.0.1:7860
    PLATE_WEIGHTS=models/best_yolov8n.pt python app.py
    (PowerShell: $env:PLATE_WEIGHTS="models/best_yolov8n.pt"; python app.py)

The models are loaded once at import time; handlers only reconfigure them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from src.pipeline import PlateOCRPipeline, draw_results
from src.video_pipeline import VideoPlateOCR

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
GITHUB_URL = "https://github.com/CoDa-git19/-vn-plate-ocr"
MAX_FRAMES = 300          # video frames processed per request (~10 s at 30 fps)
SLOW_VIDEO_SECONDS = 15   # warn above this duration
TABLE_HEADERS = ["Plate", "Det Conf", "OCR Conf", "Valid", "Rectified"]

TITLE = "🇻🇳 Vietnamese License Plate Recognition"
HEADER = f"""
# {TITLE}
Two-stage pipeline: **YOLOv8** finds the plates -> optional **perspective rectification** ->
**EasyOCR** reads the characters -> position-aware fixes + VN plate-format check.
Videos add **ByteTrack** tracking and a confidence-weighted vote over several readings per plate.

Runs on CPU (expect ~1 s per image, ~4-5 frames/s on video). Code: [GitHub]({GITHUB_URL})
"""


# --------------------------------------------------------------------------- #
# Models: loaded ONCE
# --------------------------------------------------------------------------- #
def _weights_path() -> Path:
    p = Path(os.environ.get("PLATE_WEIGHTS", "models/best.pt"))
    return p if p.is_absolute() else ROOT / p


WEIGHTS = _weights_path()
IMAGE_PIPE: PlateOCRPipeline | None = None
VIDEO_OCR: VideoPlateOCR | None = None
LOAD_ERROR: str | None = None
try:
    IMAGE_PIPE = PlateOCRPipeline(WEIGHTS, conf=0.25)
    # Separate YOLO instance for video: Ultralytics attaches tracker callbacks to the model's
    # predictor on model.track(), which would otherwise leak into still-image detection.
    VIDEO_OCR = VideoPlateOCR(WEIGHTS)
    IMAGE_PIPE.warmup()  # loads EasyOCR (shared by both) + first YOLO call
except Exception as e:  # keep the UI up and explain instead of crashing at startup
    traceback.print_exc()
    LOAD_ERROR = (f"Could not load the models from `{WEIGHTS}` ({type(e).__name__}: {e}). "
                  "Set the PLATE_WEIGHTS environment variable to a valid .pt file.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_rgb(image: np.ndarray) -> np.ndarray:
    """Gray / RGBA / RGB -> RGB uint8. Transparent pixels are composited on white."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        rgb = image[..., :3].astype(np.float32)
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_HEADERS)


def _browser_playable(path: Path) -> tuple[Path, str]:
    """Re-encode the mp4v output to H.264 so browsers can play it (needs ffmpeg on PATH)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return path, "ffmpeg not found: the result is MPEG-4 (mp4v) - download it if the player stays black."
    out = path.with_name(path.stem + "_h264.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        path.unlink(missing_ok=True)
        return out, ""
    except (subprocess.SubprocessError, OSError) as e:
        return path, f"H.264 re-encoding failed ({type(e).__name__}); returning the mp4v file."


def _video_info(path: str | None) -> tuple[float, float, int] | None:
    """``(duration_s, fps, frames)`` or ``None`` if unreadable."""
    if not path:
        return None
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frames / fps, fps, frames
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def predict_image(image: np.ndarray | None, conf: float, use_rectify: bool):
    """-> (annotated RGB image, plate table, latency markdown)."""
    if LOAD_ERROR:
        return None, _empty_table(), f"⚠️ {LOAD_ERROR}"
    if image is None:
        return None, _empty_table(), "Upload an image or take a webcam snapshot, then press **Run**."
    try:
        rgb = to_rgb(image)
        IMAGE_PIPE.detector.conf_thres = float(conf)
        IMAGE_PIPE.use_rectify = bool(use_rectify)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = IMAGE_PIPE.process(bgr)
        annotated = cv2.cvtColor(draw_results(bgr, results), cv2.COLOR_BGR2RGB)

        rows = [[r["plate_text"] or r["raw_text"] or "(unreadable)", round(r["det_conf"], 3),
                 round(r["ocr_conf"], 3), "✅" if r["is_valid_format"] else "❌",
                 "yes" if r["rectified"] else "no"] for r in results]
        lat = IMAGE_PIPE.last_latency_ms
        if results:
            md = (f"**{len(results)} plate(s)** · detect {lat['detect']:.0f} ms · rectify {lat['rectify']:.0f} ms"
                  f" · OCR {lat['ocr']:.0f} ms · **total {lat['total']:.0f} ms** (CPU)")
        else:
            md = (f"No plate found (detect {lat['detect']:.0f} ms). "
                  "Try a lower confidence or a closer, sharper photo.")
        return annotated, pd.DataFrame(rows, columns=TABLE_HEADERS), md
    except Exception as e:
        traceback.print_exc()
        return None, _empty_table(), f"⚠️ Something went wrong while processing this image ({type(e).__name__}: {e})."


def video_notice(path: str | None) -> str:
    """Shown as soon as a video is chosen: duration + CPU warning."""
    info = _video_info(path)
    if info is None:
        return f"Only the first **{MAX_FRAMES} frames** are processed."
    duration, fps, frames = info
    used = min(frames, MAX_FRAMES)
    msg = (f"Video: {duration:.1f} s, {fps:.0f} fps, {frames} frames -> processing the first "
           f"**{used} frames** ({used / fps:.1f} s).")
    if duration > SLOW_VIDEO_SECONDS:
        msg = (f"⚠️ This video is longer than {SLOW_VIDEO_SECONDS} s: processing is slow on CPU "
               f"(~4-5 frames/s, i.e. about {used / 4.5:.0f} s here).\n\n" + msg)
    return msg


def predict_video(video_path: str | None, ocr_every_n: int, min_votes: int,
                  progress=gr.Progress(track_tqdm=True)):  # noqa: B008 - Gradio injects the tracker
    """-> (annotated video path, stats JSON, notice markdown)."""
    if LOAD_ERROR:
        return None, None, f"⚠️ {LOAD_ERROR}"
    if not video_path:
        return None, None, "Upload a video (or pick the example), then press **Run**."
    try:
        VIDEO_OCR.ocr_every_n = int(ocr_every_n)
        VIDEO_OCR.min_votes = int(min_votes)
        out = Path(tempfile.mkdtemp(prefix="plate_video_")) / "result.mp4"
        res = VIDEO_OCR.process_video(video_path, out, max_frames=MAX_FRAMES)
        playable, note = _browser_playable(out)

        stats = {k: v for k, v in res.items() if k != "tracks"}
        stats["tracks"] = [
            {"id": tid, "plate": t["text"], "final": t["confirmed"], "readings": t["n_readings"],
             "frames": [t["first_frame"], t["last_frame"]]}
            for tid, t in res["tracks"].items() if t["n_readings"]
        ]
        md = video_notice(video_path) + (f"\n\n{note}" if note else "")
        return str(playable), stats, md
    except Exception as e:
        traceback.print_exc()
        return None, None, f"⚠️ Something went wrong while processing this video ({type(e).__name__}: {e})."


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_demo() -> gr.Blocks:
    image_examples = [str(EXAMPLES / f) for f in ("1.jpg", "2.png", "crop_angled.jpg") if (EXAMPLES / f).is_file()]
    video_examples = [str(EXAMPLES / "video1.mp4")] if (EXAMPLES / "video1.mp4").is_file() else []

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(HEADER)
        if LOAD_ERROR:
            gr.Markdown(f"> ⚠️ **{LOAD_ERROR}**")

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    img_in = gr.Image(sources=["upload", "webcam"], type="numpy", label="Input image")
                    img_out = gr.Image(type="numpy", label="Result", interactive=False)
                with gr.Row():
                    conf_sl = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Detection confidence")
                    rectify_cb = gr.Checkbox(value=False, label="Perspective rectification")
                img_btn = gr.Button("Run", variant="primary")
                img_md = gr.Markdown()
                img_table = gr.Dataframe(value=_empty_table(), headers=TABLE_HEADERS, label="Plates",
                                         interactive=False)
                if image_examples:
                    gr.Examples(examples=image_examples, inputs=[img_in], label="Examples")
                img_btn.click(predict_image, [img_in, conf_sl, rectify_cb], [img_out, img_table, img_md])

            with gr.Tab("Video"):
                gr.Markdown(f"Videos longer than {SLOW_VIDEO_SECONDS} s are slow on CPU; only the first "
                            f"{MAX_FRAMES} frames are processed.")
                with gr.Row():
                    vid_in = gr.Video(label="Input video", sources=["upload"])
                    vid_out = gr.Video(label="Result", interactive=False)
                with gr.Row():
                    every_sl = gr.Slider(1, 10, value=3, step=1, label="OCR every N frames")
                    votes_sl = gr.Slider(1, 5, value=3, step=1, label="Votes needed per plate")
                vid_btn = gr.Button("Run", variant="primary")
                vid_md = gr.Markdown()
                vid_json = gr.JSON(label="Statistics")
                if video_examples:
                    gr.Examples(examples=video_examples, inputs=[vid_in], label="Example")
                vid_in.change(video_notice, vid_in, vid_md)
                vid_btn.click(predict_video, [vid_in, every_sl, votes_sl], [vid_out, vid_json, vid_md])

    return demo


demo = build_demo()

if __name__ == "__main__":
    # one request at a time: the models (and the video tracker state) are shared globals
    demo.queue(default_concurrency_limit=1)
    demo.launch()
