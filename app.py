"""Gradio demo: image and video tabs.

Usage:
    python app.py            # http://127.0.0.1:7860
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from src.pipeline import PipelineConfig, PlatePipeline

TITLE = "🇻🇳 Vietnamese License Plate Recognition"
DESCRIPTION = "YOLOv8 detection → perspective rectification → EasyOCR, with ByteTrack voting for video."

_pipeline: PlatePipeline | None = None


def get_pipeline(use_rectify: bool = True) -> PlatePipeline:
    """Lazily build (and cache) the pipeline; rebuild if config changes."""
    # TODO
    raise NotImplementedError


def predict_image(image: np.ndarray, use_rectify: bool, det_conf: float) -> tuple[np.ndarray, list[list]]:
    """Gradio handler for the image tab.

    Args:
        image: RGB image from ``gr.Image(type="numpy")``.

    Returns:
        ``(annotated_rgb_image, table_rows)`` where rows are
        ``[plate_text, det_conf, ocr_conf, rectified]``.
    """
    # TODO: RGB→BGR, pipeline.process_image, draw_results, BGR→RGB
    raise NotImplementedError


def predict_video(video_path: str, stride: int) -> tuple[str, list[list]]:
    """Gradio handler for the video tab → ``(annotated_video_path, track_rows)``."""
    # TODO: video_pipeline.process_video(video_path, output_path=..., stride=stride)
    raise NotImplementedError


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

        with gr.Tab("Image"):
            with gr.Row():
                img_in = gr.Image(type="numpy", label="Input")
                img_out = gr.Image(type="numpy", label="Result")
            with gr.Row():
                rectify_cb = gr.Checkbox(value=True, label="Perspective rectification")
                conf_sl = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Detection confidence")
            img_table = gr.Dataframe(headers=["plate", "det_conf", "ocr_conf", "rectified"], label="Plates")
            gr.Button("Run").click(predict_image, [img_in, rectify_cb, conf_sl], [img_out, img_table])
            # TODO: gr.Examples(examples=[...], inputs=[img_in])

        with gr.Tab("Video"):
            with gr.Row():
                vid_in = gr.Video(label="Input")
                vid_out = gr.Video(label="Result")
            stride_sl = gr.Slider(1, 10, value=2, step=1, label="Frame stride")
            vid_table = gr.Dataframe(headers=["track_id", "plate", "agreement", "frames"], label="Tracks")
            gr.Button("Run").click(predict_video, [vid_in, stride_sl], [vid_out, vid_table])

    return demo


if __name__ == "__main__":
    build_demo().queue().launch()
