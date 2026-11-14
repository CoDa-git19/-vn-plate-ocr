# Prompts used during development

VN-PlateOCR was built with an AI coding assistant (Claude Code). The prompts were written in Vietnamese; below is a short English summary per phase, in build order. Constraints repeated in most prompts: read the existing modules and reuse them instead of rewriting, the YOLO model has two classes that are both license plates, and everything must run on a Windows laptop CPU.

## 1. Project scaffold and data

Scaffold the package layout (`src/`, `scripts/`, `tests/`), then write `scripts/prepare_data.py`: download the Roboflow dataset, merge several datasets with MD5 de-duplication, validate labels and plot EDA figures (sample grid, box-size distribution).

## 2. Detector training

Write `scripts/train.py` to fine-tune COCO-pretrained YOLOv8 on Colab, print mAP50 / mAP50-95 / precision / recall of `best.pt` on the validation split, and copy the weights to `models/best_<model>.pt`. A follow-up relaxed dependency pins for newer Python versions.

## 3. OCR module

Write `src/ocr.py`: `read_plate()` runs EasyOCR on the raw crop and on an upscaled + CLAHE + thresholded copy and keeps the more confident reading; `normalize_vn_plate()` fixes OCR confusions by character position and validates one-line and two-line Vietnamese formats, with the series letter restricted to the 20 civilian letters.

## 4. Image pipeline

Write `src/pipeline.py` (`PlateOCRPipeline`): detect, crop with 5% padding, OCR, normalize; return one dict per plate with confidences, validity and the YOLO class id; time each stage; draw green/orange boxes; never crash on images without plates; add a CLI and explain why two stages beat whole-image OCR.

## 5. Perspective rectification

Write `src/rectify.py`: Canny edges, largest contour, `minAreaRect` corners, ordered points, perspective warp, with a fallback to the original crop when no contour is found, the plate is already straight (< 5 degrees), the contour is too small (< 30% of the crop) or the warped aspect ratio is implausible. Then call it from the pipeline, changing nothing else.

## 6. Labelling tool

Write `scripts/label_tool.py` for the private manual test set: show each image with matplotlib (no `cv2.imshow`), type the plate and numeric tags in the terminal, append to a UTF-8 CSV after every image, resume where it stopped, and store portable relative paths. Optimized for labelling 50 images in 20 minutes.

## 7. Evaluation harness

Extend `src/evaluate.py` into an ablation harness: four cumulative configurations (baseline, + preprocess, + rectify, + rules), per-tag slices with sample counts and a warning below 5 samples, `--compare` for several weights with mAP50 and median latency, a failure-case figure with optional masking of plate characters, and raw results in `docs/results.json`. Score the closest prediction when an image has several plates.

## 8. Video pipeline

Write `src/video_pipeline.py` (`VideoPlateOCR`): ByteTrack tracking, OCR only every N frames on large enough boxes until a track has enough votes, a confidence-weighted vote that prefers valid formats, green/gray labels, and statistics including OCR calls saved and single-frame vs voted accuracy against `--gt`.

## 9. Gradio demo

Write `app.py` with Image and Video tabs (upload or webcam, confidence slider, rectification switch, results table, latency, examples; video sliders, JSON statistics, a warning for clips over 15 s and a 300-frame cap). Load the models once, read weights from `PLATE_WEIGHTS`, convert RGBA to RGB and return friendly error messages.

## 10. Static project page

Create `hf-static/index.html` for a static Hugging Face Space: description, GitHub link, demo image and GIF, pipeline diagram, the result tables as HTML and PowerShell commands to run locally. Inline responsive CSS only, no JavaScript or CDN, with a note that live inference runs locally.

## 11. Latency benchmark

Write `scripts/benchmark.py`: export ONNX (opset 12, simplified, static input), report median and p95 (not mean) for PyTorch CPU, ONNX Runtime CPU and PyTorch GPU if available, time detect / rectify / OCR separately to find the bottleneck, and warn if the two backends disagree on box count or by more than 1 px on the same images.

## 12. Documentation

Write this README in English from the measured numbers in `docs/results_draft.md`, with a `[FILL]` placeholder only where a number or link is still missing, commands for both bash and PowerShell, an honest Limitations section, and this prompt summary.
