# vn-plate-ocr

Detect and read Vietnamese license plates in images and videos.

**YOLOv8** (plate detection) → **perspective rectification** → **EasyOCR** (text) → VN-plate post-processing.
For video, **ByteTrack** tracking + per-track OCR voting gives one stable reading per vehicle.

> 🚧 Work in progress — results tables below are filled in by `src/evaluate.py` and `scripts/benchmark.py`.

## Pipeline

```
image/frame ─► YOLOv8 detect ─► crop ─► rectify (4-point warp) ─► EasyOCR ─► normalize + regex fix ─► plate text
                    │
                    └─(video)─► ByteTrack IDs ─► per-track voting ─► final plate per vehicle
```

## Project structure

```
src/
  detect.py          YOLOv8 wrapper: detect_plates(image) -> [(box, conf)]
  rectify.py         rectify_plate(crop) -> (warped, ok)
  ocr.py             read_plate(crop) -> (text, conf), VN plate normalization
  pipeline.py        detect + rectify + OCR for still images
  video_pipeline.py  tracking + voting for video
  evaluate.py        metrics, ablation, per-slice breakdown
scripts/
  prepare_data.py    raw annotations -> YOLO format
  train.py           fine-tune YOLOv8 (+ W&B, ONNX export)
  benchmark.py       latency: PyTorch vs ONNX Runtime
  label_tool.py      label OCR ground truth for crops
tests/               pytest unit tests
app.py               Gradio demo
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Usage

```bash
# 1. Data
python scripts/prepare_data.py --src datasets/raw --out datasets/plates

# 2. Train detector
python scripts/train.py --data datasets/plates/data.yaml --epochs 50 --wandb --export-onnx

# 3. Evaluate (ablation + slices)
python -m src.evaluate --gt datasets/test_gt.csv --ablation --slices

# 4. Benchmark
python scripts/benchmark.py --images datasets/plates/images/test \
    --weights weights/plate_yolov8n.pt weights/plate_yolov8n.onnx

# 5. Demo
python app.py
```

## Results

### Ablation (end-to-end, test set)

| Variant | Exact match | CER | FPS |
|---|---|---|---|
| baseline | – | – | – |
| + postprocess | – | – | – |
| + rectify | – | – | – |
| full | – | – | – |

### Per-slice (full pipeline)

| Slice | Exact match | CER |
|---|---|---|
| one-line / two-line | – | – |
| day / night | – | – |
| frontal / oblique | – | – |

### Latency (CPU)

| Backend | Detect (ms) | OCR (ms) | End-to-end FPS |
|---|---|---|---|
| PyTorch | – | – | – |
| ONNX Runtime | – | – | – |

## Limitations & next steps

- TODO

## Tests

```bash
pytest -q
ruff check .
```
