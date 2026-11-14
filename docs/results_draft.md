## Ablation - models/best.pt (n=45)

|Configuration|Exact Match|CER|Detection Recall|Delta EM vs previous|
|-|-:|-:|-:|-:|
|1. baseline|6.7%|0.282|100.0%|-|
|2. + preprocess|6.7%|0.255|100.0%|+0.0 pts|
|3. + rectify|6.7%|0.240|100.0%|+0.0 pts|
|4. + rules|15.6%|0.224|100.0%|+8.9 pts|

## Slices - + rules - models/best.pt

|Slice|n|Exact Match|
|-|-:|-:|
|two\_line|15|6.7%|
|angled|11|18.2%|
|lowlight|3 (!)|33.3%|
|small|6|0.0%|
|blurry|16|12.5%|
|clean|12|25.0%|

Note: lowlight has fewer than 5 samples (not reliable).

## Video - multi-frame voting

| Clip | Single-frame accuracy | Voted accuracy | OCR calls saved |
|---|---:|---:|---:|
| video1 | 0.0323 | 0.0833 | 485 |
| video2 | 0.0 | 0.0 | 481 |
| video3 | 0.0 | 0.0 | 580 |



## Detector latency - CPU (30 timed calls, 5 warm-up)

| Configuration | median ms | p95 ms | speedup vs PyTorch CPU |
|---|---:|---:|---:|
| PyTorch CPU | 196.3 | 455.7 | 1.00x |
| ONNX Runtime CPU | 231.5 | 574.1 | 0.85x |

Note: static ONNX always runs a 640x640 input; PyTorch letterboxes to the smallest multiple of 32 (fewer pixels).

## Pipeline stages per image (30 images, 47 plates)

| Stage | median ms | p95 ms | share of total time |
|---|---:|---:|---:|
| detect | 137.2 | 187.9 | 13% |
| rectify | 1.5 | 4.4 | 0% |
| ocr | 845.2 | 1755.3 | 87% |
| total | 982.7 | 1878.0 | 100% |

Bottleneck: OCR (87% of total time).

Correctness: PyTorch and ONNX Runtime give the same box count and < 1 px difference on identical input for 10/10 images.

## Detection on dataset test split (169 images, 172 plates)

| Model | Params | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 3.1M | 0.978 | 0.975 | 0.981 | 0.782 |
| YOLOv8s | 11.1M | 0.983 | 0.951 | 0.973 | 0.768 |

## Model comparison (manual test set, full pipeline, CPU)

| Weights | mAP50 (test) | Exact Match | Detect p50 (ms) | End-to-end p50 (ms) |
|---|---:|---:|---:|---:|
| best_yolov8n.pt | 0.981 | 15.6% | 206 | 1400 |
| best_yolov8s.pt | 0.973 | 17.8% | 654 | 2295 |

Model choice: YOLOv8n. Same or better detection (mAP50 0.981 vs 0.973) with 3.6x fewer parameters and about 3x faster detection on CPU. The +2.2 pt Exact Match of v8s is 1 image out of 45, within noise.
