## Ablation - models/best.pt (n=45)

| Configuration | Exact Match | CER | Detection Recall | Delta EM vs previous |
|---|---:|---:|---:|---:|
| 1. baseline | 6.7% | 0.282 | 100.0% | - |
| 2. + preprocess | 6.7% | 0.255 | 100.0% | +0.0 pts |
| 3. + rectify | 6.7% | 0.240 | 100.0% | +0.0 pts |
| 4. + rules | 15.6% | 0.224 | 100.0% | +8.9 pts |

## Slices - + rules - models/best.pt

| Slice | n | Exact Match |
|---|---:|---:|
| two_line | 15 | 6.7% |
| angled | 11 | 18.2% |
| lowlight | 3 (!) | 33.3% |
| small | 6 | 0.0% |
| blurry | 16 | 12.5% |
| clean | 12 | 25.0% |

Note: lowlight has fewer than 5 samples (not reliable).
